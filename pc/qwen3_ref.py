"""Qwen3 in NumPy, step by step as the PS4 GPU will do it (one token at a time, KV cache, optional Q8 weights).

It is the yardstick for the GPU: same operations, same order, same weight format. It is itself checked
against Hugging Face Transformers.

Run: python qwen3_ref.py            -> compare with Transformers and print a short greedy generation
"""

import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B"


class LazyWeights(dict):
    """Tensor name -> float32 array, read from the safetensors files only when asked (a 4B model is 16 GB in FP32)."""

    def __init__(self, model_dir):
        from safetensors import safe_open
        super().__init__()
        self.files = {}
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            h = safe_open(str(f), framework="pt")
            for k in h.keys():
                self.files[k] = h

    def __getitem__(self, k):
        return self.files[k].get_tensor(k).float().numpy()

    def get(self, k, default=None):
        return self[k] if k in self.files else default

    def __contains__(self, k):
        return k in self.files


def load_weights(model_dir=MODEL):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    return LazyWeights(model_dir), cfg


def quantize_q8(w):
    """Per 32 consecutive weights of a row: int8 values and an FP32 scale (max |w| / 127)."""
    blocks = w.reshape(w.shape[0], -1, 32)
    scale = (np.abs(blocks).max(axis=2) / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(blocks / scale[..., None]), -127, 127).astype(np.int8)
    return q.reshape(w.shape), scale


def dequantize_q8(q, scale):
    return (q.reshape(q.shape[0], -1, 32).astype(np.float32) * scale[..., None]).reshape(q.shape)


class Q4Matrix:
    """A Q4 weight kept packed; dequantized only while it is used."""

    def __init__(self, w):
        from q4 import quantize_q4
        self.packed, self.d16 = quantize_q4(w)

    def __matmul__(self, x):
        from q4 import dequantize_q4
        return dequantize_q4(self.packed, self.d16) @ x

    def row(self, i):
        from q4 import dequantize_q4
        return dequantize_q4(self.packed[i:i + 1], self.d16[i:i + 1])[0]


class Qwen3:
    def __init__(self, weights, cfg, q8=False, max_seq=2048, q4=False):
        self.cfg = cfg
        self.n_layers, self.hidden = cfg["num_hidden_layers"], cfg["hidden_size"]
        self.n_heads, self.n_kv, self.head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        lin = Q4Matrix if q4 else (lambda w: dequantize_q8(*quantize_q8(w))) if q8 else (lambda w: w)
        g = weights.get
        self.embed = lin(weights["model.embed_tokens.weight"])
        self.lm_head = self.embed if cfg["tie_word_embeddings"] else lin(weights["lm_head.weight"])
        self.final_norm = weights["model.norm.weight"]
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            self.layers.append(dict(
                attn_norm=g(p + "input_layernorm.weight"), mlp_norm=g(p + "post_attention_layernorm.weight"),
                q_norm=g(p + "self_attn.q_norm.weight"), k_norm=g(p + "self_attn.k_norm.weight"),
                wqkv=lin(np.concatenate([g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"),
                                         g(p + "self_attn.v_proj.weight")])),
                wo=lin(g(p + "self_attn.o_proj.weight")),
                wgu=lin(np.concatenate([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")])),
                wdown=lin(g(p + "mlp.down_proj.weight"))))
        half = self.head_dim // 2
        inv = cfg["rope_theta"] ** (-np.arange(half, dtype=np.float64) * 2 / self.head_dim)
        ang = np.arange(max_seq, dtype=np.float64)[:, None] * inv[None, :]
        self.cos, self.sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        self.kcache = np.zeros((self.n_layers, max_seq, self.n_kv, self.head_dim), np.float32)
        self.vcache = np.zeros_like(self.kcache)

    def rmsnorm(self, x, w):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps) * w

    def rope(self, x, pos):           # x [heads, head_dim], rotate_half pairs (i, i + head_dim / 2)
        h = self.head_dim // 2
        c, s = self.cos[pos], self.sin[pos]
        a, b = x[:, :h], x[:, h:]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=1)

    def step(self, token, pos):
        x = self.embed.row(token) if isinstance(self.embed, Q4Matrix) else self.embed[token].astype(np.float32)
        nq, nkv, hd = self.n_heads, self.n_kv, self.head_dim
        for i, L in enumerate(self.layers):
            h = self.rmsnorm(x, L["attn_norm"])
            qkv = L["wqkv"] @ h
            q = qkv[:nq * hd].reshape(nq, hd)
            k = qkv[nq * hd:(nq + nkv) * hd].reshape(nkv, hd)
            v = qkv[(nq + nkv) * hd:].reshape(nkv, hd)
            q = self.rope(self.rmsnorm(q, L["q_norm"]), pos)
            k = self.rope(self.rmsnorm(k, L["k_norm"]), pos)
            self.kcache[i, pos], self.vcache[i, pos] = k, v
            out = np.empty((nq, hd), np.float32)
            for head in range(nq):
                kv = head // (nq // nkv)
                scores = self.kcache[i, :pos + 1, kv] @ q[head] / np.sqrt(hd)
                p = np.exp(scores - scores.max())
                out[head] = (p / p.sum()) @ self.vcache[i, :pos + 1, kv]
            x = x + L["wo"] @ out.reshape(-1)
            h = self.rmsnorm(x, L["mlp_norm"])
            gu = L["wgu"] @ h
            gate, up = gu[:len(gu) // 2], gu[len(gu) // 2:]
            x = x + L["wdown"] @ (gate / (1 + np.exp(-gate)) * up)
        return self.lm_head @ self.rmsnorm(x, self.final_norm)

    def generate(self, prompt_ids, n):
        logits = None
        for pos, t in enumerate(prompt_ids):
            logits = self.step(t, pos)
        out = []
        for i in range(n):
            t = int(np.argmax(logits))
            out.append(t)
            logits = self.step(t, len(prompt_ids) + i)
        return out


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of Peru is"
    ids = tok(prompt)["input_ids"]
    weights, cfg = load_weights()

    hf = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.float32)
    with torch.no_grad():
        hf_logits = hf(torch.tensor([ids])).logits[0, -1].numpy()
        hf_ids = hf.generate(torch.tensor([ids]), max_new_tokens=12, do_sample=False)[0, len(ids):].tolist()
    for q8 in (False, True):
        m = Qwen3(weights, cfg, q8=q8)
        logits = None
        for pos, t in enumerate(ids):
            logits = m.step(t, pos)
        err = np.max(np.abs(logits - hf_logits)) / np.max(np.abs(hf_logits))
        m = Qwen3(weights, cfg, q8=q8)
        gen = m.generate(ids, 12)
        print(f"{'Q8  ' if q8 else 'FP32'} reference: logits vs Transformers max rel diff {err:.2e}, "
              f"same top token {np.argmax(logits) == np.argmax(hf_logits)}; greedy {'matches' if gen == hf_ids else 'differs'}: "
              f"{tok.decode(gen)!r}")
    print(f"Transformers greedy: {tok.decode(hf_ids)!r}")


if __name__ == "__main__":
    main()
