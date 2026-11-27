"""NumPy reference for Maple-Preview (one token at a time), straight from the GGUF.

Follows deepgrove-ai/llama.cpp src/models/maple.cpp (saved in ../ref/maple.cpp):
  x = embed(token)
  per layer: h = rms(x) * attn_norm; q,k,v = Wq h, Wk h, Wv h; per-head rms norm on q and k (q_norm, k_norm);
             sliding layers (pattern True): NEOX RoPE on the first 64 of 128 dims, theta 10000, window 512;
             global layers: no RoPE, full context; GQA 16 q heads / 4 kv heads, scale 1/sqrt(128); x += Wo attn
             h = rms(x) * ffn_norm; p = softmax(Wr h) over 256 experts; top-8; w = p / sum(p_top8)
             y = sum_e w_e * Wd_e (silu(min(Wg_e h, 7)) * clamp(Wu_e h, -7, 7)); x += y
  logits = Wout (rms(x) * output_norm)

This is the "truth" the converter and the PS4 kernels are checked against. Slow (dequantizes on the fly).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_maple import GGUF  # noqa: E402

MODEL = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "models" / "maple" / "maple-preview-TQ1_0-head-Q4_K.gguf")


class MapleRef:
    def __init__(self, path=MODEL, max_ctx=4096):
        self.g = g = GGUF(path)
        m = g.meta
        self.L = m["maple.block_count"]
        self.D = m["maple.embedding_length"]
        self.H = m["maple.attention.head_count"]
        self.KV = m["maple.attention.head_count_kv"]
        self.HD = m["maple.attention.key_length"]
        self.E = m["maple.expert_count"]
        self.K = m["maple.expert_used_count"]
        self.FF = m["maple.expert_feed_forward_length"]
        self.rot = m["maple.rope.dimension_count"]
        self.theta = m["maple.rope.freq_base"]
        self.eps = m["maple.attention.layer_norm_rms_epsilon"]
        self.swa = m["maple.attention.sliding_window_pattern"]
        self.window = m["maple.attention.sliding_window"]
        self.clamp = m["maple.swiglu_clamp_exp"]
        self.V = m["maple.vocab_size"]
        self.cache = {}
        self.kc = np.zeros((self.L, max_ctx, self.KV, self.HD), np.float32)
        self.vc = np.zeros((self.L, max_ctx, self.KV, self.HD), np.float32)
        self.pos = 0
        self.last_experts = []

    def W(self, name, expert=None):
        key = (name, expert)
        if key not in self.cache:
            self.cache[key] = self.g.matrix(name, expert)
            if len(self.cache) > 4000:          # keep memory bounded: experts change per token
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]

    def vec(self, name):
        return self.g.matrix(name).reshape(-1) if len(self.g.tensors[name]["dims"]) > 1 else \
            self.g.raw(name).view(np.float32).copy()

    def rms(self, x, w):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps) * w

    def rope(self, x, pos):            # x [heads, 128]; NEOX: pairs (i, i + rot/2) for i < rot/2
        half = self.rot // 2
        inv = self.theta ** (-np.arange(half, dtype=np.float64) * 2.0 / self.rot)
        ang = pos * inv
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        y = x.copy()
        a, b = x[:, :half], x[:, half:self.rot]
        y[:, :half] = a * c - b * s
        y[:, half:self.rot] = b * c + a * s
        return y

    def embed(self, tok):
        t = self.g.tensors["token_embd.weight"]
        rb = self.D * 2
        raw = np.frombuffer(self.g.mm, dtype=np.float16, count=self.D, offset=t["abs"] + tok * rb)
        return raw.astype(np.float32)

    def step(self, tok):
        p = self.pos
        x = self.embed(tok)
        self.last_experts = []
        for i in range(self.L):
            b = f"blk.{i}."
            h = self.rms(x, self.vec(b + "attn_norm.weight"))
            q = (self.W(b + "attn_q.weight") @ h).reshape(self.H, self.HD)
            k = (self.W(b + "attn_k.weight") @ h).reshape(self.KV, self.HD)
            v = (self.W(b + "attn_v.weight") @ h).reshape(self.KV, self.HD)
            q = self.rms(q, self.vec(b + "attn_q_norm.weight"))
            k = self.rms(k, self.vec(b + "attn_k_norm.weight"))
            if self.swa[i]:
                q, k = self.rope(q, p), self.rope(k, p)
            self.kc[i, p], self.vc[i, p] = k, v
            lo = max(0, p - self.window + 1) if self.swa[i] else 0
            ks, vs = self.kc[i, lo:p + 1], self.vc[i, lo:p + 1]          # [n, KV, HD]
            g = self.H // self.KV
            att = np.empty((self.H, self.HD), np.float32)
            for hh in range(self.H):
                kv = hh // g
                sc = ks[:, kv] @ q[hh] / np.sqrt(self.HD)
                sc = np.exp(sc - sc.max())
                att[hh] = (sc / sc.sum()) @ vs[:, kv]
            x = x + self.W(b + "attn_output.weight") @ att.reshape(-1)
            h = self.rms(x, self.vec(b + "ffn_norm.weight"))
            logits = self.W(b + "ffn_gate_inp.weight") @ h
            pr = np.exp(logits - logits.max())
            pr /= pr.sum()
            top = np.argsort(-pr)[:self.K]
            w = pr[top] / pr[top].sum()
            self.last_experts.append(top)
            lim = self.clamp[i]
            y = np.zeros(self.D, np.float32)
            for e, we in zip(top, w):
                gt = self.W(b + "ffn_gate_exps.weight", e) @ h
                up = self.W(b + "ffn_up_exps.weight", e) @ h
                gt = np.minimum(gt, lim)
                up = np.clip(up, -lim, lim)
                act = gt / (1.0 + np.exp(-gt)) * up
                y += we * (self.W(b + "ffn_down_exps.weight", e) @ act)
            x = x + y
        self.pos += 1
        self.x_final = x
        return self.W("output.weight") @ self.rms(x, self.vec("output_norm.weight"))


if __name__ == "__main__":
    import time
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(__import__("pathlib").Path(MODEL).parent / "tokenizer.json"))
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is 17 * 23?"
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text, add_special_tokens=False).ids
    print("prompt tokens:", len(ids), ids[:12], "...")
    m = MapleRef()
    t0 = time.time()
    for t in ids:
        logits = m.step(t)
    print(f"prefill {len(ids)} tokens in {time.time() - t0:.1f}s")
    out = []
    for _ in range(n_new):
        nxt = int(np.argmax(logits))
        out.append(nxt)
        if nxt == 151645:
            break
        logits = m.step(nxt)
        print(tok.decode(out), flush=True, end="\r")
    print("\ngenerated ids:", out)
    print("text:", repr(tok.decode(out)))
