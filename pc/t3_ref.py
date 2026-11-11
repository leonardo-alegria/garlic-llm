"""NumPy reference of Chatterbox T3 decoding, from models/chatterbox-t3.bin (q4i weights dequantized exactly as the
GPU reads them), one token at a time with a KV cache, as the PS4 runs it (no CFG: one row).

Sequence: [34 voice prefix rows | text tokens (start 255 ... stop 0), each text_emb + text_pos[i] |
           start-of-speech 6561 + speech_pos[0], twice (as the official code) | sampled tokens t_k + speech_pos[k+1]]
RoPE positions run over the whole sequence (llama3 frequencies). Logits = speech_head(norm(h)) over 6563 tokens.

Run (needs the official `chatterbox` package for its text tokenizer; --torch also needs torch):
  python t3_ref.py "Hola, ¿cómo estás?" es [--torch] [--greedy N]
  --torch  also runs the official T3 (fp32) on the same input and compares the first logits and the greedy tokens
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CB = ROOT / "models" / "chatterbox"
NAME = "chatterbox-t3"


def dequant_q4i(raw, rows, cols, scales_off):
    nb = cols // 32
    nib = np.frombuffer(raw[:scales_off], np.uint8).reshape(rows // 64, nb, 64, 16).transpose(0, 2, 1, 3)
    sc = np.frombuffer(raw[scales_off:], np.float16).reshape(rows // 64, nb, 64).transpose(0, 2, 1)
    nib = nib.reshape(rows, nb, 16)
    q = np.concatenate([nib & 15, nib >> 4], axis=2).astype(np.float32) - 8.0   # [rows][nb][32]
    return (q * sc.reshape(rows, nb, 1).astype(np.float32)).reshape(rows, cols)


class T3Ref:
    def __init__(self):
        self.meta = json.loads((ROOT / "models" / f"{NAME}.json").read_text())
        self.blob = (ROOT / "models" / f"{NAME}.bin").read_bytes()
        self.cache = {}
        from convert_chatterbox_t3 import rope_inv_freq
        self.inv = rope_inv_freq()

    def t(self, key):
        if key in self.cache:
            return self.cache[key]
        e = self.meta["tensors"][key]
        raw = self.blob[e["offset"]:e["offset"] + e["size"]]
        if e["fmt"] == "q4i":
            v = dequant_q4i(raw, e["rows"], e["cols"], e["scales"])
        elif e["fmt"] == "f16":
            v = np.frombuffer(raw, np.float16).reshape(e["rows"], e["cols"]).astype(np.float32)
        else:
            v = np.frombuffer(raw, np.float32).copy()
        self.cache[key] = v
        return v

    def inputs(self, text_tokens):
        rows = [self.t("prefix")]
        te, tp = self.t("text_emb"), self.t("text_pos")
        rows.append(te[text_tokens] + tp[np.arange(len(text_tokens))])
        bos = self.t("speech_emb")[6561] + self.t("speech_pos")[0]
        rows.append(np.stack([bos, bos]))
        return np.concatenate(rows).astype(np.float32)

    @staticmethod
    def rms(x, w, eps=1e-5):
        return x / np.sqrt(np.mean(x * x) + eps) * w

    def rope(self, v, pos):   # v [heads][64], rotate-half
        a = pos * self.inv
        c, s = np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)
        x1, x2 = v[:, :32], v[:, 32:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=1)

    def step(self, x, pos, kc, vc):
        """One position: x [1024] input embedding -> logits [6563]. kc / vc: per layer lists of [pos][16][64]."""
        h = x.astype(np.float32)
        for i in range(30):
            a = self.rms(h, self.t(f"{i}.in_norm"))
            qkv = self.t(f"{i}.qkv") @ a
            q, k, v = qkv[:1024].reshape(16, 64), qkv[1024:2048].reshape(16, 64), qkv[2048:].reshape(16, 64)
            q, k = self.rope(q, pos), self.rope(k, pos)
            kc[i].append(k)
            vc[i].append(v)
            K, V = np.stack(kc[i]), np.stack(vc[i])            # [n][16][64]
            sc = np.einsum("hd,nhd->hn", q, K) / 8.0
            sc = np.exp(sc - sc.max(axis=1, keepdims=True))
            sc /= sc.sum(axis=1, keepdims=True)
            o = np.einsum("hn,nhd->hd", sc, V).reshape(1024)
            h = h + self.t(f"{i}.o") @ o
            a = self.rms(h, self.t(f"{i}.post_norm"))
            gu = self.t(f"{i}.gu") @ a
            g, u = gu[:4096], gu[4096:]
            h = h + self.t(f"{i}.down") @ (g / (1 + np.exp(-g)) * u)
        return self.t("head")[:6563] @ self.rms(h, self.t("norm"))

    def generate(self, text_tokens, n_max=400, greedy=False, seed=0, temperature=0.8, min_p=0.05, rep=1.2):
        kc, vc = [[] for _ in range(30)], [[] for _ in range(30)]
        emb = self.inputs(text_tokens)
        for p in range(len(emb)):
            logits = self.step(emb[p], p, kc, vc)
        rng = np.random.default_rng(seed)
        out, hist = [], [6561]
        first_logits = logits.copy()
        for k in range(n_max):
            if greedy:
                tok = int(np.argmax(logits))
            else:
                tok = sample(logits, hist, rng, temperature, min_p, rep)
            if tok == 6562:
                break
            out.append(tok)
            hist.append(tok)
            x = self.t("speech_emb")[tok] + self.t("speech_pos")[k + 1]
            logits = self.step(x, len(emb) + k, kc, vc)
        return out, first_logits


def sample(logits, hist, rng, temperature=0.8, min_p=0.05, rep=1.2):
    """Repetition penalty (HF style), temperature, min_p, then a draw: the same steps as the official code."""
    l = logits.astype(np.float64).copy()
    for t in set(hist):
        l[t] = l[t] / rep if l[t] > 0 else l[t] * rep
    l /= temperature
    p = np.exp(l - l.max())
    p /= p.sum()
    p[p < min_p * p.max()] = 0
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def text_tokens(text, lang):
    sys.path.insert(0, str(CB / "src" / "src"))
    from chatterbox.models.tokenizers import MTLTokenizer
    from chatterbox.mtl_tts import punc_norm
    tok = MTLTokenizer(str(CB / "hf" / "grapheme_mtl_merged_expanded_v1.json"))
    ids = tok.text_to_tokens(punc_norm(text), language_id=lang)[0].tolist()
    return [255] + ids + [0]


def torch_first_logits(tokens, n_greedy):
    """The official T3 (fp32) on the same input: first logits and greedy tokens (cfg 0, one row)."""
    import torch
    from safetensors.torch import load_file
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(load_file(str(CB / "hf" / "t3_mtl23ls_v3.safetensors")))
    t3.eval()
    ref = T3Ref()
    emb = torch.from_numpy(ref.inputs(tokens))[None]
    with torch.inference_mode():
        pre = torch.from_numpy(ref.t("prefix")).float()[None]
        te = t3.text_emb(torch.tensor([tokens])) + t3.text_pos_emb(torch.tensor([tokens]))
        bos = t3.speech_emb(torch.tensor([[6561]])) + t3.speech_pos_emb.get_fixed_embedding(0)
        emb = torch.cat([pre, te, bos, bos], dim=1)
        out = t3.tfmr(inputs_embeds=emb, use_cache=True, output_hidden_states=True)
        logits = t3.speech_head(out.hidden_states[-1][:, -1])[0, :6563]
        first = logits.numpy().copy()
        toks, past = [], out.past_key_values
        for k in range(n_greedy):
            tok = int(torch.argmax(logits))
            if tok == 6562:
                break
            toks.append(tok)
            x = t3.speech_emb(torch.tensor([[tok]])) + t3.speech_pos_emb.get_fixed_embedding(k + 1)
            out = t3.tfmr(inputs_embeds=x, past_key_values=past, use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            logits = t3.speech_head(out.hidden_states[-1][:, -1])[0, :6563]
    return first, toks


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    text, lang = args[0], args[1]
    n = int(sys.argv[sys.argv.index("--greedy") + 1]) if "--greedy" in sys.argv else 40
    toks = text_tokens(text, lang)
    print("text tokens", len(toks), toks[:20])
    ref = T3Ref()
    mine, first = ref.generate(toks, n_max=n, greedy=True)
    print("q4 greedy", mine[:30], len(mine))
    if "--torch" in sys.argv:
        tf, ttoks = torch_first_logits(toks, n)
        print(f"first logits: max |diff| {np.abs(tf - first).max():.3f}, top1 torch {tf.argmax()} ours {first.argmax()}, "
              f"corr {np.corrcoef(tf, first)[0, 1]:.5f}")
        print("fp32 greedy", ttoks[:30], len(ttoks))
        same = sum(1 for a, b in zip(mine, ttoks) if a == b)
        print(f"same tokens from the start: {next((i for i, (a, b) in enumerate(zip(mine, ttoks)) if a != b), min(len(mine), len(ttoks)))}, total equal {same}/{min(len(mine), len(ttoks))}")
    out = ROOT / "models" / "chatterbox" / "t3_port"
    out.mkdir(exist_ok=True)
    (out / "ref_tokens.json").write_text(json.dumps(dict(text=text, lang=lang, text_tokens=toks, greedy=mine)))


if __name__ == "__main__":
    main()
