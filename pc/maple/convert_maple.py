"""Maple-Preview GGUF (TQ1_0 experts/attention, Q4_K head, F16 embedding) -> a Garlic Server model folder.

  python pc/maple/convert_maple.py --t1i [gguf] [out_dir]   (defaults: models/maple/..., models/maple-ps4)
  --t1i: rows interleaved in tiles of 64 (layout 2, what the current kernels are tuned for)

Writes:
  maple-ternary.bin  everything the GPU reads (loaded whole into GPU memory), 256-byte aligned tensors:
                     per layer: attn_norm, ffn_norm f32[2048]; q_norm, k_norm f32[128]; wqkv t1p [3072 x 2048]
                     (q 2048 rows, k 512, v 512); wo t1p [2048 x 2048]; router f16 [256 x 2048];
                     experts gate|up t1p [256][1024 x 2048] (gate rows 0..511, up 512..1023);
                     experts down t1p [256][2048 x 512]; then output_norm f32[2048], head Q4_K rows (as in the GGUF)
  maple-embed.bin    the input embedding, f16 rows of 2048 (4096 bytes per token), read by the CPU
  model.idx          "PMAP": shape, file names, every tensor's offset (chat/engine_maple.inc reads it)
  tokenizer.bin      Garlic's byte-level BPE tables (pc/export_chat.py format) for Maple's Qwen2 vocabulary

t1p = TQ1_0 bits, laid out per row for dword loads: the row's blocks' qs (48 bytes each), then their qh (4 bytes
each), then their f16 scales. Same bits, same size as TQ1_0 (54 bytes per 256 weights); nothing is re-quantized.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]   # the repository
sys.path.insert(0, str(ROOT / "pc"))
from gguf_maple import GGUF, TYPE_F16, TYPE_F32, TYPE_Q4_K, TYPE_TQ1_0, dequant_q4_k  # noqa: E402

SRC = ROOT / "models" / "maple" / "maple-preview-TQ1_0-head-Q4_K.gguf"
OUT = ROOT / "models" / "maple-ps4"
MAX_SEQ = 16384
DEVICE_DIR = "/data/garlic/models/maple"


LAYOUT = 2 if "--t1i" in sys.argv else 1   # 2: rows interleaved in tiles of 64 (see il64)


def il64(rows_bytes):
    """Rows [R, rb] (R % 64 == 0, rb % 4 == 0) -> tiles of 64 rows, dword-interleaved: tile t, dword d, lane l at
    ((t * (rb / 4) + d) * 64 + l) * 4. A wave (lane = row) then reads 256 contiguous bytes per dword of its rows."""
    R, rb = rows_bytes.shape
    assert R % 64 == 0 and rb % 4 == 0, (R, rb)
    u = np.ascontiguousarray(rows_bytes).view("<u4").reshape(R // 64, 64, rb // 4)
    return np.ascontiguousarray(u.transpose(0, 2, 1)).view(np.uint8).reshape(-1)


def lay(rows_bytes):
    return il64(rows_bytes) if LAYOUT == 2 else rows_bytes


def t1p(raw):
    """TQ1_0 rows [rows, nb*54] -> t1p rows [rows, nb*54]."""
    rows = raw.shape[0]
    blk = raw.reshape(rows, -1, 54)
    return np.concatenate([blk[:, :, :48].reshape(rows, -1), blk[:, :, 48:52].reshape(rows, -1),
                           blk[:, :, 52:54].reshape(rows, -1)], axis=1)


class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.off = 0

    def put(self, data):
        pad = (-self.off) % 256
        if pad:
            self.f.write(bytes(pad))
            self.off += pad
        start = self.off
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data).tobytes()
        self.f.write(data)
        self.off += len(data)
        return start


def export_tokenizer(model_dir, out_path):
    """Garlic's pc/export_chat.py export_tokenizer, for another tokenizer.json."""
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {chr(c): b for b, c in zip(bs, cs)}
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab, merges = tj["model"]["vocab"], tj["model"]["merges"]
    dec = bytes_to_unicode()
    n = max(max(vocab.values()), max(a["id"] for a in tj["added_tokens"])) + 1
    raw = [b""] * n
    for tok, i in vocab.items():
        raw[i] = bytes(dec[ch] for ch in tok)
    special = [0] * n
    for a in tj["added_tokens"]:
        raw[a["id"]] = a["content"].encode()
        special[a["id"]] = 1
    byte_id = [vocab[next(ch for ch, b in dec.items() if b == x)] for x in range(256)]
    mrec = []
    for m in merges:
        a, b = m.split(" ") if isinstance(m, str) else m
        mrec.append(struct.pack("<III", vocab[a], vocab[b], vocab[a + b]))
    out = struct.pack("<4sII", b"PTOK", n, len(mrec)) + struct.pack("<256I", *byte_id)
    out += b"".join(struct.pack("<BH", special[i], len(r)) + r for i, r in enumerate(raw))
    out += b"".join(mrec)
    out_path.write_bytes(out)
    return n


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(pos[0]) if len(pos) > 0 else SRC
    out = Path(pos[1]) if len(pos) > 1 else OUT
    out.mkdir(parents=True, exist_ok=True)
    g = GGUF(src)
    m = g.meta
    L, D = m["maple.block_count"], m["maple.embedding_length"]
    H, KV, HD = m["maple.attention.head_count"], m["maple.attention.head_count_kv"], m["maple.attention.key_length"]
    E, K, FF = m["maple.expert_count"], m["maple.expert_used_count"], m["maple.expert_feed_forward_length"]
    V = m["maple.vocab_size"]
    swa = m["maple.attention.sliding_window_pattern"]
    swa_mask = sum(1 << i for i, s in enumerate(swa) if s)
    vocab = json.loads((src.parent / "tokenizer.json").read_text())
    added = {a["content"]: a["id"] for a in vocab["added_tokens"]}
    eos, im_end, im_start = added["<|endoftext|>"], added["<|im_end|>"], added["<|im_start|>"]

    def f32(name):
        t = g.tensors[name]
        assert t["type"] == TYPE_F32, name
        return g.raw(name).copy()

    wname = "maple-ternary-t1i.bin" if LAYOUT == 2 else "maple-ternary.bin"
    w = Writer(out / wname)
    table = []
    for i in range(L):
        b = f"blk.{i}."
        offs = [w.put(f32(b + "attn_norm.weight")), w.put(f32(b + "ffn_norm.weight")),
                w.put(f32(b + "attn_q_norm.weight")), w.put(f32(b + "attn_k_norm.weight"))]
        for n in ("attn_q", "attn_k", "attn_v", "attn_output"):
            assert g.tensors[b + n + ".weight"]["type"] == TYPE_TQ1_0
        qkv = np.concatenate([g.rows_raw(b + "attn_q.weight"), g.rows_raw(b + "attn_k.weight"),
                              g.rows_raw(b + "attn_v.weight")])
        offs.append(w.put(lay(t1p(qkv))))
        offs.append(w.put(lay(t1p(g.rows_raw(b + "attn_output.weight")))))
        offs.append(w.put(g.matrix(b + "ffn_gate_inp.weight").astype(np.float16)))      # router [256, 2048]
        gu_start = None
        for e in range(E):   # gate | up per expert
            o = w.put(lay(np.concatenate([t1p(g.rows_raw(b + "ffn_gate_exps.weight", e)),
                                          t1p(g.rows_raw(b + "ffn_up_exps.weight", e))])))
            gu_start = o if gu_start is None else gu_start
        offs.append(gu_start)
        dn_start = None
        for e in range(E):
            o = w.put(lay(t1p(g.rows_raw(b + "ffn_down_exps.weight", e))))
            dn_start = o if dn_start is None else dn_start
        offs.append(dn_start)
        table.append(offs)
        print(f"layer {i}: {w.off / 2**20:.0f} MiB", flush=True)
    out_norm = w.put(f32("output_norm.weight"))
    # head: the GGUF's Q4_K rows as they are (144 bytes per 256 weights; shaders/maple.cl mv_q4k reads them)
    assert g.tensors["output.weight"]["type"] == TYPE_Q4_K
    head_raw = g.rows_raw("output.weight")
    if LAYOUT == 2:
        parts = [il64(head_raw[r0:r0 + 8192]) for r0 in range(0, head_raw.shape[0], 8192)]
        head_nib = w.put(b"".join(p.tobytes() for p in parts))
    else:
        head_nib = w.put(head_raw)
    head_sc = 0
    weights_size = w.off
    w.f.close()
    print(f"{wname}: {weights_size / 2**20:.1f} MiB (layout {LAYOUT})")

    # embedding: f16 rows straight from the GGUF
    te = g.tensors["token_embd.weight"]
    assert te["type"] == TYPE_F16 and te["dims"] == [D, V]
    if (out / "maple-embed.bin").exists() and (out / "maple-embed.bin").stat().st_size == V * D * 2:
        print("maple-embed.bin: already there")
    else:
      with open(out / "maple-embed.bin", "wb") as f:
        for r0 in range(0, V, 16384):
            n = min(16384, V - r0)
            f.write(g.mm[te["abs"] + r0 * D * 2: te["abs"] + (r0 + n) * D * 2])
    print(f"maple-embed.bin: {V * D * 2 / 2**20:.1f} MiB")

    n_tok = export_tokenizer(src.parent, out / "tokenizer.bin")
    hdr = struct.pack("<4sI", b"PMAP", LAYOUT)   # version = weight layout: 1 row-major t1p, 2 tile-interleaved
    hdr += struct.pack("<17I", L, D, H, KV, HD, E, K, FF, V, MAX_SEQ, m["maple.attention.sliding_window"],
                       m["maple.rope.dimension_count"], eos, im_end, im_start, swa_mask, n_tok)
    hdr += struct.pack("<3f", m["maple.attention.layer_norm_rms_epsilon"], m["maple.rope.freq_base"],
                       m["maple.swiglu_clamp_exp"][0])
    hdr += f"{DEVICE_DIR}/{wname}".encode().ljust(128, b"\0")
    hdr += f"{DEVICE_DIR}/maple-embed.bin".encode().ljust(128, b"\0")
    hdr += struct.pack("<QQ", weights_size, D * 2)
    for offs in table:
        hdr += struct.pack("<9Q", *offs)
    hdr += struct.pack("<3Q", out_norm, head_nib, head_sc)
    (out / "model.idx").write_bytes(hdr)
    print(f"model.idx: {len(hdr)} bytes; tokenizer.bin: {n_tok} tokens; folder {out}")


if __name__ == "__main__":
    main()
