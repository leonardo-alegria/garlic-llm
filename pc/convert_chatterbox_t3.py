"""Convert Chatterbox Multilingual V3's T3 (the text -> speech-token Llama) to the PS4 format, and export its text
tokenizer for the app.

T3: Llama, 30 layers, hidden 1024, 16 heads of 64 (no GQA), SwiGLU 4096, RMSNorm 1e-5, RoPE with llama3 scaling
(theta 500000, factor 8, low 1, high 4, original 8192). Input embeddings are learned tables (text / speech tokens
plus learned absolute positions, each restarting at 0); the voice is a precomputed 34 x 1024 prefix.

models/chatterbox-t3.bin (tensors on 256-byte boundaries):
  GPU part (gpu_size bytes, loaded into GPU memory):
    {i}.qkv   q4i [3072][1024]  q, k, v projections stacked
    {i}.o     q4i [1024][1024]
    {i}.gu    q4i [8192][1024]  gate rows then up rows (silu_mul)
    {i}.down  q4i [1024][4096]
    {i}.in_norm, {i}.post_norm, norm   fp32 [1024]
    head      q4i [6592][1024]  speech_head rows 0..6562 (codes + start + stop), zero rows to a multiple of 64
  CPU part (after gpu_size; the app reads single rows from the file):
    text_emb  fp16 [2454][1024], text_pos fp16 [512][1024], speech_emb fp16 [6563][1024],
    speech_pos fp16 [1024][1024], prefix fp16 [34][1024] (the built-in voice: speaker + perceiver + emotion 0.5)
models/chatterbox-t3.json: the index.

Run: python convert_chatterbox_t3.py            -> models/chatterbox-t3.bin + .json
     python convert_chatterbox_t3.py --assets   -> chat/assets/t3.idx and chat/assets/tts_tok.bin
"""

import json
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np

from q4 import interleave_q4, quantize_q4_mse

ROOT = Path(__file__).resolve().parent.parent
CB = ROOT / "models" / "chatterbox"
NAME = "chatterbox-t3"
N_SPEECH = 6563          # 6561 codes + start (6561) + stop (6562)
HEAD_ROWS = 6592         # padded to a multiple of 64
TEXT_POS, SPEECH_POS = 512, 1024


def load_t3():
    from safetensors.numpy import load_file
    return load_file(str(CB / "hf" / "t3_mtl23ls_v3.safetensors"))


def load_voice():
    from safetensors.numpy import load_file
    return load_file(str(CB / "results" / "voice_builtin_reclone" / "voice_blob.safetensors"))


def convert():
    w = load_t3()
    voice = load_voice()
    out = open(ROOT / "models" / f"{NAME}.bin", "wb")
    index, offset = {}, 0

    def put(key, data, **meta):
        nonlocal offset
        pad = (-offset) % 256
        out.write(bytes(pad))
        offset += pad
        out.write(data)
        index[key] = dict(offset=offset, size=len(data), **meta)
        offset += len(data)

    def q4i(key, m):
        m = np.ascontiguousarray(m, dtype=np.float32)
        rows = m.shape[0]
        if rows % 64:
            m = np.concatenate([m, np.zeros((64 - rows % 64, m.shape[1]), np.float32)])
        packed, d16 = quantize_q4_mse(m)
        nib, sc = interleave_q4(packed, d16)
        put(key, nib + sc, rows=m.shape[0], cols=m.shape[1], scales=len(nib), fmt="q4i")

    def f32(key, a):
        put(key, np.ascontiguousarray(a, dtype=np.float32).tobytes(), n=int(np.size(a)), fmt="f32")

    def f16(key, a):
        a = np.ascontiguousarray(a, dtype=np.float16)
        put(key, a.tobytes(), rows=a.shape[0], cols=a.shape[1], fmt="f16")

    for i in range(30):
        p = f"tfmr.layers.{i}."
        a = p + "self_attn."
        q4i(f"{i}.qkv", np.concatenate([w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"]]))
        q4i(f"{i}.o", w[a + "o_proj.weight"])
        q4i(f"{i}.gu", np.concatenate([w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]]))
        q4i(f"{i}.down", w[p + "mlp.down_proj.weight"])
        f32(f"{i}.in_norm", w[p + "input_layernorm.weight"])
        f32(f"{i}.post_norm", w[p + "post_attention_layernorm.weight"])
        print(f"layer {i}, {offset / 2**20:.0f} MiB", end="\r")
    f32("norm", w["tfmr.norm.weight"])
    q4i("head", w["speech_head.weight"][:N_SPEECH])
    gpu_size = (offset + 255) & ~255
    f16("text_emb", w["text_emb.weight"])
    f16("text_pos", w["text_pos_emb.emb.weight"][:TEXT_POS])
    f16("speech_emb", w["speech_emb.weight"][:N_SPEECH])
    f16("speech_pos", w["speech_pos_emb.emb.weight"][:SPEECH_POS])
    f16("prefix", voice["t3_cond_prefix_fp16"])
    out.close()
    meta = dict(size=offset, gpu_size=gpu_size, tensors=index, layers=30, hidden=1024, heads=16, head_dim=64,
                ffn=4096, eps=1e-5, start_text=255, stop_text=0, start_speech=6561, stop_speech=6562,
                n_speech=N_SPEECH, text_pos=TEXT_POS, speech_pos=SPEECH_POS, prefix=34)
    (ROOT / "models" / f"{NAME}.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote models/{NAME}.bin: {offset / 2**20:.1f} MiB ({gpu_size / 2**20:.1f} MiB for the GPU)")


def rope_inv_freq():
    """Llama 3 RoPE frequencies (as transformers' _compute_llama3_parameters) for head_dim 64."""
    dim, theta = 64, 500000.0
    inv = 1.0 / theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    factor, low, high, old = 8.0, 1.0, 4.0, 8192
    wavelen = 2 * np.pi / inv
    out = np.where(wavelen > old / low, inv / factor, inv)
    smooth = (old / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * out / factor + smooth * out
    medium = ~(wavelen < old / high) & ~(wavelen > old / low)
    return np.where(medium, smoothed, out)


def export_app_assets(out_dir=ROOT / "chat" / "assets"):
    """t3.idx: "PT3I", u32 count, char[128] weights path, u64 file size, u64 gpu size, then per tensor
    <32sQQIIQ (name, offset, size, rows, cols, scales offset); then 32 floats: the RoPE inverse frequencies.
    tts_tok.bin: the text tokenizer (see export_tokenizer)."""
    meta = json.loads((ROOT / "models" / f"{NAME}.json").read_text())
    recs = b"".join(struct.pack("<32sQQIIQ", k.encode(), e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0)) for k, e in meta["tensors"].items())
    head = struct.pack("<4sI128sQQ", b"PT3I", len(meta["tensors"]), f"/data/{NAME}.bin".encode(), meta["size"],
                       meta["gpu_size"])
    (out_dir / "t3.idx").write_bytes(head + recs + rope_inv_freq().astype(np.float32).tobytes())
    export_tokenizer(out_dir / "tts_tok.bin")
    print(f"t3.idx: {len(meta['tensors'])} tensors")


def export_tokenizer(path):
    """The grapheme BPE of Chatterbox Multilingual, for a small C implementation (chat/tts_tok.c):
      "PTTK", u32 n_vocab, then per id: u8 length + UTF-8
      u32 n_merges, then per merge in rank order: <HHH (left id, right id, merged id)
      u32 n_added, then per added token (longest first): <H id, u8 length + UTF-8
      u32 n_lower, then pairs <II (codepoint, lower case)            -- Python str.lower(), codepoints < 0x3000
      u32 n_nfkd, then per codepoint: <I cp, u8 n, n x <I            -- NFKD decompositions, codepoints < 0x3000
      word bitmap for codepoints 0..0x2FFF (\\w of the Whitespace pre-tokenizer: letters, marks, numbers, '_')
    """
    j = json.loads((CB / "hf" / "grapheme_mtl_merged_expanded_v1.json").read_text(encoding="utf-8"))
    vocab = j["model"]["vocab"]
    n = max(vocab.values()) + 1
    toks = [b""] * n
    for s, i in vocab.items():
        toks[i] = s.encode()
    blob = struct.pack("<4sI", b"PTTK", n) + b"".join(struct.pack("<B", len(t)) + t for t in toks)
    merges = j["model"]["merges"]
    mrec = b""
    for m in merges:
        a, b = m.split(" ") if isinstance(m, str) else m
        mrec += struct.pack("<HHH", vocab[a], vocab[b], vocab[a + b])
    blob += struct.pack("<I", len(merges)) + mrec
    added = sorted(j["added_tokens"], key=lambda a: -len(a["content"].encode()))
    blob += struct.pack("<I", len(added)) + b"".join(
        struct.pack("<HB", a["id"], len(a["content"].encode())) + a["content"].encode() for a in added)
    lower = [(c, ord(chr(c).lower())) for c in range(0x3000) if len(chr(c).lower()) == 1 and chr(c).lower() != chr(c)]
    blob += struct.pack("<I", len(lower)) + b"".join(struct.pack("<II", a, b) for a, b in lower)
    nfkd = []
    for c in range(0x80, 0x3000):
        d = unicodedata.normalize("NFKD", chr(c))
        if d != chr(c):
            nfkd.append((c, [ord(x) for x in d]))
    blob += struct.pack("<I", len(nfkd)) + b"".join(
        struct.pack("<IB", c, len(d)) + struct.pack(f"<{len(d)}I", *d) for c, d in nfkd)
    bits = bytearray(0x3000 // 8)
    for c in range(0x3000):
        ch = chr(c)
        if ch.isalnum() or ch == "_" or unicodedata.category(ch)[0] in "LMN" or unicodedata.category(ch) == "Pc":
            bits[c >> 3] |= 1 << (c & 7)
    blob += bytes(bits)
    Path(path).write_bytes(blob)
    print(f"tts_tok.bin: {n} tokens, {len(merges)} merges, {len(added)} added, {len(lower)} lower, {len(nfkd)} nfkd, "
          f"{len(blob)} bytes")


if __name__ == "__main__":
    if "--assets" in sys.argv:
        export_app_assets()
    else:
        convert()
