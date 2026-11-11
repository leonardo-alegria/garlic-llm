"""Convert Parakeet TDT 0.6B v3 (models/parakeet-tdt-0.6b-v3/model.safetensors) to the PS4 format: one file,
tensors on 256-byte boundaries, index in models/<name>.json (as the Qwen files).

  q4i   all big matrices (shaders/parakeet.cl: mm_q4i), rows padded to a multiple of 64
  f32   norms, biases, bias_u / bias_v, depthwise kernels, BatchNorm folded to scale / shift, the small
        subsampling convolutions (pointwise ones transposed: Wt[c][o]), the 128 x 257 mel filter bank

Run: python convert_parakeet.py   -> models/parakeet-v3-q4i.bin + .json
"""

import json
from pathlib import Path

import numpy as np

from parakeet_ref import load_weights, mel_filters
from q4 import interleave_q4, quantize_q4

ROOT = Path(__file__).resolve().parent.parent
NAME = "parakeet-v3-q4i"


def main():
    w = load_weights()
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

    def f32(key, a):
        put(key, np.ascontiguousarray(a, dtype=np.float32).tobytes(), n=int(np.size(a)))

    def q4i(key, m):
        m = np.ascontiguousarray(m, dtype=np.float32)
        rows = m.shape[0]
        if rows % 64:
            m = np.concatenate([m, np.zeros((64 - rows % 64, m.shape[1]), np.float32)])
        packed, d16 = quantize_q4(m)
        nib, sc = interleave_q4(packed, d16)
        put(key, nib + sc, rows=m.shape[0], cols=m.shape[1], scales=len(nib), fmt="q4i", true_rows=rows)

    S = "encoder.subsampling."
    f32("mel", mel_filters())                                          # [128][257]
    f32("sub0_w", w[S + "layers.0.weight"].reshape(256, 9))
    f32("sub0_b", w[S + "layers.0.bias"])
    for k, (dw, pw) in enumerate(((2, 3), (5, 6)), start=1):
        f32(f"sub{k}_dw", w[S + f"layers.{dw}.weight"].reshape(256, 9))
        f32(f"sub{k}_dwb", w[S + f"layers.{dw}.bias"])
        f32(f"sub{k}_pwt", w[S + f"layers.{pw}.weight"].reshape(256, 256).T)
        f32(f"sub{k}_pwb", w[S + f"layers.{pw}.bias"])
    # linear 4096 -> 1024 over (channel, freq) flattened channel-major; the GPU has [freq][channel]
    lin = w[S + "linear.weight"].reshape(1024, 256, 16).transpose(0, 2, 1).reshape(1024, 4096)
    q4i("sub_lin", lin)
    f32("sub_lin_b", w[S + "linear.bias"])
    for i in range(24):
        p = f"encoder.layers.{i}."
        for n in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            f32(f"{i}.{n}.w", w[p + n + ".weight"])
            f32(f"{i}.{n}.b", w[p + n + ".bias"])
        for ff in ("feed_forward1", "feed_forward2"):
            q4i(f"{i}.{ff}.1", w[p + ff + ".linear1.weight"])
            q4i(f"{i}.{ff}.2", w[p + ff + ".linear2.weight"])
        a = p + "self_attn."
        q4i(f"{i}.wqkv", np.concatenate([w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"]]))
        q4i(f"{i}.wpos", w[a + "relative_k_proj.weight"])
        q4i(f"{i}.wo", w[a + "o_proj.weight"])
        f32(f"{i}.bias_u", w[a + "bias_u"])
        f32(f"{i}.bias_v", w[a + "bias_v"])
        c = p + "conv."
        q4i(f"{i}.pw1", w[c + "pointwise_conv1.weight"][:, :, 0])
        q4i(f"{i}.pw2", w[c + "pointwise_conv2.weight"][:, :, 0])
        f32(f"{i}.dw", w[c + "depthwise_conv.weight"][:, 0, :])
        scale = w[c + "norm.weight"] / np.sqrt(w[c + "norm.running_var"] + 1e-5)
        f32(f"{i}.bn_scale", scale)
        f32(f"{i}.bn_shift", w[c + "norm.bias"] - w[c + "norm.running_mean"] * scale)
        print(f"layer {i}, {offset / 2**20:.0f} MiB", end="\r")
    q4i("enc_proj", w["encoder_projector.weight"])
    f32("enc_proj_b", w["encoder_projector.bias"])
    q4i("dec_emb", w["decoder.embedding.weight"])
    for l in range(2):
        q4i(f"lstm{l}", np.concatenate([w[f"decoder.lstm.weight_ih_l{l}"], w[f"decoder.lstm.weight_hh_l{l}"]], axis=1))
        f32(f"lstm{l}_b", w[f"decoder.lstm.bias_ih_l{l}"] + w[f"decoder.lstm.bias_hh_l{l}"])
    q4i("dec_proj", w["decoder.decoder_projector.weight"])
    f32("dec_proj_b", w["decoder.decoder_projector.bias"])
    q4i("joint", w["joint.head.weight"])
    jb = np.zeros(index["joint"]["rows"], np.float32)
    jb[:len(w["joint.head.bias"])] = w["joint.head.bias"]
    f32("joint_b", jb)
    out.close()
    meta = dict(size=offset, tensors=index, blank=8192, vocab=8193, n_dur=5)
    (ROOT / "models" / f"{NAME}.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote models/{NAME}.bin: {offset / 2**20:.1f} MiB, {len(index)} tensors")


if __name__ == "__main__" and "--assets" not in __import__("sys").argv:
    main()


def export_app_assets(out_dir=ROOT / "chat" / "assets"):
    """asr.idx: "PASR", u32 count, char[128] weights path, u64 file size, then per tensor <32sQQIIQ (name, offset,
    size, rows, cols, scales offset). asr_vocab.bin: "PVOC", u32 count, then per token <H length + UTF-8 piece
    (U+2581 already a space; special tokens empty)."""
    import struct
    meta = json.loads((ROOT / "models" / f"{NAME}.json").read_text())
    recs = b"".join(struct.pack("<32sQQIIQ", k.encode(), e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0)) for k, e in meta["tensors"].items())
    head = struct.pack("<4sI128sQ", b"PASR", len(meta["tensors"]), f"/data/{NAME}.bin".encode(), meta["size"])
    (out_dir / "asr.idx").write_bytes(head + recs)
    tj = json.loads((ROOT / "models" / "parakeet-tdt-0.6b-v3" / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    special = {a["id"] for a in tj["added_tokens"] if a.get("special", True)}
    pieces = [b""] * 8192
    for p, i in vocab.items():
        if i < 8192 and i not in special and not (p.startswith("<") and p.endswith(">")):
            pieces[i] = p.replace("▁", " ").encode()
    blob = struct.pack("<4sI", b"PVOC", 8192) + b"".join(struct.pack("<H", len(p)) + p for p in pieces)
    (out_dir / "asr_vocab.bin").write_bytes(blob)
    print(f"asr.idx: {len(meta['tensors'])} tensors; asr_vocab.bin: {len(blob)} bytes, {len(special)} special")


if __name__ == "__main__" and "--assets" in __import__("sys").argv:
    export_app_assets()
