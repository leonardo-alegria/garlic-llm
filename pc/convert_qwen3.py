"""Convert a Qwen3 checkpoint into the app's model file format: one file the PS4 loads straight into GPU memory.

Every linear layer (and the tied embedding / output head) becomes Q8 (int8 W[rows][cols] followed by FP32
scales[rows][cols / 32]) or, with --q4, Q4 (see q4.py: packed nibbles, then fp16 scales). Norm weights stay FP32. q/k/v and gate/up are concatenated so each needs one
matrix-vector product. Tensors start on 256-byte boundaries; the index goes into <name>.json.

Run: python convert_qwen3.py [model dir] [output name] [--q4]   -> models/<output name>.bin + .json
"""

import json
import sys
from pathlib import Path

import numpy as np

from q4 import quantize_q4
from qwen3_ref import MODEL, load_weights, quantize_q8

OUT_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_q4 = "--q4" in sys.argv
    model_dir = Path(pos[0]) if pos else MODEL
    name = pos[1] if len(pos) > 1 else "qwen3-0.6b-q8"
    w, cfg = load_weights(model_dir)
    index, offset = {}, 0
    out = open(OUT_DIR / f"{name}.bin", "wb")

    def put(key, data, **meta):
        nonlocal offset
        pad = (-offset) % 256
        out.write(bytes(pad))
        offset += pad
        out.write(data)
        index[key] = dict(offset=offset, size=len(data), **meta)
        offset += len(data)

    def put_q8(key, mat):
        if use_q4:
            packed, d16 = quantize_q4(mat.astype(np.float32))
            put(key, packed.tobytes() + d16.tobytes(), rows=mat.shape[0], cols=mat.shape[1], scales=packed.nbytes,
                fmt="q4")
            return
        q, scale = quantize_q8(mat.astype(np.float32))
        put(key, q.tobytes() + scale.tobytes(), rows=mat.shape[0], cols=mat.shape[1], scales=q.nbytes, fmt="q8")

    put_q8("embed", w["model.embed_tokens.weight"])
    if not cfg["tie_word_embeddings"]:
        put_q8("lm_head", w["lm_head.weight"])
    put("final_norm", w["model.norm.weight"].astype(np.float32).tobytes())
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        put(f"{i}.attn_norm", w[p + "input_layernorm.weight"].astype(np.float32).tobytes())
        put(f"{i}.mlp_norm", w[p + "post_attention_layernorm.weight"].astype(np.float32).tobytes())
        put(f"{i}.q_norm", w[p + "self_attn.q_norm.weight"].astype(np.float32).tobytes())
        put(f"{i}.k_norm", w[p + "self_attn.k_norm.weight"].astype(np.float32).tobytes())
        put_q8(f"{i}.wqkv", np.concatenate([w[p + "self_attn.q_proj.weight"], w[p + "self_attn.k_proj.weight"],
                                             w[p + "self_attn.v_proj.weight"]]))
        put_q8(f"{i}.wo", w[p + "self_attn.o_proj.weight"])
        put_q8(f"{i}.wgu", np.concatenate([w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]]))
        put_q8(f"{i}.wdown", w[p + "mlp.down_proj.weight"])
        print(f"layer {i} done, {offset / 2**20:.0f} MiB", end="\r")
    out.close()
    (OUT_DIR / f"{name}.json").write_text(json.dumps(dict(config=cfg, size=offset, tensors=index), indent=1))
    print(f"\nwrote {OUT_DIR / (name + '.bin')}: {offset / 2**20:.1f} MiB, {len(index)} tensors")


if __name__ == "__main__":
    main()
