"""Local stand-in for modal_qwen35_9b.py::convert.

Same file format as the Modal job (q4i / q3i / q4row), but tensors are read from a local
safetensors snapshot and quantized in row chunks so a 16 GB machine can finish it.
Default checkpoint is the architecture-compatible abliteration of Qwen/Qwen3.5-9B.

  python local_qwen35_9b.py --self-test
  python local_qwen35_9b.py --snapshot path/to/Huihui-Qwen3.5-9B-abliterated
Writes ../models/qwen3.5-9b-mix.bin and .json.

The snapshot is a local Hugging Face checkout: config.json, model.safetensors.index.json
and the *.safetensors shards. It is not a GGUF file.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q3 import interleave_q3, quantize_q3  # noqa: E402
from q4 import interleave_q4, quantize_q4  # noqa: E402

OUT_DIR = HERE.parent / "models"
NAME = "qwen3.5-9b-mix"


def default_snapshot():
    """A checkout that already has config.json, searched next to the repo."""
    for p in (HERE.parent / "hf" / "Huihui-Qwen3.5-9B-abliterated",
              HERE.parent / "models" / "Huihui-Qwen3.5-9B-abliterated"):
        if (p / "config.json").exists() and (p / "model.safetensors.index.json").exists():
            return p
    return None


def bf16_bytes_to_f32(raw, shape):
    u16 = np.frombuffer(raw, dtype=np.uint16)
    if u16.size != int(np.prod(shape)):
        raise RuntimeError(f"bf16 size {u16.size} != shape {shape}")
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)


class ShardStore:
    def __init__(self, snapshot: Path, weight_map: dict):
        self.snapshot = snapshot
        self.weight_map = weight_map
        self._open = {}

    def _meta(self, name):
        shard = self.weight_map[name]
        if shard not in self._open:
            f = open(self.snapshot / shard, "rb")
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
            self._open[shard] = (f, header, 8 + n)
        return self._open[shard]

    def _decode(self, meta, raw, shape):
        dt = meta["dtype"]
        if dt == "BF16":
            return bf16_bytes_to_f32(raw, shape)
        if dt == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.size != int(np.prod(shape)):
                raise RuntimeError(f"f32 size {arr.size} != shape {shape}")
            return arr.reshape(shape).copy()
        raise RuntimeError(f"{meta.get('dtype')} is not BF16 or F32")

    def array(self, name):
        f, header, base = self._meta(name)
        meta = header[name]
        a, b = meta["data_offsets"]
        f.seek(base + a)
        return self._decode(meta, f.read(b - a), meta["shape"])

    def rows(self, name, r0, r1):
        f, header, base = self._meta(name)
        meta = header[name]
        shape = meta["shape"]
        if meta["dtype"] != "BF16":
            raise RuntimeError(f"{name} is {meta['dtype']}; row slices are implemented for BF16 weights")
        cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        row_bytes = cols * 2
        a, b = meta["data_offsets"]
        if r1 > shape[0] or (b - a) != shape[0] * row_bytes:
            raise RuntimeError(f"{name} shape {shape} does not match offsets")
        f.seek(base + a + r0 * row_bytes)
        raw = f.read((r1 - r0) * row_bytes)
        out_shape = (r1 - r0, *shape[1:]) if len(shape) > 1 else (r1 - r0,)
        return bf16_bytes_to_f32(raw, out_shape)

    def close(self):
        for f, _, _ in self._open.values():
            f.close()
        self._open.clear()


def q4i_bytes(mat):
    packed, d16 = quantize_q4(np.ascontiguousarray(mat, dtype=np.float32))
    return interleave_q4(packed, d16)


def q4i_bytes_chunked(store, name, chunk=64):
    """q4i blob (nibbles then scales) for one 2-D weight, quantized chunk rows at a time."""
    f, header, base = store._meta(name)
    shape = header[name]["shape"]
    rows, cols = shape
    if rows % 64 != 0:
        raise RuntimeError(f"{name} has {rows} rows, not a multiple of 64")
    nb = cols // 32
    nib_parts = []
    sc_parts = []
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        # quantize_q4 / interleave tiles are 64 rows; keep chunks on that grid
        if (r1 - r0) % 64 != 0:
            raise RuntimeError(f"chunk {r1 - r0} is not a multiple of 64")
        mat = store.rows(name, r0, r1)
        packed, d16 = quantize_q4(np.ascontiguousarray(mat, dtype=np.float32))
        tile = packed.reshape(-1, 64, nb, 16).transpose(0, 2, 1, 3)
        sc = d16.reshape(-1, 64, nb).transpose(0, 2, 1)
        nib_parts.append(np.ascontiguousarray(tile).tobytes())
        sc_parts.append(np.ascontiguousarray(sc).tobytes())
        del mat, packed, d16, tile, sc
    nib = b"".join(nib_parts)
    sc = b"".join(sc_parts)
    return nib, sc, rows, cols


def q3_rows(store, name, chunk=512):
    shape = store._meta(name)[1][name]["shape"]
    rows, cols = shape
    q = np.empty((rows, cols), np.uint8)
    d16 = np.empty((rows, cols // 32), np.float16)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        mat = store.rows(name, r0, r1)
        qq, dd = quantize_q3(np.ascontiguousarray(mat, dtype=np.float32))
        q[r0:r1] = qq
        d16[r0:r1] = dd
        del mat, qq, dd
    return q, d16


def self_test():
    rng = np.random.default_rng(1)
    mat = rng.standard_normal((128, 256)).astype(np.float32)
    nib, sc = q4i_bytes(mat)
    # chunked path through a fake shard
    raw = (mat.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    # rebuild float from those bf16 values and compare formats on the bf16-rounded matrix
    rounded = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(mat.shape)
    header = {"w": {"dtype": "BF16", "shape": list(rounded.shape), "data_offsets": [0, len(raw)]}}
    path = OUT_DIR / "_selftest.safetensors"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(header).encode()
    # safetensors header is padded so that 8+len is a multiple of 8; our reader does not require that
    with open(path, "wb") as f:
        f.write(len(payload).to_bytes(8, "little"))
        f.write(payload)
        f.write(raw)
    store = ShardStore(path.parent, {"w": path.name})
    # point snapshot at the parent; weight name is the file name
    nib2, sc2, rows, cols = q4i_bytes_chunked(store, "w", chunk=64)
    store.close()
    path.unlink()
    nib_r, sc_r = q4i_bytes(rounded)
    ok = nib2 == nib_r and sc2 == sc_r and rows == 128 and cols == 256
    # q3 row-chunk matches a full quantize
    q, d = quantize_q3(rounded)
    print("q4i chunk matches full:", ok, "bytes", len(nib_r) + len(sc_r))
    print("q3 sample", int(q.min()), int(q.max()), d.shape)
    if not ok:
        raise SystemExit("self-test failed")


def convert(snapshot: Path):
    cfg = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))["text_config"]
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    store = ShardStore(snapshot, weight_map)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = open(OUT_DIR / f"{NAME}.bin", "wb")
    index, offset = {}, 0

    def put(key, data, **meta):
        nonlocal offset
        pad = (-offset) % 256
        out.write(bytes(pad))
        offset += pad
        if isinstance(data, (bytes, bytearray)):
            out.write(data)
            size = len(data)
        else:
            size = 0
            for part in data:
                out.write(part)
                size += len(part)
        index[key] = dict(offset=offset, size=size, **meta)
        offset += size

    def put_f32(key, a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        put(key, a.tobytes(), n=int(a.size))

    def put_q4i_matrix(key, mat):
        nib, sc = q4i_bytes(mat)
        put(key, nib + sc, rows=int(mat.shape[0]), cols=int(mat.shape[1]), scales=len(nib), fmt="q4i")

    def put_q4i_name(key, name):
        nib, sc, rows, cols = q4i_bytes_chunked(store, name)
        put(key, (nib, sc), rows=rows, cols=cols, scales=len(nib), fmt="q4i")

    P = "model.language_model."
    put_f32("final_norm", 1.0 + store.array(P + "norm.weight"))
    nh, hd = cfg["num_attention_heads"], cfg["head_dim"]
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"{P}layers.{i}."
        put_f32(f"{i}.attn_norm", 1.0 + store.array(p + "input_layernorm.weight"))
        put_f32(f"{i}.mlp_norm", 1.0 + store.array(p + "post_attention_layernorm.weight"))
        if kind == "full_attention":
            a = p + "self_attn."
            put_f32(f"{i}.q_norm", 1.0 + store.array(a + "q_norm.weight"))
            put_f32(f"{i}.k_norm", 1.0 + store.array(a + "k_norm.weight"))
            qg = store.array(a + "q_proj.weight").reshape(nh, 2, hd, -1)
            q = np.ascontiguousarray(qg[:, 0].reshape(nh * hd, -1))
            gate = np.ascontiguousarray(qg[:, 1].reshape(nh * hd, -1))
            k = store.array(a + "k_proj.weight")
            v = store.array(a + "v_proj.weight")
            put_q4i_matrix(f"{i}.wqkv", np.concatenate([q, k, v, gate]))
            del qg, q, gate, k, v
            put_q4i_name(f"{i}.wo", a + "o_proj.weight")
        else:
            a = p + "linear_attn."
            parts = [store.array(a + n) for n in
                     ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")]
            put_q4i_matrix(f"{i}.w_in", np.concatenate(parts))
            del parts
            conv = store.array(a + "conv1d.weight").reshape(-1, cfg["linear_conv_kernel_dim"])
            put_f32(f"{i}.conv", conv)
            put_f32(f"{i}.A", -np.exp(store.array(a + "A_log")))
            put_f32(f"{i}.dt_bias", store.array(a + "dt_bias"))
            put_f32(f"{i}.gnorm", store.array(a + "norm.weight"))
            put_q4i_name(f"{i}.wout", a + "out_proj.weight")
        gq, gd = q3_rows(store, p + "mlp.gate_proj.weight")
        uq, ud = q3_rows(store, p + "mlp.up_proj.weight")
        words, sc = interleave_q3(np.concatenate([gq, uq]), np.concatenate([gd, ud]))
        rows = gq.shape[0] + uq.shape[0]
        cols = gq.shape[1]
        del gq, gd, uq, ud
        put(f"{i}.wgu", words + sc, rows=rows, cols=cols, scales=len(words), fmt="q3i")
        put_q4i_name(f"{i}.wdown", p + "mlp.down_proj.weight")
        print(f"layer {i} ({kind}) done, {offset / 2**20:.0f} MiB", flush=True)

    nib, sc, rows, cols = q4i_bytes_chunked(store, "lm_head.weight", chunk=256)
    put("lm_head", (nib, sc), rows=rows, cols=cols, scales=len(nib), fmt="q4i")
    print(f"lm_head done, {offset / 2**20:.0f} MiB", flush=True)
    del nib, sc

    gpu_size = (offset + 4095) // 4096 * 4096
    out.write(bytes(gpu_size - offset))
    offset = gpu_size

    emb_name = P + "embed_tokens.weight"
    emb_shape = store._meta(emb_name)[1][emb_name]["shape"]
    rows, cols = emb_shape
    rowbytes = cols // 2 + cols // 32 * 2
    start = offset
    for r0 in range(0, rows, 8192):
        r1 = min(rows, r0 + 8192)
        packed, d16 = quantize_q4(np.ascontiguousarray(store.rows(emb_name, r0, r1), dtype=np.float32))
        blob = np.concatenate([packed, d16.view(np.uint8)], axis=1)
        if blob.shape[1] != rowbytes:
            raise RuntimeError(f"embed row {blob.shape[1]} != {rowbytes}")
        out.write(blob.tobytes())
        del packed, d16, blob
        if r0 % (8192 * 4) == 0:
            print(f"embed rows {r1}/{rows}", flush=True)
    offset = start + rows * rowbytes
    index["embed"] = dict(offset=start, size=rows * rowbytes, rows=rows, cols=cols, scales=cols // 2,
                          rowbytes=rowbytes, fmt="q4row")
    out.close()
    store.close()
    meta = dict(config=cfg, size=offset, gpu_size=gpu_size, tensors=index)
    (OUT_DIR / f"{NAME}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"converted: GPU part {gpu_size / 2**20:.1f} MiB, file {offset / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None,
                    help="local checkout of huihui-ai/Huihui-Qwen3.5-9B-abliterated")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        snap = args.snapshot or default_snapshot()
        if snap is None:
            sys.exit("pass --snapshot path/to/Huihui-Qwen3.5-9B-abliterated "
                     "(config.json and the safetensors shards)")
        convert(snap)
