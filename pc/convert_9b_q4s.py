"""Qwen3.5-9B for MTP + SWAR: every matrix in "q4s" (shaders/spec.h), gate/up in 4 bits instead of q3i, and lm_head
trimmed to the rows of a keep list (KEEP below: a .npy of token ids, most frequent first).

From models/qwen3.5-9b-mix.bin (written by pc/modal_qwen35_9b.py): every q4i matrix is laid out again as q4s (same
nibbles, same scales: bit for bit the same weights); gate/up come again from the original bf16 weights
(Qwen/Qwen3.5-9B on Hugging Face, only those tensors, read by byte ranges) and are quantized with pc/q4.py like every
other matrix of the app. The input embedding rows (q4row, read from the disk by the CPU) stay as they are: every token
can still be read.

  python convert_9b_q4s.py            -> models/qwen3.5-9b-q4s.bin/.json and model.idx (in IDX_OUT)
  python convert_9b_q4s.py --check    -> compares a few rows of each converted matrix with the source

q4s (shaders/spec.h, SWAR_MVS): tiles of 128 rows (lane l = rows l and 64 + l), per tile and block of 32 columns two
runs of 64 uint4; word k of a lane = columns 4k..4k+3, nibble i = row A, nibble 4 + i = row B. Scales: one dword per
tile, block and lane = fp16 d_A | fp16 d_B << 16. Rows padded with zeros to a multiple of 128 (the DeltaNet input
projection, 12 352 -> 12 416 rows; lm_head).

File: [GPU part: final_norm, layers, lm_head][embedding rows (q4row)][ids: uint32 per lm_head row = the token id].
model.idx: as pc/export_chat.py (PMQ5), flags bit 2 = "ids" record after lm_head; vocab = the rows that count.
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from q4 import quantize_q4  # noqa: E402

SRC = ROOT / "models" / "qwen3.5-9b-mix.bin"
META = json.loads((ROOT / "models" / "qwen3.5-9b-mix.json").read_text())
NAME = "qwen3.5-9b-q4s"
OUT = ROOT / "models" / f"{NAME}.bin"
CACHE = ROOT / "models" / "q4s_cache"     # gate/up in q4s per layer (re-runs do not fetch them again)
IDX_OUT = ROOT / "chat" / "assets_9b"
HF = "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/"
HF_INDEX = json.loads((ROOT / "models" / "Qwen3.5-9B" / "model.safetensors.index.json").read_text())["weight_map"]
FMT = {"q4": 2, "q8": 1, "q4i": 3, "q4row": 4, "q3i": 5, "q4s": 6, "ids": 7, "q3s": 8}


def to_q4s(q, d):
    """q [rows, cols] nibbles 0..15, d [rows, nb] fp16 -> (q4s words bytes, scale bytes); rows % 128 == 0."""
    rows, cols = q.shape
    T, nb = rows // 128, cols // 32
    q = q.astype(np.uint32).reshape(T, 2, 64, nb, 8, 4)          # tile, A/B, lane, block, word k, nibble i
    half = q[..., 0] | (q[..., 1] << 4) | (q[..., 2] << 8) | (q[..., 3] << 12)
    words = half[:, 0] | (half[:, 1] << 16)                       # T, lane, block, word
    words = words.reshape(T, 64, nb, 2, 4).transpose(0, 2, 3, 1, 4)    # T, block, h, lane, 4 words
    d = d.astype(np.float16).view(np.uint16).astype(np.uint32).reshape(T, 2, 64, nb)
    sc = (d[:, 0] | (d[:, 1] << 16)).transpose(0, 2, 1)          # T, block, lane
    return np.ascontiguousarray(words).tobytes(), np.ascontiguousarray(sc).tobytes()


def from_q4s(wb, sb, rows, cols):
    """Inverse of to_q4s (for --check)."""
    T, nb = rows // 128, cols // 32
    words = np.frombuffer(wb, np.uint32).reshape(T, nb, 2, 64, 4).transpose(0, 3, 1, 2, 4).reshape(T, 64, nb, 8)
    q = np.zeros((T, 2, 64, nb, 8, 4), np.uint8)
    for h in range(2):
        for i in range(4):
            q[:, h, :, :, :, i] = (words >> (16 * h + 4 * i)) & 15
    sc = np.frombuffer(sb, np.uint32).reshape(T, nb, 64).transpose(0, 2, 1)
    d = np.stack([(sc & 0xFFFF).astype(np.uint16), (sc >> 16).astype(np.uint16)], axis=1).view(np.float16)
    return q.reshape(rows, cols), d.reshape(rows, nb)


def read_q4i(f, t, r0=None, r1=None):
    """Rows r0..r1 (multiples of 64) of a q4i tensor -> q [rows, cols] nibbles, d [rows, nb] fp16."""
    rows, cols = t["rows"], t["cols"]
    r0, r1 = r0 or 0, r1 or rows
    nb = cols // 32
    t0, t1 = r0 // 64, r1 // 64
    f.seek(t["offset"] + t0 * nb * 64 * 16)
    nib = np.frombuffer(f.read((t1 - t0) * nb * 64 * 16), dtype=np.uint8)
    f.seek(t["offset"] + t["scales"] + t0 * nb * 64 * 2)
    sc = np.frombuffer(f.read((t1 - t0) * nb * 64 * 2), dtype=np.float16)
    packed = nib.reshape(t1 - t0, nb, 64, 16).transpose(0, 2, 1, 3).reshape(r1 - r0, nb, 16)
    q = np.concatenate([packed & 15, packed >> 4], axis=2).reshape(r1 - r0, cols)
    d = sc.reshape(t1 - t0, nb, 64).transpose(0, 2, 1).reshape(r1 - r0, nb)
    return q, d


def read_q3i(f, t):
    """A q3i tensor (pc/q3.py) -> q [rows, cols] in 0..7, d [rows, nb] fp16."""
    rows, cols = t["rows"], t["cols"]
    nb = cols // 32
    f.seek(t["offset"])
    words = np.frombuffer(f.read(rows * nb * 12), np.uint32).reshape(rows // 64, nb, 64, 3).transpose(0, 2, 1, 3)
    words = words.reshape(rows, nb, 3)
    f.seek(t["offset"] + t["scales"])
    d = np.frombuffer(f.read(rows * nb * 2), np.float16).reshape(rows // 64, nb, 64).transpose(0, 2, 1).reshape(rows, nb)
    q = np.zeros((rows, nb, 32), np.uint8)
    for half in range(2):
        for m in range(4):
            for k in range(4):
                q[..., 16 * half + 4 * m + k] = (words[..., half] >> (8 * k + 2 * m)) & 3
    for m in range(8):
        for k in range(4):
            q[..., 4 * m + k] |= (((words[..., 2] >> (8 * k + m)) & 1) << 2).astype(np.uint8)
    return q.reshape(rows, cols), d


def to_q3s(q, d):
    """q [rows, cols] in 0..7, d [rows, nb] fp16 -> q3s words and scales (shaders/spec.h SWAR3_MVS)."""
    rows, cols = q.shape
    T, nb = rows // 128, cols // 32
    q = q.astype(np.uint32).reshape(T, 2, 64, nb, 32)              # tile, A/B, lane, block, column
    w = np.zeros((T, 64, nb, 6), np.uint32)
    for h in range(2):
        for k in range(6):
            for i in range(5):
                w[..., k] |= q[:, h, :, :, 5 * k + i] << (16 * h + 3 * i)
        for j in range(3):   # columns 30 and 31: bit j in the top bit of each half of words j and 3 + j
            w[..., j] |= ((q[:, h, :, :, 30] >> j) & 1) << (16 * h + 15)
            w[..., 3 + j] |= ((q[:, h, :, :, 31] >> j) & 1) << (16 * h + 15)
    run0 = w[..., :4].transpose(0, 2, 1, 3)                        # T, block, lane, 4 words
    run1 = w[..., 4:].transpose(0, 2, 1, 3)                        # T, block, lane, 2 words
    words = np.concatenate([run0.reshape(T, nb, 256), run1.reshape(T, nb, 128)], axis=2)
    dd = d.astype(np.float16).view(np.uint16).astype(np.uint32).reshape(T, 2, 64, nb)
    sc = (dd[:, 0] | (dd[:, 1] << 16)).transpose(0, 2, 1)
    return np.ascontiguousarray(words).tobytes(), np.ascontiguousarray(sc).tobytes()


def q3s_emulate(wb, sb, rows, cols, x8, init):
    """What SWAR3_MVS computes (integer part per block, as the GPU: 16-bit fields), for checking to_q3s."""
    T, nb = rows // 128, cols // 32
    words = np.frombuffer(wb, np.uint32).reshape(T, nb, 384)
    run0 = words[..., :256].reshape(T, nb, 64, 4)
    run1 = words[..., 256:].reshape(T, nb, 64, 2)
    ws = np.concatenate([run0, run1], axis=3).astype(np.int64)     # T, block, lane, 6
    p = [((ws[..., k] >> (3 * i)) & 0x00070007) for k in range(6) for i in range(5)]
    p.append(((ws[..., 0] >> 15) & 0x00010001) | ((ws[..., 1] >> 14) & 0x00020002) | ((ws[..., 2] >> 13) & 0x00040004))
    p.append(((ws[..., 3] >> 15) & 0x00010001) | ((ws[..., 4] >> 14) & 0x00020002) | ((ws[..., 5] >> 13) & 0x00040004))
    a = (init.reshape(1, nb, 1) >> 1).astype(np.int64)
    for j in range(32):
        a = a + p[j] * x8.reshape(nb, 32)[None, :, None, j]
    a = ((a + 2**31) % 2**32) - 2**31                             # exact modulo 2^32, as int32
    lo = ((a << 48) >> 48)
    hi = (a - lo) >> 16
    return lo, hi                                                  # [T, nb, 64] sums for rows A and B


def pad128(q, d):
    rows = q.shape[0]
    p = (-rows) % 128
    if p:
        q = np.concatenate([q, np.full((p, q.shape[1]), 8, np.uint8)])          # (q - 8) * d = 0
        d = np.concatenate([d, np.full((p, d.shape[1]), 6.2e-5, np.float16)])
    return q, d


_headers = {}


def hf_tensor(name):
    """One bf16 tensor of Qwen/Qwen3.5-9B by HTTP range -> float32 numpy."""
    shard = HF_INDEX[name]
    if shard not in _headers:
        b = urllib.request.urlopen(urllib.request.Request(HF + shard, headers={"Range": "bytes=0-7"})).read()
        n = struct.unpack("<Q", b)[0]
        h = urllib.request.urlopen(urllib.request.Request(HF + shard, headers={"Range": f"bytes=8-{8 + n - 1}"})).read()
        _headers[shard] = (8 + n, json.loads(h))
    base, h = _headers[shard]
    e = h[name]
    assert e["dtype"] == "BF16", e
    a, b = e["data_offsets"]
    for attempt in range(5):
        try:
            req = urllib.request.Request(HF + shard, headers={"Range": f"bytes={base + a}-{base + b - 1}"})
            raw = urllib.request.urlopen(req, timeout=600).read()
            if len(raw) == b - a:
                break
        except Exception as ex:   # (retry: a long download may drop)
            print("  retry", name, ex, flush=True)
    u16 = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32).reshape(e["shape"])


def q4_of(w):
    packed, d16 = quantize_q4(w)
    rows = w.shape[0]
    p = packed.reshape(rows, -1, 16)
    return np.concatenate([p & 15, p >> 4], axis=2).reshape(rows, -1), d16


def gate_up(i):
    """Layer i's gate|up (24 576 x 4096) in q4s, from the bf16 weights (cached)."""
    CACHE.mkdir(exist_ok=True)
    cw, cs = CACHE / f"{i}.wgu.w", CACHE / f"{i}.wgu.s"
    if cw.exists() and cs.exists():
        return cw.read_bytes(), cs.read_bytes()
    p = f"model.language_model.layers.{i}.mlp."
    q, d = [], []
    for part in ("gate_proj", "up_proj"):
        qq, dd = q4_of(hf_tensor(p + part + ".weight"))
        q.append(qq)
        d.append(dd)
    wb, sb = to_q4s(np.concatenate(q), np.concatenate(d))
    cw.write_bytes(wb)
    cs.write_bytes(sb)
    return wb, sb


def arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


KEEP = Path(arg("keep", str(ROOT / "pc" / "vocab_order.npy")))   # every token id, most frequent first
# gate/up layers that stay q3i (the source file's bytes) when the memory does not reach for all: --q3=12-19 or --q3=none
_q3 = arg("q3", "0-31")
Q3_LAYERS = set() if _q3 == "none" else set(range(int(_q3.split("-")[0]), int(_q3.split("-")[1]) + 1))
GU3 = arg("gu3", "q3s")   # those layers in q3s (SWAR, same bytes per weight) or q3i (the matvec_q3i kernels)


def main():
    keep = np.load(KEEP).astype(np.int64)
    c, src = META["config"], META["tensors"]
    order = ["final_norm"]
    for i, kind in enumerate(c["layer_types"]):
        mixer = ("q_norm", "k_norm", "wqkv", "wo") if kind == "full_attention" else (
            "w_in", "conv", "A", "dt_bias", "gnorm", "wout")
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", *mixer, "wgu", "wdown")]
    order.append("lm_head")
    tensors = {}
    f = open(SRC, "rb")
    out = open(OUT, "wb")

    def put(key, wb, sb=b"", **meta):
        off = out.tell()
        out.write(wb)
        out.write(sb)
        tensors[key] = dict(offset=off, size=len(wb) + len(sb), **meta)
        pad = (-out.tell()) % 256
        out.write(b"\0" * pad)

    for k in order:
        t = src[k]
        if k == "lm_head":
            q, d = read_q4i(f, t)
            q, d = pad128(q[keep], d[keep])
            wb, sb = b"", b""
            parts = []
            for r in range(0, q.shape[0], 8192):
                parts.append(to_q4s(q[r:r + 8192], d[r:r + 8192]))
            wb = b"".join(p[0] for p in parts)
            sb = b"".join(p[1] for p in parts)
            put(k, wb, sb, rows=q.shape[0], cols=t["cols"], scales=len(wb), fmt="q4s", vocab=len(keep))
        elif k.endswith(".wgu") and int(k.split(".")[0]) in Q3_LAYERS and GU3 == "q3s":   # the source q3i, SWAR layout
            wb, sb = to_q3s(*read_q3i(f, t))
            put(k, wb, sb, rows=t["rows"], cols=t["cols"], scales=len(wb), fmt="q3s")
        elif k.endswith(".wgu") and int(k.split(".")[0]) in Q3_LAYERS:   # as in the source file (q3i)
            f.seek(t["offset"])
            put(k, f.read(t["size"]), **{x: y for x, y in t.items() if x not in ("offset", "size")})
        elif k.endswith(".wgu"):
            wb, sb = gate_up(int(k.split(".")[0]))
            put(k, wb, sb, rows=t["rows"], cols=t["cols"], scales=len(wb), fmt="q4s")
        elif t.get("fmt") == "q4i":
            q, d = pad128(*read_q4i(f, t))
            wb, sb = to_q4s(q, d)
            put(k, wb, sb, rows=q.shape[0], cols=t["cols"], scales=len(wb), fmt="q4s")
        else:
            f.seek(t["offset"])
            put(k, f.read(t["size"]), **{x: y for x, y in t.items() if x not in ("offset", "size")})
        print(k, tensors[k]["size"], tensors[k].get("fmt", ""), flush=True)
    gpu_size = out.tell()
    e = src["embed"]   # the input embedding rows, as they are
    f.seek(e["offset"])
    off = out.tell()
    left = e["size"]
    while left:
        b = f.read(min(left, 64 << 20))
        out.write(b)
        left -= len(b)
    tensors["embed"] = dict(e, offset=off)
    pad = (-out.tell()) % 256
    out.write(b"\0" * pad)
    ids = keep.astype(np.uint32).tobytes()
    tensors["ids"] = dict(offset=out.tell(), size=len(ids), rows=len(keep), cols=1, scales=0, fmt="ids")
    out.write(ids)
    size = out.tell()
    out.close()
    meta = {"config": dict(c, vocab_size=len(keep)), "size": size, "gpu_size": gpu_size, "tensors": tensors,
            "source": SRC.name, "vocab_kept": len(keep), "keep": KEEP.name, "q3_layers": sorted(Q3_LAYERS)}
    (ROOT / "models" / f"{NAME}.json").write_text(json.dumps(meta, indent=0))
    write_index(meta)
    print(f"{OUT.name}: {size / 2**20:.0f} MiB, GPU part {gpu_size / 2**20:.0f} MiB (was "
          f"{META['gpu_size'] / 2**20:.0f}), lm_head {len(keep)} rows")


def write_index(meta):
    """model.idx like pc/export_chat.py export_model_index_qwen35, + flags bit 2 and the ids record."""
    c, t = meta["config"], meta["tensors"]
    order, mask = ["embed", "final_norm"], 0
    for i, kind in enumerate(c["layer_types"]):
        full = kind == "full_attention"
        mask |= full << i
        mixer = ("q_norm", "k_norm", "wqkv", "wo") if full else ("w_in", "conv", "A", "dt_bias", "gnorm", "wout")
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", *mixer, "wgu", "wdown")]
    order += ["lm_head", "ids"]
    flags = 1 | 2 | 4
    rp = c["rope_parameters"]
    rope_dim = int(c["head_dim"] * rp.get("partial_rotary_factor", 1.0))
    path = f"/data/{NAME}.bin".encode()
    head = struct.pack("<4s10I2f128sQI", b"PMQ5", c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"],
                       c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"],
                       8192, 248046, 248044, c["rms_norm_eps"], rp["rope_theta"], path, meta["gpu_size"], len(order))
    head += struct.pack("<8I", c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"],
                        c["linear_value_head_dim"], rope_dim, mask, 248045, flags)
    recs = []
    for k in order:
        e = t[k]
        recs.append(struct.pack("<QQIIQI", e["offset"], e["size"], e.get("rows", 0), e.get("cols", 0),
                                e.get("scales", 0), FMT.get(e.get("fmt"), 0)))
    IDX_OUT.mkdir(parents=True, exist_ok=True)
    (IDX_OUT / "model.idx").write_bytes(head + b"".join(recs))
    print(f"model.idx: {len(order)} records, flags {flags}, weights at {path.decode()}")


def check():
    meta = json.loads((ROOT / "models" / f"{NAME}.json").read_text())
    keep = np.load(KEEP).astype(np.int64)
    f, g = open(SRC, "rb"), open(OUT, "rb")
    for k in ("0.w_in", "3.wqkv", "31.wdown", "lm_head"):
        t, s = meta["tensors"][k], META["tensors"][k]
        g.seek(t["offset"])
        wb = g.read(t["scales"])
        sb = g.read(t["size"] - t["scales"])
        rows = t["rows"]
        q, d = from_q4s(wb, sb, rows, t["cols"])
        q0, d0 = read_q4i(f, s)
        if k == "lm_head":
            q0, d0 = q0[keep], d0[keep]
        n = q0.shape[0]
        print(k, "nibbles equal:", bool((q[:n] == q0).all()), "scales equal:", bool((d[:n] == d0).all()))
    t = meta["tensors"]["0.wgu"]
    g.seek(t["offset"])
    wb = g.read(t["scales"])
    sb = g.read(t["size"] - t["scales"])
    q, d = from_q4s(wb, sb, t["rows"], t["cols"])
    w = hf_tensor("model.language_model.layers.0.mlp.gate_proj.weight")[:256]
    deq = (q[:256].astype(np.float32) - 8) * np.repeat(d[:256].astype(np.float32), 32, axis=1)
    print("0.wgu gate rows 0..255: relative error of q4 vs bf16",
          float(np.linalg.norm(deq - w) / np.linalg.norm(w)))


MTP_NAME = "qwen3.5-9b-mtp"
MTP_ORDER = ["pre_e", "pre_h", "fc", "attn_norm", "mlp_norm", "q_norm", "k_norm", "wqkv", "wo", "wgu", "wdown", "norm"]
ND = 32768   # draft rows: the first ND rows of the trimmed lm_head


def mtp():
    """The MTP head (mtp.*, which pc/modal_qwen35_9b.py does not convert) in q4s, norms as 1 + w like the 9B's, q_proj
    split into q | k | v | gate rows like wqkv; then the embedding rows (q4row) of the first ND kept tokens: the draft's
    tokens, which the GPU reads itself (chat/engine_spec.inc). -> models/qwen3.5-9b-mtp.bin and mtp.idx (in IDX_OUT)"""
    keep = np.load(KEEP).astype(np.int64)
    c = META["config"]
    nh, hd = c["num_attention_heads"], c["head_dim"]
    P = "mtp."
    out = open(ROOT / "models" / f"{MTP_NAME}.bin", "wb")
    recs = {}

    def put(key, data, rows=0, cols=0, scales=0, fmt=0):
        off = out.tell()
        out.write(data)
        recs[key] = (off, len(data), rows, cols, scales, fmt)
        out.write(b"\0" * ((-out.tell()) % 256))

    def f32(key, w):
        put(key, np.ascontiguousarray(w, np.float32).tobytes(), cols=w.size)

    def q4s(key, w):
        q, d = pad128(*q4_of(w))
        wb, sb = to_q4s(q, d)
        put(key, wb + sb, rows=q.shape[0], cols=w.shape[1], scales=len(wb), fmt=6)

    f32("pre_e", 1.0 + hf_tensor(P + "pre_fc_norm_embedding.weight"))
    f32("pre_h", 1.0 + hf_tensor(P + "pre_fc_norm_hidden.weight"))
    q4s("fc", hf_tensor(P + "fc.weight"))
    L = P + "layers.0."
    f32("attn_norm", 1.0 + hf_tensor(L + "input_layernorm.weight"))
    f32("mlp_norm", 1.0 + hf_tensor(L + "post_attention_layernorm.weight"))
    f32("q_norm", 1.0 + hf_tensor(L + "self_attn.q_norm.weight"))
    f32("k_norm", 1.0 + hf_tensor(L + "self_attn.k_norm.weight"))
    qg = hf_tensor(L + "self_attn.q_proj.weight").reshape(nh, 2, hd, -1)
    q, gate = qg[:, 0].reshape(nh * hd, -1), qg[:, 1].reshape(nh * hd, -1)
    q4s("wqkv", np.concatenate([q, hf_tensor(L + "self_attn.k_proj.weight"), hf_tensor(L + "self_attn.v_proj.weight"),
                                gate]))
    q4s("wo", hf_tensor(L + "self_attn.o_proj.weight"))
    q4s("wgu", np.concatenate([hf_tensor(L + "mlp.gate_proj.weight"), hf_tensor(L + "mlp.up_proj.weight")]))
    q4s("wdown", hf_tensor(L + "mlp.down_proj.weight"))
    f32("norm", 1.0 + hf_tensor(P + "norm.weight"))
    wsize = out.tell()
    e = META["tensors"]["embed"]   # the draft rows' input embeddings, in lm_head row order
    rb = e["rowbytes"]
    with open(SRC, "rb") as f:
        for tokid in keep[:ND]:
            f.seek(e["offset"] + int(tokid) * rb)
            out.write(f.read(rb))
    tsize = out.tell() - wsize
    out.close()
    path = f"/data/{MTP_NAME}.bin".encode()
    head = struct.pack("<4sI128sQQI", b"PMTP", ND, path, wsize, tsize, len(MTP_ORDER))
    body = b"".join(struct.pack("<QQIIQI", *recs[k]) for k in MTP_ORDER)
    IDX_OUT.mkdir(parents=True, exist_ok=True)
    (IDX_OUT / "mtp.idx").write_bytes(head + body)
    print(f"{MTP_NAME}.bin: tensors {wsize / 2**20:.1f} MiB + draft table {tsize / 2**20:.1f} MiB; mtp.idx {len(head + body)} B")


if __name__ == "__main__":
    if "--fetch" in sys.argv:   # only gate/up of every layer into the cache (no keep.npy needed)
        import time
        for i in range(META["config"]["num_hidden_layers"]):
            t0 = time.time()
            gate_up(i)
            print(f"layer {i} gate/up in q4s: {time.time() - t0:.0f} s", flush=True)
    elif "--check" in sys.argv:
        check()
    elif "--mtp" in sys.argv:
        mtp()
    elif "--test-q3s" in sys.argv:   # to_q3s against a plain integer dot product (layer 0's gate/up, 256 rows)
        with open(SRC, "rb") as f:
            q, d = read_q3i(f, META["tensors"]["0.wgu"])
        q, d = q[:256], d[:256]
        rng = np.random.default_rng(1)
        x8 = rng.integers(-127, 128, 4096)
        nb = 128
        init = -8 * x8.reshape(nb, 32).sum(1) * 65537
        wb, sb = to_q3s(q, d)
        lo, hi = q3s_emulate(wb, sb, 256, 4096, x8, init)
        ref = ((q.astype(np.int64) - 4).reshape(256, nb, 32) * x8.reshape(nb, 32)).sum(2)     # [rows, nb]
        got = np.zeros((256, nb), np.int64)
        for T in range(2):
            got[T * 128:T * 128 + 64] = lo[T].T
            got[T * 128 + 64:T * 128 + 128] = hi[T].T
        print("q3s emulation equals the integer sums:", bool((got == ref).all()))
        print("q3i read back = q3.py quantize of nothing to compare; values in 0..7:", int(q.min()), int(q.max()))
    else:
        main()
