"""Read Maple-Preview's GGUF (deepgrove-ai/llama.cpp, arch "maple") and dequantize its tensors with NumPy.

The decoders follow ggml exactly (ggml/src/ggml-quants.c of the deepgrove fork, saved in ../ref):
  TQ1_0 (type 34): blocks of 256 ternary weights, 54 bytes: qs[48] (5 trits per byte), qh[4] (4 trits per byte), f16 d
  Q4_K  (type 12): blocks of 256 weights, 144 bytes: f16 d, f16 dmin, scales[12] (6-bit), qs[128] (4-bit)
  F16 (1), F32 (0)

GGUF stores a 2-D weight as dims [ne0, ne1] = [in, out]: rows of ne0 contiguous values, ne1 rows (row-major out x in).
3-D expert tensors are [ne0, ne1, ne2] = [in, out, expert].
"""
import mmap
import struct
from pathlib import Path

import numpy as np

QK_K = 256
TYPE_F32, TYPE_F16, TYPE_Q4_K, TYPE_TQ1_0, TYPE_TQ2_0 = 0, 1, 12, 34, 35
BLOCK = {TYPE_Q4_K: (256, 144), TYPE_TQ1_0: (256, 54), TYPE_TQ2_0: (256, 66), TYPE_F16: (1, 2), TYPE_F32: (1, 4)}


class GGUF:
    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.p = 0
        assert self.mm[:4] == b"GGUF", "not a GGUF file"
        self.version = struct.unpack_from("<I", self.mm, 4)[0]
        self.p = 8
        nt, nkv = self._rd("Q"), self._rd("Q")
        self.meta = {}
        for _ in range(nkv):
            k = self._str()
            self.meta[k] = self._val(self._rd("I"))
        self.tensors = {}
        for _ in range(nt):
            name = self._str()
            nd = self._rd("I")
            dims = [self._rd("Q") for _ in range(nd)]
            typ = self._rd("I")
            off = self._rd("Q")
            self.tensors[name] = dict(dims=dims, type=typ, off=off)
        align = self.meta.get("general.alignment", 32)
        self.data_start = (self.p + align - 1) // align * align
        for t in self.tensors.values():
            n = int(np.prod(t["dims"]))
            per, size = BLOCK[t["type"]]
            t["nbytes"] = n // per * size
            t["abs"] = self.data_start + t["off"]

    def _rd(self, fmt):
        v = struct.unpack_from("<" + fmt, self.mm, self.p)
        self.p += struct.calcsize("<" + fmt)
        return v[0]

    def _str(self):
        n = self._rd("Q")
        s = self.mm[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n
        return s

    def _val(self, t):
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self._rd("I"), self._rd("Q")
            return [self._val(et) for _ in range(n)]
        return self._rd({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t])

    def raw(self, name):
        t = self.tensors[name]
        return np.frombuffer(self.mm, dtype=np.uint8, count=t["nbytes"], offset=t["abs"])

    def rows_raw(self, name, expert=None):
        """Raw bytes of a 2-D tensor (or of one expert of a 3-D tensor) as [rows, row_bytes]."""
        t = self.tensors[name]
        ne0 = t["dims"][0]
        ne1 = t["dims"][1]
        per, size = BLOCK[t["type"]]
        rb = ne0 // per * size
        if expert is None:
            n_rows = int(np.prod(t["dims"][1:]))
            return np.frombuffer(self.mm, dtype=np.uint8, count=n_rows * rb, offset=t["abs"]).reshape(n_rows, rb)
        return np.frombuffer(self.mm, dtype=np.uint8, count=ne1 * rb, offset=t["abs"] + expert * ne1 * rb).reshape(ne1, rb)

    def matrix(self, name, expert=None, rows=None):
        """Dequantized float32 matrix [out, in] (one expert of a 3-D tensor if expert is given)."""
        t = self.tensors[name]
        raw = self.rows_raw(name, expert)
        if rows is not None:
            raw = raw[rows]
        return dequant(raw, t["type"], t["dims"][0])


_POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint8)


def _trits(b, n):
    """ggml's trit extraction: digit n of each byte (q = b * 3^n mod 256; xi = (q * 3) >> 8) -> -1, 0, +1."""
    q = (b.astype(np.uint16) * int(_POW3[n])) & 0xFF
    return ((q * 3) >> 8).astype(np.int8) - 1


def tq1_0_codes(raw):
    """raw [rows, nb*54] -> (codes int8 [rows, nb*256] in {-1,0,1}, d float32 [rows, nb])."""
    rows = raw.shape[0]
    blk = raw.reshape(rows, -1, 54)
    qs, qh = blk[:, :, :48], blk[:, :, 48:52]
    d = blk[:, :, 52:54].copy().view(np.float16)[..., 0].astype(np.float32)
    parts = []
    a = qs[:, :, :32]                       # first 32 bytes: 5 digits x 32 -> 160 values
    for n in range(5):
        parts.append(_trits(a, n))
    b = qs[:, :, 32:48]                     # next 16 bytes: 5 digits x 16 -> 80 values
    for n in range(5):
        parts.append(_trits(b, n))
    for n in range(4):                      # qh: 4 digits x 4 -> 16 values
        parts.append(_trits(qh, n))
    codes = np.concatenate(parts, axis=2)   # [rows, nb, 256]
    return codes.reshape(rows, -1), d


def dequant_tq1_0(raw, ncols):
    codes, d = tq1_0_codes(raw)
    rows = raw.shape[0]
    return (codes.reshape(rows, -1, 256).astype(np.float32) * d[:, :, None]).reshape(rows, ncols)


def dequant_q4_k(raw, ncols):
    rows = raw.shape[0]
    blk = raw.reshape(rows, -1, 144)
    d = blk[:, :, 0:2].copy().view(np.float16)[..., 0].astype(np.float32)
    dmin = blk[:, :, 2:4].copy().view(np.float16)[..., 0].astype(np.float32)
    sc = blk[:, :, 4:16].astype(np.uint8)
    qs = blk[:, :, 16:144]
    scale = np.empty(sc.shape[:2] + (8,), np.float32)
    mn = np.empty(sc.shape[:2] + (8,), np.float32)
    for j in range(8):
        if j < 4:
            s, m = sc[:, :, j] & 63, sc[:, :, j + 4] & 63
        else:
            s = (sc[:, :, j + 4] & 0xF) | ((sc[:, :, j - 4] >> 6) << 4)
            m = (sc[:, :, j + 4] >> 4) | ((sc[:, :, j] >> 6) << 4)
        scale[:, :, j], mn[:, :, j] = s, m
    out = np.empty((rows, blk.shape[1], 256), np.float32)
    for g in range(4):                      # 4 groups of 64: low nibbles (sub-block 2g), high nibbles (2g + 1)
        q = qs[:, :, 32 * g: 32 * g + 32]
        lo, hi = (q & 0xF).astype(np.float32), (q >> 4).astype(np.float32)
        out[:, :, 64 * g: 64 * g + 32] = d[..., None] * scale[:, :, 2 * g, None] * lo - dmin[..., None] * mn[:, :, 2 * g, None]
        out[:, :, 64 * g + 32: 64 * g + 64] = d[..., None] * scale[:, :, 2 * g + 1, None] * hi - dmin[..., None] * mn[:, :, 2 * g + 1, None]
    return out.reshape(rows, ncols)


def dequant(raw, typ, ncols):
    if typ == TYPE_TQ1_0:
        return dequant_tq1_0(raw, ncols)
    if typ == TYPE_Q4_K:
        return dequant_q4_k(raw, ncols)
    if typ == TYPE_F16:
        return raw.copy().view(np.float16).astype(np.float32).reshape(raw.shape[0], ncols)
    if typ == TYPE_F32:
        return raw.copy().view(np.float32).reshape(raw.shape[0], ncols)
    raise ValueError(f"type {typ} not supported")


if __name__ == "__main__":
    import sys
    g = GGUF(sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[2] / "models" / "maple" / "maple-preview-TQ1_0-head-Q4_K.gguf"))
    print("GGUF v%d, %d tensors, data at %d" % (g.version, len(g.tensors), g.data_start))
    for k, v in g.meta.items():
        if not k.startswith("tokenizer.ggml.") or k in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
            print("  %s = %s" % (k, v if not isinstance(v, list) or len(v) < 30 else f"[{len(v)} items]"))
    for name in list(g.tensors)[:20]:
        t = g.tensors[name]
        print("  %-36s type %2d dims %s  %.1f MiB" % (name, t["type"], t["dims"], t["nbytes"] / 2**20))
    last = list(g.tensors)[-1]
    t = g.tensors[last]
    end = t["abs"] + t["nbytes"]
    print("last tensor ends at %d, file size %d" % (end, g.path.stat().st_size))
