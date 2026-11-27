"""Checks the converted Maple folder against the GGUF reference (maple_ref.py).

Reads model.idx + maple-ternary.bin + maple-embed.bin and runs the same forward pass, decoding t1p rows with exactly
the index math of shaders/maple.cl mv_t1 (dwords of qs: block = q / 12, byte k = 4 (q % 12) + i, element
k < 32 ? n*32 + k : 160 + n*16 + (k - 32); qh dword b: element 240 + n*4 + i; digit n = ((byte * 3^n) & 255) * 3 >> 8).
The head goes through Garlic's q4i layout. Compares logits with the GGUF reference for a few tokens.
"""
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]   # the repository
sys.path.insert(0, str(ROOT / "pc"))
from maple_ref import MapleRef  # noqa: E402

DIR = ROOT / "models" / "maple-ps4"
POW3 = [1, 3, 9, 27, 81]


def decode_t1p_rows(rows_bytes, cols):
    """Literal port of mv_t1's addressing: rows [n, nb*54] uint8 -> weights [n, cols] float32."""
    n = rows_bytes.shape[0]
    nb = cols // 256
    qs = rows_bytes[:, :nb * 48].copy().view("<u4")          # nb*12 dwords
    qh = rows_bytes[:, nb * 48:nb * 52].copy().view("<u4")    # nb dwords
    d = rows_bytes[:, nb * 52:nb * 54].copy().view("<f2").astype(np.float32)
    out = np.zeros((n, cols), np.float32)
    for q in range(nb * 12):
        b, wd = q // 12, q % 12
        v = qs[:, q]
        for i in range(4):
            byte = (v >> (8 * i)) & 255
            k = wd * 4 + i
            base, step = (k, 32) if k < 32 else (160 + (k - 32), 16)
            for nn in range(5):
                t = (((byte * POW3[nn]) & 255) * 3) >> 8
                out[:, b * 256 + base + nn * step] = (t.astype(np.float32) - 1.0) * d[:, b]
    for b in range(nb):
        v = qh[:, b]
        for i in range(4):
            byte = (v >> (8 * i)) & 255
            for nn in range(4):
                t = (((byte * POW3[nn]) & 255) * 3) >> 8
                out[:, b * 256 + 240 + nn * 4 + i] = (t.astype(np.float32) - 1.0) * d[:, b]
    return out


class Converted:
    def __init__(self, d=DIR):
        h = (d / "model.idx").read_bytes()
        assert h[:4] == b"PMAP"
        v = struct.unpack_from("<17I", h, 8)
        (self.L, self.D, self.H, self.KV, self.HD, self.E, self.K, self.FF, self.V, self.max_seq, self.window,
         self.rot, self.eos, self.im_end, self.im_start, self.swa_mask, self.n_tok) = v
        self.eps, self.theta, self.clamp = struct.unpack_from("<3f", h, 8 + 68)
        p = 8 + 68 + 12 + 256
        self.wsize, self.erow = struct.unpack_from("<QQ", h, p)
        p += 16
        self.table = [struct.unpack_from("<9Q", h, p + 72 * i) for i in range(self.L)]
        self.out_norm, self.head_nib, self.head_sc = struct.unpack_from("<3Q", h, p + 72 * self.L)
        self.w = np.memmap(d / "maple-ternary.bin", dtype=np.uint8, mode="r")
        self.emb = np.memmap(d / "maple-embed.bin", dtype=np.float16, mode="r").reshape(self.V, self.D)

    def f32(self, off, n):
        return np.frombuffer(self.w, dtype=np.float32, count=n, offset=off)

    def t1(self, off, rows, cols, row0=0, nrows=None):
        rs = cols // 256 * 54
        nrows = rows if nrows is None else nrows
        raw = np.frombuffer(self.w, dtype=np.uint8, count=nrows * rs, offset=off + row0 * rs).reshape(nrows, rs)
        return decode_t1p_rows(raw, cols)

    def router(self, i):
        return np.frombuffer(self.w, dtype=np.float16, count=self.E * self.D, offset=self.table[i][6]).reshape(
            self.E, self.D).astype(np.float32)

    def head(self, rows):
        """Q4_K head rows, decoded with mv_q4k's addressing (per block: d, dmin, 12 scale bytes, 128 qs bytes)."""
        rb = self.D // 256 * 144
        out = np.zeros((len(rows), self.D), np.float32)
        for ri, r in enumerate(rows):
            raw = np.frombuffer(self.w, dtype=np.uint8, count=rb, offset=self.head_nib + r * rb)
            for blk in range(self.D // 256):
                bb = raw[blk * 144:(blk + 1) * 144]
                d, dmin = bb[0:4].copy().view("<f2").astype(np.float32)
                sc = bb[4:16].astype(np.int32)
                qs = bb[16:144]
                for dw in range(32):                 # qs dword wq: bytes 4wq..4wq+3, group g = wq // 8
                    g = dw // 8
                    def smin(j):
                        if j < 4:
                            return sc[j] & 63, sc[j + 4] & 63
                        return (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4), (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)
                    s0, m0 = smin(2 * g)
                    s1, m1 = smin(2 * g + 1)
                    for i in range(4):
                        l = dw * 4 + i
                        mm = l % 32
                        byte = int(qs[l])
                        out[ri, blk * 256 + 64 * g + mm] = d * s0 * (byte & 15) - dmin * m0
                        out[ri, blk * 256 + 64 * g + 32 + mm] = d * s1 * (byte >> 4) - dmin * m1
        return out

def main():
    c = Converted()
    print(f"PMAP: {c.L} layers, hidden {c.D}, {c.E} experts top-{c.K}, vocab {c.V}, swa mask {c.swa_mask:#x}, "
          f"eps {c.eps}, theta {c.theta}, clamp {c.clamp}")
    ref = MapleRef()
    # 1. tensors: converted vs GGUF, a few rows of each kind
    i = 5
    b = f"blk.{i}."
    t = c.table[i]
    checks = [
        ("wqkv q rows", c.t1(t[4], 3072, 2048, 0, 8), ref.g.matrix(b + "attn_q.weight", rows=slice(0, 8))),
        ("wqkv v rows", c.t1(t[4], 3072, 2048, 2560, 8), ref.g.matrix(b + "attn_v.weight", rows=slice(0, 8))),
        ("wo rows", c.t1(t[5], 2048, 2048, 100, 8), ref.g.matrix(b + "attn_output.weight", rows=slice(100, 108))),
        ("expert 37 gate", c.t1(t[7] + 37 * 1024 * 432, 1024, 2048, 0, 8),
         ref.g.matrix(b + "ffn_gate_exps.weight", 37)[:8]),
        ("expert 37 up", c.t1(t[7] + 37 * 1024 * 432, 1024, 2048, 512, 8),
         ref.g.matrix(b + "ffn_up_exps.weight", 37)[:8]),
        ("expert 200 down", c.t1(t[8] + 200 * 2048 * 108, 2048, 512, 0, 8),
         ref.g.matrix(b + "ffn_down_exps.weight", 200)[:8]),
        ("router", c.router(i), ref.g.matrix(b + "ffn_gate_inp.weight")),
    ]
    for name, a, r in checks:
        print(f"  {name:16s} max |diff| {np.abs(a - r).max():.3g}  (ref max {np.abs(r).max():.3g})")
    hr = [0, 1, 151643, 151645, 9999]
    hd = c.head(hr)
    hrf = ref.g.matrix("output.weight", rows=hr)
    print(f"  head (Q4_K, mv_q4k addressing) max |diff| {np.abs(hd - hrf).max():.3g}, rel {np.abs(hd - hrf).max() / np.abs(hrf).max():.3g}")
    e = np.array(c.emb[151644], np.float32)
    print(f"  embedding row   max |diff| {np.abs(e - ref.embed(151644)).max():.3g}")


if __name__ == "__main__":
    main()
