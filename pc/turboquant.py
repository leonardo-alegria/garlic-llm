"""TurboQuant KV-cache format (Zandieh et al., arXiv 2504.19874, "TurboQuant_mse"): the reference in NumPy and the
source of the shader constants.

Per head vector x (256 dims; k after its RMS norm and RoPE, or v):
  1. rotate: x' = H (s * x), H the 256-point Walsh-Hadamard matrix / 16 (orthonormal, symmetric), s fixed random
     signs (SIGNS, the same for every layer and head). q is rotated the same way, so q' . k' = q . k; values stay
     rotated and the attention output is rotated back once: o = s * (H o').
  2. each coordinate of x' / |x| is distributed like a coordinate of a random unit vector of R^256 (density
     ~ (1 - t^2)^((d-3)/2), close to N(0, 1/d)); scaled by sqrt(d) it gets the Lloyd-Max codebook of that
     distribution with 2^b levels (CODEBOOKS[b], no training, no calibration data).
  3. stored: b-bit indices + one float per vector, the scale |x| / sqrt(d) (or, with ls=True, the least-squares
     scale <x', c> / <c, c>, which never has a larger error).
Dequantized: x' ~ scale * c[idx], and x = s * (H x').
"""

import numpy as np

D = 256


def _lowbias32(x):
    x = np.asarray(x, np.uint64) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


# Random signs: bit i of SIGN_WORDS[i // 32] set = sign -1 (shaders/attn.h has the same 8 words)
_bits = (_lowbias32(np.arange(D) + 0x7A3C0000) >> 7) & 1
SIGN_WORDS = [int(sum(int(_bits[32 * w + k]) << k for k in range(32))) for w in range(D // 32)]
SIGNS = (1.0 - 2.0 * _bits).astype(np.float32)


def hadamard(n=D):
    h = np.array([[1.0]], np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


H = hadamard()


def _lloyd_max(bits, d=D, iters=3000):
    """Lloyd-Max levels for t = sqrt(d) * (a coordinate of a uniform unit vector in R^d)."""
    t = np.linspace(-np.sqrt(d) + 1e-9, np.sqrt(d) - 1e-9, 400001)
    p = np.exp((d - 3) / 2 * np.log1p(-t * t / d))
    p /= p.sum()
    n = 1 << bits
    # start from Gaussian quantiles
    from math import erf, sqrt
    cdf = np.cumsum(p)
    c = np.array([t[np.searchsorted(cdf, (i + 0.5) / n)] for i in range(n)])
    for _ in range(iters):
        b = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(b, t)
        w = np.bincount(idx, p, n)
        m = np.bincount(idx, p * t, n)
        c_new = m / w
        if np.max(np.abs(c_new - c)) < 1e-12:
            c = c_new
            break
        c = c_new
    c = (c - c[::-1]) / 2                    # exactly symmetric
    b = (c[1:] + c[:-1]) / 2
    idx = np.searchsorted(b, t)
    mse = float(np.sum(p * (t - c[idx]) ** 2))
    return c.astype(np.float32), mse


CODEBOOKS, MSE = {}, {}
for _b in (2, 3, 4):
    CODEBOOKS[_b], MSE[_b] = _lloyd_max(_b)
# The 4-bit levels the PS4 uses (chat/engine.c TQ_TABLE), frozen as float32 bit patterns
_FROZEN4 = [0xc02d6e45, 0xc0039f58, 0xbfce1f1d, 0xbfa0262e, 0xbf705f2d, 0xbf2790a4, 0xbec60879, 0xbe030d12,
            0x3e030d12, 0x3ec60879, 0x3f2790a4, 0x3f705f2d, 0x3fa0262e, 0x3fce1f1d, 0x40039f58, 0x402d6e45]
_f4 = np.array(_FROZEN4, np.uint32).view(np.float32)
assert np.max(np.abs(_f4 - CODEBOOKS[4])) < 1e-5, "codebook changed"
CODEBOOKS[4] = _f4


def boundaries(bits):
    c = CODEBOOKS[bits]
    return ((c[1:] + c[:-1]) / 2).astype(np.float32)


def rotate(x):
    """x [..., 256] -> H (s * x) along the last axis (float32)."""
    return (x.astype(np.float32) * SIGNS) @ H


def unrotate(y):
    return (y.astype(np.float32) @ H) * SIGNS


def quantize(x, bits, ls=False):
    """x [..., 256] (unrotated) -> (idx uint8 [..., 256], scale float32 [...])."""
    y = rotate(x)
    norm = np.sqrt(np.sum(y.astype(np.float64) ** 2, axis=-1)).astype(np.float32)
    inv = np.where(norm > 0, np.float32(np.sqrt(D)) / np.maximum(norm, 1e-30), 0).astype(np.float32)
    idx = np.searchsorted(boundaries(bits), y * inv[..., None]).astype(np.uint8)
    c = CODEBOOKS[bits][idx]
    if ls:
        cc = np.sum(c * c, axis=-1)
        scale = np.where(cc > 0, np.sum(y * c, axis=-1) / np.maximum(cc, 1e-30), 0).astype(np.float32)
    else:
        scale = (norm / np.float32(np.sqrt(D))).astype(np.float32)
    return idx, scale


def dequantize_rotated(idx, scale, bits):
    return CODEBOOKS[bits][idx] * scale[..., None]


def dequantize(idx, scale, bits):
    return unrotate(dequantize_rotated(idx, scale, bits))


def pack_k(idx, scale):
    """4-bit k cache of one layer as shaders/attn_tq.h lays it out: idx [max_seq, n_kv, 256], scale [max_seq, n_kv]
    -> bytes of words [n_kv][32][max_seq] then scales [n_kv][max_seq]."""
    ms, nkv, d = idx.shape
    q = idx.astype(np.uint32).reshape(ms, nkv, d // 8, 8)
    words = np.zeros((ms, nkv, d // 8), np.uint32)
    for k in range(8):
        words |= q[..., k] << (4 * k)
    return (np.ascontiguousarray(words.transpose(1, 2, 0)).tobytes() +
            np.ascontiguousarray(scale.T, dtype=np.float32).tobytes())


def unpack_k(data, ms, nkv, d=D):
    words = np.frombuffer(data[: nkv * d // 8 * ms * 4], np.uint32).reshape(nkv, d // 8, ms).transpose(2, 0, 1)
    idx = np.stack([(words >> (4 * k)) & 15 for k in range(8)], axis=-1).reshape(ms, nkv, d).astype(np.uint8)
    scale = np.frombuffer(data[nkv * d // 8 * ms * 4:][: nkv * ms * 4], np.float32).reshape(nkv, ms).T
    return idx, scale


def pack_v(idx, scale):
    """4-bit v cache: idx [max_seq, n_kv, 256] -> bytes [max_seq][n_kv][128] (dim 2b low nibble), then scales
    [max_seq][n_kv]."""
    b = (idx[..., 0::2].astype(np.uint8) | (idx[..., 1::2].astype(np.uint8) << 4))
    return np.ascontiguousarray(b).tobytes() + np.ascontiguousarray(scale, dtype=np.float32).tobytes()


def unpack_v(data, ms, nkv, d=D):
    b = np.frombuffer(data[: ms * nkv * d // 2], np.uint8).reshape(ms, nkv, d // 2)
    idx = np.stack([b & 15, b >> 4], axis=-1).reshape(ms, nkv, d)
    scale = np.frombuffer(data[ms * nkv * d // 2:][: ms * nkv * 4], np.float32).reshape(ms, nkv)
    return idx, scale


def table_words(bits=4):
    """The 40 words the kernels read (shaders/attn_tq.h `tq`): levels, boundaries, 0, sign words."""
    f = lambda a: [int(w) for w in np.asarray(a, np.float32).view(np.uint32)]
    return f(CODEBOOKS[bits]) + f(boundaries(bits)) + [0] + SIGN_WORDS


def c_table(bits=4):
    """C text for chat/engine.c (bit patterns, so the floats are exactly these)."""
    w = table_words(bits)
    lines = [", ".join(f"0x{v:08x}u" for v in w[i:i + 8]) for i in range(0, 40, 8)]
    return "#define TQ_TABLE \\\n    " + ", \\\n    ".join(lines)


if __name__ == "__main__":
    for b in (2, 3, 4):
        print(b, "bits: mse/coord", f"{MSE[b]:.6f}", "levels", np.round(CODEBOOKS[b], 4).tolist())
    print("signs: -1 count", int(_bits.sum()), "words", [hex(w) for w in SIGN_WORDS])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4096, D)).astype(np.float32)
    x[:, 7] *= 30                                            # an outlier channel
    assert np.allclose(unrotate(rotate(x)), x, atol=1e-4)
    for b in (2, 3, 4):
        for ls in (False, True):
            i, s = quantize(x, b, ls)
            r = dequantize(i, s, b)
            print(b, "ls" if ls else "norm", "relative mse", float(np.mean((r - x) ** 2) / np.mean(x ** 2)))
    print(c_table(4))
