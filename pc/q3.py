"""Q3 weight format ("q3i"), for the largest matrices of Qwen3.5-9B (MLP gate/up): 3.5 bits per weight.

Per block of 32 weights: q = round(w / d) + 4 in 0..7 (value (q - 4) * d), d an fp16 scale picked among a few
candidates to minimise the block's squared error. 12 bytes per block, three little-endian words:
  d0  2 low bits of weights 0..15:  weight 4m + k (m, k in 0..3) at bits 8k + 2m, 8k + 2m + 1
  d1  the same for weights 16..31
  d2  high bit of weight 4m + k (m in 0..7, k in 0..3) at bit 8k + m
so (d >> 2m) & 0x03030303 holds 4 consecutive weights' low bits in its 4 bytes, and the high bits move to bit 2
of each byte with one shift (shaders/common.h: matvec_q3i).
Interleaved like q4i: rows in tiles of 64; tile t, block b, row t * 64 + l at ((t * nb + b) * 64 + l) * 12 bytes,
its scale at index (t * nb + b) * 64 + l.
"""

import numpy as np

FP16_MIN_NORMAL = 6.2e-5
FACTORS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


def quantize_q3(w):
    """w [rows, cols] -> (q uint8 [rows, cols] in 0..7, d16 float16 [rows, cols / 32])."""
    rows, cols = w.shape
    b = w.astype(np.float32).reshape(rows, cols // 32, 32)
    idx = np.abs(b).argmax(axis=2)
    peak = np.take_along_axis(b, idx[..., None], axis=2)[..., 0]
    best_q = best_d = best_err = None
    for f in FACTORS:
        d = (peak / -4.0 * f)
        d = np.where(np.abs(d) < FP16_MIN_NORMAL, np.where(d < 0, -FP16_MIN_NORMAL, FP16_MIN_NORMAL), d)
        d16 = d.astype(np.float16)
        df = d16.astype(np.float32)[..., None]
        q = np.clip(np.rint(b / df) + 4, 0, 7)
        err = (((q - 4) * df - b) ** 2).sum(axis=2)
        if best_q is None:
            best_q, best_d, best_err = q, d16, err
        else:
            m = err < best_err
            best_q = np.where(m[..., None], q, best_q)
            best_d = np.where(m, d16, best_d)
            best_err = np.where(m, err, best_err)
    return best_q.astype(np.uint8).reshape(rows, cols), best_d


def dequantize_q3(q, d16):
    rows, cols = q.shape
    return ((q.reshape(rows, cols // 32, 32).astype(np.float32) - 4) * d16.astype(np.float32)[..., None]).reshape(rows, cols)


def pack_q3(q):
    """q [rows, cols] (0..7) -> [rows, cols / 32, 3] uint32 words d0, d1, d2 per block."""
    rows, cols = q.shape
    b = q.reshape(rows, cols // 32, 32).astype(np.uint32)
    lo, hi = b & 3, b >> 2
    words = np.zeros((rows, cols // 32, 3), np.uint32)
    for half in range(2):
        for m in range(4):
            for k in range(4):
                words[..., half] |= lo[..., 16 * half + 4 * m + k] << (8 * k + 2 * m)
    for m in range(8):
        for k in range(4):
            words[..., 2] |= hi[..., 4 * m + k] << (8 * k + m)
    return words


def interleave_q3(q, d16):
    """-> (packed bytes, scale bytes) in the q3i layout."""
    rows, cols = q.shape
    nb = cols // 32
    assert rows % 64 == 0, rows
    words = pack_q3(q).reshape(rows // 64, 64, nb, 3).transpose(0, 2, 1, 3)
    sc = d16.reshape(rows // 64, 64, nb).transpose(0, 2, 1)
    return np.ascontiguousarray(words).tobytes(), np.ascontiguousarray(sc).tobytes()
