"""Q4 weight format shared by the converters and the references.

Per row, blocks of 32 weights: q = round(w / d) + 8 in 0..15, d stored as fp16 (one per block, separate array).
Packing: byte j of a block = q[j] | q[j + 16] << 4. Dequantized weight: (q - 8) * d.
"""

import numpy as np

FP16_MIN_NORMAL = 6.2e-5


def quantize_q4(w):
    """w [rows, cols] float -> (packed uint8 [rows, cols / 2], scales float16 [rows, cols / 32])."""
    rows, cols = w.shape
    blocks = w.astype(np.float32).reshape(rows, cols // 32, 32)
    idx = np.abs(blocks).argmax(axis=2)
    peak = np.take_along_axis(blocks, idx[..., None], axis=2)[..., 0]
    d = peak / -8.0                                          # largest magnitude maps to q = 0 (value -8)
    d = np.where(np.abs(d) < FP16_MIN_NORMAL, np.where(d < 0, -FP16_MIN_NORMAL, FP16_MIN_NORMAL), d)
    d16 = d.astype(np.float16)
    q = np.clip(np.rint(blocks / d16.astype(np.float32)[..., None]) + 8, 0, 15).astype(np.uint8)
    packed = q[..., :16] | (q[..., 16:] << 4)
    return packed.reshape(rows, cols // 2), d16


def dequantize_q4(packed, d16):
    rows = packed.shape[0]
    p = packed.reshape(rows, -1, 16)
    q = np.concatenate([p & 15, p >> 4], axis=2).astype(np.float32) - 8.0
    return (q * d16.astype(np.float32)[..., None]).reshape(rows, -1)


def interleave_q4(packed, d16):
    """Q4 -> "q4i" layout for matvec_q4i: rows in tiles of 64; per tile, per block, the 64 rows' 16-byte nibble
    blocks back to back, then (separately) the fp16 scales in the same order. Returns (nibbles, scales) bytes."""
    rows, half = packed.shape
    nb = half // 16
    assert rows % 64 == 0, rows
    nib = packed.reshape(rows // 64, 64, nb, 16).transpose(0, 2, 1, 3)
    sc = d16.reshape(rows // 64, 64, nb).transpose(0, 2, 1)
    return np.ascontiguousarray(nib).tobytes(), np.ascontiguousarray(sc).tobytes()


def quantize_q4_mse(w, factors=(1.0, 0.95, 0.9, 0.85, 0.8)):
    """Same format as quantize_q4, but each block's scale is the one (of a few) with the least squared error."""
    rows, cols = w.shape
    b = w.astype(np.float32).reshape(rows, cols // 32, 32)
    idx = np.abs(b).argmax(axis=2)
    peak = np.take_along_axis(b, idx[..., None], axis=2)[..., 0]
    best_q = best_d = best_err = None
    for f in factors:
        d = peak / -8.0 * f
        d = np.where(np.abs(d) < FP16_MIN_NORMAL, np.where(d < 0, -FP16_MIN_NORMAL, FP16_MIN_NORMAL), d)
        d16 = d.astype(np.float16)
        df = d16.astype(np.float32)[..., None]
        q = np.clip(np.rint(b / df) + 8, 0, 15)
        err = (((q - 8) * df - b) ** 2).sum(axis=2)
        if best_q is None:
            best_q, best_d, best_err = q, d16, err
        else:
            m = err < best_err
            best_q = np.where(m[..., None], q, best_q)
            best_d = np.where(m, d16, best_d)
            best_err = np.where(m, err, best_err)
    q = best_q.astype(np.uint8)
    packed = q[..., :16] | (q[..., 16:] << 4)
    return packed.reshape(rows, cols // 2), best_d
