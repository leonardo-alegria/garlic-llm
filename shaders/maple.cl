// Maple-Preview (deepgrove, 20B-A1B ternary MoE) decode kernels for the PS4 GPU (GCN gfx701), compiled by LLVM from
// OpenCL C 1.2 without its standard headers. Weights come from pc/convert_maple.py (maple/convert_maple.py); the
// engine side is chat/engine_maple.inc. One token per pass: per layer rmsnorm -> qkv (ternary) -> q/k norm (+RoPE on
// sliding layers) -> attention (sliding window 512, ring of 576 positions; or global, no RoPE) -> o (ternary, added
// to x) -> rmsnorm -> router (f16) -> top-8 -> 8 experts' gate|up (ternary) -> silu(min(g,7)) * clamp(u,7) * weight
// -> 8 experts' down (ternary) -> sum into x. Then rmsnorm -> head (Q4_K) -> sample.

#include "common.h"

#define HEAD_DIM 128
// sliding-window layers
#define ROPE_DIM 64
#define MAX_SEQ 576
#define ATTN_WINDOW
#define ATTN_NAME(x) swa_##x
#include "attn.h"
#undef ATTN_WINDOW
#undef ATTN_NAME
#undef ROPE_DIM
#undef MAX_SEQ
// global layers (no RoPE)
#define ROPE_DIM 0
#define MAX_SEQ 16384
#define ATTN_NAME(x) glb_##x
#include "attn.h"
#undef ATTN_NAME

// Sum over the 64 lanes of a wave, without LDS round trips or barriers: a butterfly over each half (ds_swizzle
// SWAP 1..16), then the two halves (readlane). The result is the same in every lane.
static float wave_reduce(float v) {
    v += __builtin_astype(__builtin_amdgcn_ds_swizzle(__builtin_astype(v, int), 0x041F), float);
    v += __builtin_astype(__builtin_amdgcn_ds_swizzle(__builtin_astype(v, int), 0x081F), float);
    v += __builtin_astype(__builtin_amdgcn_ds_swizzle(__builtin_astype(v, int), 0x101F), float);
    v += __builtin_astype(__builtin_amdgcn_ds_swizzle(__builtin_astype(v, int), 0x201F), float);
    v += __builtin_astype(__builtin_amdgcn_ds_swizzle(__builtin_astype(v, int), 0x401F), float);
    return __builtin_astype(__builtin_amdgcn_readlane(__builtin_astype(v, int), 0), float) +
           __builtin_astype(__builtin_amdgcn_readlane(__builtin_astype(v, int), 32), float);
}

// Ternary matrix x vector, t1p rows (TQ1_0 bits, per row: the blocks' qs [48 bytes each], qh [4 each], f16 d [2 each]).
// 256 threads = 4 waves = 4 rows per group, one row per wave (the layout proven on the console; bigger groups and
// per-wave row loops gave wrong rows). The input is read straight from global memory (cached in L1/L2: 8 KB),
// not copied to LDS for every 4 rows. Slot s of R rows uses expert ids[s] (w + ids[s] * es) and input x + s * xstride
// (dense: ids -> zeros, es = 0, xstride = 0, R = rows). y[s * R + r] = (add ? y : 0) + sum_c w[r][c] x[c].
// rpg is unused (the engine passes 4).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mv_t1(global const uchar* w, uint rs, ulong es, global const uint* ids, global const float* x, uint xstride,
           global float* y, uint R, uint cols, uint add, uint rpg) {
    uint t = LID, wv = t >> 6, l = t & 63;
    uint row0 = GID * 4, slot = row0 / R;
    global const float* xs = x + slot * xstride;
    uint r = row0 + wv - slot * R, nb = cols >> 8;
    global const uchar* rowp = w + (ulong)ids[slot] * es + (ulong)r * rs;
    global const uint* qs = (global const uint*)rowp;
    global const uint* qh = qs + nb * 12;
    global const half_t* dd = (global const half_t*)(qh + nb);
    float acc = 0.0f;
    for (uint q = l; q < nb * 12; q += 64) {
        uint v = qs[q], b = q / 12, wd = q - b * 12;
        global const float* xb = xs + b * 256;
        float s = 0.0f;
#pragma unroll
        for (uint i = 0; i < 4; i++) {
            uint byte = (v >> (8 * i)) & 255u, k = wd * 4 + i;
            uint base = k < 32 ? k : 128 + k, step = k < 32 ? 32 : 16;   // (160 + (k - 32) = 128 + k)
            uint m3 = 1;
#pragma unroll
            for (uint n = 0; n < 5; n++) {
                uint d3 = (((byte * m3) & 255u) * 3u) >> 8;   // ggml's trit: 0, 1, 2 for -1, 0, +1
                s += ((float)d3 - 1.0f) * xb[base + n * step];
                m3 *= 3u;
            }
        }
        acc += (float)dd[b] * s;
    }
    if (l < nb) {
        uint v = qh[l];
        global const float* xb = xs + l * 256 + 240;
        float s = 0.0f;
#pragma unroll
        for (uint i = 0; i < 4; i++) {
            uint byte = (v >> (8 * i)) & 255u, m3 = 1;
#pragma unroll
            for (uint n = 0; n < 4; n++) {
                uint d3 = (((byte * m3) & 255u) * 3u) >> 8;
                s += ((float)d3 - 1.0f) * xb[n * 4 + i];
                m3 *= 3u;
            }
        }
        acc += (float)dd[l] * s;
    }
    float sum = wave_reduce(acc);
    if (l == 0) {
        uint o = slot * R + r;
        y[o] = (add ? y[o] : 0.0f) + sum;
    }
}

// Q4_K matrix x vector (the head, rows as in the GGUF: per 256 weights f16 d, f16 dmin, 12 bytes of 6-bit scales and
// mins, 128 bytes of nibbles). 4 rows per group, one row per wave, input from global memory; rpg unused.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mv_q4k(global const uchar* w, global const float* x, global float* y, uint cols, uint rows, uint rpg) {
    uint t = LID, wv = t >> 6, l = t & 63;
    uint r = GID * 4 + wv, nb = cols >> 8;
    float acc = 0.0f;
    if (r < rows) {
        global const uint* rowp = (global const uint*)(w + (ulong)r * nb * 144);
        for (uint dw = l; dw < nb * 32; dw += 64) {
            uint b = dw >> 5, wq = dw & 31, g = wq >> 3;
            global const uint* bp = rowp + b * 36;
            global const half_t* dh = (global const half_t*)bp;
            float d = (float)dh[0], dmin = (float)dh[1];
            global const uchar* sc = (global const uchar*)(bp + 1);
            uint j0 = 2 * g, j1 = 2 * g + 1, s0, m0, s1, m1;
            if (j0 < 4) {
                s0 = sc[j0] & 63u, m0 = sc[j0 + 4] & 63u;
                s1 = sc[j1] & 63u, m1 = sc[j1 + 4] & 63u;
            } else {
                s0 = (sc[j0 + 4] & 15u) | ((sc[j0 - 4] >> 6) << 4), m0 = (sc[j0 + 4] >> 4) | ((sc[j0] >> 6) << 4);
                s1 = (sc[j1 + 4] & 15u) | ((sc[j1 - 4] >> 6) << 4), m1 = (sc[j1 + 4] >> 4) | ((sc[j1] >> 6) << 4);
            }
            uint v = bp[4 + wq];
            global const float* x0 = x + b * 256 + 64 * g + ((wq * 4) & 31u);
            global const float* x1 = x0 + 32;
            float sl = 0.0f, sh = 0.0f, xs0 = 0.0f, xs1 = 0.0f;
#pragma unroll
            for (uint i = 0; i < 4; i++) {
                uint byte = (v >> (8 * i)) & 255u;
                sl += (float)(byte & 15u) * x0[i];
                sh += (float)(byte >> 4) * x1[i];
                xs0 += x0[i];
                xs1 += x1[i];
            }
            acc += d * ((float)s0 * sl + (float)s1 * sh) - dmin * ((float)m0 * xs0 + (float)m1 * xs1);
        }
    }
    float sum = wave_reduce(acc);
    if (l == 0 && r < rows) y[r] = sum;
}

// f16 matrix x vector (the router, 256 x 2048): one row per wave, 4 per group; each lane reads 16 bytes (8 weights)
// per step, 512 columns per step (cols a multiple of 512).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mv_f16(global const half_t* w, global const float* x, global float* y, uint cols) {
    uint t = LID, wv = t >> 6, l = t & 63, r = GID * 4 + wv;
    global const uint4v* wr = (global const uint4v*)(w + (ulong)r * cols);
    float a = 0.0f;
    uint steps = cols >> 9;
    for (uint k = 0; k < steps; k++) {
        uint4v v = wr[k * 64 + l];
        global const float* xc = x + k * 512 + l * 8;
        a += (float)__builtin_astype((ushort)(v.x & 0xFFFFu), half_t) * xc[0];
        a += (float)__builtin_astype((ushort)(v.x >> 16), half_t) * xc[1];
        a += (float)__builtin_astype((ushort)(v.y & 0xFFFFu), half_t) * xc[2];
        a += (float)__builtin_astype((ushort)(v.y >> 16), half_t) * xc[3];
        a += (float)__builtin_astype((ushort)(v.z & 0xFFFFu), half_t) * xc[4];
        a += (float)__builtin_astype((ushort)(v.z >> 16), half_t) * xc[5];
        a += (float)__builtin_astype((ushort)(v.w & 0xFFFFu), half_t) * xc[6];
        a += (float)__builtin_astype((ushort)(v.w >> 16), half_t) * xc[7];
    }
    float sum = wave_reduce(a);
    if (l == 0) y[r] = sum;
}

// Top-k experts of n (<= 256) router logits (k <= 8); weights = softmax over the chosen ones (the same as softmax over
// all, then renormalised over the top k). Two levels of ranking (a logit beats another if larger, or equal with a
// smaller index; loops kept rolled: this kernel runs as one group, so every line of its code is an instruction-cache
// miss each time, and a short program is what makes it fast): within each wave, by reading the 64 values from registers (readlane: no LDS, no barrier), keeping
// each wave's top k; then the 4k candidates ranked among themselves.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void moe_topk(global const float* logits, uint n, uint k, global uint* ids, global float* wts) {
    local float cv[32];
    local uint ci[32];
    local float top[8];
    uint t = LID, wv = t >> 6, l = t & 63;
    logits += GID * n, ids += GID * k, wts += GID * k;   // (a batch: group g = token g)
    float mine = t < n ? logits[t] : -3.0e38f;
    int mi = __builtin_astype(mine, int);
    uint rank = 0;
#pragma unroll 1
    for (uint j = 0; j < 64; j++) {
        float o = __builtin_astype(__builtin_amdgcn_readlane(mi, j), float);
        rank += (o > mine || (o == mine && j < l)) ? 1u : 0u;
    }
    if (rank < k) {
        cv[wv * 8 + rank] = mine;
        ci[wv * 8 + rank] = t;
    }
    BARRIER();
    if (t < 32 && (t & 7) < k) {
        float c = cv[t];
        uint cidx = ci[t], r2 = 0;
#pragma unroll 1
        for (uint j = 0; j < 32; j++) {
            if ((j & 7) < k) {
                float o = cv[j];
                uint oi = ci[j];
                r2 += (o > c || (o == c && oi < cidx)) ? 1u : 0u;
            }
        }
        if (r2 < k && cidx < n) {
            ids[r2] = cidx;
            top[r2] = c;
        }
    }
    BARRIER();
    if (t < k) {
        float m = top[0], sum = 0.0f;
#pragma unroll 1
        for (uint j = 0; j < k; j++) sum += exp_(top[j] - m);
        wts[t] = exp_(top[t] - m) / sum;
    }
}

// h[s][i] = wts[s] * silu(min(gate, lim)) * clamp(up, -lim, lim); gu[s] = [gate ff | up ff]. n = k * ff threads.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void moe_act(global const float* gu, global const float* wts, global float* h, uint ff, uint n, float lim) {
    uint i = GID * 256 + LID;
    if (i >= n) return;
    uint s = i / ff, j = i - s * ff;
    float g = __builtin_fminf(gu[s * 2 * ff + j], lim);
    float u = __builtin_fminf(__builtin_fmaxf(gu[s * 2 * ff + ff + j], -lim), lim);
    h[i] = wts[s] * (g / (1.0f + exp_(-g))) * u;
}

// x[i] += sum over the k experts' outputs y[s][i].
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void moe_sum(global float* x, global const float* y, uint n, uint k) {
    uint i = GID * 256 + LID;
    if (i >= n) return;
    float a = 0.0f;
    for (uint s = 0; s < k; s++) a += y[s * n + i];
    x[i] += a;
}

// ---- tile-interleaved layout (model.idx version 2, convert_maple.py --t1i) -------------------------------------------
// Rows in tiles of 64, dword-interleaved: dword d of row (tile, lane) at ((tile * rd + d) * 64 + lane) * 4, rd = dwords
// per row. Lane = row: one load fetches 256 contiguous bytes for 64 rows. A group (4 waves) takes one tile; wave w takes
// the row's blocks w, w + 4, ...; the input is staged in LDS once per 64 rows and read at uniform addresses (broadcast);
// the 4 waves' partial sums meet in LDS. No per-wave row loop, no cross-lane shuffles.

static float half_bits(uint h) { return (float)__builtin_astype((ushort)h, half_t); }

// ternary: row = nb blocks of qs (12 dwords each), then nb qh dwords, then nb f16 scales (nb / 2 dwords).
// Decode by table: lo[byte] = its first 4 trits + 1 as bytes, hi[byte] = the 5th + 1 (two arrays: lanes spread over all
// 32 LDS banks); v_cvt_f32_ubyteN makes each a float in one instruction; sum (t + 1) x - sum x = sum t x, sum x per
// dword (xsum) once per group. Input values are the same for every lane (all 64 rows use x[c] together): uniform
// addresses, scalar loads into SGPRs, which v_mad takes directly.

// one block of a row: 12 qs dwords + qh dword in v[0..12], input xb (uniform), scale d, xsum xsb[0..12]
static inline __attribute__((always_inline)) float t1_block(const uint* v, global const float* restrict xb, float d,
                                                             local const float* xsb, local const uint* lo_t,
                                                             local const uint* hi_t) {
    float s = 0.0f, sx = 0.0f;
#pragma unroll
    for (uint j = 0; j < 12; j++) {
        const uint base = j < 8 ? 4 * j : 128 + 4 * j, step = j < 8 ? 32 : 16;
#pragma unroll
        for (uint i = 0; i < 4; i++) {
            uint byte = (v[j] >> (8 * i)) & 255u;
            uint lo = lo_t[byte], hi = hi_t[byte];
            s += (float)(lo & 255u) * xb[base + i];
            s += (float)((lo >> 8) & 255u) * xb[base + step + i];
            s += (float)((lo >> 16) & 255u) * xb[base + 2 * step + i];
            s += (float)(lo >> 24) * xb[base + 3 * step + i];
            s += (float)hi * xb[base + 4 * step + i];
        }
        sx += xsb[j];
    }
#pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint lo = lo_t[(v[12] >> (8 * i)) & 255u];
        s += (float)(lo & 255u) * xb[240 + i];
        s += (float)((lo >> 8) & 255u) * xb[244 + i];
        s += (float)((lo >> 16) & 255u) * xb[248 + i];
        s += (float)(lo >> 24) * xb[252 + i];
    }
    sx += xsb[12];
    return d * (s - sx);
}

// dwords J0 .. J1-1 of one block (12 = the qh dword), static: v[j - J0]
#define T1_PART(J0, J1)                                                                                                 \
    static inline __attribute__((always_inline)) float t1_part_##J0(const uint* v, global const float* restrict xb,   \
                                                                    float d, local const float* xsb,                 \
                                                                    local const uint* lo_t, local const uint* hi_t) { \
        float s = 0.0f, sx = 0.0f;                                                                                    \
        _Pragma("unroll") for (uint j = J0; j < J1; j++) {                                                            \
            uint vj = v[j - J0];                                                                                      \
            if (j < 12) {                                                                                             \
                const uint base = j < 8 ? 4 * j : 128 + 4 * j, step = j < 8 ? 32 : 16;                                \
                _Pragma("unroll") for (uint i = 0; i < 4; i++) {                                                      \
                    uint byte = (vj >> (8 * i)) & 255u;                                                               \
                    uint lo = lo_t[byte], hi = hi_t[byte];                                                            \
                    s += (float)(lo & 255u) * xb[base + i];                                                           \
                    s += (float)((lo >> 8) & 255u) * xb[base + step + i];                                             \
                    s += (float)((lo >> 16) & 255u) * xb[base + 2 * step + i];                                        \
                    s += (float)(lo >> 24) * xb[base + 3 * step + i];                                                 \
                    s += (float)hi * xb[base + 4 * step + i];                                                         \
                }                                                                                                     \
            } else {                                                                                                  \
                _Pragma("unroll") for (uint i = 0; i < 4; i++) {                                                      \
                    uint lo = lo_t[(vj >> (8 * i)) & 255u];                                                           \
                    s += (float)(lo & 255u) * xb[240 + i];                                                            \
                    s += (float)((lo >> 8) & 255u) * xb[244 + i];                                                     \
                    s += (float)((lo >> 16) & 255u) * xb[248 + i];                                                    \
                    s += (float)(lo >> 24) * xb[252 + i];                                                             \
                }                                                                                                     \
            }                                                                                                         \
            sx += xsb[j];                                                                                             \
        }                                                                                                             \
        return d * (s - sx);                                                                                          \
    }
T1_PART(0, 7)
T1_PART(7, 13)

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mv_t1i(global const uint* w, uint rd, ulong es, global const uint* ids, global const float* restrict x, uint xstride,
            global float* y, uint R, uint cols, uint add) {
    local float xsum[8 * 13];
    local uint lo_t[256], hi_t[256];
    local float part[256];
    uint t = LID, wv = __builtin_amdgcn_readfirstlane(t >> 6), l = t & 63;
    uint row0 = GID * 64, slot = __builtin_amdgcn_readfirstlane(row0 / R);
    global const float* restrict xs = x + slot * xstride;
    {   // the table: trits of byte value t, + 1
        uint e[5], m3 = 1;
        for (uint n = 0; n < 5; n++) {
            e[n] = (((t * m3) & 255u) * 3u) >> 8;
            m3 *= 3u;
        }
        lo_t[t] = e[0] | (e[1] << 8) | (e[2] << 16) | (e[3] << 24);
        hi_t[t] = e[4];
    }
    uint nb = cols >> 8;
    if (t < nb * 13) {   // xsum[b * 13 + j]: the elements dword j of block b multiplies
        uint b = t / 13, j = t - b * 13;
        global const float* xb = xs + b * 256;
        float sx = 0.0f;
        if (j < 12) {
            uint base = j < 8 ? 4 * j : 128 + 4 * j, step = j < 8 ? 32 : 16;
            for (uint n = 0; n < 5; n++)
                for (uint i = 0; i < 4; i++) sx += xb[base + n * step + i];
        } else {
            for (uint i = 0; i < 16; i++) sx += xb[240 + i];
        }
        xsum[t] = sx;
    }
    BARRIER();
    uint tile = (row0 - slot * R) >> 6;
    global const uint* tp = (global const uint*)((global const uchar*)w + (ulong)ids[slot] * es) + (ulong)tile * rd * 64 + l;
    float acc = 0.0f;
    if (nb == 8) {   // 2048 columns: wave wv owns blocks wv and wv + 4; every offset is a constant
        uint b0 = wv, b1 = wv + 4;
        uint v0[13], v1[13];
#pragma unroll
        for (uint j = 0; j < 12; j++) {
            v0[j] = tp[(ulong)(b0 * 12 + j) * 64];
            v1[j] = tp[(ulong)(b1 * 12 + j) * 64];
        }
        v0[12] = tp[(ulong)(96 + b0) * 64];
        v1[12] = tp[(ulong)(96 + b1) * 64];
        uint s0 = tp[(ulong)(104 + (b0 >> 1)) * 64], s1 = tp[(ulong)(104 + (b1 >> 1)) * 64];
        float d0 = half_bits(b0 & 1 ? s0 >> 16 : s0 & 0xFFFFu), d1 = half_bits(b1 & 1 ? s1 >> 16 : s1 & 0xFFFFu);
        acc = t1_block(v0, xs + b0 * 256, d0, xsum + b0 * 13, lo_t, hi_t) +
              t1_block(v1, xs + b1 * 256, d1, xsum + b1 * 13, lo_t, hi_t);
    } else if (nb == 2) {   // 512 columns (expert down): wave wv takes half of block wv >> 1, statically
        uint b = wv >> 1, h = wv & 1;
        uint sdw = tp[(ulong)(26 + 0) * 64];   // both blocks' scales are in dword nb * 13 = 26
        float d = half_bits(b ? sdw >> 16 : sdw & 0xFFFFu);
        global const float* restrict xb = xs + b * 256;
        if (h == 0) {
            uint v[7];
#pragma unroll
            for (uint j = 0; j < 7; j++) v[j] = tp[(ulong)(b * 12 + j) * 64];
            acc = t1_part_0(v, xb, d, xsum + b * 13, lo_t, hi_t);
        } else {
            uint v[6];
#pragma unroll
            for (uint j = 7; j < 12; j++) v[j - 7] = tp[(ulong)(b * 12 + j) * 64];
            v[5] = tp[(ulong)(24 + b) * 64];   // qh dword of block b: nb * 12 + b
            acc = t1_part_7(v, xb, d, xsum + b * 13, lo_t, hi_t);
        }
    } else {   // other widths: units u = block u / 13, dword u % 13, wave wv takes wv, wv + 4, ...
        uint nu = nb * 13;
        uint vv[26], sd[26];
#pragma unroll
        for (uint k = 0; k < 26; k++) {
            uint u = wv + 4 * k;
            if (u < nu) {
                uint b = u / 13, jj = u - b * 13;
                vv[k] = tp[(ulong)(jj < 12 ? b * 12 + jj : nb * 12 + b) * 64];
                sd[k] = tp[(ulong)(nb * 13 + (b >> 1)) * 64];
            }
        }
#pragma unroll
        for (uint k = 0; k < 26; k++) {
            uint u = wv + 4 * k;
            if (u < nu) {
                uint b = u / 13, jj = u - b * 13;
                global const float* restrict xb = xs + b * 256;
                float d = half_bits(b & 1 ? sd[k] >> 16 : sd[k] & 0xFFFFu);
                uint v = vv[k];
                float s = 0.0f;
                if (jj < 12) {
                    uint base = jj < 8 ? 4 * jj : 128 + 4 * jj, step = jj < 8 ? 32 : 16;
#pragma unroll
                    for (uint i = 0; i < 4; i++) {
                        uint byte = (v >> (8 * i)) & 255u;
                        uint lo = lo_t[byte], hi = hi_t[byte];
                        s += (float)(lo & 255u) * xb[base + i];
                        s += (float)((lo >> 8) & 255u) * xb[base + step + i];
                        s += (float)((lo >> 16) & 255u) * xb[base + 2 * step + i];
                        s += (float)(lo >> 24) * xb[base + 3 * step + i];
                        s += (float)hi * xb[base + 4 * step + i];
                    }
                } else {
#pragma unroll
                    for (uint i = 0; i < 4; i++) {
                        uint lo = lo_t[(v >> (8 * i)) & 255u];
                        s += (float)(lo & 255u) * xb[240 + i];
                        s += (float)((lo >> 8) & 255u) * xb[244 + i];
                        s += (float)((lo >> 16) & 255u) * xb[248 + i];
                        s += (float)(lo >> 24) * xb[252 + i];
                    }
                }
                acc += d * (s - xsum[u]);
            }
        }
    }
    part[t] = acc;
    BARRIER();
    if (wv == 0) {
        float r = part[l] + part[64 + l] + part[128 + l] + part[192 + l];
        uint o = row0 + l;
        y[o] = (add ? y[o] : 0.0f) + r;
    }
}

// Q4_K head, interleaved: row = nb blocks of 36 dwords (d | dmin, 3 dwords of scales, 32 dwords of nibbles)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mv_q4ki(global const uint* w, global const float* x, global float* y, uint cols, uint rows) {
    local float xl[2048];
    local float part[256];
    uint t = LID, wv = __builtin_amdgcn_readfirstlane(t >> 6), l = t & 63;
    for (uint i = t; i < cols; i += 256) xl[i] = x[i];
    BARRIER();
    uint nb = cols >> 8, rd = nb * 36;
    global const uint* tp = w + (ulong)GID * rd * 64 + l;
    float acc = 0.0f;
    for (uint b = wv; b < nb; b += 4) {
        global const uint* bp = tp + (ulong)b * 36 * 64;
        uint dw = bp[0];
        float d = half_bits(dw & 0xFFFFu), dmin = half_bits(dw >> 16);
        uint sw[3] = {bp[64], bp[128], bp[192]};
        float sc[8], mn[8];
#pragma unroll
        for (uint j = 0; j < 8; j++) {
            uint s6, m6;
#define SCB(n) ((sw[(n) >> 2] >> (8 * ((n) & 3))) & 255u)
            if (j < 4) s6 = SCB(j) & 63u, m6 = SCB(j + 4) & 63u;
            else s6 = (SCB(j + 4) & 15u) | ((SCB(j - 4) >> 6) << 4), m6 = (SCB(j + 4) >> 4) | ((SCB(j) >> 6) << 4);
#undef SCB
            sc[j] = d * (float)s6, mn[j] = dmin * (float)m6;
        }
        local const float* xb = xl + b * 256;
#pragma unroll
        for (uint g = 0; g < 4; g++) {
            float sl = 0.0f, sh = 0.0f, x0s = 0.0f, x1s = 0.0f;
#pragma unroll
            for (uint q = 0; q < 8; q++) {
                uint v = bp[(ulong)(4 + g * 8 + q) * 64];
                local const float* x0 = xb + 64 * g + q * 4;
#pragma unroll
                for (uint i = 0; i < 4; i++) {
                    uint byte = (v >> (8 * i)) & 255u;
                    sl += (float)(byte & 15u) * x0[i];
                    sh += (float)(byte >> 4) * x0[32 + i];
                    x0s += x0[i];
                    x1s += x0[32 + i];
                }
            }
            acc += sc[2 * g] * sl + sc[2 * g + 1] * sh - mn[2 * g] * x0s - mn[2 * g + 1] * x1s;
        }
    }
    part[t] = acc;
    BARRIER();
    if (wv == 0) {
        uint r = GID * 64 + l;
        if (r < rows) y[r] = part[l] + part[64 + l] + part[128 + l] + part[192 + l];
    }
}

// Q4_K head for 2048 columns (8 blocks), interleaved like mv_q4ki; wave wv takes blocks wv and wv + 4 statically.
// The input is uniform across lanes (scalar loads, as in mv_t1i); the mins term needs only the sum of x over each
// 32-element half (xs32, once per group); nibbles: v & 0x0F0F0F0F and (v >> 4) & 0x0F0F0F0F, each byte -> float in one
// instruction. Also writes hmax[group] = the largest of the group's 64 logits (for the sampler's min-p floor).
static inline __attribute__((always_inline)) float q4k_block(global const uint* bp, global const float* restrict xb,
                                                              local const float* xsb) {
    uint dw = bp[0];
    uint sw[3] = {bp[64], bp[128], bp[192]};
    float d = half_bits(dw & 0xFFFFu), dmin = half_bits(dw >> 16);
    float acc = 0.0f;
#pragma unroll
    for (uint g = 0; g < 4; g++) {
        uint s6a, m6a, s6b, m6b;
#define SCB(n) ((sw[(n) >> 2] >> (8 * ((n) & 3))) & 255u)
#define SCM(j, s6, m6)                                                                              \
    if ((j) < 4) s6 = SCB(j) & 63u, m6 = SCB((j) + 4) & 63u;                                       \
    else s6 = (SCB((j) + 4) & 15u) | ((SCB((j) - 4) >> 6) << 4), m6 = (SCB((j) + 4) >> 4) | ((SCB(j) >> 6) << 4);
        SCM(2 * g, s6a, m6a)
        SCM(2 * g + 1, s6b, m6b)
#undef SCM
#undef SCB
        float sl = 0.0f, sh = 0.0f;
        uint nib[8];
#pragma unroll
        for (uint q = 0; q < 8; q++) nib[q] = bp[(ulong)(4 + g * 8 + q) * 64];
#pragma unroll
        for (uint q = 0; q < 8; q++) {
            uint v = nib[q], lo = v & 0x0F0F0F0Fu, hi = (v >> 4) & 0x0F0F0F0Fu;
            global const float* restrict x0 = xb + 64 * g + 4 * q;
#pragma unroll
            for (uint i = 0; i < 4; i++) {
                sl += (float)((lo >> (8 * i)) & 255u) * x0[i];
                sh += (float)((hi >> (8 * i)) & 255u) * x0[32 + i];
            }
        }
        acc += d * ((float)s6a * sl + (float)s6b * sh) - dmin * ((float)m6a * xsb[2 * g] + (float)m6b * xsb[2 * g + 1]);
    }
    return acc;
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mv_q4kh(global const uint* w, global const float* restrict x, global float* y, uint cols, uint rows,
             global float* hmax) {
    local float xs32[64];
    local float part[256];
    local float mx[64];
    uint t = LID, wv = __builtin_amdgcn_readfirstlane(t >> 6), l = t & 63;
    if (t < 64) {
        global const float* xc = x + t * 32;
        float sx = 0.0f;
        for (uint i = 0; i < 32; i++) sx += xc[i];
        xs32[t] = sx;
    }
    BARRIER();
    global const uint* tp = w + (ulong)GID * (8 * 36) * 64 + l;
    uint b0 = wv, b1 = wv + 4;
    float acc = q4k_block(tp + (ulong)b0 * 36 * 64, x + b0 * 256, xs32 + b0 * 8) +
                q4k_block(tp + (ulong)b1 * 36 * 64, x + b1 * 256, xs32 + b1 * 8);
    part[t] = acc;
    BARRIER();
    if (wv == 0) {
        uint r = GID * 64 + l;
        float v = part[l] + part[64 + l] + part[128 + l] + part[192 + l];
        if (r < rows) y[r] = v;
        mx[l] = r < rows ? v : -3.0e38f;
    }
    BARRIER();
    if (t == 0) {
        float m = mx[0];
        for (uint i = 1; i < 64; i++) m = __builtin_fmaxf(m, mx[i]);
        hmax[GID] = m;
    }
}

// The sampler in two steps (the one-group sampler read all 152k logits on one compute unit: 0.7 ms).
// sample_part: group g takes logits g * 1024 .. + 1024; the same rule as sample() (greedy, or Gumbel-max over the
// tokens within min_gap of the top logit, the top from the head's per-group maxima); writes its best (value, index).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sample_part(global const float* logits, uint n, float inv_temp, uint seed, global const float* hmax, uint nh,
                 global uint* pairs, float min_gap) {
    local float bv[256];
    local uint bi[256];
    const float LN2 = 0.69314718f;
    uint t = LID, salt = hash32(seed);
    float floor_ = -3.0e38f;
    if (inv_temp > 0.0f) {
        float top = -3.0e38f;
        for (uint i = t; i < nh; i += 256) top = __builtin_fmaxf(top, hmax[i]);
        bv[t] = top;
        BARRIER();
        for (uint s = 128; s > 0; s >>= 1) {
            if (t < s) bv[t] = __builtin_fmaxf(bv[t], bv[t + s]);
            BARRIER();
        }
        floor_ = bv[0] - min_gap;
        BARRIER();
    }
    float best = -3.0e38f;
    uint idx = 0xFFFFFFFFu;
#pragma unroll
    for (uint j = 0; j < 4; j++) {
        uint i = GID * 1024 + j * 256 + t;
        if (i < n) {
            float v = logits[i];
            bool ok = true;
            if (inv_temp > 0.0f) {
                ok = v >= floor_;
                float u = ((float)(hash32(salt ^ i) >> 8) + 0.5f) * (1.0f / 16777216.0f);
                v = v * inv_temp - __builtin_log2f(-__builtin_log2f(u) * LN2) * LN2;
            }
            if (ok && (v > best || idx == 0xFFFFFFFFu)) best = v, idx = i;
        }
    }
    bv[t] = best;
    bi[t] = idx;
    BARRIER();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s && (bv[t + s] > bv[t] || (bv[t + s] == bv[t] && bi[t + s] < bi[t]))) {
            bv[t] = bv[t + s];
            bi[t] = bi[t + s];
        }
        BARRIER();
    }
    if (t == 0) {
        pairs[2 * GID] = __builtin_astype(bv[0], uint);
        pairs[2 * GID + 1] = bi[0];
    }
}

// sample_fin: the best of the ng groups' pairs -> *last, history[step].
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sample_fin(global const uint* pairs, uint ng, global uint* last, global uint* history, uint step) {
    local float bv[256];
    local uint bi[256];
    uint t = LID;
    float best = -3.0e38f;
    uint idx = 0xFFFFFFFFu;
    for (uint g = t; g < ng; g += 256) {
        float v = __builtin_astype(pairs[2 * g], float);
        uint i = pairs[2 * g + 1];
        if (i != 0xFFFFFFFFu && (idx == 0xFFFFFFFFu || v > best || (v == best && i < idx))) best = v, idx = i;
    }
    bv[t] = best;
    bi[t] = idx;
    BARRIER();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s && bi[t + s] != 0xFFFFFFFFu &&
            (bi[t] == 0xFFFFFFFFu || bv[t + s] > bv[t] || (bv[t + s] == bv[t] && bi[t + s] < bi[t]))) {
            bv[t] = bv[t + s];
            bi[t] = bi[t + s];
        }
        BARRIER();
    }
    if (t == 0) {
        uint r = bi[0] == 0xFFFFFFFFu ? 0u : bi[0];
        *last = r;
        history[step] = r;
    }
}

// a group that does nothing (the kernel lab: the cost of a dispatch by itself)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void lab_empty(global float* y) {
    if (LID == 999) y[0] = 0.0f;
}

// ---- batched prompt reading: up to 64 tokens per pass -------------------------------------------------------------
// rmsnorm of nt vectors: group g = token g
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void rms_b(global const float* x, global const float* w, global float* out, uint n, float eps) {
    local float red[256];
    x += GID * n, out += GID * n;
    float ss = 0.0f;
    for (uint i = LID; i < n; i += 256) ss += x[i] * x[i];
    float r = rsq(group_sum(red, ss) / (float)n + eps);
    for (uint i = LID; i < n; i += 256) out[i] = x[i] * r * w[i];
}

// router for nt tokens: row r per wave (4 per group), the row's weights read once (cols = 2048), then each token
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mv_f16b(global const half_t* w, global const float* x, global float* y, uint cols, uint rows, uint nt) {
    uint t = LID, wv = t >> 6, l = t & 63, r = GID * 4 + wv;
    global const uint4v* wr = (global const uint4v*)(w + (ulong)r * cols);
    uint4v v0 = wr[l], v1 = wr[64 + l], v2 = wr[128 + l], v3 = wr[192 + l];
    uint wv32[16] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w};
    float f[32];
#pragma unroll
    for (uint i = 0; i < 16; i++) {
        f[2 * i] = half_bits(wv32[i] & 0xFFFFu);
        f[2 * i + 1] = half_bits(wv32[i] >> 16);
    }
#pragma unroll 1
    for (uint q = 0; q < nt; q++) {
        global const float* xq = x + q * cols + l * 8;
        float a = 0.0f;
#pragma unroll
        for (uint k = 0; k < 4; k++)
#pragma unroll
            for (uint i = 0; i < 8; i++) a += f[k * 8 + i] * xq[k * 512 + i];
        float sum = wave_reduce(a);
        if (l == 0) y[q * rows + r] = sum;
    }
}

// per expert, the slots (token * K + k) that chose it: thread e scans the ids (uniform reads)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void moe_lists(global const uint* ids, uint nids, uint E, uint cap, global uint* counts, global uint* lists) {
    uint e = LID, c = 0;
#pragma unroll 1
    for (uint s = 0; s < nids; s++)
        if (ids[s] == e && c < cap) lists[e * cap + c++] = s;
    if (e < E) counts[e] = c;
}

// x[t][i] += sum over the K experts of token t: y[t * K + k][i]; nt tokens
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void moe_sum_b(global float* x, global const float* y, uint n, uint k, uint nt) {
    uint i = GID * 256 + LID;
    if (i >= n * nt) return;
    uint t = i / n, j = i - t * n;
    float a = 0.0f;
    for (uint s = 0; s < k; s++) a += y[(t * k + s) * n + j];
    x[i] += a;
}

// Ternary product for many inputs: group = (expert e, tile of 64 rows); the expert's slots come from lists (dense: one
// "expert", slots = tokens). The wave's weights stay in registers; slots go 4 at a time, each 4 sharing one decode.
// Slot s reads x + (in_by_slot ? s : s / K) * xstride and writes y[s * R + row] (+= when add).
// one weight dword (4 bytes = 20 trits, or the qh dword's 16) against NQ_ inputs; j (0..12) may vary at run time.
// Trit-major order: per trit n, NQ_ x 4 input values (scalar loads), so few are live at once.
#define T1G_DWORD(NQ_)                                                                                                 \
    static inline __attribute__((always_inline)) void t1g_dword##NQ_(uint vj, uint j, global const float* const* xb,  \
                                                                      float* s, local const uint* lo_t,               \
                                                                      local const uint* hi_t) {                       \
        uint lo[4], hi[4];                                                                                             \
        _Pragma("unroll") for (uint i = 0; i < 4; i++) {                                                               \
            uint byte = (vj >> (8 * i)) & 255u;                                                                        \
            lo[i] = lo_t[byte], hi[i] = hi_t[byte];                                                                    \
        }                                                                                                              \
        if (j < 12) {                                                                                                  \
            uint base = j < 8 ? 4 * j : 128 + 4 * j, step = j < 8 ? 32 : 16;                                           \
            _Pragma("unroll") for (uint n = 0; n < 5; n++) {                                                           \
                _Pragma("unroll") for (uint q = 0; q < NQ_; q++) {                                                     \
                    global const float* restrict x0 = xb[q] + base + n * step;                                         \
                    _Pragma("unroll") for (uint i = 0; i < 4; i++) {                                                   \
                        float f = n < 4 ? (float)((lo[i] >> (8 * n)) & 255u) : (float)hi[i];                           \
                        s[q] += f * x0[i];                                                                             \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            _Pragma("unroll") for (uint n = 0; n < 4; n++) {                                                           \
                _Pragma("unroll") for (uint q = 0; q < NQ_; q++) {                                                     \
                    global const float* restrict x0 = xb[q] + 240 + 4 * n;                                             \
                    _Pragma("unroll") for (uint i = 0; i < 4; i++) s[q] += (float)((lo[i] >> (8 * n)) & 255u) * x0[i];   \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }
T1G_DWORD(4)
T1G_DWORD(1)

// this wave's part of NQ_ inputs' dot products (blocks / dwords as in mv_t1g), the input sums taken off
#define T1G_PART(NQ_)                                                                                                  \
    static inline __attribute__((always_inline)) void t1g_part##NQ_(                                                  \
        global const uint* tp, uint nb, uint nblk, uint wv, uint ja, uint jn, global const float* const* xq,          \
        local const float (*xsum)[8 * 13], local const uint* lo_t, local const uint* hi_t, float* acc) {               \
        _Pragma("unroll") for (uint q = 0; q < NQ_; q++) acc[q] = 0.0f;                                                \
        _Pragma("unroll 1") for (uint bi = 0; bi < nblk; bi++) {                                                       \
            uint b = nb == 8 ? wv + 4 * bi : wv >> 1;                                                                  \
            uint sdw = tp[(ulong)(nb * 13 + (b >> 1)) * 64];                                                           \
            float d = half_bits(b & 1 ? sdw >> 16 : sdw & 0xFFFFu);                                                    \
            global const float* xb[NQ_];                                                                               \
            _Pragma("unroll") for (uint q = 0; q < NQ_; q++) xb[q] = xq[q] + b * 256;                                  \
            _Pragma("unroll 1") for (uint jj = 0; jj < jn; jj++) {                                                     \
                uint j = ja + jj;                                                                                      \
                uint vj = tp[(ulong)(j < 12 ? b * 12 + j : nb * 12 + b) * 64];                                         \
                float s[NQ_];                                                                                          \
                _Pragma("unroll") for (uint q = 0; q < NQ_; q++) s[q] = 0.0f;                                          \
                t1g_dword##NQ_(vj, j, xb, s, lo_t, hi_t);                                                              \
                _Pragma("unroll") for (uint q = 0; q < NQ_; q++) acc[q] += d * (s[q] - xsum[q][b * 13 + j]);           \
            }                                                                                                          \
        }                                                                                                              \
    }
T1G_PART(4)
T1G_PART(1)

// Ternary product for many inputs: group = (expert e, tile of 64 rows, split c); the expert's slots come from lists
// (dense: one "expert", slots = tokens). Split c takes the expert's slots c*4.., (c + nsplit)*4.., 4 at a time
// sharing one decode, a last 1..3 one at a time. Slot s reads x + (in_by_slot ? s : s / K) * xstride and writes
// y[s * R + row] (+= when add).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mv_t1g(global const uint* w, uint rd, ulong es, global const uint* counts, global const uint* lists, uint cap,
            uint K, global const float* restrict x, uint xstride, uint in_by_slot, global float* y, uint R, uint cols,
            uint add, uint nsplit) {
    local uint lo_t[256], hi_t[256];
    local float xsum[4][8 * 13];
    local float part[4][256];
    uint t = LID, wv = __builtin_amdgcn_readfirstlane(t >> 6), l = t & 63;
    uint tiles = R >> 6, g = __builtin_amdgcn_readfirstlane(GID / nsplit), csplit = GID - g * nsplit;
    uint e = __builtin_amdgcn_readfirstlane(g / tiles), tile = g - e * tiles;
    uint m = counts[e];
    if (csplit * 4 >= m) return;
    global const uint* lst = lists + e * cap;
    {
        uint ev[5], m3 = 1;
        for (uint n = 0; n < 5; n++) {
            ev[n] = (((t * m3) & 255u) * 3u) >> 8;
            m3 *= 3u;
        }
        lo_t[t] = ev[0] | (ev[1] << 8) | (ev[2] << 16) | (ev[3] << 24);
        hi_t[t] = ev[4];
    }
    uint nb = cols >> 8, nu = nb * 13;
    global const uint* tp = (global const uint*)((global const uchar*)w + (ulong)e * es) + (ulong)tile * rd * 64 + l;
    uint nblk = nb == 8 ? 2 : 1;
    uint ja = nb == 8 ? 0 : ((wv & 1) ? 7 : 0), jn = nb == 8 ? 13 : ((wv & 1) ? 6 : 7);
#pragma unroll 1
    for (uint c0 = csplit * 4; c0 < m; c0 += 4 * nsplit) {
        uint nq = m - c0 >= 4 ? 4 : 1;   // a full four, or the rest one at a time
#pragma unroll 1
        for (uint c1 = c0; c1 < c0 + 4 && c1 < m; c1 += nq) {
            uint sl[4];
            global const float* xq[4];
#pragma unroll
            for (uint q = 0; q < 4; q++) {
                sl[q] = lst[c1 + q < m ? c1 + q : m - 1];
                xq[q] = x + (in_by_slot ? sl[q] : sl[q] / K) * xstride;
            }
            for (uint idx = t; idx < nq * nu; idx += 256) {   // xsum[q][b * 13 + j]
                uint q = idx / nu, u = idx - q * nu, b = u / 13, j = u - b * 13;
                global const float* xb = q == 0 ? xq[0] : q == 1 ? xq[1] : q == 2 ? xq[2] : xq[3];
                xb += b * 256;
                float sx = 0.0f;
                if (j < 12) {
                    uint base = j < 8 ? 4 * j : 128 + 4 * j, step = j < 8 ? 32 : 16;
                    for (uint n = 0; n < 5; n++)
                        for (uint i = 0; i < 4; i++) sx += xb[base + n * step + i];
                } else {
                    for (uint i = 0; i < 16; i++) sx += xb[240 + i];
                }
                xsum[q][u] = sx;
            }
            BARRIER();
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (nq == 4) t1g_part4(tp, nb, nblk, wv, ja, jn, xq, xsum, lo_t, hi_t, acc);
            else t1g_part1(tp, nb, nblk, wv, ja, jn, xq, xsum, lo_t, hi_t, acc);
#pragma unroll
            for (uint q = 0; q < 4; q++) part[q][t] = acc[q];
            BARRIER();
            if (wv == 0) {
                uint row = tile * 64 + l;
#pragma unroll
                for (uint q = 0; q < 4; q++)
                    if (q < nq) {
                        float r = part[q][l] + part[q][64 + l] + part[q][128 + l] + part[q][192 + l];
                        global float* o = y + (ulong)sl[q] * R + row;
                        *o = (add ? *o : 0.0f) + r;
                    }
            }
            BARRIER();
        }
    }
}

// ---- kernel lab (engine_maple.inc mp_lab): what limits mv_t1i? Same grid and addressing, parts taken out ----------

// loads only: mv_t1i's weight loads (units, all issued first), no decode / LDS math
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void lab_load(global const uint* w, uint rd, ulong es, global const uint* ids, global const float* x, uint xstride,
              global float* y, uint R, uint cols, uint add) {
    local float part[256];
    uint t = LID, wv = __builtin_amdgcn_readfirstlane(t >> 6), l = t & 63;
    uint row0 = GID * 64, slot = row0 / R, nb = cols >> 8, tile = (row0 - slot * R) >> 6, nu = nb * 13;
    global const uint* tp = (global const uint*)((global const uchar*)w + (ulong)ids[slot] * es) + (ulong)tile * rd * 64 + l;
    uint acc = 0;
#pragma unroll
    for (uint k = 0; k < 26; k++) {
        uint u = wv + 4 * k;
        if (u < nu) {
            uint b = u / 13, jj = u - b * 13;
            acc += tp[(ulong)(jj < 12 ? b * 12 + jj : nb * 12 + b) * 64] ^ tp[(ulong)(nb * 13 + (b >> 1)) * 64];
        }
    }
    part[t] = (float)acc;
    BARRIER();
    if (wv == 0) y[row0 + l] = part[l] + part[64 + l] + part[128 + l] + part[192 + l];
}

// setup only: mv_t1i's prologue (input to LDS, table, x sums, barriers), no row work
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void lab_stage(global const uint* w, uint rd, ulong es, global const uint* ids, global const float* x, uint xstride,
               global float* y, uint R, uint cols, uint add) {
    local float xl[2048];
    local float xsum[8 * 13];
    local uint lut[512];
    uint t = LID, l = t & 63, row0 = GID * 64, slot = row0 / R, nb = cols >> 8;
    global const float* xs = x + slot * xstride;
    for (uint i = t; i < cols; i += 256) xl[i] = xs[i];
    lut[2 * t] = t * 3u, lut[2 * t + 1] = t;
    BARRIER();
    if (t < nb * 13) {
        float sx = 0.0f;
        for (uint i = 0; i < 20; i++) sx += xl[(t * 20 + i) & 2047];
        xsum[t] = sx;
    }
    BARRIER();
    if (t < 64) y[row0 + l] = xl[l] + xsum[l % (nb * 13)] + (float)lut[l];
}

// math only: mv_t1i's decode and LDS arithmetic on made-up weights (no weight loads)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void lab_alu(global const uint* w, uint rd, ulong es, global const uint* ids, global const float* x, uint xstride,
             global float* y, uint R, uint cols, uint add) {
    local float xl[2048];
    local float xsum[8 * 13];
    local uint lut[512];
    local float part[256];
    uint t = LID, wv = __builtin_amdgcn_readfirstlane(t >> 6), l = t & 63;
    uint row0 = GID * 64, slot = row0 / R, nb = cols >> 8, nu = nb * 13;
    global const float* xs = x + slot * xstride;
    for (uint i = t; i < cols; i += 256) xl[i] = xs[i];
    {
        uint e[5], m3 = 1;
        for (uint n = 0; n < 5; n++) e[n] = (((t * m3) & 255u) * 3u) >> 8, m3 *= 3u;
        lut[2 * t] = e[0] | (e[1] << 8) | (e[2] << 16) | (e[3] << 24);
        lut[2 * t + 1] = e[4];
    }
    BARRIER();
    if (t < nu) xsum[t] = 0.5f;
    BARRIER();
    float acc = 0.0f;
#pragma unroll
    for (uint k = 0; k < 26; k++) {
        uint u = wv + 4 * k;
        if (u < nu) {
            uint b = u / 13, jj = u - b * 13;
            local const float* xb = xl + b * 256;
            uint v = (l * 0x9E3779B9u) ^ (k * 0x85EBCA6Bu);
            float s = 0.0f;
            uint base = jj < 8 ? 4 * jj : 128 + 4 * (jj % 12), step = jj < 8 ? 32 : 16;
#pragma unroll
            for (uint i = 0; i < 4; i++) {
                uint byte = (v >> (8 * i)) & 255u;
                uint lo = lut[2 * byte], hi = lut[2 * byte + 1];
                s += (float)(lo & 255u) * xb[base + i];
                s += (float)((lo >> 8) & 255u) * xb[base + step + i];
                s += (float)((lo >> 16) & 255u) * xb[base + 2 * step + i];
                s += (float)(lo >> 24) * xb[base + 3 * step + i];
                s += (float)hi * xb[(base + 4 * step + i) & 255];
            }
            acc += s - xsum[u];
        }
    }
    part[t] = acc;
    BARRIER();
    if (wv == 0) y[row0 + l] = part[l] + part[64 + l] + part[128 + l] + part[192 + l];
}

// raw stream: the same bytes read as 16-byte loads, contiguous (the best this much data can go)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void lab_stream(global const uint4v* p, uint n4, global float* y) {
    uint i = GID * 256 + LID, stride = 256 * 128;
    uint acc = 0;
    for (; i < n4; i += stride) {
        uint4v v = p[i];
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) y[0] = 1.0f;   // (keeps the loads)
}
