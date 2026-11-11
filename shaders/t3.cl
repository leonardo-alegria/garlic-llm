// Chatterbox T3 (the text -> speech-token Llama of Garlic's voice): 30 layers, 16 heads of 64 (no GQA), no norm on
// q and k, RoPE on the whole head. Included by chat.cl after qwen35.cl, so the attention kernels get other names
// (qk_rope_cache_t3, attn_part_t3, attn_combine_t3); the rest (rmsnorm, matvec_q4i, silu_mul, add) is shared.
// MAX_SEQ is one sentence: 34 voice rows + text + speech tokens (25 per second).
#undef HEAD_DIM
#undef ROPE_DIM
#undef MAX_SEQ
#undef ATTN_NAME
#define HEAD_DIM 64
#define ROPE_DIM 64
#define MAX_SEQ 512
#define ATTN_NAME(x) x##_t3
#define ATTN_NO_QK_NORM
#include "attn.h"

// ---- fused kernels for T3's decode step (fewer dispatches: each one costs a full GPU sync) -------------------
// Per layer: t3_rmsnorm_bs, t3_mv (qkv), t3_attn, t3_mv (o, added to x), t3_rmsnorm_bs, t3_mv (gate|up), t3_silu_bs,
// t3_mv (down, added to x): 8 dispatches instead of ~15. Same arithmetic as the shared kernels.

// x += the nparts partial results of a split matrix product (t3_mvk), then xn = x / rms(x) * w (1024 values), and
// bsum[b] = 8 * sum of xn over block b (for t3_mv). One group per row: GID = row of x, xn, bsum (32 per row) and
// parts [rows][4][1024] (rows = 1 in the decode step; the batched prefill passes several).
// The 4 parts' loads all go out at once, and the block sums read v with a stride of 33, so the 32 threads hit 32
// different LDS banks instead of one.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void t3_rmsnorm_bs(global float* x, global const float* parts, uint nparts, global const float* w, global float* xn,
                   global float* bsum, float eps) {
    local float red[256], v[1056];
    uint row = GID;
    x += (ulong)row * 1024, xn += (ulong)row * 1024, bsum += row * 32, parts += (ulong)row * 4096;
    float ss = 0.0f;
    float xv[4], pv[4][4];
    for (uint k = 0; k < 4; k++) xv[k] = x[LID + 256 * k];
    if (nparts) {
        for (uint p = 0; p < 4; p++)   // (the parts buffer always holds 4: those past nparts are not added)
            for (uint k = 0; k < 4; k++) pv[p][k] = parts[p * 1024 + LID + 256 * k];
        for (uint k = 0; k < 4; k++) {
            float a = xv[k];
            for (uint p = 0; p < 4; p++)
                if (p < nparts) a += pv[p][k];
            x[LID + 256 * k] = a;
            xv[k] = a;
        }
    }
    for (uint k = 0; k < 4; k++) ss += xv[k] * xv[k];
    float r = rsq(group_sum(red, ss) / 1024.0f + eps);
    for (uint k = 0; k < 4; k++) {
        uint i = LID + 256 * k;
        float y = xv[k] * r * w[i];
        xn[i] = y;
        v[i + (i >> 5)] = y;
    }
    BARRIER();
    if (LID < 32) {
        float s = 0.0f;
        for (uint k = 0; k < 32; k++) s += v[LID * 33 + k];
        bsum[LID] = 8.0f * s;
    }
}

// h = silu(gate) * up over 4096 values (gu = gate | up), and its block sums. 16 groups per row (GID / 16 = row:
// gu [rows][8192], h [rows][4096], bsum [rows][128]).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void t3_silu_bs(global const float* gu, global float* h, global float* bsum) {
    local float v[264];
    uint row = GID / 16, g16 = GID % 16;
    gu += (ulong)row * 8192, h += (ulong)row * 4096, bsum += row * 128;
    uint i = g16 * 256 + LID;
    float g = gu[i], y = g / (1.0f + exp_(-g)) * gu[4096 + i];
    h[i] = y;
    v[LID + (LID >> 5)] = y;
    BARRIER();
    if (LID < 8) {
        float s = 0.0f;
        for (uint k = 0; k < 32; k++) s += v[LID * 33 + k];
        bsum[g16 * 8 + LID] = 8.0f * s;
    }
}

// y = W x (accumulate = 0) or y += W x (1): matvec_q4i8 with the residual add folded in.
kernel __attribute__((reqd_work_group_size(512, 1, 1)))
void t3_mv(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict x,
           global const float* restrict bsum, global float* restrict y, uint cols, uint accumulate) {
    local float part[8 * 64];
    uint l = LID & 63, nb = cols / 32;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    ulong tile = (ulong)GID * nb * 64;
    global const uint4v* wp = w + tile + l;
    global const half_t* sp = sc + tile + l;
    float acc = 0.0f;
    for (uint b = wv; b < nb; b += 8) {
        uint4v q = wp[(ulong)b * 64];
        float d = (float)sp[(ulong)b * 64];
        global const float* xb = x + b * 32;
        float s = dot_q4_word_s(q.x, xb, xb + 16) + dot_q4_word_s(q.y, xb + 4, xb + 20) +
                  dot_q4_word_s(q.z, xb + 8, xb + 24) + dot_q4_word_s(q.w, xb + 12, xb + 28);
        acc += d * (s - bsum[b]);
    }
    part[wv * 64 + l] = acc;
    BARRIER();
    if (wv == 0) {
        float r = 0.0f;
        for (uint k = 0; k < 8; k++) r += part[k * 64 + l];
        global float* yo = y + GID * 64 + l;
        *yo = accumulate ? *yo + r : r;
    }
}

// Split-K form of t3_mv for the short wide matrices (few rows, so few groups): group g takes tile g % tiles and the
// blocks of its K slice g / tiles (nks slices); part[slice][row] gets the slice's sum (t3_rmsnorm_bs adds them).
kernel __attribute__((reqd_work_group_size(512, 1, 1)))
void t3_mvk(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict x,
            global const float* restrict bsum, global float* restrict part, uint cols, uint rows, uint nks) {
    local float pr[8 * 64];
    uint l = LID & 63, nb = cols / 32, tiles = rows / 64;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tl = GID % tiles, ks = GID / tiles, per = nb / nks;
    ulong tile = (ulong)tl * nb * 64;
    global const uint4v* wp = w + tile + l;
    global const half_t* sp = sc + tile + l;
    float acc = 0.0f;
    for (uint b = ks * per + wv; b < (ks + 1) * per; b += 8) {
        uint4v q = wp[(ulong)b * 64];
        float d = (float)sp[(ulong)b * 64];
        global const float* xb = x + b * 32;
        float s = dot_q4_word_s(q.x, xb, xb + 16) + dot_q4_word_s(q.y, xb + 4, xb + 20) +
                  dot_q4_word_s(q.z, xb + 8, xb + 24) + dot_q4_word_s(q.w, xb + 12, xb + 28);
        acc += d * (s - bsum[b]);
    }
    pr[wv * 64 + l] = acc;
    BARRIER();
    if (wv == 0) {
        float r = 0.0f;
        for (uint k = 0; k < 8; k++) r += pr[k * 64 + l];
        part[(ulong)ks * rows + tl * 64 + l] = r;
    }
}

// ---- batched prefill: T3_NB positions per group, the weights read once for all of them ----------------------
// Each position gets exactly the arithmetic of t3_mv / t3_mvk (the same products and sums in the same order: the
// nibbles are converted once and multiplied as dot_q4_word_s does), so the prefill gives the same bits as one
// position per pass. x [n][cols], bsum [n][cols / 32].
#define T3_NB 16

static inline __attribute__((always_inline)) void t3_unpack(uint4v q, float f[32]) {   // as dot_q4_word_s reads them
    uint ws[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (uint k = 0; k < 4; k++) {
        uint lo = ws[k] & 0x0F0F0F0Fu, hi = (ws[k] >> 4) & 0x0F0F0F0Fu;
        __asm__("" : "+v"(lo), "+v"(hi));
        f[8 * k] = (float)(lo & 0xFFu);
        f[8 * k + 1] = (float)((lo >> 8) & 0xFFu);
        f[8 * k + 2] = (float)((lo >> 16) & 0xFFu);
        f[8 * k + 3] = (float)(lo >> 24);
        f[8 * k + 4] = (float)(hi & 0xFFu);
        f[8 * k + 5] = (float)((hi >> 8) & 0xFFu);
        f[8 * k + 6] = (float)((hi >> 16) & 0xFFu);
        f[8 * k + 7] = (float)(hi >> 24);
    }
}

static inline __attribute__((always_inline)) float t3_dot8(const float* f, global const float* restrict xl,
                                                           global const float* restrict xh) {   // dot_q4_word_s
    float s = f[0] * xl[0];
    s += f[1] * xl[1];
    s += f[2] * xl[2];
    s += f[3] * xl[3];
    s += f[4] * xh[0];
    s += f[5] * xh[1];
    s += f[6] * xh[2];
    s += f[7] * xh[3];
    return s;
}

// Blocks [b0, b1) of a q4i tile for positions r0 .. r0 + nv - 1 (8 waves: wave wv takes b0 + wv, b0 + wv + 8, ...),
// then out(v, row sum) for each position, the 8 waves' sums added in order (16 KB of LDS, 8 positions a round).
#define T3_MVB_BODY(OUT)                                                                                           \
    local float part[8][8][64];                                                                                    \
    float acc[T3_NB];                                                                                              \
    _Pragma("unroll") for (uint v = 0; v < T3_NB; v++) acc[v] = 0.0f;                                              \
    for (uint b = b0 + wv; b < b1; b += 8) {                                                                       \
        uint4v q = wp[(ulong)b * 64];                                                                              \
        float d = (float)sp[(ulong)b * 64];                                                                        \
        float f[32];                                                                                               \
        t3_unpack(q, f);                                                                                           \
        _Pragma("unroll") for (uint v = 0; v < T3_NB; v++) {                                                       \
            if (v < nv) {                                                                                          \
                global const float* xb = x + (ulong)(r0 + v) * cols + b * 32;                                      \
                float s = t3_dot8(f, xb, xb + 16) + t3_dot8(f + 8, xb + 4, xb + 20) +                              \
                          t3_dot8(f + 16, xb + 8, xb + 24) + t3_dot8(f + 24, xb + 12, xb + 28);                    \
                acc[v] += d * (s - bsum[(ulong)(r0 + v) * nb + b]);                                                \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
    _Pragma("unroll") for (uint h8 = 0; h8 < T3_NB / 8; h8++) {                                                    \
        if (h8 * 8 < nv) {   /* (nv is the same for the whole group) */                                            \
            _Pragma("unroll") for (uint v = 0; v < 8; v++) part[wv][v][l] = acc[h8 * 8 + v];                       \
            BARRIER();                                                                                             \
            if (wv == 0) {                                                                                         \
                _Pragma("unroll") for (uint v = 0; v < 8; v++) {                                                   \
                    if (h8 * 8 + v < nv) {                                                                         \
                        float r = 0.0f;                                                                            \
                        for (uint k = 0; k < 8; k++) r += part[k][v][l];                                           \
                        OUT(r0 + h8 * 8 + v, r);                                                                   \
                    }                                                                                              \
                }                                                                                                  \
            }                                                                                                      \
            BARRIER();                                                                                             \
        }                                                                                                          \
    }

// y[r] = W x[r] (accumulate = 0) or y[r] += W x[r] (1), r < n: t3_mv for n positions. y [n][rows].
// Groups: rows / 64 tiles x ceil(n / T3_NB).
kernel __attribute__((reqd_work_group_size(512, 1, 1)))
void t3_mv_b(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict x,
             global const float* restrict bsum, global float* restrict y, uint cols, uint rows, uint n,
             uint accumulate) {
    uint l = LID & 63, nb = cols / 32, tiles = rows / 64;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tl = GID % tiles, r0 = (GID / tiles) * T3_NB, nv = n - r0 < T3_NB ? n - r0 : T3_NB;
    ulong tile = (ulong)tl * nb * 64;
    global const uint4v* wp = w + tile + l;
    global const half_t* sp = sc + tile + l;
    uint b0 = 0, b1 = nb;
#define T3_OUT_MV(r, val)                                         \
    {                                                             \
        global float* yo = y + (ulong)(r) * rows + tl * 64 + l;   \
        *yo = accumulate ? *yo + (val) : (val);                   \
    }
    T3_MVB_BODY(T3_OUT_MV)
#undef T3_OUT_MV
}

// t3_mvk for n positions: part [n][4][rows] (the layout t3_rmsnorm_bs reads). Groups: tiles x nks x ceil(n / T3_NB).
kernel __attribute__((reqd_work_group_size(512, 1, 1)))
void t3_mvk_b(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict x,
              global const float* restrict bsum, global float* restrict out, uint cols, uint rows, uint nks, uint n) {
    uint l = LID & 63, nb = cols / 32, tiles = rows / 64;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tl = GID % tiles, ks = (GID / tiles) % nks, r0 = GID / tiles / nks * T3_NB, per = nb / nks;
    uint nv = n - r0 < T3_NB ? n - r0 : T3_NB;
    ulong tile = (ulong)tl * nb * 64;
    global const uint4v* wp = w + tile + l;
    global const half_t* sp = sc + tile + l;
    uint b0 = ks * per, b1 = (ks + 1) * per;
#define T3_OUT_MVK(r, val) out[((ulong)(r) * 4 + ks) * rows + tl * 64 + l] = (val);
    T3_MVB_BODY(T3_OUT_MVK)
#undef T3_OUT_MVK
}

// The attention of head h at position pos once its q (scaled by 1/8), its k as stored (fp16) and the current v as
// stored (vcur, lane = dimension) are known: scores against positions 0..pos, softmax, weighted sum of v -> att[h],
// and the two block sums of this head's 64 outputs. 256 threads.
// Scores: thread = position, all 64 dimensions, so the 64 loads of a position go out together (256 positions per
// round). Values: wave w takes positions w, w + 4, ..., lane = dimension, 16 positions' loads in flight. Order of
// the sums: the 4 quarter sums of 16 dimensions are added quarter 0 first, and o grows 4 positions at a time.
static inline __attribute__((always_inline)) void t3_attn_core(local float* q, local float* k, local float* p,
                                                               local float* red, local float* part, float vcur,
                                                               global const half_t* kc, global const schar* vc,
                                                               global float* att, global float* bsum, uint h,
                                                               uint pos) {
    uint t = LID, wv = t >> 6, l = t & 63;
    global const float* vscale = (global const float*)(vc + 512 * 16 * 64);
    float m = -3.0e38f;
    for (uint j = t; j <= pos; j += 256) {
        float a[4];
        if (j < pos) {
            global const half_t* kp = kc + (ulong)h * 64 * 512 + j;
#pragma unroll
            for (uint w = 0; w < 4; w++) {
                float s = 0.0f;
#pragma unroll
                for (uint d = 0; d < 16; d++) s += q[w * 16 + d] * (float)kp[(ulong)(w * 16 + d) * 512];
                a[w] = s;
            }
        } else {
#pragma unroll
            for (uint w = 0; w < 4; w++) {
                float s = 0.0f;
                for (uint d = 0; d < 16; d++) s += q[w * 16 + d] * k[w * 16 + d];
                a[w] = s;
            }
        }
        float sc = a[0] + a[1] + a[2] + a[3];
        p[j] = sc;
        m = __builtin_fmaxf(m, sc);
    }
    m = group_max(red, m);
    float sum = 0.0f;
    for (uint j = t; j <= pos; j += 256) {
        float e = exp_(p[j] - m);
        p[j] = e;
        sum += e;
    }
    sum = group_sum(red, sum);   // (its barriers also make every p[j] visible)
    float o = 0.0f;
    uint j = wv;
    for (; j + 60 < pos; j += 64) {   // 4 of the steps below, their 16 loads in flight together
        float a[16];
#pragma unroll
        for (uint u = 0; u < 16; u++) {
            uint jj = j + 4 * u;
            a[u] = (float)vc[((ulong)jj * 16 + h) * 64 + l] * vscale[jj * 16 + h];
        }
#pragma unroll
        for (uint s = 0; s < 4; s++) {
            uint js = j + 16 * s;
            o += p[js] * a[4 * s] + p[js + 4] * a[4 * s + 1] + p[js + 8] * a[4 * s + 2] + p[js + 12] * a[4 * s + 3];
        }
    }
    for (; j + 12 < pos; j += 16) {   // 4 positions per step
        global const schar* v0 = vc + ((ulong)j * 16 + h) * 64 + l;
        float a0 = (float)v0[0] * vscale[j * 16 + h], a1 = (float)v0[4 * 16 * 64] * vscale[(j + 4) * 16 + h];
        float a2 = (float)v0[8 * 16 * 64] * vscale[(j + 8) * 16 + h];
        float a3 = (float)v0[12 * 16 * 64] * vscale[(j + 12) * 16 + h];
        o += p[j] * a0 + p[j + 4] * a1 + p[j + 8] * a2 + p[j + 12] * a3;
    }
    for (; j < pos; j += 4) o += p[j] * (float)vc[((ulong)j * 16 + h) * 64 + l] * vscale[j * 16 + h];
    if (wv == pos % 4) o += p[pos] * vcur;
    BARRIER();
    part[wv * 64 + l] = o;
    BARRIER();
    if (wv == 0) {
        float r = (part[l] + part[64 + l] + part[128 + l] + part[192 + l]) / sum;
        att[h * 64 + l] = r;
        red[l] = r;
        __builtin_amdgcn_wave_barrier();
        for (uint st = 16; st > 0; st >>= 1) {   // two sums of 32: lanes 0..31 and 32..63
            if ((l & 31) < st) red[l] += red[l + st];
            __builtin_amdgcn_wave_barrier();
        }
        if ((l & 31) == 0) bsum[2 * h + l / 32] = 8.0f * red[l];
    }
}

// RoPE on q and k of head h at pos, k (fp16) and v (int8 + a scale per position and head) into the cache; q / 8,
// k and v as stored -> q, k, cur (lane = dimension). One wave.
static inline __attribute__((always_inline)) void t3_rope_head(global const float* qkv, global const float* rope,
                                                               global half_t* kc, global schar* vc, local float* q,
                                                               local float* k, local float* red, local float* cur,
                                                               uint h, uint pos) {
    uint l = LID & 63;
    global float* vscale = (global float*)(vc + 512 * 16 * 64);
    global const float* rp = rope + (ulong)pos * 64;
    float qv = qkv[h * 64 + l], kv = qkv[1024 + h * 64 + l], vv = qkv[2048 + h * 64 + l];
    q[l] = qv;
    k[l] = kv;
    red[l] = __builtin_fabsf(vv);
    float c = rp[l % 32], s = rp[32 + l % 32];
    // (a wave runs in lockstep: q and k are complete before they are read; the barriers below keep LLVM honest)
    __builtin_amdgcn_wave_barrier();
    float qr = l < 32 ? qv * c - q[l + 32] * s : qv * c + q[l - 32] * s;
    float kr = l < 32 ? kv * c - k[l + 32] * s : kv * c + k[l - 32] * s;
    half_t kh = (half_t)kr;
    kc[((ulong)h * 64 + l) * 512 + pos] = kh;
    __builtin_amdgcn_wave_barrier();
    q[l] = qr * 0.125f;
    k[l] = (float)kh;
    for (uint st = 32; st > 0; st >>= 1) {
        if (l < st) red[l] = __builtin_fmaxf(red[l], red[l + st]);
        __builtin_amdgcn_wave_barrier();
    }
    float mx = red[0];
    float inv = mx > 0.0f ? 127.0f / mx : 0.0f, vs = mx / 127.0f;
    float q8 = __builtin_rintf(vv * inv);
    vc[((ulong)pos * 16 + h) * 64 + l] = (schar)q8;
    if (l == 0) vscale[pos * 16 + h] = vs;
    cur[l] = q8 * vs;   // the current position's v, as stored
}

// The whole attention of one head (one group of 256 threads) in the decode step: t3_rope_head (wave 0), then
// t3_attn_core.
// Cache layout as attn.h with MAX_SEQ 512, 16 kv heads of 64: k [16][64][512], v [512][16][64] then scales [512][16].
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void t3_attn(global const float* qkv, global const float* rope, global half_t* kc, global schar* vc,
             global float* att, global float* bsum, uint pos) {
    local float q[64], k[64], p[512], red[256], part[4 * 64];
    uint h = GID, wv = LID >> 6, l = LID & 63;
    if (wv == 0) t3_rope_head(qkv, rope, kc, vc, q, k, red, part, h, pos);
    BARRIER();
    float vcur = part[l];
    BARRIER();
    t3_attn_core(q, k, p, red, part, vcur, kc, vc, att, bsum, h, pos);
}

// Batched prefill, positions p0 + r (r < rows): t3_rope_head for each (head, position) -> the caches and q / 8
// in qb [rows][1024]. Groups: 16 x rows of 64 threads (GID = r * 16 + h).
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void t3_rope_b(global const float* qkv, global const float* rope, global half_t* kc, global schar* vc,
               global float* qb, uint p0) {
    local float q[64], k[64], red[64], cur[64];
    uint h = GID % 16, r = GID / 16, l = LID;
    t3_rope_head(qkv + (ulong)r * 3072, rope, kc, vc, q, k, red, cur, h, p0 + r);
    __builtin_amdgcn_wave_barrier();
    qb[(ulong)r * 1024 + h * 64 + l] = q[l];
}

// Batched prefill: t3_attn_core for each (head, position p0 + r), the caches already written by t3_rope_b (k and v
// of the position itself read back as stored: the same values t3_attn keeps in registers). att [rows][1024], bsum
// [rows][32]. Groups: 16 x rows of 256 threads (GID = r * 16 + h).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void t3_attn_b(global const float* qb, global const half_t* kc, global const schar* vc, global float* att,
               global float* bsum, uint p0) {
    local float q[64], k[64], p[512], red[256], part[4 * 64];
    uint h = GID % 16, r = GID / 16, pos = p0 + r, wv = LID >> 6, l = LID & 63;
    global const float* vscale = (global const float*)(vc + 512 * 16 * 64);
    if (wv == 0) {
        q[l] = qb[(ulong)r * 1024 + h * 64 + l];
        k[l] = (float)kc[((ulong)h * 64 + l) * 512 + pos];
        part[l] = (float)vc[((ulong)pos * 16 + h) * 64 + l] * vscale[pos * 16 + h];
    }
    BARRIER();
    float vcur = part[l];
    BARRIER();
    t3_attn_core(q, k, p, red, part, vcur, kc, vc, att + (ulong)r * 1024, bsum + r * 32, h, pos);
}

// T3's sampling on the GPU, as the official code: repetition penalty 1.2 on the tokens already said (seen[i] != 0),
// temperature 0.8, min_p 0.05 (only tokens at least 5 % as likely as the top one), then a draw: the argmax of the
// scaled logits + Gumbel noise is an exact sample of that distribution. greedy: argmax of the raw logits. The token
// goes to res[0] and is marked seen. One group of 256 threads.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void t3_sample(global const float* logits, global uchar* seen, global uint* res, uint seed, uint greedy) {
    local float bv[256];
    local uint bi[256];
    const float LN2 = 0.69314718f;
    uint t = LID;
    float top = -3.0e38f;
    for (uint i = t; i < 6563; i += 256) {
        float l = logits[i];
        if (!greedy && seen[i]) l = l > 0.0f ? l / 1.2f : l * 1.2f;
        top = __builtin_fmaxf(top, greedy ? l : l / 0.8f);
    }
    bv[t] = top;
    BARRIER();
    for (uint st = 128; st > 0; st >>= 1) {
        if (t < st) bv[t] = __builtin_fmaxf(bv[t], bv[t + st]);
        BARRIER();
    }
    float floor_ = bv[0] - 2.9957323f;   // ln(1 / 0.05)
    BARRIER();
    float best = -3.0e38f;
    uint idx = 0;
    for (uint i = t; i < 6563; i += 256) {
        float l = logits[i];
        float v;
        if (greedy) v = l;
        else {
            if (seen[i]) l = l > 0.0f ? l / 1.2f : l * 1.2f;
            v = l / 0.8f;
            if (v < floor_) continue;
            float u = ((float)(hash32(seed ^ (i * 0x9E3779B9u)) >> 8) + 0.5f) * (1.0f / 16777216.0f);
            v -= __builtin_log2f(-__builtin_log2f(u) * LN2) * LN2;   // + Gumbel(0, 1)
        }
        if (v > best) best = v, idx = i;
    }
    bv[t] = best;
    bi[t] = idx;
    BARRIER();
    for (uint st = 128; st > 0; st >>= 1) {
        if (t < st && (bv[t + st] > bv[t] || (bv[t + st] == bv[t] && bi[t + st] < bi[t]))) {
            bv[t] = bv[t + st];
            bi[t] = bi[t + st];
        }
        BARRIER();
    }
    if (t == 0) {
        res[0] = bi[0];
        seen[bi[0]] = 1;
    }
}
