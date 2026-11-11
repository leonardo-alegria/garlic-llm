// Kernels shared by the Qwen3 and Qwen3.5 decoders (qwen3.cl, qwen35.cl): norms, embedding, Q4 matrix x
// vector, SiLU, add, sampling. GCN gfx701, compiled by LLVM from OpenCL C 1.2 without its standard headers.

#ifndef COMMON_H
#define COMMON_H

typedef unsigned int uint;
typedef unsigned long ulong;
typedef signed char schar;

#define LID __builtin_amdgcn_workitem_id_x()
#define GID __builtin_amdgcn_workgroup_id_x()
#define BARRIER()                                                    \
    do {                                                             \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");       \
        __builtin_amdgcn_s_barrier();                                \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");       \
    } while (0)
#define LOG2E 1.44269504f
#define NO_TOKEN 0xFFFFFFFFu

static float rsq(float x) { return __builtin_amdgcn_rsqf(x); }
static float exp_(float x) { return __builtin_exp2f(x * LOG2E); }

// Sum / max of one value per thread over a 256-thread group.
static float group_sum(local float* red, float v) {
    uint t = LID;
    red[t] = v;
    BARRIER();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        BARRIER();
    }
    float r = red[0];
    BARRIER();
    return r;
}

static float group_max(local float* red, float v) {
    uint t = LID;
    red[t] = v;
    BARRIER();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) red[t] = __builtin_fmaxf(red[t], red[t + s]);
        BARRIER();
    }
    float r = red[0];
    BARRIER();
    return r;
}

// x = dequantized row `token` of the Q8 embedding (token from `forced`, or from the GPU's last argmax).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void embed(global const schar* q, global const float* scales, global const uint* last, uint forced,
           global float* x, uint cols) {
    uint tok = forced != NO_TOKEN ? forced : *last;
    ulong row = (ulong)tok * cols;
    for (uint i = LID; i < cols; i += 256) x[i] = (float)q[row + i] * scales[(row + i) >> 5];
}

// out = x / rms(x) * w, one group of 256 threads.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void rmsnorm(global const float* x, global const float* w, global float* out, uint n, float eps) {
    local float red[256];
    float ss = 0.0f;
    for (uint i = LID; i < n; i += 256) ss += x[i] * x[i];
    float r = rsq(group_sum(red, ss) / (float)n + eps);
    for (uint i = LID; i < n; i += 256) out[i] = x[i] * r * w[i];
}

typedef _Float16 half_t;

// h[i] = silu(gate[i]) * up[i], with gate = gu[0..n), up = gu[n..2n).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void silu_mul(global const float* gu, global float* h, uint n) {
    uint i = GID * 256 + LID;
    if (i < n) {
        float g = gu[i];
        h[i] = g / (1.0f + exp_(-g)) * gu[n + i];
    }
}

// dst[i] = src[i]: the token's embedding row, prepared by the CPU (Qwen3.5-9B keeps the embedding out of GPU memory)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void copy_f32(global const float* src, global float* dst, uint n) {
    uint i = GID * 256 + LID;
    if (i < n) dst[i] = src[i];
}

// x[i] += y[i]
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void add(global float* x, global const float* y, uint n) {
    uint i = GID * 256 + LID;
    if (i < n) x[i] += y[i];
}

// Greedy pick: index of the largest logit -> *last and history[step]. One group of 256 threads.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void argmax(global const float* logits, uint n, global uint* last, global uint* history, uint step) {
    local float bv[256];
    local uint bi[256];
    uint t = LID;
    float best = -3.0e38f;
    uint idx = 0;
    for (uint i = t; i < n; i += 256) {
        float v = logits[i];
        if (v > best) {
            best = v;
            idx = i;
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
        *last = bi[0];
        history[step] = bi[0];
    }
}


// ---- 4-bit weights (Q4) ---------------------------------------------------------------------------------------
// Per row: blocks of 32 weights = 16 bytes; byte j holds element j (low nibble) and element j + 16 (high nibble),
// stored as q = round(w / d) + 8 in 0..15. One fp16 scale d per block, in a separate array [rows][cols / 32].
// w = (q - 8) * d, so a block's dot product is d * (sum q_j x_j - 8 * sum x_j).

typedef unsigned short ushort;
typedef unsigned char uchar;
typedef uint uint4v __attribute__((ext_vector_type(4)));
typedef float float4v __attribute__((ext_vector_type(4)));

static float h2f(uint h) {                     // fp16 -> fp32 (normal numbers; zero/denormal -> 0)
    uint e = (h >> 10) & 0x1F;
    uint f = e ? ((h & 0x8000u) << 16) | ((e + 112u) << 23) | ((h & 0x3FFu) << 13) : ((h & 0x8000u) << 16);
    return __builtin_bit_cast(float, f);
}

static float dot_q4_block(uint4v w, const float* xb) {   // sum q_j * x_j over the 32 weights of a block
    uint words[4] = {w.x, w.y, w.z, w.w};
    float s = 0.0f;
    for (uint k = 0; k < 4; k++) {
        uint lo = words[k] & 0x0F0F0F0Fu, hi = (words[k] >> 4) & 0x0F0F0F0Fu;
        for (uint b = 0; b < 4; b++) {
            s += (float)((lo >> (8 * b)) & 0xFFu) * xb[4 * k + b];
            s += (float)((hi >> (8 * b)) & 0xFFu) * xb[16 + 4 * k + b];
        }
    }
    return s;
}

// y = W x for 4 rows per 64-thread group; lanes stride over the row's blocks.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void matvec_q4(global const uchar* w, global const ushort* scales, global const float* x, global float* y,
               uint cols) {
    local float red[4][64];
    uint lane = LID, r0 = GID * 4, nb = cols / 32;
    ulong rowbytes = cols / 2;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = lane; b < nb; b += 64) {
        float xb[32];
        global const float4v* xv = (global const float4v*)(x + b * 32);
        for (uint k = 0; k < 8; k++) {
            float4v v = xv[k];
            xb[4 * k] = v.x; xb[4 * k + 1] = v.y; xb[4 * k + 2] = v.z; xb[4 * k + 3] = v.w;
        }
        float sx = 0.0f;
        for (uint k = 0; k < 32; k++) sx += xb[k];
        for (uint r = 0; r < 4; r++) {
            ulong row = r0 + r;
            uint4v q = *(global const uint4v*)(w + row * rowbytes + b * 16);
            float d = h2f(scales[row * nb + b]);
            acc[r] += d * (dot_q4_block(q, xb) - 8.0f * sx);
        }
    }
    for (uint r = 0; r < 4; r++) red[r][lane] = acc[r];
    BARRIER();
    for (uint s = 32; s > 0; s >>= 1) {
        if (lane < s)
            for (uint r = 0; r < 4; r++) red[r][lane] += red[r][lane + s];
        BARRIER();
    }
    if (lane < 4) y[r0 + lane] = red[lane][0];
}

// ---- Q4 matrix x vector, interleaved layout ("q4i") ---------------------------------------------------------
// Rows in tiles of 64. Tile t, block b, row t * 64 + l: its 16 bytes of nibbles at ((t * nb + b) * 64 + l) * 16 and
// its fp16 scale at index (t * nb + b) * 64 + l. A wave reads 1 KiB of consecutive weights per block, and each lane
// owns one row, so x (the same for the whole wave) comes through the scalar unit and no cross-lane sum is needed.
// bsum[b] = 8 * sum of x over block b (block_sums), so w = (q - 8) d gives d (sum q x - bsum).
// Group = 4 waves = one tile of 64 rows; wave w takes blocks w, w + 4, ...

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void block_sums(global const float* x, global float* bsum, uint n) {
    uint b = GID * 256 + LID;
    if (b * 32 >= n) return;
    global const float4v* xv = (global const float4v*)(x + b * 32);
    float4v t = xv[0] + xv[1] + xv[2] + xv[3] + xv[4] + xv[5] + xv[6] + xv[7];
    bsum[b] = 8.0f * (t.x + t.y + t.z + t.w);
}

static float dot_q4_word_s(uint w, global const float* restrict xl, global const float* restrict xh) {   // as dot_q4_word, x from scalar regs
    uint lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
    __asm__("" : "+v"(lo), "+v"(hi));
    float s = (float)(lo & 0xFFu) * xl[0];
    s += (float)((lo >> 8) & 0xFFu) * xl[1];
    s += (float)((lo >> 16) & 0xFFu) * xl[2];
    s += (float)(lo >> 24) * xl[3];
    s += (float)(hi & 0xFFu) * xh[0];
    s += (float)((hi >> 8) & 0xFFu) * xh[1];
    s += (float)((hi >> 16) & 0xFFu) * xh[2];
    s += (float)(hi >> 24) * xh[3];
    return s;
}

// WAVES waves per tile of 64 rows: 4 for tall matrices, 8 when there are few rows (more groups in flight).
static inline __attribute__((always_inline)) void mvq4i(global const uint4v* restrict w,
                                                        global const half_t* restrict sc,
                                                        global const float* restrict x,
                                                        global const float* restrict bsum, global float* restrict y,
                                                        uint cols, local float* part, const uint WAVES) {
    uint l = LID & 63, nb = cols / 32;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);   // uniform per wave: x addresses become scalar loads
    ulong tile = (ulong)GID * nb * 64;
    global const uint4v* wp = w + tile + l;
    global const half_t* sp = sc + tile + l;
    float acc = 0.0f;
    for (uint b = wv; b < nb; b += WAVES) {
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
        for (uint k = 0; k < WAVES; k++) r += part[k * 64 + l];
        y[GID * 64 + l] = r;
    }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void matvec_q4i(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict x,
                global const float* restrict bsum, global float* restrict y, uint cols) {
    local float part[4 * 64];
    mvq4i(w, sc, x, bsum, y, cols, part, 4);
}

kernel __attribute__((reqd_work_group_size(512, 1, 1)))
void matvec_q4i8(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict x,
                 global const float* restrict bsum, global float* restrict y, uint cols) {
    local float part[8 * 64];
    mvq4i(w, sc, x, bsum, y, cols, part, 8);
}

// ---- Q3 matrix x vector, interleaved layout ("q3i", pc/q3.py) ------------------------------------------------
// 12 bytes per block (d0, d1: 2 low bits; d2: high bits), w = (q - 4) d, q in 0..7. Same tiles, waves and scalar x
// as q4i; bsum is 8 * sum x, so the -4 offset is bsum / 2.

static float dot_q3_group(uint lo, uint hi, global const float* restrict xg) {   // 4 weights' bytes . 4 x
    uint q = lo | hi;
    __asm__("" : "+v"(q));
    float s = (float)(q & 0xFFu) * xg[0];
    s += (float)((q >> 8) & 0xFFu) * xg[1];
    s += (float)((q >> 16) & 0xFFu) * xg[2];
    s += (float)(q >> 24) * xg[3];
    return s;
}

static float dot_q3_block(uint d0, uint d1, uint d2, global const float* restrict xb) {
    float s = dot_q3_group(d0 & 0x03030303u, (d2 << 2) & 0x04040404u, xb);
    s += dot_q3_group((d0 >> 2) & 0x03030303u, (d2 << 1) & 0x04040404u, xb + 4);
    s += dot_q3_group((d0 >> 4) & 0x03030303u, d2 & 0x04040404u, xb + 8);
    s += dot_q3_group((d0 >> 6) & 0x03030303u, (d2 >> 1) & 0x04040404u, xb + 12);
    s += dot_q3_group(d1 & 0x03030303u, (d2 >> 2) & 0x04040404u, xb + 16);
    s += dot_q3_group((d1 >> 2) & 0x03030303u, (d2 >> 3) & 0x04040404u, xb + 20);
    s += dot_q3_group((d1 >> 4) & 0x03030303u, (d2 >> 4) & 0x04040404u, xb + 24);
    s += dot_q3_group((d1 >> 6) & 0x03030303u, (d2 >> 5) & 0x04040404u, xb + 28);
    return s;
}

static inline __attribute__((always_inline)) void mvq3i(global const uint* restrict w,
                                                        global const half_t* restrict sc,
                                                        global const float* restrict x,
                                                        global const float* restrict bsum, global float* restrict y,
                                                        uint cols, local float* part, const uint WAVES) {
    uint l = LID & 63, nb = cols / 32;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    ulong tile = (ulong)GID * nb * 64;
    global const uint* wp = w + (tile + l) * 3;
    global const half_t* sp = sc + tile + l;
    float acc = 0.0f;
    for (uint b = wv; b < nb; b += WAVES) {
        global const uint* q = wp + (ulong)b * 64 * 3;
        uint d0 = q[0], d1 = q[1], d2 = q[2];
        float d = (float)sp[(ulong)b * 64];
        acc += d * (dot_q3_block(d0, d1, d2, x + b * 32) - 0.5f * bsum[b]);
    }
    part[wv * 64 + l] = acc;
    BARRIER();
    if (wv == 0) {
        float r = 0.0f;
        for (uint k = 0; k < WAVES; k++) r += part[k * 64 + l];
        y[GID * 64 + l] = r;
    }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void matvec_q3i(global const uint* restrict w, global const half_t* restrict sc, global const float* restrict x,
                global const float* restrict bsum, global float* restrict y, uint cols) {
    local float part[4 * 64];
    mvq3i(w, sc, x, bsum, y, cols, part, 4);
}

kernel __attribute__((reqd_work_group_size(512, 1, 1)))
void matvec_q3i8(global const uint* restrict w, global const half_t* restrict sc, global const float* restrict x,
                 global const float* restrict bsum, global float* restrict y, uint cols) {
    local float part[8 * 64];
    mvq3i(w, sc, x, bsum, y, cols, part, 8);
}

// x = dequantized row `token` of a Q4 embedding in the interleaved layout (token from `forced` or the last pick).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void embed_q4i(global const uchar* w, global const half_t* scales, global const uint* last, uint forced,
               global float* x, uint cols) {
    uint tok = forced != NO_TOKEN ? forced : *last;
    uint nb = cols / 32, t = tok / 64, l = tok % 64;
    for (uint i = LID; i < cols; i += 256) {
        uint blk = i >> 5, j = i & 31;
        ulong at = ((ulong)t * nb + blk) * 64 + l;
        uint byte = w[at * 16 + (j & 15)];
        uint q = j < 16 ? (byte & 15u) : (byte >> 4);
        x[i] = ((float)q - 8.0f) * (float)scales[at];
    }
}

// x = dequantized row `token` of a Q4 embedding (token from `forced` or the last argmax).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void embed_q4(global const uchar* w, global const ushort* scales, global const uint* last, uint forced,
              global float* x, uint cols) {
    uint tok = forced != NO_TOKEN ? forced : *last;
    global const uchar* row = w + (ulong)tok * (cols / 2);
    global const ushort* sc = scales + (ulong)tok * (cols / 32);
    for (uint i = LID; i < cols; i += 256) {
        uint blk = i >> 5, j = i & 31;
        uint byte = row[blk * 16 + (j & 15)];
        uint q = j < 16 ? (byte & 15u) : (byte >> 4);
        x[i] = ((float)q - 8.0f) * h2f(sc[blk]);
    }
}

// ---- sampling ---------------------------------------------------------------------------------------------------
// Draw the next token from softmax(logits / T) without reading the logits back: the argmax of
// logits / T + Gumbel noise is an exact sample (Gumbel-max trick). inv_temp = 0 gives greedy decoding.
// Noise comes from a hash of (seed, step, index), so a run is reproducible for a given seed.

static uint hash32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sample(global const float* logits, uint n, float inv_temp, uint seed, global uint* last, global uint* history,
            uint step, float min_gap) {
    local float bv[256];
    local uint bi[256];
    const float LN2 = 0.69314718f;
    uint t = LID, salt = hash32(seed ^ (step * 0x9E3779B9u));
    // min-p: only tokens whose probability is at least min_p times the top one may be drawn, i.e. whose logit
    // is within min_gap = T * ln(1 / min_p) of the largest. Without it, sampling over 150k tokens now and then
    // picks a very unlikely one (with Qwen, often a Chinese character in the middle of a Spanish answer).
    float top = -3.0e38f;
    for (uint i = t; i < n; i += 256) top = __builtin_fmaxf(top, logits[i]);
    bv[t] = top;
    BARRIER();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) bv[t] = __builtin_fmaxf(bv[t], bv[t + s]);
        BARRIER();
    }
    float floor_ = bv[0] - min_gap;
    BARRIER();
    float best = -3.0e38f;
    uint idx = 0;
    for (uint i = t; i < n; i += 256) {
        float v = logits[i];
        if (inv_temp > 0.0f) {
            if (v < floor_) continue;
            float u = ((float)(hash32(salt ^ i) >> 8) + 0.5f) * (1.0f / 16777216.0f);   // (0, 1)
            v = v * inv_temp - __builtin_log2f(-__builtin_log2f(u) * LN2) * LN2;          // + Gumbel(0, 1)
        }
        if (v > best) {
            best = v;
            idx = i;
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
        *last = bi[0];
        history[step] = bi[0];
    }
}

#endif
