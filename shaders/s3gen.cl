// Chatterbox S3Gen (the meanflow decoder of chatterbox-turbo) + HiFT vocoder on the PS4 GPU: speech tokens ->
// mel -> 24 kHz waveform. Reference: pc/s3gen_ref.py.
// Activations are fp32, frames-major [T][C]. A Conv1d is a matrix product whose input row for output
// frame t is the k*C floats starting at frame t*stride - pad of a zero-padded buffer (X points at frame -pad,
// ldx = stride * C), so the matrix kernels take ldx / ldy and never see a convolution.

#include "common.h"

#define S3_NB 16          // frames per group in the matrix kernels (each weight is unpacked once for S3_NB frames)
#define S3_MAXT 2048      // most frames an attention sees (T25 or T50 of one sentence)
#define S3_PI 3.14159265358979f

// ---- elementwise functions, as pc/s3gen_ref.py -------------------------------------------------------------
static float s3_erfc_pos(float z) {   // erfc(z), z >= 0, relative error < 1.2e-7 (Numerical Recipes erfcc)
    float t = 1.0f / (1.0f + 0.5f * z);
    float p = 0.17087277f;
    p = -0.82215223f + t * p;
    p = 1.48851587f + t * p;
    p = -1.13520398f + t * p;
    p = 0.27886807f + t * p;
    p = -0.18628806f + t * p;
    p = 0.09678418f + t * p;
    p = 0.37409196f + t * p;
    p = 1.00002368f + t * p;
    return t * exp_(-z * z - 1.26551223f + t * p);
}

static float s3_expm1(float x) {   // accurate near 0 (Kahan): (e^x - 1) x / ln(e^x)
    float u = exp_(x);
    if (u == 1.0f) return x;
    float um1 = u - 1.0f;
    if (um1 == -1.0f) return -1.0f;
    return um1 * x / (__builtin_log2f(u) * 0.69314718f);
}

static float s3_mish(float x) {
    if (x > 20.0f) return x;
    float n = exp_(x), p = n * (n + 2.0f);
    return x * p / (p + 2.0f);
}

static float s3_tanh(float x) {
    float e = exp_(-2.0f * __builtin_fabsf(x));
    float r = (1.0f - e) / (1.0f + e);
    return x < 0.0f ? -r : r;
}

static float s3_sin2pi(float x) { return __builtin_amdgcn_sinf(x - __builtin_floorf(x)); }   // sin(2 pi x)
static float s3_cos2pi(float x) { return __builtin_amdgcn_cosf(x - __builtin_floorf(x)); }

// 0 none, 1 SiLU, 3 GELU (erf), 4 leaky ReLU 0.01, 5 ELU, 6 |x|
static float s3_act(float v, uint act) {
    switch (act) {
    case 1: return v / (1.0f + exp_(-v));
    case 3: {
        float c = 0.5f * s3_erfc_pos(__builtin_fabsf(v) * 0.7071067811865476f);
        return v >= 0.0f ? v * (1.0f - c) : v * c;
    }
    case 4: return v >= 0.0f ? v : 0.01f * v;
    case 5: return v > 0.0f ? v : s3_expm1(v);
    case 6: return __builtin_fabsf(v);
    default: return v;
    }
}

// ---- matrix kernels -----------------------------------------------------------------------------------------
// Y[r*ldy + o] = beta * Y[r*ldy + o] + alpha * act(sum_c W[o][c] X[r*ldx + c] + bias[o]), r < n.
// W q4i (shaders/common.h layout, rows a multiple of 64). Group = 4 waves = 64 outputs x S3_NB frames; lane =
// output; the waves split the blocks of 32 inputs. X loads are uniform per wave (scalar).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_mm_q4i(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict bias,
               global const float* restrict X, global float* restrict Y, uint cols, uint rows, uint n, uint ldx,
               uint ldy, uint act, float alpha, float beta) {
    local float part[4][S3_NB][64];
    uint l = LID & 63, nb = cols / 32, tiles = rows / 64;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tile = GID % tiles, r0 = (GID / tiles) * S3_NB;
    uint nv = n - r0 < S3_NB ? n - r0 : S3_NB;
    ulong tb = (ulong)tile * nb * 64;
    global const uint4v* wp = w + tb + l;
    global const half_t* sp = sc + tb + l;
    float acc[S3_NB];
    for (uint v = 0; v < S3_NB; v++) acc[v] = 0.0f;
    for (uint b = wv; b < nb; b += 4) {
        uint4v q = wp[(ulong)b * 64];
        float d = (float)sp[(ulong)b * 64], m8 = -8.0f * d;
        uint ws[4] = {q.x, q.y, q.z, q.w};
        float wf[32];
        for (uint k = 0; k < 4; k++) {
            uint lo = ws[k] & 0x0F0F0F0Fu, hi = (ws[k] >> 4) & 0x0F0F0F0Fu;
            __asm__("" : "+v"(lo), "+v"(hi));
            wf[4 * k] = (float)(lo & 0xFFu) * d + m8;
            wf[4 * k + 1] = (float)((lo >> 8) & 0xFFu) * d + m8;
            wf[4 * k + 2] = (float)((lo >> 16) & 0xFFu) * d + m8;
            wf[4 * k + 3] = (float)(lo >> 24) * d + m8;
            wf[16 + 4 * k] = (float)(hi & 0xFFu) * d + m8;
            wf[16 + 4 * k + 1] = (float)((hi >> 8) & 0xFFu) * d + m8;
            wf[16 + 4 * k + 2] = (float)((hi >> 16) & 0xFFu) * d + m8;
            wf[16 + 4 * k + 3] = (float)(hi >> 24) * d + m8;
        }
        for (uint v = 0; v < S3_NB; v++) {
            if (v < nv) {
                global const float* xb = X + (ulong)(r0 + v) * ldx + b * 32;
                float s = 0.0f;
                for (uint j = 0; j < 32; j++) s += wf[j] * xb[j];
                acc[v] += s;
            }
        }
    }
    for (uint v = 0; v < S3_NB; v++) part[wv][v][l] = acc[v];
    BARRIER();
    if (wv == 0) {
        uint o = tile * 64 + l;
        float bo = bias ? bias[o] : 0.0f;
        for (uint v = 0; v < nv; v++) {
            float r = s3_act(part[0][v][l] + part[1][v][l] + part[2][v][l] + part[3][v][l] + bo, act);
            global float* y = Y + (ulong)(r0 + v) * ldy + o;
            *y = (beta != 0.0f ? beta * *y : 0.0f) + alpha * r;
        }
    }
}

// The same with fp16 weights (f16i: tile g of 64 outputs, column c, lane l at ((g * cols) + c) * 64 + l) and a
// dilated tap gather: column c = j*C + ch reads X[r*ldx + j*dil*C + ch] (C a multiple of 8; dil 1 = contiguous).
// No activation (HiFT applies leaky ReLU / Snake before its convs).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_mm_f16i(global const half_t* restrict w, global const float* restrict bias, global const float* restrict X,
                global float* restrict Y, uint cols, uint rows, uint n, uint ldx, uint ldy, uint C, uint dil,
                float alpha, float beta) {
    local float part[4][S3_NB][64];
    uint l = LID & 63, tiles = rows / 64;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tile = GID % tiles, r0 = (GID / tiles) * S3_NB;
    uint nv = n - r0 < S3_NB ? n - r0 : S3_NB;
    global const half_t* wp = w + (ulong)tile * cols * 64 + l;
    float acc[S3_NB];
    for (uint v = 0; v < S3_NB; v++) acc[v] = 0.0f;
    for (uint c0 = wv * 8; c0 < cols; c0 += 32) {   // 8 columns at a time, all in the same tap
        float wf[8];
        for (uint j = 0; j < 8; j++) wf[j] = (float)wp[(ulong)(c0 + j) * 64];
        uint xo = dil == 1 ? c0 : (c0 / C) * dil * C + c0 % C;
        for (uint v = 0; v < S3_NB; v++) {
            if (v < nv) {
                global const float* xb = X + (ulong)(r0 + v) * ldx + xo;
                float s = 0.0f;
                for (uint j = 0; j < 8; j++) s += wf[j] * xb[j];
                acc[v] += s;
            }
        }
    }
    for (uint v = 0; v < S3_NB; v++) part[wv][v][l] = acc[v];
    BARRIER();
    if (wv == 0) {
        uint o = tile * 64 + l;
        float bo = bias ? bias[o] : 0.0f;
        for (uint v = 0; v < nv; v++) {
            float r = part[0][v][l] + part[1][v][l] + part[2][v][l] + part[3][v][l] + bo;
            global float* y = Y + (ulong)(r0 + v) * ldy + o;
            *y = (beta != 0.0f ? beta * *y : 0.0f) + alpha * r;
        }
    }
}

// fp32 weights stored transposed, Wt[c][o] (o padded to rows, a multiple of 64): small precision-critical
// matrices. One thread per output (r, o < nout); X[r] is a scalar load for the wave.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void s3_mm_f32t(global const float* restrict wt, global const float* restrict bias, global const float* restrict X,
                global float* restrict Y, uint cols, uint rows, uint n, uint ldx, uint ldy, uint nout, uint act) {
    uint tiles = rows / 64, o = (GID % tiles) * 64 + LID;
    uint r = __builtin_amdgcn_readfirstlane(GID / tiles);
    if (r >= n || o >= nout) return;
    global const float* x = X + (ulong)r * ldx;
    float s = bias ? bias[o] : 0.0f;
    for (uint c = 0; c < cols; c++) s += wt[(ulong)c * rows + o] * x[c];
    Y[(ulong)r * ldy + o] = s3_act(s, act);
}

// Y[r] = LayerNorm(X[r]) * w + b (eps), then Mish if act = 1, then + tb if given. One group per row.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_ln(global const float* X, global const float* w, global const float* b, global const float* tb,
           global float* Y, uint cols, uint ldx, uint ldy, float eps, uint act) {
    local float red[256];
    global const float* x = X + (ulong)GID * ldx;
    float s = 0.0f;
    for (uint i = LID; i < cols; i += 256) s += x[i];
    float mean = group_sum(red, s) / (float)cols;
    float v = 0.0f;
    for (uint i = LID; i < cols; i += 256) v += (x[i] - mean) * (x[i] - mean);
    float r = 1.0f / __builtin_sqrtf(group_sum(red, v) / (float)cols + eps);
    for (uint i = LID; i < cols; i += 256) {
        float y = (x[i] - mean) * r * w[i] + b[i];
        if (act == 1) y = s3_mish(y);
        if (tb) y += tb[i];
        Y[(ulong)GID * ldy + i] = y;
    }
}

// Row tok[r] of a q4i embedding [rows][cols] -> X[r] (one group per token).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_embed_q4i(global const uint4v* w, global const half_t* sc, global const uint* tok, global float* X,
                  uint cols) {
    uint r = tok[GID], tile = r / 64, lane = r % 64, nb = cols / 32;
    for (uint e = LID; e < cols; e += 256) {
        uint b = e / 32, j = e % 32;
        ulong idx = ((ulong)tile * nb + b) * 64 + lane;
        uint4v q = w[idx];
        uint k = (j % 16) / 4, wd = k == 0 ? q.x : k == 1 ? q.y : k == 2 ? q.z : q.w;
        uint byte = (wd >> ((j % 4) * 8)) & 0xFFu;
        uint nib = j < 16 ? byte & 15u : byte >> 4;
        X[(ulong)GID * cols + e] = ((float)nib - 8.0f) * (float)sc[idx];
    }
}

// ESPnet relative positions pe [2T-1][512]: row i = position p = T-1-i, pe[i][2k] = sin(p d_k), [2k+1] = cos(p d_k),
// p d_k rounded to fp32 as torch does (div = d_k from the host), the angle reduced in double (it reaches ~1e3 rad).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_relpos(global const float* div, global float* pe, uint T) {
    uint g = GID * 256 + LID;
    if (g >= (2 * T - 1) * 256) return;
    uint i = g / 256, k = g % 256;
    float x = (float)((int)T - 1 - (int)i) * div[k];
    double rev = (double)x * 0.15915494309189535;
    rev -= __builtin_floor(rev);
    float fr = (float)rev;
    pe[(ulong)i * 512 + 2 * k] = __builtin_amdgcn_sinf(fr);
    pe[(ulong)i * 512 + 2 * k + 1] = __builtin_amdgcn_cosf(fr);
}

// ---- attention (8 heads x 64 over 512) -----------------------------------------------------------------------
// Encoder: relative positions (ESPnet / Transformer-XL). One group of 64 threads per (query frame i, head h).
// QKV [T][1536] = q | k | v (biases added), R [2T-1][512] = pe x linear_pos (row T-1-i+j = distance i-j).
// score(i, j) = ((q_i + u) . k_j + (q_i + v) . r_{i-j}) / 8, softmax over j; O [T][512].
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void s3_relattn(global const float* QKV, global const float* R, global const float* bu, global const float* bv,
                global float* O, uint T) {
    local float qu[64], qv[64], p[S3_MAXT], red[64];
    uint h = GID % 8, i = GID / 8, t = LID;
    float q = QKV[(ulong)i * 1536 + h * 64 + t];
    qu[t] = q + bu[h * 64 + t];
    qv[t] = q + bv[h * 64 + t];
    BARRIER();
    float mx = -3.0e38f;
    for (uint j = t; j < T; j += 64) {
        global const float* k = QKV + (ulong)j * 1536 + 512 + h * 64;
        global const float* r = R + (ulong)(T - 1 - i + j) * 512 + h * 64;
        float s = 0.0f;
#pragma unroll 8
        for (uint d = 0; d < 64; d++) s += qu[d] * k[d] + qv[d] * r[d];
        s *= 0.125f;
        p[j] = s;
        mx = __builtin_fmaxf(mx, s);
    }
    red[t] = mx;
    BARRIER();
    for (uint st = 32; st > 0; st >>= 1) {
        if (t < st) red[t] = __builtin_fmaxf(red[t], red[t + st]);
        BARRIER();
    }
    mx = red[0];
    BARRIER();
    float sum = 0.0f;
    for (uint j = t; j < T; j += 64) {
        float e = exp_(p[j] - mx);
        p[j] = e;
        sum += e;
    }
    red[t] = sum;
    BARRIER();
    for (uint st = 32; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        BARRIER();
    }
    sum = red[0];
    float o = 0.0f;
    for (uint j = 0; j < T; j++) o += p[j] * QKV[(ulong)j * 1536 + 1024 + h * 64 + t];
    O[(ulong)i * 512 + h * 64 + t] = o / sum;
}

// Encoder attention, faster: rb[h][row] = ((v - u) . r_row) / 8 for every row of R (one thread per (row, head)),
// then s3_relattn2: one wave per (block of 64 queries, head), one query per thread with (q + u) / 8 and the output
// in registers; per tile of 16 key frames, K, V and the 79 R rows that tile meets go through LDS (R rows padded to
// 65 floats: lane t reads row 63 - t + r, so the lanes hit different banks). score = qu . (k_j + r_{i-j}) + rb.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void s3_relbias(global const float* R, global const float* bu, global const float* bv, global float* rb, uint T) {
    uint g = GID * 64 + LID, rows = 2 * T - 1;
    if (g >= rows * 8) return;
    uint h = g / rows, row = g % rows;
    global const float* r = R + (ulong)row * 512 + h * 64;
    float s = 0.0f;
    for (uint d = 0; d < 64; d++) s += (bv[h * 64 + d] - bu[h * 64 + d]) * r[d];
    rb[g] = s * 0.125f;
}

#define S3_RT 16
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void s3_relattn2(global const float* QKV, global const float* R, global const float* bu, global const float* rb,
                 global float* O, uint T) {
    local float ks[S3_RT][64], vs[S3_RT][64], rs[64 + S3_RT - 1][65], cs[64 + S3_RT];
    uint h = GID % 8, t = LID, i0 = (GID / 8) * 64, i = i0 + t;
    uint ic = i < T ? i : T - 1;
    float q[64], o[64];
#pragma unroll
    for (uint d = 0; d < 64; d++) {
        q[d] = (QKV[(ulong)ic * 1536 + h * 64 + d] + bu[h * 64 + d]) * 0.125f;
        o[d] = 0.0f;
    }
    float m = -3.0e38f, lsum = 0.0f;
    uint rows = 2 * T - 1;
    for (uint j0 = 0; j0 < T; j0 += S3_RT) {
        BARRIER();
        for (uint r = 0; r < S3_RT; r++) {
            uint j = j0 + r < T ? j0 + r : T - 1;
            global const float* kp = QKV + (ulong)j * 1536 + h * 64;
            ks[r][t] = kp[512 + t];
            vs[r][t] = kp[1024 + t];
        }
        // R rows T-1-i+j for i in [i0, i0+63], j in [j0, j0+15]: base = T-1-(i0+63)+j0 (may be < 0 or past the end
        // for frames outside the sentence: clamped, those scores are not used)
        int base = (int)T - 1 - (int)(i0 + 63) + (int)j0;
        for (uint r = 0; r < 64 + S3_RT - 1; r++) {
            int row = base + (int)r;
            row = row < 0 ? 0 : row >= (int)rows ? (int)rows - 1 : row;
            rs[r][t] = R[(ulong)row * 512 + h * 64 + t];
        }
        for (uint r = t; r < 64 + S3_RT - 1; r += 64) {
            int row = base + (int)r;
            row = row < 0 ? 0 : row >= (int)rows ? (int)rows - 1 : row;
            cs[r] = rb[(ulong)h * rows + row];
        }
        BARRIER();
        uint cnt = T - j0 < S3_RT ? T - j0 : S3_RT;
        float s[S3_RT];
        float mt = m;
#pragma unroll
        for (uint r = 0; r < S3_RT; r++) {
            uint lr = 63 - t + r;
            float a = cs[lr];
#pragma unroll
            for (uint d = 0; d < 64; d++) a += q[d] * (ks[r][d] + rs[lr][d]);
            s[r] = r < cnt ? a : -3.0e38f;
            mt = __builtin_fmaxf(mt, s[r]);
        }
        float corr = exp_(m - mt);
        lsum *= corr;
#pragma unroll
        for (uint d = 0; d < 64; d++) o[d] *= corr;
#pragma unroll
        for (uint r = 0; r < S3_RT; r++) {
            float pr = r < cnt ? exp_(s[r] - mt) : 0.0f;
            lsum += pr;
#pragma unroll
            for (uint d = 0; d < 64; d++) o[d] += pr * vs[r][d];
        }
        m = mt;
    }
    if (i < T) {
        float inv = 1.0f / lsum;
        global float* op = O + (ulong)i * 512 + h * 64;
#pragma unroll
        for (uint d = 0; d < 64; d++) op[d] = o[d] * inv;
    }
}

// CFM: plain full attention, scale 1/8, QKV [T][1536] (no biases) -> O [T][512]. Flash style: one wave per (block
// of 64 queries, head), one query per thread with q and the output in registers; K and V go through LDS in tiles
// of 32 frames, softmax online (running max and sum).
#define S3_KT 16
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void s3_attn(global const float* QKV, global float* O, uint T) {
    local float4v ks[S3_KT][16], vs[S3_KT][16];
    uint h = GID % 8, t = LID, i = (GID / 8) * 64 + t;
    float4v q[16], o[16];
    uint ic = i < T ? i : T - 1;
#pragma unroll
    for (uint d = 0; d < 16; d++) {
        global const float* qp = QKV + (ulong)ic * 1536 + h * 64 + 4 * d;
        q[d] = (float4v)(qp[0], qp[1], qp[2], qp[3]) * 0.125f;
        o[d] = (float4v)(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float m = -3.0e38f, lsum = 0.0f;
    for (uint k0 = 0; k0 < T; k0 += S3_KT) {
        BARRIER();
        for (uint r = 0; r < S3_KT; r++) {   // lane t loads dimension t of each row
            uint j = k0 + r < T ? k0 + r : T - 1;
            global const float* kp = QKV + (ulong)j * 1536 + h * 64;
            ((local float*)ks[r])[t] = kp[512 + t];
            ((local float*)vs[r])[t] = kp[1024 + t];
        }
        BARRIER();
        uint cnt = T - k0 < S3_KT ? T - k0 : S3_KT;
        float s[S3_KT];
        float mt = m;
#pragma unroll
        for (uint r = 0; r < S3_KT; r++) {
            float4v a = q[0] * ks[r][0];
#pragma unroll
            for (uint d = 1; d < 16; d++) a += q[d] * ks[r][d];
            s[r] = r < cnt ? a.x + a.y + a.z + a.w : -3.0e38f;
            mt = __builtin_fmaxf(mt, s[r]);
        }
        float corr = exp_(m - mt);
        lsum *= corr;
#pragma unroll
        for (uint d = 0; d < 16; d++) o[d] *= corr;
#pragma unroll
        for (uint r = 0; r < S3_KT; r++) {
            float pr = r < cnt ? exp_(s[r] - mt) : 0.0f;
            lsum += pr;
#pragma unroll
            for (uint d = 0; d < 16; d++) o[d] += pr * vs[r][d];
        }
        m = mt;
    }
    if (i < T) {
        float inv = 1.0f / lsum;
        global float* op = O + (ulong)i * 512 + h * 64;
#pragma unroll
        for (uint d = 0; d < 16; d++) {
            op[4 * d] = o[d].x * inv;
            op[4 * d + 1] = o[d].y * inv;
            op[4 * d + 2] = o[d].z * inv;
            op[4 * d + 3] = o[d].w * inv;
        }
    }
}

// The same attention with K and V read as scalars: one wave per (block of 64 queries, head), lane =
// query, and each key row is loaded once per wave into SGPRs (its address is the same for every lane), so a value
// feeds the 64 lanes' FMAs for 4 bytes of the scalar cache. In s3_attn every K / V value goes through LDS as a
// broadcast read, which moves 64 x 4 bytes per value: that kernel is bound by LDS bandwidth (and its 224 VGPRs
// leave one wave per SIMD). Online softmax over steps of S3_KS keys. engine_s3g.inc times both at load and keeps the
// faster (s3_attn_tune), after checking that they agree.
#define S3_KS 8
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void s3_attn_s(global const float* restrict QKV, global float* restrict O, uint T) {
    uint h = GID % 8, t = LID, i = (GID / 8) * 64 + t;
    uint ic = i < T ? i : T - 1;
    float q[64], o[64];
    global const float* qp = QKV + (ulong)ic * 1536 + h * 64;
#pragma unroll
    for (uint d = 0; d < 64; d++) q[d] = qp[d] * 0.125f, o[d] = 0.0f;
    float m = -3.0e38f, lsum = 0.0f;
    for (uint k0 = 0; k0 < T; k0 += S3_KS) {
        uint cnt = T - k0 < S3_KS ? T - k0 : S3_KS;
        float s[S3_KS];
        float mt = m;
#pragma unroll
        for (uint r = 0; r < S3_KS; r++) {
            uint j = k0 + r < T ? k0 + r : T - 1;
            global const float* kp = QKV + (ulong)j * 1536 + 512 + h * 64;
            float a = 0.0f;
#pragma unroll
            for (uint d = 0; d < 64; d++) a += q[d] * kp[d];
            s[r] = r < cnt ? a : -3.0e38f;
            mt = __builtin_fmaxf(mt, s[r]);
        }
        float corr = exp_(m - mt);
        lsum *= corr;
#pragma unroll
        for (uint d = 0; d < 64; d++) o[d] *= corr;
#pragma unroll
        for (uint r = 0; r < S3_KS; r++) {
            uint j = k0 + r < T ? k0 + r : T - 1;
            global const float* vp = QKV + (ulong)j * 1536 + 1024 + h * 64;
            float pr = r < cnt ? exp_(s[r] - mt) : 0.0f;
            lsum += pr;
#pragma unroll
            for (uint d = 0; d < 64; d++) o[d] += pr * vp[d];
        }
        m = mt;
    }
    if (i < T) {
        float inv = 1.0f / lsum;
        global float* op = O + (ulong)i * 512 + h * 64;
#pragma unroll
        for (uint d = 0; d < 64; d++) op[d] = o[d] * inv;
    }
}

// s3_attn_s split between 2 waves (group of 128 threads): wave w holds dimensions 32w .. 32w + 31 of q and of the
// output, so each lane keeps 64 values instead of 128 and 2 waves fit per SIMD (to hide the scalar loads: with
// one wave per SIMD, s3_attn_s is slower than s3_attn). The two halves of each
// score meet in LDS (double buffered: one barrier per step of S3_KS keys) and are added in the same order by both
// waves, so both see the same softmax.
kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void s3_attn_s2(global const float* restrict QKV, global float* restrict O, uint T) {
    local float ps[2][2][S3_KS][64];   // [buffer][wave][key][lane]: half scores
    uint h = GID % 8, t = LID & 63, i = (GID / 8) * 64 + t;
    uint w = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint ic = i < T ? i : T - 1;
    float q[32], o[32];
    global const float* qp = QKV + (ulong)ic * 1536 + h * 64 + 32 * w;
#pragma unroll
    for (uint d = 0; d < 32; d++) q[d] = qp[d] * 0.125f, o[d] = 0.0f;
    float m = -3.0e38f, lsum = 0.0f;
    uint buf = 0;
    for (uint k0 = 0; k0 < T; k0 += S3_KS, buf ^= 1) {
        uint cnt = T - k0 < S3_KS ? T - k0 : S3_KS;
#pragma unroll
        for (uint r = 0; r < S3_KS; r++) {
            uint j = k0 + r < T ? k0 + r : T - 1;
            global const float* kp = QKV + (ulong)j * 1536 + 512 + h * 64 + 32 * w;
            float a = 0.0f;
#pragma unroll
            for (uint d = 0; d < 32; d++) a += q[d] * kp[d];
            ps[buf][w][r][t] = a;
        }
        BARRIER();
        float s[S3_KS];
        float mt = m;
#pragma unroll
        for (uint r = 0; r < S3_KS; r++) {
            float sc = ps[buf][0][r][t] + ps[buf][1][r][t];
            s[r] = r < cnt ? sc : -3.0e38f;
            mt = __builtin_fmaxf(mt, s[r]);
        }
        float corr = exp_(m - mt);
        lsum *= corr;
#pragma unroll
        for (uint d = 0; d < 32; d++) o[d] *= corr;
#pragma unroll
        for (uint r = 0; r < S3_KS; r++) {
            uint j = k0 + r < T ? k0 + r : T - 1;
            global const float* vp = QKV + (ulong)j * 1536 + 1024 + h * 64 + 32 * w;
            float pr = r < cnt ? exp_(s[r] - mt) : 0.0f;
            lsum += pr;
#pragma unroll
            for (uint d = 0; d < 32; d++) o[d] += pr * vp[d];
        }
        m = mt;
    }
    if (i < T) {
        float inv = 1.0f / lsum;
        global float* op = O + (ulong)i * 512 + h * 64 + 32 * w;
#pragma unroll
        for (uint d = 0; d < 32; d++) op[d] = o[d] * inv;
    }
}

// ---- small elementwise kernels ------------------------------------------------------------------------------
// Y[t*ldy + c] = act(alpha * Y[t*ldy + c] + beta * X[t*ldx + c]) for t < T, c < C; act: slope of a leaky ReLU
// (1 = none). X may be Y (in place); with beta = 0, X is not read.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_mix(global float* Y, global const float* X, uint T, uint C, uint ldy, uint ldx, float alpha, float beta,
            float slope) {
    uint g = GID * 256 + LID;
    if (g >= T * C) return;
    uint t = g / C, c = g % C;
    float y = alpha != 0.0f ? alpha * Y[(ulong)t * ldy + c] : 0.0f;
    if (beta != 0.0f) y += beta * X[(ulong)t * ldx + c];
    if (y < 0.0f) y *= slope;
    Y[(ulong)t * ldy + c] = y;
}

// Snake: Y = X + ia[c] sin^2(a[c] X) (ia = 1 / (a + 1e-9)).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_snake(global const float* X, global float* Y, global const float* a, global const float* ia, uint T, uint C,
              uint ldx, uint ldy) {
    uint g = GID * 256 + LID;
    if (g >= T * C) return;
    uint t = g / C, c = g % C;
    float x = X[(ulong)t * ldx + c];
    float s = s3_sin2pi(a[c] * x * (0.5f / S3_PI));
    Y[(ulong)t * ldy + c] = x + ia[c] * s * s;
}

// CFM input xin [T][320] = noise | mu | spk | cond: noise and the prompt mel (first Tp frames) from the host, spk the
// same 80 values on every frame; mu (columns 80..159) is written by the encoder.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_cfm_init(global float* xin, global const float* noise, global const float* spk, global const float* pmel,
                 uint T, uint Tp) {
    uint g = GID * 256 + LID;
    if (g >= T * 80) return;
    uint t = g / 80, c = g % 80;
    global float* x = xin + (ulong)t * 320;
    x[c] = noise[g];
    x[160 + c] = spk[c];
    x[240 + c] = t < Tp ? pmel[g] : 0.0f;
}

// HiFT F0 head: f0[t] = |sum_c w[c] x[t][c] + b|, 512 inputs, one wave per frame.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void s3_f0(global const float* X, global const float* w, global const float* b, global float* f0, uint cols) {
    local float red[64];
    global const float* x = X + (ulong)GID * cols;
    float s = 0.0f;
    for (uint c = LID; c < cols; c += 64) s += w[c] * x[c];
    red[LID] = s;
    BARRIER();
    for (uint st = 32; st > 0; st >>= 1) {
        if (LID < st) red[LID] += red[LID + st];
        BARRIER();
    }
    if (LID == 0) f0[GID] = __builtin_fabsf(red[0] + b[0]);
}

// NSF phase at the start of each mel frame, in cycles: C[m][h] = frac(sum_{m' < m} 480 f0[m'] (h+1) / 24000).
// One group; thread h < 9 scans the frames (double).
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void s3_nsf_phase(global const float* f0, global float* C, uint M) {
    uint h = LID;
    if (h >= 9) return;
    double acc = 0.0;
    for (uint m = 0; m < M; m++) {
        C[m * 9 + h] = (float)acc;
        float inc = f0[m] * (float)(h + 1) / 24000.0f;
        acc += (double)inc * 480.0;
        acc -= __builtin_floor(acc);
    }
}

static float s3_gauss(uint seed, uint key) {   // N(0, 1) from a counter (Box-Muller)
    uint a = hash32(key * 2u + seed), b = hash32(key * 2u + 1u + seed);
    float u1 = ((float)(a >> 8) + 0.5f) * (1.0f / 16777216.0f);
    float u2 = (float)(b >> 8) * (1.0f / 16777216.0f);
    return __builtin_sqrtf(-2.0f * 0.69314718f * __builtin_log2f(u1)) * s3_cos2pi(u2);
}

// NSF excitation, one thread per 24 kHz sample n = 480 m + k (n0 = first sample of this call):
// s[n] = tanh(sum_h w_h (0.1 sin(2 pi ph_h + phi_h) uv + (uv ? 0.003 : 0.1/3) g_h(n)) + b), uv = f0 > 10 Hz.
// phi: the 9 phases in cycles (host, phi_0 = 0).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_nsf_source(global const float* f0, global const float* C, global const float* w, global const float* b,
                   global const float* phi, global float* s, uint n_samples, uint seed) {
    uint n = GID * 256 + LID;
    if (n >= n_samples) return;
    uint m = n / 480, k = n % 480;
    float f = f0[m];
    int uv = f > 10.0f;
    float acc = b[0];
    for (uint h = 0; h < 9; h++) {
        float inc = f * (float)(h + 1) / 24000.0f;
        float ph = C[m * 9 + h] + (float)(k + 1) * inc;
        float sine = uv ? 0.1f * s3_sin2pi(ph + phi[h]) : 0.0f;
        acc += w[h] * (sine + (uv ? 0.003f : 0.1f / 3.0f) * s3_gauss(seed, n * 9u + h));
    }
    s[n] = s3_tanh(acc);
}

// STFT (n_fft 16, hop 4, Hann, centred, reflect): S[f][k] = sum_n B[k][n] s[4f - 8 + n], f < F, k < 18.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_stft(global const float* s, global const float* B, global float* S, uint L, uint F) {
    uint g = GID * 256 + LID;
    if (g >= F * 18) return;
    uint f = g / 18, k = g % 18;
    float acc = 0.0f;
    for (uint n = 0; n < 16; n++) {
        int i = (int)(4 * f) - 8 + (int)n;
        if (i < 0) i = -i;
        if (i >= (int)L) i = 2 * ((int)L - 1) - i;
        acc += B[k * 16 + n] * s[i];
    }
    S[g] = acc;
}

// conv_post output X [F][18] (row stride ldx; frame f of X = global spectrum frame f + fx0) -> waveform samples
// [i0, i0 + n): magnitude min(e^x, 100), phase sin(x); inverse DFT x Hann (IB [16][18]), overlap-add (hop 4),
// divided by the overlap-added Hann^2 of the F_total frames; clamped to +-0.99; then the official post-processing:
// samples < 480 -> 0, 480..959 faded in. One thread per sample.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void s3_istft(global const float* X, global const float* IB, global float* out, uint ldx, int fx0, uint i0,
              uint n, uint F_total) {
    uint g = GID * 256 + LID;
    if (g >= n) return;
    int i = (int)(i0 + g);
    float y = 0.0f, env = 0.0f;
    int fa = (i + 8 - 15 + 3) / 4, fb = (i + 8) / 4;   // frames f with 0 <= i + 8 - 4f < 16
    if (fa < 0) fa = 0;
    if (fb > (int)F_total - 1) fb = (int)F_total - 1;
    for (int f = fa; f <= fb; f++) {
        int nn = i + 8 - 4 * f;
        global const float* x = X + (long)(f - fx0) * ldx;
        float acc = 0.0f;
        for (uint k = 0; k < 9; k++) {
            float mag = __builtin_fminf(exp_(x[k]), 100.0f);
            float ph = s3_sin2pi(x[9 + k] * (0.5f / S3_PI)) * (0.5f / S3_PI);   // sin(x), in cycles
            acc += mag * (IB[nn * 18 + k] * s3_cos2pi(ph) + IB[nn * 18 + 9 + k] * s3_sin2pi(ph));
        }
        float hw = 0.5f - 0.5f * s3_cos2pi((float)nn / 16.0f);
        y += acc;
        env += hw * hw;
    }
    y = __builtin_fminf(__builtin_fmaxf(y / env, -0.99f), 0.99f);
    if (i < 480) y = 0.0f;
    else if (i < 960) y *= (s3_cos2pi(0.5f - 0.5f * (float)(i - 480) / 479.0f) + 1.0f) * 0.5f;
    out[g] = y;
}
