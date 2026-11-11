// Parakeet TDT 0.6B v3 (speech to text) on the PS4 GPU. Activations are row-major [frames][channels].
// The big matrices are q4i (shaders/common.h): mm_q4i multiplies one by up to MM_NB rows of X at a time, so each
// weight is unpacked once for MM_NB frames. Reference: pc/parakeet_ref.py.

#include "common.h"

#define MM_NB 8
#define D_MODEL 1024
#define N_HEADS 8
#define D_HEAD 128
#define MAX_T 512                 // encoder frames (80 ms each): 40 s of audio

static float act_fn(float v, uint act) {   // 0 none, 1 SiLU, 2 ReLU
    return act == 1 ? v / (1.0f + exp_(-v)) : act == 2 ? __builtin_fmaxf(v, 0.0f) : v;
}

// Y[r][o] = beta * Y[r][o] + alpha * act(sum_c W[o][c] X[r][c] + bias[o]) for r < n. W is q4i (rows a multiple of
// 64). Group = 4 waves = one tile of 64 output channels x MM_NB rows of X; lane = output channel; waves split the
// blocks of the input. Group g: tile g % tiles, rows (g / tiles) * MM_NB ...
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mm_q4i(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict bias,
            global const float* restrict X, global float* restrict Y, uint cols, uint rows, uint n, uint ldx,
            uint ldy, uint act, float alpha, float beta) {
    local float part[4][MM_NB][64];
    uint l = LID & 63, nb = cols / 32, tiles = rows / 64;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tile = GID % tiles, r0 = (GID / tiles) * MM_NB;
    uint nv = n - r0 < MM_NB ? n - r0 : MM_NB;
    ulong tb = (ulong)tile * nb * 64;
    global const uint4v* wp = w + tb + l;
    global const half_t* sp = sc + tb + l;
    float acc[MM_NB];
    for (uint v = 0; v < MM_NB; v++) acc[v] = 0.0f;
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
        for (uint v = 0; v < MM_NB; v++) {
            if (v < nv) {
                global const float* xb = X + (ulong)(r0 + v) * ldx + b * 32;
                float s = 0.0f;
                for (uint j = 0; j < 32; j++) s += wf[j] * xb[j];
                acc[v] += s;
            }
        }
    }
    for (uint v = 0; v < MM_NB; v++) part[wv][v][l] = acc[v];
    BARRIER();
    if (wv == 0) {
        uint o = tile * 64 + l;
        float bo = bias ? bias[o] : 0.0f;
        for (uint v = 0; v < nv; v++) {
            float r = act_fn(part[0][v][l] + part[1][v][l] + part[2][v][l] + part[3][v][l] + bo, act);
            global float* y = Y + (ulong)(r0 + v) * ldy + o;
            *y = (beta != 0.0f ? beta * *y : 0.0f) + alpha * r;
        }
    }
}

// The same with fp32 weights stored transposed, Wt[c][o] (small matrices: the subsampling's pointwise convs).
// One thread per output (r, o); a wave shares r, so X[r][c] is a scalar load.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void mm_f32t(global const float* restrict wt, global const float* restrict bias, global const float* restrict X,
             global float* restrict Y, uint cols, uint rows, uint n, uint act) {
    uint tiles = rows / 64, o = (GID % tiles) * 64 + LID;
    uint r = __builtin_amdgcn_readfirstlane(GID / tiles);
    if (r >= n) return;
    global const float* x = X + (ulong)r * cols;
    float s = bias ? bias[o] : 0.0f;
    for (uint c = 0; c < cols; c++) s += wt[(ulong)c * rows + o] * x[c];
    Y[(ulong)r * rows + o] = act_fn(s, act);
}

// Y[r] = LayerNorm(X[r]) * w + b, one group of 256 threads per row.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void layernorm_rows(global const float* X, global const float* w, global const float* b, global float* Y,
                    uint cols) {
    local float red[256];
    global const float* x = X + (ulong)GID * cols;
    float s = 0.0f;
    for (uint i = LID; i < cols; i += 256) s += x[i];
    float mean = group_sum(red, s) / (float)cols;
    float v = 0.0f;
    for (uint i = LID; i < cols; i += 256) v += (x[i] - mean) * (x[i] - mean);
    float r = rsq(group_sum(red, v) / (float)cols + 1e-5f);
    for (uint i = LID; i < cols; i += 256) Y[(ulong)GID * cols + i] = (x[i] - mean) * r * w[i] + b[i];
}

// Conformer conv module middle: GLU of X [T][2048] -> depthwise conv of 9 over time (zero padded) -> BatchNorm
// (folded: scale, shift) -> SiLU. Two kernels: glu_rows writes G [T][1024]; dwconv9 reads it.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void glu_rows(global const float* X, global float* G, uint T) {
    uint i = GID * 256 + LID;
    if (i >= T * D_MODEL) return;
    uint t = i / D_MODEL, c = i % D_MODEL;
    float a = X[(ulong)t * 2 * D_MODEL + c], g = X[(ulong)t * 2 * D_MODEL + D_MODEL + c];
    G[i] = a / (1.0f + exp_(-g));
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void dwconv9_bn_silu(global const float* G, global const float* w, global const float* scale,
                     global const float* shift, global float* Y, uint T) {
    uint i = GID * 256 + LID;
    if (i >= T * D_MODEL) return;
    uint t = i / D_MODEL, c = i % D_MODEL;
    float s = 0.0f;
    for (int k = 0; k < 9; k++) {
        int tt = (int)t + k - 4;
        if (tt >= 0 && tt < (int)T) s += G[(ulong)tt * D_MODEL + c] * w[c * 9 + k];
    }
    s = s * scale[c] + shift[c];
    Y[i] = s / (1.0f + exp_(-s));
}

// Relative-position self-attention (Transformer-XL): one group of 128 threads per (query frame i, head h).
// QKV: [T][3 * 1024] (q | k | v), R: [2T - 1][1024] = positional embeddings x linear_pos (row T - 1 - i + j holds
// distance i - j). score(i, j) = (q_i + bias_u) . k_j + (q_i + bias_v) . r_{i-j}, scaled by 1 / sqrt(128).
kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void relattn(global const float* QKV, global const float* R, global const float* bu, global const float* bv,
             global float* O, uint T) {
    local float qu[D_HEAD], qv[D_HEAD], p[MAX_T], red[128];
    uint h = GID % N_HEADS, i = GID / N_HEADS, t = LID;
    float q = QKV[(ulong)i * 3 * D_MODEL + h * D_HEAD + t];
    qu[t] = q + bu[h * D_HEAD + t];
    qv[t] = q + bv[h * D_HEAD + t];
    BARRIER();
    float mx = -3.0e38f;
    for (uint j = t; j < T; j += 128) {
        global const float* k = QKV + (ulong)j * 3 * D_MODEL + D_MODEL + h * D_HEAD;
        global const float* r = R + (ulong)(T - 1 - i + j) * D_MODEL + h * D_HEAD;
        float s = 0.0f;
        for (uint d = 0; d < D_HEAD; d++) s += qu[d] * k[d] + qv[d] * r[d];
        s *= 0.08838834764831845f;
        p[j] = s;
        mx = __builtin_fmaxf(mx, s);
    }
    red[t] = mx;
    BARRIER();
    for (uint st = 64; st > 0; st >>= 1) {
        if (t < st) red[t] = __builtin_fmaxf(red[t], red[t + st]);
        BARRIER();
    }
    mx = red[0];
    BARRIER();
    float sum = 0.0f;
    for (uint j = t; j < T; j += 128) {
        float e = exp_(p[j] - mx);
        p[j] = e;
        sum += e;
    }
    red[t] = sum;
    BARRIER();
    for (uint st = 64; st > 0; st >>= 1) {
        if (t < st) red[t] += red[t + st];
        BARRIER();
    }
    sum = red[0];
    float o = 0.0f;
    for (uint j = 0; j < T; j++) o += p[j] * QKV[(ulong)j * 3 * D_MODEL + 2 * D_MODEL + h * D_HEAD + t];
    O[(ulong)i * D_MODEL + h * D_HEAD + t] = o / sum;
}

// ---- subsampling (x8 in time, x8 in frequency) -------------------------------------------------------------
// Activations [time][freq][256]. conv 3x3 stride 2 pad 1, 1 -> 256 channels, + ReLU, from the mels [T][128].
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sub_conv1(global const float* F, global const float* w, global const float* b, global float* Y, uint T,
               uint T1) {
    uint i = GID * 256 + LID, c = i % 256, f = (i / 256) % 64, t = i / (256 * 64);
    if (t >= T1) return;
    float s = b[c];
    for (int ki = 0; ki < 3; ki++) {
        int tt = 2 * (int)t + ki - 1;
        if (tt < 0 || tt >= (int)T) continue;
        for (int kj = 0; kj < 3; kj++) {
            int ff = 2 * (int)f + kj - 1;
            if (ff >= 0 && ff < 128) s += F[(ulong)tt * 128 + ff] * w[c * 9 + ki * 3 + kj];
        }
    }
    Y[i] = __builtin_fmaxf(s, 0.0f);
}

// conv1 (+ ReLU) and the first depthwise conv in one pass, so the conv1 output [T/2][64][256] never exists:
// each depthwise output (t2, f2, c) recomputes the 3x3 conv1 outputs of channel c it needs. -> [T2][32][256]
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sub_conv1_dw1(global const float* F, global const float* w0, global const float* b0, global const float* w1,
                   global const float* b1, global float* Y, uint T, uint T1, uint T2) {
    uint i = GID * 256 + LID, c = i % 256, f2 = (i / 256) % 32, t2 = i / (256 * 32);
    if (t2 >= T2) return;
    // branch-free (clamped indices, zero weight outside): with `continue`s here LLVM produced wrong results
    float s = b1[c];
    for (int ki = 0; ki < 3; ki++) {
        int t1 = 2 * (int)t2 + ki - 1;
        float vt1 = (t1 >= 0 && t1 < (int)T1) ? 1.0f : 0.0f;
        for (int kj = 0; kj < 3; kj++) {
            int f1 = 2 * (int)f2 + kj - 1;
            float v1 = (f1 >= 0 && f1 < 64) ? vt1 : 0.0f;
            float a = b0[c];                               // conv1 output (t1, f1, c)
            for (int ui = 0; ui < 3; ui++) {
                int tt = 2 * t1 + ui - 1;
                float vt = (tt >= 0 && tt < (int)T) ? 1.0f : 0.0f;
                int tc = tt < 0 ? 0 : tt >= (int)T ? (int)T - 1 : tt;
                for (int uj = 0; uj < 3; uj++) {
                    int ff = 2 * f1 + uj - 1;
                    float v = (ff >= 0 && ff < 128) ? vt : 0.0f;
                    int fc = ff < 0 ? 0 : ff > 127 ? 127 : ff;
                    a += v * F[(ulong)tc * 128 + fc] * w0[c * 9 + ui * 3 + uj];
                }
            }
            s += v1 * __builtin_fmaxf(a, 0.0f) * w1[c * 9 + ki * 3 + kj];
        }
    }
    Y[i] = s;
}

// depthwise conv 3x3 stride 2 pad 1 on [H][W][256] -> [H2][W2][256] (no activation: ReLU comes after pointwise)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sub_dwconv(global const float* X, global const float* w, global const float* b, global float* Y, uint H,
                uint W, uint H2, uint W2) {
    uint i = GID * 256 + LID, c = i % 256, x2 = (i / 256) % W2, y2 = i / (256 * W2);
    if (y2 >= H2) return;
    float s = b[c];
    for (int ki = 0; ki < 3; ki++) {
        int yy = 2 * (int)y2 + ki - 1;
        if (yy < 0 || yy >= (int)H) continue;
        for (int kj = 0; kj < 3; kj++) {
            int xx = 2 * (int)x2 + kj - 1;
            if (xx >= 0 && xx < (int)W) s += X[((ulong)yy * W + xx) * 256 + c] * w[c * 9 + ki * 3 + kj];
        }
    }
    Y[i] = s;
}

// ---- TDT decoder ------------------------------------------------------------------------------------------
// LSTM cell: gates [2560] = i | f | g | o; c [640] updated; h written to h_a and h_b.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void lstm_cell(global const float* gates, global float* c, global float* h_a, global float* h_b) {
    uint k = GID * 64 + LID;
    if (k >= 640) return;
    float ig = 1.0f / (1.0f + exp_(-gates[k])), fg = 1.0f / (1.0f + exp_(-gates[640 + k]));
    float gg = gates[1280 + k], og = 1.0f / (1.0f + exp_(-gates[1920 + k]));
    gg = 2.0f / (1.0f + exp_(-2.0f * gg)) - 1.0f;   // tanh
    float cc = fg * c[k] + ig * gg;
    c[k] = cc;
    float h = og * (2.0f / (1.0f + exp_(-2.0f * cc)) - 1.0f);
    h_a[k] = h;
    h_b[k] = h;
}

// joint input: relu(encp[t] + dec), 640 values
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void joint_in(global const float* encp, global const float* dec, global float* out, uint t) {
    uint k = GID * 64 + LID;
    if (k < 640) out[k] = __builtin_fmaxf(encp[(ulong)t * 640 + k] + dec[k], 0.0f);
}

// res[0] = argmax of logits[0 .. vocab), res[1] = argmax of the n_dur duration logits after them. 256 threads.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void argmax_tdt(global const float* logits, global uint* res, uint vocab, uint n_dur) {
    local float bv[256];
    local uint bi[256];
    uint t = LID;
    float best = -3.0e38f;
    uint idx = 0;
    for (uint i = t; i < vocab; i += 256)
        if (logits[i] > best) {
            best = logits[i];
            idx = i;
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
        res[0] = bi[0];
        uint dbest = 0;
        for (uint k = 1; k < n_dur; k++)
            if (logits[vocab + k] > logits[vocab + dbest]) dbest = k;
        res[1] = dbest;
    }
}
