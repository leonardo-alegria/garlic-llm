// ---- Reading the prompt in batches (chat/engine.c prefill) -----------------------------------------------------
// Include in qwen35.cl after common.h (DK, DV defined). Up to 32 prompt tokens go through each layer together:
// the matrices are read and unpacked once for all of them (matrix x matrix instead of 32 matrix x vector), the
// Gated DeltaNet state stays in registers across the tokens (one load and one store per batch instead of per
// token), and the attention runs token by token over the cache (attn.h / attn_tq.h, one dispatch each).
// Every row kernel computes exactly what its one-token version does (same expressions, same order), so the
// batch gives the same numbers up to the summation order of the matrix products.
// Rows: X [n][cols] row-major, consecutive tokens.

#define MM_ROWS 16      // tokens per group in the matrix products (acc[16] + 32 unpacked weights in registers)

// Y[v][o] (+)= sum_c W[o][c] X[v][c] for v < n; W q4i (shaders/common.h), ldx = cols, ldy = rows. Group = 4 waves
// = one tile of 64 output rows x MM_ROWS tokens; lane = output row; wave w takes blocks w, w + 4, ...; x comes
// through the scalar unit (the same for the whole wave). Groups of the same tile are consecutive (L2 reuse).
// The weights are unpacked once per group: w = q d - 8 d, the same value as (q - 8) d.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mm_q4i_rows(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict X,
                 global float* restrict Y, uint cols, uint rows, uint n, uint accumulate) {
    local float part[4][MM_ROWS][64];
    uint l = LID & 63, nb = cols / 32, nch = (n + MM_ROWS - 1) / MM_ROWS;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tile = GID / nch, r0 = (GID % nch) * MM_ROWS;
    uint nv = n - r0 < MM_ROWS ? n - r0 : MM_ROWS;
    ulong tb = (ulong)tile * nb * 64;
    global const uint4v* wp = w + tb + l;
    global const half_t* sp = sc + tb + l;
    global const float* x0 = X + (ulong)r0 * cols;
    float acc[MM_ROWS];
    for (uint v = 0; v < MM_ROWS; v++) acc[v] = 0.0f;
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
#pragma unroll
        for (uint v = 0; v < MM_ROWS; v++) {
            if (v < nv) {
                global const float* xb = x0 + (ulong)v * cols + b * 32;
                float s = 0.0f;
                for (uint j = 0; j < 32; j++) s += wf[j] * xb[j];
                acc[v] += s;
            }
        }
    }
    for (uint v = 0; v < MM_ROWS; v++) part[wv][v][l] = acc[v];
    BARRIER();
    // all four waves write: wave w the tokens w, w + 4, ...
    for (uint v = wv; v < nv; v += 4) {
        float r = part[0][v][l] + part[1][v][l] + part[2][v][l] + part[3][v][l];
        global float* y = Y + (ulong)(r0 + v) * rows + tile * 64 + l;
        *y = accumulate ? *y + r : r;
    }
}

// The same for q3i weights (shaders/common.h: 12 bytes per block, w = (q - 4) d with q in 0..7).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mm_q3i_rows(global const uint* restrict w, global const half_t* restrict sc, global const float* restrict X,
                 global float* restrict Y, uint cols, uint rows, uint n, uint accumulate) {
    local float part[4][MM_ROWS][64];
    uint l = LID & 63, nb = cols / 32, nch = (n + MM_ROWS - 1) / MM_ROWS;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tile = GID / nch, r0 = (GID % nch) * MM_ROWS;
    uint nv = n - r0 < MM_ROWS ? n - r0 : MM_ROWS;
    ulong tb = (ulong)tile * nb * 64;
    global const uint* wp = w + (tb + l) * 3;
    global const half_t* sp = sc + tb + l;
    global const float* x0 = X + (ulong)r0 * cols;
    float acc[MM_ROWS];
    for (uint v = 0; v < MM_ROWS; v++) acc[v] = 0.0f;
    for (uint b = wv; b < nb; b += 4) {
        global const uint* qp = wp + (ulong)b * 64 * 3;
        uint d0 = qp[0], d1 = qp[1], d2 = qp[2];
        float d = (float)sp[(ulong)b * 64], m4 = -4.0f * d;
        // group g (x[4g .. 4g + 3]): 2 low bits from d0 (g < 4) or d1, the high bit from d2
        uint g8[8] = {(d0 & 0x03030303u) | ((d2 << 2) & 0x04040404u), ((d0 >> 2) & 0x03030303u) | ((d2 << 1) & 0x04040404u),
                      ((d0 >> 4) & 0x03030303u) | (d2 & 0x04040404u), ((d0 >> 6) & 0x03030303u) | ((d2 >> 1) & 0x04040404u),
                      (d1 & 0x03030303u) | ((d2 >> 2) & 0x04040404u), ((d1 >> 2) & 0x03030303u) | ((d2 >> 3) & 0x04040404u),
                      ((d1 >> 4) & 0x03030303u) | ((d2 >> 4) & 0x04040404u), ((d1 >> 6) & 0x03030303u) | ((d2 >> 5) & 0x04040404u)};
        float wf[32];
        for (uint g = 0; g < 8; g++) {
            uint qq = g8[g];
            __asm__("" : "+v"(qq));
            wf[4 * g] = (float)(qq & 0xFFu) * d + m4;
            wf[4 * g + 1] = (float)((qq >> 8) & 0xFFu) * d + m4;
            wf[4 * g + 2] = (float)((qq >> 16) & 0xFFu) * d + m4;
            wf[4 * g + 3] = (float)(qq >> 24) * d + m4;
        }
#pragma unroll
        for (uint v = 0; v < MM_ROWS; v++) {
            if (v < nv) {
                global const float* xb = x0 + (ulong)v * cols + b * 32;
                float s = 0.0f;
                for (uint j = 0; j < 32; j++) s += wf[j] * xb[j];
                acc[v] += s;
            }
        }
    }
    for (uint v = 0; v < MM_ROWS; v++) part[wv][v][l] = acc[v];
    BARRIER();
    for (uint v = wv; v < nv; v += 4) {
        float r = part[0][v][l] + part[1][v][l] + part[2][v][l] + part[3][v][l];
        global float* y = Y + (ulong)(r0 + v) * rows + tile * 64 + l;
        *y = accumulate ? *y + r : r;
    }
}

// ---- other shapes of the same products, timed against each other when the engine starts (engine.c mm_tune) ----
// mm_q4i_r<R> / mm_q3i_r<R>: as above with R tokens per group (more tokens per unpacked weight, more registers);
// mm_q4i_w<R> / mm_q3i_w<R>: the weights unpacked a word (8 or 4 weights) at a time, so that only those stay in
// registers next to the R sums. All of them reduce the 4 waves' sums through 4 KiB of LDS, 4 tokens at a time.
// The sums run in another order than mm_q4i_rows (one accumulator per token over the whole row).
#define MM_STORE(R)                                                                               \
    _Pragma("unroll") for (uint v0 = 0; v0 < R; v0 += 4) {                                       \
        _Pragma("unroll") for (uint u = 0; u < 4; u++) red[(wv * 4 + u) * 64 + l] = acc[v0 + u]; \
        BARRIER();                                                                                \
        uint v = v0 + wv;                                                                         \
        if (v < nv) {                                                                             \
            float r = red[wv * 64 + l] + red[(4 + wv) * 64 + l] + red[(8 + wv) * 64 + l] +       \
                      red[(12 + wv) * 64 + l];                                                    \
            global float* y = Y + (ulong)(r0 + v) * rows + tile * 64 + l;                         \
            *y = accumulate ? *y + r : r;                                                         \
        }                                                                                         \
        BARRIER();                                                                                \
    }

#define MM_HEAD(R)                                                                       \
    local float red[16 * 64];                                                            \
    uint l = LID & 63, nb = cols / 32, nch = (n + (R) - 1) / (R);                        \
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);                                  \
    uint tile = GID / nch, r0 = (GID % nch) * (R);                                       \
    uint nv = n - r0 < (R) ? n - r0 : (R);                                               \
    ulong tb = (ulong)tile * nb * 64;                                                    \
    global const half_t* sp = sc + tb + l;                                               \
    global const float* x0 = X + (ulong)r0 * cols;                                       \
    float acc[R];                                                                        \
    _Pragma("unroll") for (uint v = 0; v < (R); v++) acc[v] = 0.0f;

// 32 weights of a q4i block -> wf (as mm_q4i_rows)
#define MM_Q4_UNPACK32(q, d)                                                     \
    float m8 = -8.0f * (d);                                                      \
    uint ws[4] = {q.x, q.y, q.z, q.w};                                           \
    float wf[32];                                                                \
    _Pragma("unroll") for (uint k = 0; k < 4; k++) {                             \
        uint lo = ws[k] & 0x0F0F0F0Fu, hi = (ws[k] >> 4) & 0x0F0F0F0Fu;         \
        __asm__("" : "+v"(lo), "+v"(hi));                                        \
        wf[4 * k] = (float)(lo & 0xFFu) * (d) + m8;                              \
        wf[4 * k + 1] = (float)((lo >> 8) & 0xFFu) * (d) + m8;                   \
        wf[4 * k + 2] = (float)((lo >> 16) & 0xFFu) * (d) + m8;                  \
        wf[4 * k + 3] = (float)(lo >> 24) * (d) + m8;                            \
        wf[16 + 4 * k] = (float)(hi & 0xFFu) * (d) + m8;                         \
        wf[16 + 4 * k + 1] = (float)((hi >> 8) & 0xFFu) * (d) + m8;              \
        wf[16 + 4 * k + 2] = (float)((hi >> 16) & 0xFFu) * (d) + m8;             \
        wf[16 + 4 * k + 3] = (float)(hi >> 24) * (d) + m8;                       \
    }

// the 8 groups of 4 weights of a q3i block (mm_q3i_rows)
#define MM_Q3_GROUPS(d0, d1, d2)                                                                                     \
    uint g8[8] = {(d0 & 0x03030303u) | ((d2 << 2) & 0x04040404u), ((d0 >> 2) & 0x03030303u) | ((d2 << 1) & 0x04040404u), \
                  ((d0 >> 4) & 0x03030303u) | (d2 & 0x04040404u), ((d0 >> 6) & 0x03030303u) | ((d2 >> 1) & 0x04040404u), \
                  (d1 & 0x03030303u) | ((d2 >> 2) & 0x04040404u), ((d1 >> 2) & 0x03030303u) | ((d2 >> 3) & 0x04040404u), \
                  ((d1 >> 4) & 0x03030303u) | ((d2 >> 4) & 0x04040404u), ((d1 >> 6) & 0x03030303u) | ((d2 >> 5) & 0x04040404u)};

#define MM_Q4_R(NAME, R)                                                                                          \
    kernel __attribute__((reqd_work_group_size(256, 1, 1)))                                                      \
    void NAME(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict X,  \
              global float* restrict Y, uint cols, uint rows, uint n, uint accumulate) {                         \
        MM_HEAD(R)                                                                                                \
        global const uint4v* wp = w + tb + l;                                                                     \
        for (uint b = wv; b < nb; b += 4) {                                                                       \
            uint4v q = wp[(ulong)b * 64];                                                                         \
            float d = (float)sp[(ulong)b * 64];                                                                   \
            MM_Q4_UNPACK32(q, d)                                                                                  \
            _Pragma("unroll") for (uint v = 0; v < (R); v++) {                                                   \
                if (v < nv) {                                                                                     \
                    global const float* xb = x0 + (ulong)v * cols + b * 32;                                       \
                    float s = 0.0f;                                                                               \
                    _Pragma("unroll") for (uint j = 0; j < 32; j++) s += wf[j] * xb[j];                          \
                    acc[v] += s;                                                                                  \
                }                                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
        MM_STORE(R)                                                                                               \
    }

#define MM_Q4_W(NAME, R)                                                                                          \
    kernel __attribute__((reqd_work_group_size(256, 1, 1)))                                                      \
    void NAME(global const uint4v* restrict w, global const half_t* restrict sc, global const float* restrict X,  \
              global float* restrict Y, uint cols, uint rows, uint n, uint accumulate) {                         \
        MM_HEAD(R)                                                                                                \
        global const uint4v* wp = w + tb + l;                                                                     \
        for (uint b = wv; b < nb; b += 4) {                                                                       \
            uint4v q = wp[(ulong)b * 64];                                                                         \
            float d = (float)sp[(ulong)b * 64], m8 = -8.0f * d;                                                   \
            uint ws[4] = {q.x, q.y, q.z, q.w};                                                                    \
            _Pragma("unroll") for (uint k = 0; k < 4; k++) {                                                     \
                uint lo = ws[k] & 0x0F0F0F0Fu, hi = (ws[k] >> 4) & 0x0F0F0F0Fu;                                  \
                __asm__("" : "+v"(lo), "+v"(hi));                                                                 \
                float a0 = (float)(lo & 0xFFu) * d + m8, a1 = (float)((lo >> 8) & 0xFFu) * d + m8;                \
                float a2 = (float)((lo >> 16) & 0xFFu) * d + m8, a3 = (float)(lo >> 24) * d + m8;                 \
                float c0 = (float)(hi & 0xFFu) * d + m8, c1 = (float)((hi >> 8) & 0xFFu) * d + m8;                \
                float c2 = (float)((hi >> 16) & 0xFFu) * d + m8, c3 = (float)(hi >> 24) * d + m8;                 \
                _Pragma("unroll") for (uint v = 0; v < (R); v++) {                                               \
                    if (v < nv) {                                                                                 \
                        global const float* xr = x0 + (ulong)v * cols + b * 32 + 4 * k;                           \
                        acc[v] += a0 * xr[0] + a1 * xr[1] + a2 * xr[2] + a3 * xr[3] + c0 * xr[16] + c1 * xr[17] + \
                                  c2 * xr[18] + c3 * xr[19];                                                      \
                    }                                                                                             \
                }                                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
        MM_STORE(R)                                                                                               \
    }

#define MM_Q3_R(NAME, R)                                                                                          \
    kernel __attribute__((reqd_work_group_size(256, 1, 1)))                                                      \
    void NAME(global const uint* restrict w, global const half_t* restrict sc, global const float* restrict X,    \
              global float* restrict Y, uint cols, uint rows, uint n, uint accumulate) {                         \
        MM_HEAD(R)                                                                                                \
        global const uint* wp = w + (tb + l) * 3;                                                                 \
        for (uint b = wv; b < nb; b += 4) {                                                                       \
            global const uint* qp = wp + (ulong)b * 64 * 3;                                                       \
            uint d0 = qp[0], d1 = qp[1], d2 = qp[2];                                                              \
            float d = (float)sp[(ulong)b * 64], m4 = -4.0f * d;                                                   \
            MM_Q3_GROUPS(d0, d1, d2)                                                                              \
            float wf[32];                                                                                         \
            _Pragma("unroll") for (uint g = 0; g < 8; g++) {                                                     \
                uint qq = g8[g];                                                                                  \
                __asm__("" : "+v"(qq));                                                                           \
                wf[4 * g] = (float)(qq & 0xFFu) * d + m4;                                                         \
                wf[4 * g + 1] = (float)((qq >> 8) & 0xFFu) * d + m4;                                              \
                wf[4 * g + 2] = (float)((qq >> 16) & 0xFFu) * d + m4;                                             \
                wf[4 * g + 3] = (float)(qq >> 24) * d + m4;                                                       \
            }                                                                                                     \
            _Pragma("unroll") for (uint v = 0; v < (R); v++) {                                                   \
                if (v < nv) {                                                                                     \
                    global const float* xb = x0 + (ulong)v * cols + b * 32;                                       \
                    float s = 0.0f;                                                                               \
                    _Pragma("unroll") for (uint j = 0; j < 32; j++) s += wf[j] * xb[j];                          \
                    acc[v] += s;                                                                                  \
                }                                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
        MM_STORE(R)                                                                                               \
    }

#define MM_Q3_W(NAME, R)                                                                                          \
    kernel __attribute__((reqd_work_group_size(256, 1, 1)))                                                      \
    void NAME(global const uint* restrict w, global const half_t* restrict sc, global const float* restrict X,    \
              global float* restrict Y, uint cols, uint rows, uint n, uint accumulate) {                         \
        MM_HEAD(R)                                                                                                \
        global const uint* wp = w + (tb + l) * 3;                                                                 \
        for (uint b = wv; b < nb; b += 4) {                                                                       \
            global const uint* qp = wp + (ulong)b * 64 * 3;                                                       \
            uint d0 = qp[0], d1 = qp[1], d2 = qp[2];                                                              \
            float d = (float)sp[(ulong)b * 64], m4 = -4.0f * d;                                                   \
            MM_Q3_GROUPS(d0, d1, d2)                                                                              \
            _Pragma("unroll") for (uint g = 0; g < 8; g++) {                                                     \
                uint qq = g8[g];                                                                                  \
                __asm__("" : "+v"(qq));                                                                           \
                float a0 = (float)(qq & 0xFFu) * d + m4, a1 = (float)((qq >> 8) & 0xFFu) * d + m4;                \
                float a2 = (float)((qq >> 16) & 0xFFu) * d + m4, a3 = (float)(qq >> 24) * d + m4;                 \
                _Pragma("unroll") for (uint v = 0; v < (R); v++) {                                               \
                    if (v < nv) {                                                                                 \
                        global const float* xr = x0 + (ulong)v * cols + b * 32 + 4 * g;                           \
                        acc[v] += a0 * xr[0] + a1 * xr[1] + a2 * xr[2] + a3 * xr[3];                              \
                    }                                                                                             \
                }                                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
        MM_STORE(R)                                                                                               \
    }

MM_Q4_R(mm_q4i_r8, 8)
MM_Q4_R(mm_q4i_r32, 32)
MM_Q4_W(mm_q4i_w16, 16)
MM_Q4_W(mm_q4i_w32, 32)
MM_Q3_R(mm_q3i_r8, 8)
MM_Q3_R(mm_q3i_r32, 32)
MM_Q3_W(mm_q3i_w16, 16)
MM_Q3_W(mm_q3i_w32, 32)

// rmsnorm (common.h) of row GID: out[r] = x[r] / rms(x[r]) * w.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void rmsnorm_rows(global const float* x, global const float* w, global float* out, uint n, float eps) {
    local float red[256];
    x += (ulong)GID * n;
    out += (ulong)GID * n;
    float ss = 0.0f;
    for (uint i = LID; i < n; i += 256) ss += x[i] * x[i];
    float r = rsq(group_sum(red, ss) / (float)n + eps);
    for (uint i = LID; i < n; i += 256) out[i] = x[i] * r * w[i];
}

// silu_mul (common.h) over rows: h[v][j] = silu(gu[v][j]) * gu[v][n + j], gu rows of 2n.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void silu_mul_rows(global const float* gu, global float* h, uint n, uint rows) {
    uint i = GID * 256 + LID;
    if (i >= n * rows) return;
    uint v = i / n, j = i - v * n;
    global const float* r = gu + (ulong)v * 2 * n;
    float g = r[j];
    h[i] = g / (1.0f + exp_(-g)) * r[n + j];
}

// mul_sigmoid (qwen35.cl) over rows: x[v][j] *= sigmoid(g[v * ldg + j]), x rows of n.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mul_sigmoid_rows(global float* x, global const float* g, uint n, uint ldg, uint rows) {
    uint i = GID * 256 + LID;
    if (i >= n * rows) return;
    uint v = i / n, j = i - v * n;
    x[i] = x[i] / (1.0f + exp_(-g[(ulong)v * ldg + j]));
}

// conv_silu (qwen35.cl) for `rows` consecutive tokens: channel c keeps its 3 inputs of state in registers.
// x rows of ldx floats (the first n channels are q, k, v).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void conv_silu_rows(global float* x, global const float* w, global float* state, uint n, uint ldx, uint rows) {
    uint c = GID * 256 + LID;
    if (c >= n) return;
    global float* s = state + 3 * (ulong)c;
    global const float* wc = w + 4 * (ulong)c;
    float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float s0 = s[0], s1 = s[1], s2 = s[2];
    for (uint t = 0; t < rows; t++) {
        global float* xt = x + (ulong)t * ldx + c;
        float v = *xt;
        float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * v;
        s0 = s1;
        s1 = s2;
        s2 = v;
        *xt = y / (1.0f + exp_(-y));
    }
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
}

// delta_step (qwen35.cl) for `rows` consecutive tokens: the state of value head GID is read once, updated token by
// token in registers (the same operations in the same order as delta_step) and written once. x rows of ldx floats,
// out rows of n_vh * DV.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void delta_rows(global const float* x, global const float* A, global const float* dt_bias, global float* S_all,
                global const float* gw, global float* out, uint n_kh, uint n_vh, float eps, uint ldx, uint rows) {
    local float qs[DK], ks[DK];
    local float red[256];
    uint h = GID, t = LID, kh = h / (n_vh / n_kh);
    uint kd = n_kh * DK, vd = n_vh * DV;
    uint vi = t & 127, k0 = (t >> 7) * 64;
    global float* S = S_all + (ulong)h * DK * DV + (ulong)k0 * DV + vi;
    float st[64];
#pragma unroll
    for (uint i = 0; i < 64; i++) st[i] = S[(ulong)i * DV];
    float Ah = A[h], dth = dt_bias[h], gwt = t < DV ? gw[t] : 0.0f;
    for (uint r = 0; r < rows; r++) {
        global const float* xr = x + (ulong)r * ldx;
        global const float* vin = xr + 2 * kd + h * DV;
        float b = xr[2 * kd + 2 * vd + h], a = xr[2 * kd + 2 * vd + n_vh + h];

        float val = t < DK ? xr[kh * DK + t] : xr[kd + kh * DK + (t - DK)];
        red[t] = val * val;
        BARRIER();
        for (uint s = 64; s > 0; s >>= 1) {
            if ((t & 127) < s) red[t] += red[t + s];
            BARRIER();
        }
        float nrm = rsq(red[t & 128] + 1e-6f);
        if (t < DK) qs[t] = val * nrm * 0.08838834764831845f;     // 1 / sqrt(128)
        else ks[t - DK] = val * nrm;
        BARRIER();

        float beta = 1.0f / (1.0f + exp_(-b));
        float sp = a + dth;
        sp = sp > 20.0f ? sp : __builtin_log2f(1.0f + exp_(sp)) * 0.69314718f;   // softplus
        float decay = exp_(Ah * sp);

        float kv = 0.0f;
#pragma unroll
        for (uint i = 0; i < 64; i++) {
            st[i] = st[i] * decay;
            kv += st[i] * ks[k0 + i];
        }
        red[t] = kv;
        BARRIER();
        float delta = (vin[vi] - (red[vi] + red[vi + 128])) * beta;
        BARRIER();
        float o = 0.0f;
#pragma unroll
        for (uint i = 0; i < 64; i++) {
            st[i] += ks[k0 + i] * delta;
            o += st[i] * qs[k0 + i];
        }
        red[t] = o;
        BARRIER();
        o = red[vi] + red[vi + 128];
        BARRIER();

        red[t] = t < DV ? o * o : 0.0f;
        BARRIER();
        for (uint s = 64; s > 0; s >>= 1) {
            if (t < s) red[t] += red[t + s];
            BARRIER();
        }
        if (t < DV) {
            float rr = rsq(red[0] / (float)DV + eps), zz = xr[2 * kd + vd + h * DV + t];
            out[(ulong)r * vd + h * DV + t] = o * rr * gwt * (zz / (1.0f + exp_(-zz)));
        }
        BARRIER();   // (red, qs and ks are rewritten by the next token)
    }
#pragma unroll
    for (uint i = 0; i < 64; i++) S[(ulong)i * DV] = st[i];
}

