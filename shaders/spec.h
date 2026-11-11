// ---- SWAR products, speculative decoding with the MTP head ----------------------------------------------------
// Included by chat_lab.cl (the 9B's kernels) after qwen35.cl (common.h, prefill.h).
//
// 1. SWAR int4 x int8 products on "q4s" weights (pc/convert_9b_q4s.py): two 4-bit weights
//    of two rows 16 bits apart in one word times one int8 activation = two products per v_mad_i32_i24.
//    Tiles of 128 rows; lane l of a tile owns rows A = l and B = 64 + l. Per tile and block of 32 columns: two runs of
//    64 uint4; lane l's word k = columns 4k..4k+3: nibble i = row A, nibble 4 + i = row B. Scales: one dword per tile,
//    block and lane = fp16 d_A | fp16 d_B << 16.
//    Activations: quant_x8 per block of 32: dx = max|x| / 127, x8 = rint(x / dx), init = -8 * sum(x8) * 65537.
// 2. Speculative decoding: the draft's rows (embedding table of the first 32K lm_head rows, q4row), the draft's
//    sampling with its stop rule, the exact acceptance (min(1, p/q), else the residual max(0, p - q)) and the
//    Gated DeltaNet state "commit" for the tokens accepted (the verification does not store the states).

typedef int int4v __attribute__((ext_vector_type(4)));
typedef _Float16 half2v __attribute__((ext_vector_type(2)));

// x [n][cols] float -> x8 [n][cols] int32, dx [n * nb], init [n * nb]; one thread per block of 32 (nblocks = n * nb).
// gate (may be 0): when *gate != 0 nothing is done (a draft step after the draft stopped).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void quant_x8(global const float* restrict x, global int* restrict x8, global float* restrict dx,
              global int* restrict init, uint nblocks, global const uint* gate) {
    uint i = GID * 256 + LID;
    if (i >= nblocks || (gate && *gate)) return;
    global const float4v* xv = (global const float4v*)(x + (ulong)i * 32);
    float4v v[8];
    float m = 0.0f;
    for (uint k = 0; k < 8; k++) {
        v[k] = xv[k];
        m = __builtin_fmaxf(m, __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(v[k].x), __builtin_fabsf(v[k].y)),
                                               __builtin_fmaxf(__builtin_fabsf(v[k].z), __builtin_fabsf(v[k].w))));
    }
    float d = m / 127.0f, inv = m > 0.0f ? 127.0f / m : 0.0f;
    global int4v* o = (global int4v*)(x8 + (ulong)i * 32);
    int s = 0;
    for (uint k = 0; k < 8; k++) {
        int4v q = {(int)__builtin_rintf(v[k].x * inv), (int)__builtin_rintf(v[k].y * inv),
                   (int)__builtin_rintf(v[k].z * inv), (int)__builtin_rintf(v[k].w * inv)};
        o[k] = q;
        s += q.x + q.y + q.z + q.w;
    }
    dx[i] = d;
    init[i] = -8 * s * 65537;
}

// acc += p * x (signed 24-bit operands, low 32 bits of the product); x from an SGPR, read directly by the VALU.
#define SWAR_MAD(acc, p, xs) __asm__("v_mad_i32_i24 %0, %1, %2, %0" : "+v"(acc) : "v"(p), "s"(xs))

// Y[t][rows] (+)= W x_t for t < N tokens with one read and one unpacking of the weights. Group = WAVES waves = one
// tile of 128 rows; wave w takes blocks w, w + WAVES, ... SPLIT > 1 (split-K): group g = tile g / SPLIT, part
// s = g % SPLIT takes blocks s * WAVES + w, then every WAVES * SPLIT, and writes its partial sums to
// y + (s * N + t) * rows (sum_split adds them up, and does the accumulation). acc = 1: y += (SPLIT 1 only).
#define SWAR_MVS(NAME, N, WAVES, SPLIT)                                                                           \
    kernel __attribute__((reqd_work_group_size(64 * (WAVES), 1, 1)))                                             \
    void NAME(global const uint4v* restrict w, global const half2v* restrict sc, global const int* restrict x8,   \
              global const float* restrict dx, global const int* restrict init, global float* restrict y,         \
              uint cols, uint rows, uint acc, global const uint* gate) {                                          \
        local float part[(WAVES) * 2 * (N) * 64];                                                                 \
        if (gate && *gate) return;                                                                                \
        uint l = LID & 63, nb = cols / 32;                                                                        \
        uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);                                                       \
        uint tile = GID / (SPLIT), sp_ = GID % (SPLIT);                                                          \
        global const uint4v* wp = w + (ulong)tile * nb * 128 + l;                                                  \
        global const half2v* sp = sc + (ulong)tile * nb * 64 + l;                                                  \
        float accA[N], accB[N];                                                                                   \
        _Pragma("unroll") for (uint t = 0; t < (N); t++) accA[t] = accB[t] = 0.0f;                               \
        for (uint b = sp_ * (WAVES) + wv; b < nb; b += (WAVES) * (SPLIT)) {                                       \
            uint4v q0 = wp[(ulong)b * 128], q1 = wp[(ulong)b * 128 + 64];                                         \
            half2v s2 = sp[(ulong)b * 64];                                                                        \
            float dA = (float)s2.x, dB = (float)s2.y;                                                             \
            uint ws[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};                                        \
            int p[32];                                                                                            \
            _Pragma("unroll") for (uint k = 0; k < 8; k++) {                                                      \
                p[4 * k] = (int)(ws[k] & 0x000F000Fu);                                                            \
                p[4 * k + 1] = (int)((ws[k] >> 4) & 0x000F000Fu);                                                 \
                p[4 * k + 2] = (int)((ws[k] >> 8) & 0x000F000Fu);                                                 \
                p[4 * k + 3] = (int)((ws[k] >> 12) & 0x000F000Fu);                                                \
            }                                                                                                     \
            _Pragma("unroll") for (uint t = 0; t < (N); t++) {                                                    \
                global const int* xb = x8 + (ulong)t * cols + b * 32;                                             \
                int a = init[t * nb + b];                                                                         \
                _Pragma("unroll") for (uint j = 0; j < 32; j++) SWAR_MAD(a, p[j], xb[j]);                        \
                int lo = (a << 16) >> 16;                                                                         \
                int hi = a - lo; /* 65536 * sum for row B; the 1/65536 goes in at the end */                      \
                float d = dx[t * nb + b];                                                                         \
                accA[t] += (float)lo * (dA * d);                                                                  \
                accB[t] += (float)hi * (dB * d);                                                                  \
            }                                                                                                     \
        }                                                                                                         \
        _Pragma("unroll") for (uint t = 0; t < (N); t++) {                                                        \
            part[(wv * 2 * (N) + 2 * t) * 64 + l] = accA[t];                                                      \
            part[(wv * 2 * (N) + 2 * t + 1) * 64 + l] = accB[t];                                                  \
        }                                                                                                         \
        BARRIER();                                                                                                \
        if (wv == 0) {                                                                                            \
            _Pragma("unroll") for (uint t = 0; t < (N); t++) {                                                    \
                float ra = 0.0f, rb = 0.0f;                                                                       \
                for (uint k = 0; k < (WAVES); k++) {                                                              \
                    ra += part[(k * 2 * (N) + 2 * t) * 64 + l];                                                   \
                    rb += part[(k * 2 * (N) + 2 * t + 1) * 64 + l];                                               \
                }                                                                                                 \
                rb *= (1.0f / 65536.0f);                                                                          \
                global float* yt = y + ((ulong)sp_ * (N) + t) * rows + tile * 128 + l;                           \
                if ((SPLIT) == 1 && acc) {                                                                        \
                    yt[0] += ra;                                                                                  \
                    yt[64] += rb;                                                                                 \
                } else {                                                                                          \
                    yt[0] = ra;                                                                                   \
                    yt[64] = rb;                                                                                  \
                }                                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
    }

#define SWAR_SET(N)                                   \
    SWAR_MVS(swar_mv##N##_w4, N, 4, 1)                \
    SWAR_MVS(swar_mv##N##_w8, N, 8, 1)                \
    SWAR_MVS(swar_mv##N##_w16, N, 16, 1)              \
    SWAR_MVS(swar_mv##N##_w8_s2, N, 8, 2)             \
    SWAR_MVS(swar_mv##N##_w8_s4, N, 8, 4)
SWAR_SET(1)
SWAR_SET(2)
SWAR_SET(3)
SWAR_SET(4)

// The same with 3-bit weights ("q3s", pc/convert_9b_q4s.py to_q3s: the app's q3i values and scales, laid out again):
// q in 0..7, w = (q - 4) d. Tiles of 128 rows, lane l = rows A = l and B = 64 + l; per tile and block of 32 columns,
// a run of 64 uint4 (words 0..3) then a run of 64 uint2 (words 4, 5): 12 bytes per row and block, as q3i. Word k
// holds columns 5k..5k+4 in 3-bit fields (row A at bit 3i, row B at 16 + 3i): (word >> 3i) & 0x00070007 = (A, B).
// Columns 30 and 31 take the top bits of each half: column 30 = bits 15 / 31 of words 0..2 (bit j of q in word j),
// column 31 the same in words 3..5. Fields hold |sum (q - 4) x8| <= 32 * 4 * 127; the start value is init / 2.
// (No 16-wave variant: it spills registers.)
typedef uint uint2v __attribute__((ext_vector_type(2)));

#define SWAR3_MVS(NAME, N, WAVES, SPLIT)                                                                          \
    kernel __attribute__((reqd_work_group_size(64 * (WAVES), 1, 1)))                                             \
    void NAME(global const uint* restrict w, global const half2v* restrict sc, global const int* restrict x8,     \
              global const float* restrict dx, global const int* restrict init, global float* restrict y,         \
              uint cols, uint rows, uint acc, global const uint* gate) {                                          \
        local float part[(WAVES) * 2 * (N) * 64];                                                                 \
        if (gate && *gate) return;                                                                                \
        uint l = LID & 63, nb = cols / 32;                                                                        \
        uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);                                                       \
        uint tile = GID / (SPLIT), sp_ = GID % (SPLIT);                                                          \
        global const uint* wt = w + (ulong)tile * nb * 384;                                                       \
        global const half2v* sp = sc + (ulong)tile * nb * 64 + l;                                                  \
        float accA[N], accB[N];                                                                                   \
        _Pragma("unroll") for (uint t = 0; t < (N); t++) accA[t] = accB[t] = 0.0f;                               \
        for (uint b = sp_ * (WAVES) + wv; b < nb; b += (WAVES) * (SPLIT)) {                                       \
            global const uint* wb = wt + (ulong)b * 384;                                                          \
            uint4v q0 = ((global const uint4v*)wb)[l];                                                            \
            uint2v q1 = ((global const uint2v*)(wb + 256))[l];                                                    \
            half2v s2 = sp[(ulong)b * 64];                                                                        \
            float dA = (float)s2.x, dB = (float)s2.y;                                                             \
            uint ws[6] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y};                                                    \
            int p[32];                                                                                            \
            _Pragma("unroll") for (uint k = 0; k < 6; k++) {                                                      \
                _Pragma("unroll") for (uint i = 0; i < 5; i++) p[5 * k + i] = (int)((ws[k] >> (3 * i)) & 0x00070007u); \
            }                                                                                                     \
            p[30] = (int)(((ws[0] >> 15) & 0x00010001u) | ((ws[1] >> 14) & 0x00020002u) |                        \
                          ((ws[2] >> 13) & 0x00040004u));                                                         \
            p[31] = (int)(((ws[3] >> 15) & 0x00010001u) | ((ws[4] >> 14) & 0x00020002u) |                        \
                          ((ws[5] >> 13) & 0x00040004u));                                                         \
            _Pragma("unroll") for (uint t = 0; t < (N); t++) {                                                    \
                global const int* xb = x8 + (ulong)t * cols + b * 32;                                             \
                int a = init[t * nb + b] >> 1;                                                                    \
                _Pragma("unroll") for (uint j = 0; j < 32; j++) SWAR_MAD(a, p[j], xb[j]);                        \
                int lo = (a << 16) >> 16;                                                                         \
                int hi = a - lo;                                                                                  \
                float d = dx[t * nb + b];                                                                         \
                accA[t] += (float)lo * (dA * d);                                                                  \
                accB[t] += (float)hi * (dB * d);                                                                  \
            }                                                                                                     \
        }                                                                                                         \
        _Pragma("unroll") for (uint t = 0; t < (N); t++) {                                                        \
            part[(wv * 2 * (N) + 2 * t) * 64 + l] = accA[t];                                                      \
            part[(wv * 2 * (N) + 2 * t + 1) * 64 + l] = accB[t];                                                  \
        }                                                                                                         \
        BARRIER();                                                                                                \
        if (wv == 0) {                                                                                            \
            _Pragma("unroll") for (uint t = 0; t < (N); t++) {                                                    \
                float ra = 0.0f, rb = 0.0f;                                                                       \
                for (uint k = 0; k < (WAVES); k++) {                                                              \
                    ra += part[(k * 2 * (N) + 2 * t) * 64 + l];                                                   \
                    rb += part[(k * 2 * (N) + 2 * t + 1) * 64 + l];                                               \
                }                                                                                                 \
                rb *= (1.0f / 65536.0f);                                                                          \
                global float* yt = y + ((ulong)sp_ * (N) + t) * rows + tile * 128 + l;                           \
                if ((SPLIT) == 1 && acc) {                                                                        \
                    yt[0] += ra;                                                                                  \
                    yt[64] += rb;                                                                                 \
                } else {                                                                                          \
                    yt[0] = ra;                                                                                   \
                    yt[64] = rb;                                                                                  \
                }                                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
    }

#define SWAR3_SET(N)                                  \
    SWAR3_MVS(swar3_mv##N##_w4, N, 4, 1)              \
    SWAR3_MVS(swar3_mv##N##_w8, N, 8, 1)              \
    SWAR3_MVS(swar3_mv##N##_w8_s2, N, 8, 2)           \
    SWAR3_MVS(swar3_mv##N##_w8_s4, N, 8, 4)
SWAR3_SET(1)
SWAR3_SET(2)
SWAR3_SET(3)
SWAR3_SET(4)

// y[i] (+)= sum over s < split of yp[s * count + i] (the partial sums of SWAR_MVS)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void sum_split(global const float* restrict yp, global float* restrict y, uint count, uint split, uint acc,
               global const uint* gate) {
    uint i = GID * 256 + LID;
    if (i >= count || (gate && *gate)) return;
    float s = yp[i];
    for (uint k = 1; k < split; k++) s += yp[(ulong)k * count + i];
    y[i] = acc ? y[i] + s : s;
}

// ---- speculative decoding --------------------------------------------------------------------------------------
// The state block st (uint words), shared by the kernels below and engine.c (SPEC_*):
//   [0] draft stop flag   [1] tokens proposed (K)   [2..5] the draft's rows (lm_head rows, < 32768)
//   [8] accepted drafts j  [9] the next token's row (sampled after them)  [10] tokens to commit (j + 1)
//   [11] the position where a draft was rejected (or K + 1)   [12..15] stop rows (end of turn, </think>; NO_TOKEN)
#define ST_STOP 0
#define ST_K 1
#define ST_D 2
#define ST_J 8
#define ST_NEXT 9
#define ST_COMMIT 10
#define ST_REJ 11
#define ST_STOPROWS 12

// x = embedding row `row` of a q4row table (2048 bytes of nibbles as q4: byte j of a block = q[j] | q[j + 16] << 4;
// then 128 fp16 scales), row = st[sel] (or `fixed` when sel is 0xFFFF). Nothing when the row is NO_TOKEN or the gate
// is set (the CPU staged that row itself).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void embed_q4row(global const uchar* table, global const uint* st, uint sel, uint fixed, global float* x, uint cols,
                 global const uint* gate) {
    if (gate && *gate) return;
    uint row = sel == 0xFFFFu ? fixed : st[sel];
    if (row == NO_TOKEN) return;
    ulong rb = cols / 2 + cols / 16;
    global const uchar* q = table + (ulong)row * rb;
    global const half_t* sc = (global const half_t*)(q + cols / 2);
    for (uint i = LID; i < cols; i += 256) {
        uint blk = i >> 5, j = i & 31;
        uint byte = q[blk * 16 + (j & 15)];
        uint v = j < 16 ? (byte & 15u) : (byte >> 4);
        x[i] = ((float)v - 8.0f) * (float)sc[blk];
    }
}

// Softmax statistics of a logits row under Garlic's sampling (temperature, min-p as `sample` in common.h):
// top logit, and Z = sum over the tokens within min_gap of the top of exp((l - top) * inv_temp). One group.
static void row_stats(local float* red, global const float* l, uint n, float inv_temp, float min_gap, float* top,
                      float* z) {
    uint t = LID;
    float m = -3.0e38f;
    for (uint i = t; i < n; i += 256) m = __builtin_fmaxf(m, l[i]);
    m = group_max(red, m);
    float s = 0.0f;
    if (inv_temp > 0.0f)
        for (uint i = t; i < n; i += 256) {
            float v = l[i];
            if (v >= m - min_gap) s += exp_((v - m) * inv_temp);
        }
    s = group_sum(red, s);
    *top = m;
    *z = inv_temp > 0.0f ? s : 1.0f;
}

// probability of row i under those statistics (greedy: 1 for the argmax row `amax`, else 0)
static float row_p(float v, float top, float z, float inv_temp, float min_gap) {
    if (inv_temp <= 0.0f) return v >= top ? 1.0f : 0.0f;
    return v >= top - min_gap ? exp_((v - top) * inv_temp) / z : 0.0f;
}

static float uniform01(uint salt, uint i) { return ((float)(hash32(salt ^ i) >> 8) + 0.5f) * (1.0f / 16777216.0f); }

// Gumbel-max draw from the weights w_i (0 = never), or the first largest when greedy. -> index (all threads).
static uint draw(local float* bv, local uint* bi, float best, uint idx) {
    uint t = LID;
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
    uint r = bi[0];
    BARRIER();
    return r;
}

// Draft step k (0-based): draw a row from the draft's logits (the first n rows of lm_head) with Garlic's sampling;
// propose it only if the draft's top probability is at least theta (else the draft stops: st[ST_STOP] = 1, and no
// later step runs). One group of 256.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void draft_sample(global const float* restrict q, uint n, float inv_temp, uint seed, uint k, float min_gap,
                  float theta, global uint* st) {
    local float red[256];
    local float bv[256];
    local uint bi[256];
    if (st[ST_STOP]) return;
    float top, z;
    row_stats(red, q, n, inv_temp, min_gap, &top, &z);
    uint t = LID, salt = hash32(seed ^ 0x51ED270Bu ^ (k * 0x9E3779B9u));
    const float LN2 = 0.69314718f;
    float best = -3.0e38f;
    uint idx = 0;
    for (uint i = t; i < n; i += 256) {
        float v = q[i];
        if (inv_temp > 0.0f) {
            if (v < top - min_gap) continue;
            float u = uniform01(salt, i);
            v = v * inv_temp - __builtin_log2f(-__builtin_log2f(u) * LN2) * LN2;
        }
        if (v > best) best = v, idx = i;
    }
    uint r = draw(bv, bi, best, idx);
    if (t == 0) {
        if (1.0f / z < theta) {
            st[ST_STOP] = 1;
        } else {
            st[ST_D + k] = r;
            st[ST_K] = k + 1;
        }
    }
}

// The exact acceptance (speculative sampling) of the K = st[ST_K] proposed rows d_1..d_K against the target's logits
// L [K + 1][ldl] (row i = the distribution after the i-th verified token, the first being the last sampled one) and
// the draft's logits Q [K][ldq] (only its first nq rows are drawn from). d_i is kept with probability
// min(1, p_i(d)/q_i(d)); at the first refusal the token is drawn from max(0, p_i - q_i) (normalised), and if all
// are kept, from p_{K+1}. A kept stop row (end of turn, </think>) ends the run there: it becomes the next token.
// Writes st[ST_J] = drafts kept, st[ST_NEXT] = the next token's row, st[ST_COMMIT] = j + 1. One group of 256.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void spec_accept(global const float* restrict L, uint ldl, uint nv, global const float* restrict Q, uint ldq, uint nq,
                 float inv_temp, uint seed, float min_gap, global uint* st) {
    local float red[256];
    local float bv[256];
    local uint bi[256];
    local float pq[2];
    uint t = LID, K = st[ST_K], j = 0, next = NO_TOKEN, rej = K + 1;
    uint salt = hash32(seed ^ 0xACCE97EDu);
    const float LN2 = 0.69314718f;
    for (uint i = 0; i < K; i++) {
        global const float* l = L + (ulong)i * ldl;
        global const float* qq = Q + (ulong)i * ldq;
        float tp, zp, tq, zq;
        row_stats(red, l, nv, inv_temp, min_gap, &tp, &zp);
        row_stats(red, qq, nq, inv_temp, min_gap, &tq, &zq);
        uint d = st[ST_D + i];
        if (t == 0) {
            pq[0] = row_p(l[d], tp, zp, inv_temp, min_gap);
            pq[1] = row_p(qq[d], tq, zq, inv_temp, min_gap);
        }
        BARRIER();
        float p = pq[0], qd = pq[1];
        BARRIER();
        int ok;
        if (inv_temp > 0.0f) ok = qd > 0.0f && uniform01(salt, i) * qd < p;   // u < p / q
        else ok = p > 0.0f;                                                  // greedy: d is the target's argmax
        if (inv_temp <= 0.0f && ok) {   // (ties: the target's own first argmax, as `sample` picks it)
            float best = -3.0e38f;
            uint idx = 0;
            for (uint r = t; r < nv; r += 256)
                if (l[r] > best) best = l[r], idx = r;
            ok = draw(bv, bi, best, idx) == d;
        }
        if (ok) {
            int stop = 0;
            for (uint s = 0; s < 4; s++) stop |= st[ST_STOPROWS + s] == d;
            if (stop) {   // the turn (or the reasoning) ends here: d is the next token, not a committed one
                next = d;
                rej = i + 1;
                break;
            }
            j++;
            continue;
        }
        // refused: draw from max(0, p - q), q = 0 past the draft's rows
        uint salt2 = hash32(salt ^ (0x7F4A7C15u * (i + 1)));
        float best = -3.0e38f;
        uint idx = 0;
        for (uint r = t; r < nv; r += 256) {
            float pr = row_p(l[r], tp, zp, inv_temp, min_gap);
            float qr = r < nq ? row_p(qq[r], tq, zq, inv_temp, min_gap) : 0.0f;
            float wgt = pr - qr;
            if (inv_temp > 0.0f) {
                if (!(wgt > 0.0f)) continue;
                float u = uniform01(salt2, r);
                float v = __builtin_log2f(wgt) * LN2 - __builtin_log2f(-__builtin_log2f(u) * LN2) * LN2;
                if (v > best) best = v, idx = r;
            } else if (l[r] > best) {
                best = l[r], idx = r;
            }
        }
        next = draw(bv, bi, best, idx);
        rej = i + 1;
        break;
    }
    if (next == NO_TOKEN) {   // every draft kept: the token after them, from p_{K+1}
        global const float* l = L + (ulong)K * ldl;
        float tp, zp;
        row_stats(red, l, nv, inv_temp, min_gap, &tp, &zp);
        uint salt3 = hash32(salt ^ 0xB0E5u);
        float best = -3.0e38f;
        uint idx = 0;
        for (uint r = t; r < nv; r += 256) {
            float v = l[r];
            if (inv_temp > 0.0f) {
                if (v < tp - min_gap) continue;
                float u = uniform01(salt3, r);
                v = v * inv_temp - __builtin_log2f(-__builtin_log2f(u) * LN2) * LN2;
            }
            if (v > best) best = v, idx = r;
        }
        next = draw(bv, bi, best, idx);
    }
    if (t == 0) {
        st[ST_J] = j;
        st[ST_NEXT] = next;
        st[ST_COMMIT] = j + 1;
        st[ST_REJ] = rej;
    }
}

// The Gated DeltaNet pieces of a verification: conv_silu_rows / delta_rows (prefill.h) that do not store the state
// (the old one stays for the commit), the conv's raw inputs kept in raw [rows][n]; then, once the number of tokens
// accepted is known (*count, written by spec_accept), the state moved forward by those tokens only.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void conv_silu_rows_v(global float* x, global const float* w, global const float* state, uint n, uint ldx, uint rows,
                      global float* raw) {
    uint c = GID * 256 + LID;
    if (c >= n) return;
    global const float* s = state + 3 * (ulong)c;
    global const float* wc = w + 4 * (ulong)c;
    float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float s0 = s[0], s1 = s[1], s2 = s[2];
    for (uint t = 0; t < rows; t++) {
        global float* xt = x + (ulong)t * ldx + c;
        float v = *xt;
        raw[(ulong)t * n + c] = v;
        float y = w0 * s0 + w1 * s1 + w2 * s2 + w3 * v;
        s0 = s1;
        s1 = s2;
        s2 = v;
        *xt = y / (1.0f + exp_(-y));
    }
}

kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void conv_commit(global float* state, global const float* raw, uint n, global const uint* count) {
    uint c = GID * 256 + LID;
    if (c >= n) return;
    global float* s = state + 3 * (ulong)c;
    float s0 = s[0], s1 = s[1], s2 = s[2];
    uint m = *count;
    for (uint t = 0; t < m; t++) {
        s0 = s1;
        s1 = s2;
        s2 = raw[(ulong)t * n + c];
    }
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
}

// delta_rows (prefill.h) with: rows = min(rows, *count) when count is given; the state stored only when `store`;
// out = 0: no output (the commit).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void delta_rows_v(global const float* x, global const float* A, global const float* dt_bias, global float* S_all,
                  global const float* gw, global float* out, uint n_kh, uint n_vh, float eps, uint ldx, uint rows,
                  global const uint* count, uint store) {
    local float qs[DK], ks[DK];
    local float red[256];
    uint h = GID, t = LID, kh = h / (n_vh / n_kh);
    uint kd = n_kh * DK, vd = n_vh * DV;
    uint vi = t & 127, k0 = (t >> 7) * 64;
    if (count && *count < rows) rows = *count;
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
        if (t < DV && out) {
            float rr = rsq(red[0] / (float)DV + eps), zz = xr[2 * kd + vd + h * DV + t];
            out[(ulong)r * vd + h * DV + t] = o * rr * gwt * (zz / (1.0f + exp_(-zz)));
        }
        BARRIER();   // (red, qs and ks are rewritten by the next token)
    }
    if (store) {
#pragma unroll
        for (uint i = 0; i < 64; i++) S[(ulong)i * DV] = st[i];
    }
}

// dst[i] = src[i] for i < n, unless the gate is set (copy_f32 for the draft chain).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void copy_f32_g(global const float* src, global float* dst, uint n, global const uint* gate) {
    uint i = GID * 256 + LID;
    if (i < n && !(gate && *gate)) dst[i] = src[i];
}

// rmsnorm (common.h) into out + GID * ldo (several rows: x row GID at x + GID * ldx), skipped when gated.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void rmsnorm_g(global const float* x, global const float* w, global float* out, uint n, float eps, uint ldx, uint ldo,
               global const uint* gate) {
    local float red[256];
    if (gate && *gate) return;
    x += (ulong)GID * ldx;
    out += (ulong)GID * ldo;
    float ss = 0.0f;
    for (uint i = LID; i < n; i += 256) ss += x[i] * x[i];
    float r = rsq(group_sum(red, ss) / (float)n + eps);
    for (uint i = LID; i < n; i += 256) out[i] = x[i] * r * w[i];
}
