// ---- KV cache and attention, for heads of HEAD_DIM (128 or 256) with RoPE on the first ROPE_DIM dims ----------
// Include after common.h with HEAD_DIM, ROPE_DIM and MAX_SEQ defined.
// The cache is fp16 (half the memory traffic of fp32), one pair of buffers per attention layer:
//   k: [n_kv][HEAD_DIM][MAX_SEQ]  transposed, so that consecutive positions are consecutive in memory
//   v: [MAX_SEQ][n_kv][HEAD_DIM]; with V_INT8 defined, int8 instead, then a float scale per (position, kv head)
//      [MAX_SEQ][n_kv] right after the values (v = q * scale, scale = max |v| / 127): half the memory
// Attention runs flash-decoding style: attn_part splits the positions into blocks of ATT_BLOCK, one group per
// (block, kv head) handling all the q heads that share that kv head (GQA), so each K/V element is read once.
// It leaves an unnormalised partial output plus its max and sum; attn_combine merges the blocks of each head.

// It can be included more than once (heads of another size in the same program): define ATTN_NAME(x) to give
// the kernels other names, and ATTN_NO_QK_NORM for models without RMS norm on q and k (Llama).
// With ATTN_SPLIT (for this inclusion only), each layer's cache is two halves of MAX_SEQ / 2 positions with the
// layout above: positions [0, MAX_SEQ / 2) at kcache / vcache, the others `split` bytes further (one more kernel
// argument). The engine keeps all the second halves in one block that it can lend while the context is short.
#ifndef ATTN_NAME
#define ATTN_NAME(x) x
#endif
#undef ATT_MAX_BLOCKS
#undef EPT
#undef DPW
#undef HALFV
#undef CHARV
#undef V_SCALES
#undef KSTRIDE
#undef SPLIT_ARG
#undef KV_AT
#ifdef ATTN_SPLIT
#define KSTRIDE (MAX_SEQ / 2)                  // positions per half
#define SPLIT_ARG , ulong split
#define KV_AT(c, p) ((p) >= KSTRIDE ? (global void*)((global uchar*)(c) + split) : (global void*)(c))
#else
#define KSTRIDE MAX_SEQ
#define SPLIT_ARG
#define KV_AT(c, p) ((global void*)(c))
#endif
#define ATT_BLOCK 64
#define ATT_MAX_BLOCKS (MAX_SEQ / ATT_BLOCK)
#define GQ_MAX 4                                // q heads per kv head (Qwen3: 2 or 4; Qwen3.5-4B: 4)
#define EPT (HEAD_DIM / 64)                     // elements of a head per thread, with 64 threads
#define DPW (HEAD_DIM / 4)                      // dimensions per wave in the score pass
#define HALFV ATTN_NAME(halfv_t)
#define CHARV ATTN_NAME(charv_t)
#if EPT == 1
typedef _Float16 HALFV;
typedef char CHARV;
#undef VEL
#define VEL(v, d) (v)
#else
typedef _Float16 HALFV __attribute__((ext_vector_type(EPT)));
typedef char CHARV __attribute__((ext_vector_type(EPT)));
#undef VEL
#define VEL(v, d) ((v)[d])
#endif
#define V_SCALES(vc, n_kv) ((global float*)((global schar*)(vc) + (ulong)KSTRIDE * (n_kv) * HEAD_DIM))

// Groups 0..n_heads-1: q heads (RMS norm + RoPE in place). Next n_kv groups: k heads (norm + RoPE -> k cache).
// Last n_kv groups: copy v heads to the v cache. 64 threads; thread j owns elements j, j + 64, ...
// RoPE (rotate-half) on dims [0, ROPE_DIM): pairs (i, i + ROPE_DIM / 2); rope: [MAX_SEQ][ROPE_DIM] = cos then sin.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void ATTN_NAME(qk_rope_cache)(global float* qkv, global const float* q_norm, global const float* k_norm,
                   global const float* rope, global half_t* kcache, global half_t* vcache,
                   uint pos, uint n_heads, uint n_kv, float eps SPLIT_ARG) {
    local float red[64];
    local float hv[HEAD_DIM];
    uint g = GID, j = LID;
    uint lp = pos % KSTRIDE;   // position within its half of the cache
    kcache = (global half_t*)KV_AT(kcache, pos);
    vcache = (global half_t*)KV_AT(vcache, pos);
    if (g < n_heads + n_kv) {
        global float* h = qkv + g * HEAD_DIM;
        float x[EPT];
        float ss = 0.0f;
        for (uint m = 0; m < EPT; m++) {
            x[m] = h[j + 64 * m];
            ss += x[m] * x[m];
        }
        red[j] = ss;
        BARRIER();
        for (uint s = 32; s > 0; s >>= 1) {
            if (j < s) red[j] += red[j + s];
            BARRIER();
        }
#ifdef ATTN_NO_QK_NORM
        for (uint m = 0; m < EPT; m++) hv[j + 64 * m] = x[m];
#else
        float r = rsq(red[0] / (float)HEAD_DIM + eps);
        global const float* w = g < n_heads ? q_norm : k_norm;
        for (uint m = 0; m < EPT; m++) {
            x[m] *= r * w[j + 64 * m];
            hv[j + 64 * m] = x[m];
        }
#endif
        BARRIER();
        const uint rh = ROPE_DIM / 2;
        global const float* rp = rope + (ulong)pos * ROPE_DIM;
        for (uint m = 0; m < EPT; m++) {
            uint i = j + 64 * m;
            float y = x[m];
            if (i < ROPE_DIM) {
                float c = rp[i % rh], s = rp[rh + i % rh];
                y = i < rh ? y * c - hv[i + rh] * s : y * c + hv[i - rh] * s;
            }
            if (g < n_heads) h[i] = y;
            else kcache[((ulong)(g - n_heads) * HEAD_DIM + i) * KSTRIDE + lp] = (half_t)y;
        }
    } else {
        uint kv = g - n_heads - n_kv;
        global const float* v = qkv + (n_heads + n_kv + kv) * HEAD_DIM;
#ifdef V_INT8
        float vv[EPT], mx = 0.0f;
        for (uint m = 0; m < EPT; m++) {
            vv[m] = v[j + 64 * m];
            mx = __builtin_fmaxf(mx, __builtin_fabsf(vv[m]));
        }
        red[j] = mx;
        BARRIER();
        for (uint s = 32; s > 0; s >>= 1) {
            if (j < s) red[j] = __builtin_fmaxf(red[j], red[j + s]);
            BARRIER();
        }
        mx = red[0];
        float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
        global schar* dst = (global schar*)vcache + ((ulong)lp * n_kv + kv) * HEAD_DIM;
        for (uint m = 0; m < EPT; m++) dst[j + 64 * m] = (schar)__builtin_rintf(vv[m] * inv);
        if (j == 0) V_SCALES(vcache, n_kv)[(ulong)lp * n_kv + kv] = mx / 127.0f;
#else
        global half_t* dst = vcache + ((ulong)lp * n_kv + kv) * HEAD_DIM;
        for (uint m = 0; m < EPT; m++) dst[j + 64 * m] = (half_t)v[j + 64 * m];
#endif
    }
}

// Group g = block * n_kv + kv, 256 threads = 4 waves. A block is ATT_BLOCK positions:
//   scores   wave w sums dimensions DPW w .. DPW (w + 1) - 1 for the block's 64 positions (lane = position)
//   softmax  wave h = q head h, lane = position
//   v        wave w takes positions w, w + 4, ...; lane = dimensions EPT lane .. EPT lane + EPT - 1 (one load)
// A context of a single block writes the normalised result to out; otherwise part gets, per head and block,
// the unnormalised output [n_heads][ATT_MAX_BLOCKS][HEAD_DIM], then max m and sum s [n_heads][ATT_MAX_BLOCKS].
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void ATTN_NAME(attn_part)(global const float* q, global const half_t* kcache, global const half_t* vcache, global float* part,
               global float* out, uint pos, uint n_heads, uint n_kv, float scale SPLIT_ARG) {
    local float qd[GQ_MAX][HEAD_DIM];
    local float ps[4][GQ_MAX][ATT_BLOCK];      // per-wave partial scores
    local float w[GQ_MAX][ATT_BLOCK];          // softmax weights exp(score - max)
    local float red[GQ_MAX][ATT_BLOCK];
    local float acc[4][GQ_MAX][HEAD_DIM];      // per-wave partial outputs
    uint t = LID, wv = t >> 6, lane = t & 63;
    uint kv = GID % n_kv, blk = GID / n_kv, gq = n_heads / n_kv, n = pos + 1;
    uint p0 = blk * ATT_BLOCK, cnt = n - p0 < ATT_BLOCK ? n - p0 : ATT_BLOCK;
    int direct = n <= ATT_BLOCK;
    uint lp0 = p0 % KSTRIDE;   // a block never straddles the two halves (ATT_BLOCK divides KSTRIDE)
    kcache = (global const half_t*)KV_AT(kcache, p0);
    vcache = (global const half_t*)KV_AT(vcache, p0);

    for (uint i = t; i < GQ_MAX * HEAD_DIM; i += 256)
        qd[i / HEAD_DIM][i % HEAD_DIM] = i / HEAD_DIM < gq ? q[kv * gq * HEAD_DIM + i] * scale : 0.0f;
    BARRIER();

    float a[GQ_MAX] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < cnt) {
        global const half_t* k = kcache + ((ulong)kv * HEAD_DIM + wv * DPW) * KSTRIDE + lp0 + lane;
#pragma unroll
        for (uint i = 0; i < DPW; i++) {
            float kf = (float)k[(ulong)i * KSTRIDE];
            for (uint h = 0; h < GQ_MAX; h++) a[h] += qd[h][wv * DPW + i] * kf;
        }
    }
    for (uint h = 0; h < GQ_MAX; h++) ps[wv][h][lane] = a[h];
    BARRIER();

    // softmax over the block: wave wv = head wv
    float sc = lane < cnt ? ps[0][wv][lane] + ps[1][wv][lane] + ps[2][wv][lane] + ps[3][wv][lane] : -3.0e38f;
    red[wv][lane] = sc;
    BARRIER();
    for (uint st = 32; st > 0; st >>= 1) {
        if (lane < st) red[wv][lane] = __builtin_fmaxf(red[wv][lane], red[wv][lane + st]);
        BARRIER();
    }
    float m = red[wv][0];
    BARRIER();
    float e = lane < cnt ? exp_(sc - m) : 0.0f;
    w[wv][lane] = e;
    red[wv][lane] = e;
    BARRIER();
    for (uint st = 32; st > 0; st >>= 1) {
        if (lane < st) red[wv][lane] += red[wv][lane + st];
        BARRIER();
    }
    float sum = red[wv][0];

    // weighted sum of v. No branch, so the 16 loads are all in flight together: past the end of the context the
    // weight is 0 and the position is clamped to the last one written (never-written cache memory may hold NaNs).
    float o[GQ_MAX][EPT];
    for (uint h = 0; h < GQ_MAX; h++)
        for (uint d = 0; d < EPT; d++) o[h][d] = 0.0f;
#ifdef V_INT8
    global const CHARV* v = (global const CHARV*)((global const schar*)vcache + ((ulong)lp0 * n_kv + kv) * HEAD_DIM) + lane;
    global const float* vsc = V_SCALES(vcache, n_kv) + (ulong)lp0 * n_kv + kv;
#else
    global const HALFV* v = (global const HALFV*)(vcache + ((ulong)lp0 * n_kv + kv) * HEAD_DIM) + lane;
#endif
#pragma unroll
    for (uint j = 0; j < ATT_BLOCK / 4; j++) {
        uint jj = wv + 4 * j, jc = jj < cnt ? jj : cnt - 1;
        float vf[EPT];
#ifdef V_INT8
        CHARV vv = v[(ulong)jc * n_kv * (HEAD_DIM / EPT)];
        float s8 = vsc[(ulong)jc * n_kv];
        for (uint d = 0; d < EPT; d++) vf[d] = (float)VEL(vv, d) * s8;
#else
        HALFV vv = v[(ulong)jc * n_kv * (HEAD_DIM / EPT)];
        for (uint d = 0; d < EPT; d++) vf[d] = (float)VEL(vv, d);
#endif
        for (uint h = 0; h < GQ_MAX; h++) {
            float ww = w[h][jj];
            for (uint d = 0; d < EPT; d++) o[h][d] += ww * vf[d];
        }
    }
    for (uint h = 0; h < GQ_MAX; h++)
        for (uint d = 0; d < EPT; d++) acc[wv][h][EPT * lane + d] = o[h][d];
    BARRIER();
    for (uint i = t; i < gq * HEAD_DIM; i += 256) {
        uint h = i / HEAD_DIM, d = i % HEAD_DIM, qh = kv * gq + h;
        float r = acc[0][h][d] + acc[1][h][d] + acc[2][h][d] + acc[3][h][d];
        if (direct) out[qh * HEAD_DIM + d] = r / red[h][0];
        else part[((ulong)qh * ATT_MAX_BLOCKS + blk) * HEAD_DIM + d] = r;
    }
    if (!direct && lane == 0 && wv < gq) {
        uint qh = kv * gq + wv;
        global float* pm = part + (ulong)n_heads * ATT_MAX_BLOCKS * HEAD_DIM;
        pm[qh * ATT_MAX_BLOCKS + blk] = m;
        pm[(n_heads + qh) * ATT_MAX_BLOCKS + blk] = sum;
    }
}

// One group of HEAD_DIM threads per q head: merge the blocks of attn_part into out[h][d] (contexts over one block).
kernel __attribute__((reqd_work_group_size(HEAD_DIM, 1, 1)))
void ATTN_NAME(attn_combine)(global const float* part, global float* out, uint n_heads, uint n_blocks) {
    uint h = GID, t = LID;
    global const float* pm = part + (ulong)n_heads * ATT_MAX_BLOCKS * HEAD_DIM + h * ATT_MAX_BLOCKS;
    global const float* psum = pm + n_heads * ATT_MAX_BLOCKS;
    float m = -3.0e38f;
    for (uint b = 0; b < n_blocks; b++) m = __builtin_fmaxf(m, pm[b]);
    float o = 0.0f, sum = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        float wb = exp_(pm[b] - m);
        o += wb * part[((ulong)h * ATT_MAX_BLOCKS + b) * HEAD_DIM + t];
        sum += wb * psum[b];
    }
    out[h * HEAD_DIM + t] = o / sum;
}
#undef ATTN_SPLIT
