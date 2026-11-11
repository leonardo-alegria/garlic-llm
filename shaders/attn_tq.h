// ---- TurboQuant KV cache, 4 bits per element (Zandieh et al., arXiv 2504.19874; pc/turboquant.py) ------------
// Include after attn.h (heads of 256, HEAD_DIM 256, ROPE_DIM, MAX_SEQ). Kernels qk_rope_cache_tq, attn_part_tq and
// attn_combine_tq do the jobs of attn.h's three with a cache 4x smaller than fp16:
//   every k head (after its RMS norm and RoPE) and v head x is stored rotated, x' = H (s * x) / 16 (H the
//   256-point Walsh-Hadamard transform, s fixed random signs). Whatever x is, the coordinates of x' * 16 / |x|
//   follow the distribution of a random unit vector's coordinates times 16 (close to N(0, 1)); each one is
//   replaced by the nearest of that distribution's 16 Lloyd-Max levels: a 4-bit index, plus one scale |x| / 16
//   per vector. No calibration: the codebook only depends on the head size.
//   q is rotated the same way (q' . k' = q . k). The values stay rotated: the attention output is rotated back
//   once per head, in attn_combine_tq (which runs for every context length).
// tq: 40 words from the engine (pc/turboquant.py): [0, 16) the levels, [16, 31) the boundaries between them,
// [32, 40) the sign bits (bit i & 31 of word i >> 5 set: s_i = -1).
// Cache of one layer, in blocks of TQ_SEQ positions (kcache, vcache):
//   k  [n_kv][32][TQ_SEQ] uint: word w of a position = dims 8w .. 8w + 7, a nibble each (dim 8w in the lowest),
//      positions consecutive (as the fp16 cache, lane = position); then the scales [n_kv][TQ_SEQ] float
//   v  [TQ_SEQ][n_kv][128] bytes: byte b = dims 2b (low nibble) and 2b + 1; then the scales [TQ_SEQ][n_kv] float
// 8.25 MiB per layer and block of 8192 positions with 4 kv heads (fp16 k + int8 v: 24.1 MiB).
// Positions TQ_SEQ .. 2 TQ_SEQ - 1 (16K of context) are in a second block with the same layout, `split` bytes
// after the first (the engine keeps them in its lendable block, kv_hi).

#define TQ_D 256
#define TQ_WORDS (TQ_D / 8)
#define TQ_SEQ MAX_SEQ                           // positions per block
#define TQ_MAX_BLOCKS (2 * TQ_SEQ / ATT_BLOCK)   // attention blocks of up to 16K positions (layout of `part`)
#define TQ_KSC(kc, n_kv) ((global float*)((global uint*)(kc) + (ulong)(n_kv) * TQ_WORDS * TQ_SEQ))
#define TQ_VSC(vc, n_kv) ((global float*)((global uchar*)(vc) + (ulong)TQ_SEQ * (n_kv) * (TQ_D / 2)))
#define TQ_AT(c, p) ((p) >= TQ_SEQ ? (global void*)((global uchar*)(c) + split) : (global void*)(c))

// x[m] = element j + 64 m of a 256-vector held by 64 threads (one wave). Multiplies it by the signs, then applies
// the unnormalised Walsh-Hadamard transform (natural order): strides 64 and 128 inside each thread, 1 .. 32
// through hv.
static void tq_rotate64(local float* hv, float* x, uint j, global const uint* restrict sign) {
    for (uint m = 0; m < 4; m++)
        if (sign[(j >> 5) + 2 * m] >> (j & 31) & 1) x[m] = -x[m];
    float a0 = x[0] + x[1], a1 = x[0] - x[1], a2 = x[2] + x[3], a3 = x[2] - x[3];
    x[0] = a0 + a2, x[2] = a0 - a2, x[1] = a1 + a3, x[3] = a1 - a3;
    for (uint h = 1; h < 64; h <<= 1) {
        for (uint m = 0; m < 4; m++) hv[j + 64 * m] = x[m];
        BARRIER();
        for (uint m = 0; m < 4; m++) {
            float o = hv[(j ^ h) + 64 * m];
            x[m] = (j & h) ? o - x[m] : x[m] + o;
        }
        BARRIER();
    }
}

// 4-bit index of t = x' * 16 / |x|: the number of boundaries below it
static uint tq_code(float t, global const float* restrict bnd) {
    uint c = 0;
    for (uint k = 0; k < 15; k++) c += t > bnd[k] ? 1u : 0u;
    return c;
}

// Groups 0..n_heads-1: q heads (RMS norm, RoPE, rotation, in place). Next n_kv: k heads (norm, RoPE, rotation,
// 4 bits -> k cache). Last n_kv: v heads (rotation, 4 bits -> v cache). 64 threads; thread j: elements j + 64 m.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void qk_rope_cache_tq(global float* qkv, global const float* q_norm, global const float* k_norm,
                      global const float* rope, global uint* kcache, global uchar* vcache,
                      global const float* restrict tq, uint pos, uint n_heads, uint n_kv, float eps, ulong split) {
    local float red[64];
    local float hv[TQ_D];
    local uint cd[TQ_D];
    global const uint* restrict sign = (global const uint*)tq + 32;
    uint g = GID, j = LID, lp = pos % TQ_SEQ;   // position within its block of the cache
    kcache = (global uint*)TQ_AT(kcache, pos);
    vcache = (global uchar*)TQ_AT(vcache, pos);
    float x[4];
    int is_v = g >= n_heads + n_kv;
    global float* h = qkv + g * TQ_D;
    float ss = 0.0f;
    for (uint m = 0; m < 4; m++) {
        x[m] = h[j + 64 * m];
        ss += x[m] * x[m];
    }
    if (!is_v) {   // RMS norm + RoPE (rotate-half on dims [0, ROPE_DIM)), as attn.h
        red[j] = ss;
        BARRIER();
        for (uint s = 32; s > 0; s >>= 1) {
            if (j < s) red[j] += red[j + s];
            BARRIER();
        }
        float r = rsq(red[0] / (float)TQ_D + eps);
        global const float* w = g < n_heads ? q_norm : k_norm;
        for (uint m = 0; m < 4; m++) {
            x[m] *= r * w[j + 64 * m];
            hv[j + 64 * m] = x[m];
        }
        BARRIER();
        const uint rh = ROPE_DIM / 2;
        global const float* rp = rope + (ulong)pos * ROPE_DIM;
        for (uint m = 0; m < 4; m++) {
            uint i = j + 64 * m;
            if (i < ROPE_DIM) {
                float c = rp[i % rh], s = rp[rh + i % rh];
                x[m] = i < rh ? x[m] * c - hv[i + rh] * s : x[m] * c + hv[i - rh] * s;
            }
        }
        BARRIER();
    }
    tq_rotate64(hv, x, j, sign);
    for (uint m = 0; m < 4; m++) x[m] *= 0.0625f;
    if (g < n_heads) {
        for (uint m = 0; m < 4; m++) h[j + 64 * m] = x[m];
        return;
    }
    // |x'| over the group, then the indices
    ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    red[j] = ss;
    BARRIER();
    for (uint s = 32; s > 0; s >>= 1) {
        if (j < s) red[j] += red[j + s];
        BARRIER();
    }
    float norm = __builtin_sqrtf(red[0]);
    float inv = norm > 0.0f ? 16.0f / norm : 0.0f;
    for (uint m = 0; m < 4; m++) cd[j + 64 * m] = tq_code(x[m] * inv, tq + 16);
    BARRIER();
    if (!is_v) {
        uint kv = g - n_heads;
        if (j < TQ_WORDS) {
            uint wd = 0;
            for (uint k = 0; k < 8; k++) wd |= cd[8 * j + k] << (4 * k);
            kcache[((ulong)kv * TQ_WORDS + j) * TQ_SEQ + lp] = wd;
        }
        if (j == 0) TQ_KSC(kcache, n_kv)[(ulong)kv * TQ_SEQ + lp] = norm * 0.0625f;
    } else {
        uint kv = g - n_heads - n_kv;
        uint u = cd[4 * j] | cd[4 * j + 1] << 4 | cd[4 * j + 2] << 8 | cd[4 * j + 3] << 12;
        ((global ushort*)vcache)[((ulong)lp * n_kv + kv) * (TQ_D / 4) + j] = (ushort)u;
        if (j == 0) TQ_VSC(vcache, n_kv)[(ulong)lp * n_kv + kv] = norm * 0.0625f;
    }
}

// As attn_part (attn.h): group g = block * n_kv + kv, 256 threads = 4 waves, a block = ATT_BLOCK positions, q the
// rotated q heads. Always writes the partial result to part (attn_combine_tq rotates the output back).
//   scores   wave w: dimensions 64 w .. 64 w + 63 (8 words), lane = position; level from cb (LDS), times the scale
//   v        wave w: positions w, w + 4, ...; lane = dimensions 4 lane .. 4 lane + 3 (one 16-bit load)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void attn_part_tq(global const float* q, global const uint* kcache, global const uchar* vcache, global float* part,
                  global const float* restrict tq, uint pos, uint n_heads, uint n_kv, float scale, ulong split) {
    local float qd[GQ_MAX][TQ_D];
    local float ps[4][GQ_MAX][ATT_BLOCK];
    local float w[GQ_MAX][ATT_BLOCK];
    local float red[GQ_MAX][ATT_BLOCK];
    local float acc[4][GQ_MAX][TQ_D];
    local float cb[16];
    uint t = LID, wv = t >> 6, lane = t & 63;
    uint kv = GID % n_kv, blk = GID / n_kv, gq = n_heads / n_kv, n = pos + 1;
    uint p0 = blk * ATT_BLOCK, cnt = n - p0 < ATT_BLOCK ? n - p0 : ATT_BLOCK;
    uint lp0 = p0 % TQ_SEQ;   // a block never straddles two cache blocks (ATT_BLOCK divides TQ_SEQ)
    kcache = (global const uint*)TQ_AT(kcache, p0);
    vcache = (global const uchar*)TQ_AT(vcache, p0);

    if (t < 16) cb[t] = tq[t];
    for (uint i = t; i < GQ_MAX * TQ_D; i += 256)
        qd[i / TQ_D][i % TQ_D] = i / TQ_D < gq ? q[kv * gq * TQ_D + i] * scale : 0.0f;
    BARRIER();

    float a[GQ_MAX] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (lane < cnt) {
        global const uint* k = kcache + ((ulong)kv * TQ_WORDS + wv * 8) * TQ_SEQ + lp0 + lane;
        uint wd[8];
#pragma unroll
        for (uint i = 0; i < 8; i++) wd[i] = k[(ulong)i * TQ_SEQ];
        float ks = TQ_KSC(kcache, n_kv)[(ulong)kv * TQ_SEQ + lp0 + lane];
#pragma unroll
        for (uint i = 0; i < 8; i++)
#pragma unroll
            for (uint e = 0; e < 8; e++) {
                float kf = cb[(wd[i] >> (4 * e)) & 15];
                uint d = wv * 64 + 8 * i + e;
                for (uint h = 0; h < GQ_MAX; h++) a[h] += qd[h][d] * kf;
            }
        for (uint h = 0; h < GQ_MAX; h++) a[h] *= ks;
    }
    for (uint h = 0; h < GQ_MAX; h++) ps[wv][h][lane] = a[h];
    BARRIER();

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

    // weighted sum of the values (no branch: past the end the weight is 0 and the position the last one written)
    float o[GQ_MAX][4];
    for (uint h = 0; h < GQ_MAX; h++)
        for (uint d = 0; d < 4; d++) o[h][d] = 0.0f;
    global const ushort* v = (global const ushort*)vcache + ((ulong)lp0 * n_kv + kv) * (TQ_D / 4) + lane;
    global const float* vsc = TQ_VSC(vcache, n_kv) + (ulong)lp0 * n_kv + kv;
#pragma unroll
    for (uint j = 0; j < ATT_BLOCK / 4; j++) {
        uint jj = wv + 4 * j, jc = jj < cnt ? jj : cnt - 1;
        uint u = v[(ulong)jc * n_kv * (TQ_D / 4)];
        float s = vsc[(ulong)jc * n_kv];
        float vf[4];
        for (uint d = 0; d < 4; d++) vf[d] = cb[(u >> (4 * d)) & 15];
        for (uint h = 0; h < GQ_MAX; h++) {
            float ww = w[h][jj] * s;
            for (uint d = 0; d < 4; d++) o[h][d] += ww * vf[d];
        }
    }
    for (uint h = 0; h < GQ_MAX; h++)
        for (uint d = 0; d < 4; d++) acc[wv][h][4 * lane + d] = o[h][d];
    BARRIER();
    for (uint i = t; i < gq * TQ_D; i += 256) {
        uint h = i / TQ_D, d = i % TQ_D, qh = kv * gq + h;
        part[((ulong)qh * TQ_MAX_BLOCKS + blk) * TQ_D + d] = acc[0][h][d] + acc[1][h][d] + acc[2][h][d] + acc[3][h][d];
    }
    if (lane == 0 && wv < gq) {
        uint qh = kv * gq + wv;
        global float* pm = part + (ulong)n_heads * TQ_MAX_BLOCKS * TQ_D;
        pm[qh * TQ_MAX_BLOCKS + blk] = m;
        pm[(n_heads + qh) * TQ_MAX_BLOCKS + blk] = sum;
    }
}

// One group of 256 threads per q head: merge the blocks (as attn_combine), then rotate back: o = s * (H o') / 16.
kernel __attribute__((reqd_work_group_size(TQ_D, 1, 1)))
void attn_combine_tq(global const float* part, global float* out, global const float* restrict tq, uint n_heads,
                     uint n_blocks) {
    local float hv[TQ_D];
    global const uint* restrict sign = (global const uint*)tq + 32;
    uint h = GID, t = LID;
    global const float* pm = part + (ulong)n_heads * TQ_MAX_BLOCKS * TQ_D + h * TQ_MAX_BLOCKS;
    global const float* psum = pm + n_heads * TQ_MAX_BLOCKS;
    float m = -3.0e38f;
    for (uint b = 0; b < n_blocks; b++) m = __builtin_fmaxf(m, pm[b]);
    float o = 0.0f, sum = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        float wb = exp_(pm[b] - m);
        o += wb * part[((ulong)h * TQ_MAX_BLOCKS + b) * TQ_D + t];
        sum += wb * psum[b];
    }
    hv[t] = o / sum;
    BARRIER();
    for (uint st = 1; st < TQ_D; st <<= 1) {
        if (t < TQ_D / 2) {
            uint i = (t / st) * 2 * st + t % st;
            float a = hv[i], b = hv[i + st];
            hv[i] = a + b;
            hv[i + st] = a - b;
        }
        BARRIER();
    }
    float r = hv[t] * 0.0625f;
    out[h * TQ_D + t] = sign[t >> 5] >> (t & 31) & 1 ? -r : r;
}
