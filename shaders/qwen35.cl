// Qwen3.5 decode-step kernels for the PS4 GPU (GCN gfx701), compiled by LLVM from OpenCL C 1.2 without its
// standard headers. Text decoder of Qwen3.5-4B: 32 layers, every 4th full attention (heads of 256, RoPE on the
// first 64 dims, sigmoid output gate), the others Gated DeltaNet (linear attention with a 128 x 128 state per
// value head). Shared kernels: common.h; full attention: attn.h; Gated DeltaNet below.

#define HEAD_DIM 256
#define ROPE_DIM 64
#define MAX_SEQ 8192
#define V_INT8              // value cache in int8 + a scale per position and kv head (attn.h)
#define ATTN_SPLIT          // the cache in two halves (attn.h): voice mode lends the second one (chat/engine.c)

#include "common.h"
#include "attn.h"
#include "attn_tq.h"     // TurboQuant 4-bit cache (chat/engine.c tq)

// x[i] *= sigmoid(g[i]): the output gate of the full-attention layers.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mul_sigmoid(global float* x, global const float* g, uint n) {
    uint i = GID * 256 + LID;
    if (i < n) x[i] = x[i] / (1.0f + exp_(-g[i]));
}

// ---- Gated DeltaNet ---------------------------------------------------------------------------------------------
// Per layer, one input projection gives x = [q (n_kh x 128) | k (n_kh x 128) | v (n_vh x 128) | z (n_vh x 128) |
// b (n_vh) | a (n_vh)]. q, k and v go through a causal depthwise convolution of 4 taps + SiLU (conv_silu), then the
// gated delta rule updates each value head's state S [128 k][128 v] (delta_step):
//   q, k = l2norm(q), l2norm(k); q /= sqrt(128); beta = sigmoid(b); decay = exp(A * softplus(a + dt_bias))
//   S *= decay;  delta = (v - S^T k) * beta;  S += k delta^T;  o = S^T q
//   out = rmsnorm(o) * w * silu(z)
// Value head h uses key head h / (n_vh / n_kh).

#define DK 128
#define DV 128

// Causal conv over the n channels of q, k, v: y = w0 s0 + w1 s1 + w2 s2 + w3 x, then SiLU; the state keeps the
// last 3 inputs per channel. w: [n][4]; state: [n][3] (zero at the start of a conversation).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void conv_silu(global float* x, global const float* w, global float* state, uint n) {
    uint c = GID * 256 + LID;
    if (c >= n) return;
    global float* s = state + 3 * (ulong)c;
    global const float* wc = w + 4 * (ulong)c;
    float s0 = s[0], s1 = s[1], s2 = s[2], v = x[c];
    float y = wc[0] * s0 + wc[1] * s1 + wc[2] * s2 + wc[3] * v;
    s[0] = s1;
    s[1] = s2;
    s[2] = v;
    x[c] = y / (1.0f + exp_(-y));
}

// One group of 256 threads per value head. Thread t: state column v = t & 127, key rows 64 side .. 64 side + 63
// (side = t >> 7), held in registers between the two passes. S: [n_vh][DK][DV], DV contiguous (coalesced).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void delta_step(global const float* x, global const float* A, global const float* dt_bias, global float* S_all,
                global const float* gw, global float* out, uint n_kh, uint n_vh, float eps) {
    local float qs[DK], ks[DK];
    local float red[256];
    uint h = GID, t = LID, kh = h / (n_vh / n_kh);
    uint kd = n_kh * DK, vd = n_vh * DV;
    global const float* vin = x + 2 * kd + h * DV;
    global const float* z = x + 2 * kd + vd + h * DV;
    float b = x[2 * kd + 2 * vd + h], a = x[2 * kd + 2 * vd + n_vh + h];

    // l2-normalise q (threads 0..127) and k (threads 128..255)
    float val = t < DK ? x[kh * DK + t] : x[kd + kh * DK + (t - DK)];
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
    float sp = a + dt_bias[h];
    sp = sp > 20.0f ? sp : __builtin_log2f(1.0f + exp_(sp)) * 0.69314718f;   // softplus
    float decay = exp_(A[h] * sp);

    uint vi = t & 127, k0 = (t >> 7) * 64;
    global float* S = S_all + (ulong)h * DK * DV + (ulong)k0 * DV + vi;
    float st[64];
    float kv = 0.0f;
#pragma unroll
    for (uint i = 0; i < 64; i++) {
        st[i] = S[(ulong)i * DV] * decay;
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
        S[(ulong)i * DV] = st[i];
        o += st[i] * qs[k0 + i];
    }
    red[t] = o;
    BARRIER();
    o = red[vi] + red[vi + 128];
    BARRIER();

    // gated RMS norm over the head's 128 outputs (threads 0..127)
    red[t] = t < DV ? o * o : 0.0f;
    BARRIER();
    for (uint s = 64; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        BARRIER();
    }
    if (t < DV) {
        float r = rsq(red[0] / (float)DV + eps), zz = z[t];
        out[h * DV + t] = o * r * gw[t] * (zz / (1.0f + exp_(-zz)));
    }
}

#include "prefill.h"      // the prompt in batches (chat/engine.c prefill)
