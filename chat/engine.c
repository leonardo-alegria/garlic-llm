// Qwen3 / Qwen3.5 inference on the PS4 GPU from a native app (the same command packets as pc/gpu.py).
//
// One block of Garlic memory (CPU write-combined, GPU read/write) holds everything the GPU touches. Each run builds
// a command buffer: Sony's default state, then for every token ~400 compute dispatches (embed, per layer: norm,
// q/k/v, q/k norm + RoPE + cache, attention, o, add, norm, gate/up, SiLU x up, down, add; final norm, logits,
// sample), each followed by CS_PARTIAL_FLUSH and an ACQUIRE_MEM that drops stale L1 / scalar cache lines; then an
// end-of-pipe event writes a label the CPU polls. Kernels come from assets/kernels.bin (shaders/qwen3.cl or,
// for Qwen3.5, shaders/qwen35.cl, whose DeltaNet layers replace attention by conv + delta rule).

#include "engine.h"

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <orbis/GnmDriver.h>
#include <orbis/libkernel.h>

#define SH_START_X 0x204
#define SH_NUM_THREAD_X 0x207
#define SH_PGM_LO 0x20C
#define SH_PGM_RSRC1 0x212
#define SH_RESOURCE_LIMITS 0x215
#define SH_STATIC_THREAD_MGMT_SE0 0x216
#define SH_TMPRING_SIZE 0x218
#define SH_USER_DATA_0 0x240
#define EV_CS_PARTIAL_FLUSH (7 | (4 << 8))
#define EV_BOTTOM_OF_PIPE_TS (0x28 | (5 << 8))
#define EV_CACHE_FLUSH_AND_INV_TS (0x14 | (5 << 8))
#define COHER_TCL1_KCACHE ((1u << 22) | (1u << 27))
// The GPU's L2 is write-back and does not see the CPU's writes: invalidate it (writing back dirty lines) when a
// submission starts, so kernel arguments, staged rows and the like that the CPU just rewrote at the same addresses are
// fetched fresh, and write it back before the end-of-pipe label, so the CPU reads the GPU's last results.
// (CP_COHER_CNTL TC_ACTION_ENA; on the end-of-pipe event, Gnm's kCacheActionWriteBackAndInvalidateL1andL2 = 0x38 << 12.)
#define COHER_L2_TCL1_KCACHE ((1u << 22) | (1u << 23) | (1u << 27))
#define EOP_WB_INV_L1_L2 (0x38u << 12)
#define LABEL_DONE 0x600DF00Du
#define NO_TOKEN 0xFFFFFFFFu
#define DCB_BYTES (4u << 20)
#define KARGS_BYTES (4u << 20)
#define STEPS_PER_SUBMIT 40
#define PF_ROWS 32              // prompt tokens per batch (shaders/prefill.h; at most STEPS_PER_SUBMIT staging rows)
#define ALIGN_2MB (2ull << 20)
#define ATT_BLOCK 64            // ATT_BLOCK in shaders/attn.h (MAX_SEQ comes from model.idx)
#define ROW_CACHE 4096          // embedding rows kept in memory (power of two)
#define SWAR_SPLIT_ROWS 12416   // split-K (partial sums) only for matrices up to this many rows

void say(const char* fmt, ...);   // main.c
static void mm_tune(Engine* e);   // (after engine_run)
static void swar_tune(Engine* e);
static uint64_t pf_a_floats, pf_h_floats;   // sizes of the batched prompt's buffers (swar_tune uses them)
#ifdef GARLIC_SPEC
// speculative decoding with the MTP head (engine_spec.inc, at the end)
int spec_active(const Engine* e);
static void spec_reset(Engine* e);
static void spec_prefill(Engine* e, uint64_t stage, uint32_t n, uint32_t pos0);
uint64_t spec_carry(void);
#endif
static int gpu_selftest(Engine* e);   // (after begin)

static void fail(Engine* e, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->error, sizeof(e->error), fmt, ap);
    va_end(ap);
    char line[200];
    snprintf(line, sizeof(line), "[ps4ai] error: %s\n", e->error);
    sceKernelDebugOutText(0, line);
}

// Where a weights file is: inside the package (/app0/models, the all-in-one PKG) or, with an app-only package, at the
// path its index names (in /data). The first one that exists; path (129 bytes or more) is rewritten in place.
static void weights_locate(char* path) {
    const char* base = strrchr(path, '/');
    char in_pkg[129];
    snprintf(in_pkg, sizeof(in_pkg), "/app0/models/%s", base ? base + 1 : path);
    int f = open(in_pkg, O_RDONLY);
    if (f >= 0) {
        close(f);
        strcpy(path, in_pkg);
    }
    say("weights: %s", path);
}

static uint8_t* read_file(const char* path, size_t* size) {
    int f = open(path, O_RDONLY);
    if (f < 0) return NULL;
    off_t n = lseek(f, 0, SEEK_END);
    lseek(f, 0, SEEK_SET);
    uint8_t* buf = malloc(n);
    size_t done = 0;
    while (buf && done < (size_t)n) {
        ssize_t r = read(f, buf + done, n - done);
        if (r <= 0) break;
        done += r;
    }
    close(f);
    if (!buf || done != (size_t)n) {
        free(buf);
        return NULL;
    }
    *size = n;
    return buf;
}

static uint64_t gpu_alloc(Engine* e, uint64_t size) {
    uint64_t off = (e->next + 255) & ~255ull;
    if (off + size > e->mem_size) return 0;
    e->next = off + size;
    return (uint64_t)(uintptr_t)(e->mem + off);
}

int eng_margin_mb = 64;

static int reserve_garlic(Engine* e) {
    // as much of the app's direct memory as is still free (the framebuffers took theirs), 32 MiB at a time
    // (the first size that works, less eng_margin_mb left for the system dialogs: 64 MiB, or 32 when main.c took
    // 32 of them for the screen copies, which leaves the same block for the model)
    for (uint64_t m = (sceKernelGetDirectMemorySize() >> 20) & ~31ull; m >= 1024; m -= 32) {
        uint64_t size = m << 20;
        off_t off = 0;
        void* addr = NULL;
        if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, ALIGN_2MB, 3 /* WC Garlic */, &off))
            continue;
        if (sceKernelMapDirectMemory(&addr, size, 0x33 /* CPU + GPU read/write */, 0, off, ALIGN_2MB)) {
            sceKernelReleaseDirectMemory(off, size);
            continue;
        }
        sceKernelMunmap(addr, size);   // found the largest block; take it again with a 64 MiB margin
        sceKernelReleaseDirectMemory(off, size);
        size -= (uint64_t)eng_margin_mb << 20;
        if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, ALIGN_2MB, 3, &off)) return -1;
        if (sceKernelMapDirectMemory(&addr, size, 0x33, 0, off, ALIGN_2MB)) return -1;
        e->mem = addr;
        e->mem_size = size;
        return 0;
    }
    return -1;
}

static const GpuKernel* kernel(Engine* e, const char* name) {
    for (int i = 0; i < e->nk; i++)
        if (!strcmp(e->k[i].name, name)) return &e->k[i];
    fail(e, "kernel %s missing from kernels.bin", name);
    return NULL;
}

static int load_kernels(Engine* e, const char* dir) {
    char path[256];
    size_t size;
    snprintf(path, sizeof(path), "%s/kernels.bin", dir);
    uint8_t* d = read_file(path, &size);
    if (!d || memcmp(d, "PKRN", 4)) {
        fail(e, "cannot read %s", path);
        return -1;
    }
    uint32_t n, text_size;
    memcpy(&n, d + 4, 4);
    memcpy(&text_size, d + 8, 4);
    uint64_t text = gpu_alloc(e, text_size);
    memcpy((void*)(uintptr_t)text, d + 12 + n * 52, text_size);
    e->nk = 0;
    for (uint32_t i = 0; i < n && i < sizeof(e->k) / sizeof(e->k[0]); i++) {
        const uint8_t* r = d + 12 + i * 52;
        GpuKernel* k = &e->k[e->nk++];
        memcpy(k->name, r, 32);
        k->name[31] = 0;
        uint32_t entry;
        memcpy(&entry, r + 32, 4);
        memcpy(&k->rsrc1, r + 36, 4);
        memcpy(&k->rsrc2, r + 40, 4);
        memcpy(&k->user_sgprs, r + 44, 4);
        memcpy(&k->kernarg_sgpr, r + 48, 4);
        k->addr = text + entry;
    }
    free(d);
    int q4i = e->t[0].fmt == 3 || e->t[0].fmt == 4;   // the embedding's format tells the layout of the file
    e->k_embed = kernel(e, q4i ? "embed_q4i" : "embed_q4");
    if (q4i) {
        e->k_bsum = kernel(e, "block_sums");
        e->k_mvi = kernel(e, "matvec_q4i");
        e->k_mvi8 = kernel(e, "matvec_q4i8");
        if (!e->k_bsum || !e->k_mvi || !e->k_mvi8) return -1;
    }
    e->k_copy = kernel(e, "copy_f32");   // (all models: "Pensar" copies the DeltaNet state with it)
    if (e->swar) {   // q4s weights: the SWAR kernels of shaders/spec.h
        static const char* var[5] = {"w4", "w8", "w16", "w8_s2", "w8_s4"};
        e->k_qx8 = kernel(e, "quant_x8");
        e->k_sumsplit = kernel(e, "sum_split");
        if (!e->k_qx8 || !e->k_sumsplit) return -1;
        for (int n = 0; n < 4; n++)
            for (int v = 0; v < 5; v++) {
                char nm[32];
                snprintf(nm, sizeof(nm), "swar_mv%d_%s", n + 1, var[v]);
                if (!(e->k_swar[n][v] = kernel(e, nm))) return -1;
                snprintf(nm, sizeof(nm), "swar3_mv%d_%s", n + 1, var[v]);
                e->k_swar3[n][v] = kernel(e, nm);   // (q3s: only needed if the model has such matrices)
            }
        e->error[0] = 0;
    }
    if (e->cpu_embed) {
        e->k_mv3 = kernel(e, "matvec_q3i");
        e->k_mv38 = kernel(e, "matvec_q3i8");
        if (!e->k_mv3 || !e->k_mv38 || !e->k_copy) return -1;
    }
    e->k_rmsnorm = kernel(e, "rmsnorm");
    e->k_qk = kernel(e, "qk_rope_cache");
    e->k_attn = kernel(e, "attn_part");
    e->k_attn_comb = kernel(e, "attn_combine");
    e->k_silu = kernel(e, "silu_mul");
    e->k_add = kernel(e, "add");
    e->k_mv = kernel(e, "matvec_q4");
    e->k_sample = kernel(e, "sample");
    if (e->qwen35) {
        e->k_sig = kernel(e, "mul_sigmoid");
        e->k_conv = kernel(e, "conv_silu");
        e->k_delta = kernel(e, "delta_step");
        if (!e->k_sig || !e->k_conv || !e->k_delta) return -1;
    }
    if (e->qwen35) {   // TurboQuant cache (shaders/attn_tq.h) when kernels.bin has it; /data/garlic_tq.txt = 0: off
        for (int i = 0; i < e->nk; i++) {
            if (!strcmp(e->k[i].name, "qk_rope_cache_tq")) e->k_qk_tq = &e->k[i];
            if (!strcmp(e->k[i].name, "attn_part_tq")) e->k_attn_tq = &e->k[i];
            if (!strcmp(e->k[i].name, "attn_combine_tq")) e->k_comb_tq = &e->k[i];
        }
        size_t fs;
        char* off = (char*)read_file("/data/garlic_tq.txt", &fs);
        e->tq = e->k_qk_tq && e->k_attn_tq && e->k_comb_tq && !(off && fs && off[0] == '0');
        free(off);
        // a kernels.bin built with the batched prompt (mm_q4i_rows) has the attn_tq.h that takes the split
        // argument: the cache can hold 16K (positions 8192.. in kv_hi)
        for (int i = 0; i < e->nk; i++)
            if (!strcmp(e->k[i].name, "mm_q4i_rows")) e->tq16 = 1;
    }
    // the prompt in batches (shaders/prefill.h); the same kernels.bin has the TurboQuant kernels for 16K (split).
    // The embedding rows come from the disk (cpu_embed, the 9B) or from the GPU (4B).
    static const char* pf_names[7] = {"mm_q4i_rows", "mm_q3i_rows", "rmsnorm_rows", "silu_mul_rows",
                                      "mul_sigmoid_rows", "conv_silu_rows", "delta_rows"};
    const GpuKernel** pf_k[7] = {&e->k_mm4r, &e->k_mm3r, &e->k_rmsr, &e->k_silur, &e->k_sigr, &e->k_convr, &e->k_deltar};
    e->pf = e->qwen35 && (e->cpu_embed || e->k_embed);
    for (int j = 0; j < 7; j++) {
        *pf_k[j] = NULL;
        for (int i = 0; i < e->nk; i++)
            if (!strcmp(e->k[i].name, pf_names[j])) *pf_k[j] = &e->k[i];
        if (!*pf_k[j]) e->pf = 0;
    }
    {   // /data/garlic_pf.txt = 0: the prompt token by token again, without rebuilding (the 16K context stays)
        size_t fs;
        char* sw = (char*)read_file("/data/garlic_pf.txt", &fs);
        if (sw && fs && sw[0] == '0') e->pf = 0;
        free(sw);
    }
    return e->k_embed && e->k_rmsnorm && e->k_qk && e->k_attn && e->k_attn_comb && e->k_silu && e->k_add && e->k_mv && e->k_sample ? 0 : -1;
}

static int load_index(Engine* e, const char* dir) {
    char path[256];
    size_t size;
    snprintf(path, sizeof(path), "%s/model.idx", dir);
    uint8_t* d = read_file(path, &size);
    if (!d || (memcmp(d, "PMDL", 4) && memcmp(d, "PMQ5", 4))) {
        fail(e, "cannot read %s", path);
        return -1;
    }
    e->qwen35 = !memcmp(d, "PMQ5", 4);
    uint32_t v[10];
    memcpy(v, d + 4, 40);
    e->n_layers = v[0], e->hidden = v[1], e->inter = v[2], e->n_heads = v[3], e->n_kv = v[4];
    e->head_dim = v[5], e->vocab = v[6], e->max_seq = v[7], e->eos = v[8], e->eos2 = v[9];
    memcpy(&e->eps, d + 44, 4);
    memcpy(&e->rope_theta, d + 48, 4);
    memcpy(e->weights_path, d + 52, 128);
    e->weights_path[128] = 0;
    weights_locate(e->weights_path);
    memcpy(&e->weights_size, d + 180, 8);
    memcpy(&e->n_tensors, d + 188, 4);
    uint32_t rec0 = 192, vocab_map = 0;
    if (e->qwen35) {   // "<8I" after the header: n_kh, n_vh, dk, dv, rope_dim, full_mask, im_start, 0
        uint32_t x[8];
        memcpy(x, d + 192, 32);
        e->n_kh = x[0], e->n_vh = x[1], e->dk = x[2], e->dv = x[3], e->rope_dim = x[4], e->full_mask = x[5];
        e->im_start = x[6];
        e->has_head = x[7] & 1;
        e->cpu_embed = (x[7] >> 1) & 1;
        vocab_map = (x[7] >> 2) & 1;   // lm_head trimmed, an "ids" record (row -> token id) after it
        rec0 = 224;
    } else {
        e->rope_dim = e->head_dim;
        e->full_mask = 0xFFFFFFFFu;
        e->im_start = 151644;
    }
    e->im_end = e->eos;
    e->layer_base = calloc(e->n_layers, sizeof(uint32_t));
    for (uint32_t i = 0, b = 2; i < e->n_layers; i++) {   // full attention 8 tensors per layer, DeltaNet 10
        e->layer_base[i] = b;
        b += (e->full_mask >> i & 1) ? 8 : 10;
    }
    e->t = calloc(e->n_tensors, sizeof(Tensor));
    for (uint32_t i = 0; i < e->n_tensors; i++) {
        const uint8_t* r = d + rec0 + i * 36;   // "<QQIIQI" records (pc/export_chat.py)
        memcpy(&e->t[i].off, r, 8);
        memcpy(&e->t[i].size, r + 8, 8);
        memcpy(&e->t[i].rows, r + 16, 4);
        memcpy(&e->t[i].cols, r + 20, 4);
        memcpy(&e->t[i].scales, r + 24, 8);
        memcpy(&e->t[i].fmt, r + 32, 4);
        if (e->t[i].fmt == 6 || e->t[i].fmt == 8) e->swar = 1;   // q4s / q3s matrices: SWAR products (spec.h)
    }
    free(d);
    e->head_idx = e->has_head ? e->n_tensors - 1 - vocab_map : 0;
    if (vocab_map && (e->t[e->n_tensors - 1].fmt != 7 || e->t[e->n_tensors - 1].rows != e->vocab)) {
        fail(e, "model.idx: the ids record does not match the vocabulary");
        return -1;
    }
    return 0;
}

// TurboQuant 4-bit KV cache (shaders/attn_tq.h): the 16 Lloyd-Max levels for the coordinates of a rotated head of
// 256, the 15 boundaries between them, 0, then the 8 words of random sign bits; as float bit patterns, from
// pc/turboquant.py (python -c "import turboquant as t; print(t.c_table())")
#define TQ_TABLE \
    0xc02d6e45u, 0xc0039f58u, 0xbfce1f1du, 0xbfa0262eu, 0xbf705f2du, 0xbf2790a4u, 0xbec60879u, 0xbe030d12u, \
    0x3e030d12u, 0x3ec60879u, 0x3f2790a4u, 0x3f705f2du, 0x3fa0262eu, 0x3fce1f1du, 0x40039f58u, 0x402d6e45u, \
    0xc01886ceu, 0xbfeaaee6u, 0xbfb722a6u, 0xbf8c2ae2u, 0xbf4bf7e8u, 0xbf054a70u, 0xbe83c781u, 0x00000000u, \
    0x3e83c781u, 0x3f054a70u, 0x3f4bf7e8u, 0x3f8c2ae2u, 0x3fb722a6u, 0x3feaaee6u, 0x401886ceu, 0x00000000u, \
    0x76452f77u, 0x1d5aee0du, 0xda3d646du, 0x05d4b13du, 0x9642207cu, 0x5d203182u, 0x0d2a2a6du, 0xb4a5b633u

int engine_init(Engine* e, const char* dir, ProgressFn progress, void* user) {
    memset(e, 0, sizeof(*e));
    e->tsc_freq = sceKernelGetTscFrequency();
    if (load_index(e, dir)) return -1;
    if (reserve_garlic(e)) {
        fail(e, "could not reserve GPU memory");
        return -1;
    }
    char line[200];
    snprintf(line, sizeof(line), "[ps4ai] %u MiB of GPU memory at %p; model %u layers, hidden %u, vocab %u\n",
             (unsigned)(e->mem_size >> 20), e->mem, e->n_layers, e->hidden, e->vocab);
    sceKernelDebugOutText(0, line);

    e->dcb = (uint32_t*)(uintptr_t)gpu_alloc(e, DCB_BYTES);
    e->dcb_cap = DCB_BYTES / 4;
    e->kargs = (uint8_t*)(uintptr_t)gpu_alloc(e, KARGS_BYTES);
    e->kargs_cap = KARGS_BYTES;
    e->label = (volatile uint32_t*)(uintptr_t)gpu_alloc(e, 256);
    e->stamps = (volatile uint64_t*)(uintptr_t)gpu_alloc(e, 256);
    if (load_kernels(e, dir)) return -1;

    uint32_t* init = (uint32_t*)(uintptr_t)gpu_alloc(e, 1024);
    sceGnmDrawInitDefaultHardwareState350(init, 256);
    memcpy(e->init_state, init, sizeof(e->init_state));
    if (gpu_selftest(e)) return -1;   // before reading gigabytes of weights: does this console's GPU do the right thing?

    if (e->tq && (e->head_dim != 256 || e->max_seq != 8192)) e->tq = 0;   // the kernels' shape (qwen35.cl MAX_SEQ)
    // TurboQuant with a kernels.bin whose attn_tq.h takes the split argument: 16K of context, positions 8192.. in
    // kv_hi (while it is not lent to S3Gen)
    if (e->tq && e->tq16) e->max_seq = 2 * 8192;
    // RoPE table: [max_seq][rope_dim] = cos of the first half, then sin (RoPE on the first rope_dim dims of a head)
    uint32_t half = e->rope_dim / 2;
    e->rope = gpu_alloc(e, (uint64_t)e->max_seq * e->rope_dim * 4);
    float* rope = (float*)(uintptr_t)e->rope;
    for (uint32_t p = 0; p < e->max_seq; p++)
        for (uint32_t i = 0; i < half; i++) {
            double ang = p * pow((double)e->rope_theta, -2.0 * i / e->rope_dim);
            rope[(uint64_t)p * e->rope_dim + i] = (float)cos(ang);
            rope[(uint64_t)p * e->rope_dim + half + i] = (float)sin(ang);
        }

    if (e->head_dim != (e->qwen35 ? 256u : 128u) || e->n_heads / e->n_kv > 4 || e->max_seq % ATT_BLOCK) {
        fail(e, "model shape not supported by the kernels (max_seq %u, head_dim %u)", e->max_seq, e->head_dim);
        return -1;
    }
    // per layer: fp16 KV cache, k transposed [n_kv][head_dim][max_seq], v [max_seq][n_kv][head_dim] (shaders/attn.h);
    // or, for a Gated DeltaNet layer, the conv state [conv_n][3] and the delta-rule state [n_vh][dk][dv] (fp32)
    e->layer_cache = (uint64_t)e->max_seq * e->n_kv * e->head_dim * 2;
    e->conv_n = 2ull * e->n_kh * e->dk + (uint64_t)e->n_vh * e->dv;
    e->state_bytes = (uint64_t)e->n_vh * e->dk * e->dv * 4;
    e->kc = calloc(e->n_layers, sizeof(uint64_t));
    e->vc = calloc(e->n_layers, sizeof(uint64_t));
    e->ctx_limit = e->max_seq;
    e->lend_ctx = e->max_seq / 2;
    if (e->qwen35 && e->tq) {
        // TurboQuant (shaders/attn_tq.h), per full layer and block of 8192 positions: k = 4-bit words
        // [n_kv][head_dim / 8][8192] + scales [n_kv][8192]; v = 4-bit codes [8192][n_kv][head_dim / 2 bytes] +
        // scales [8192][n_kv]. Positions 0..8191 of all the layers in one block (66 MiB). kv_hi is a block of its own,
        // the size of the second halves without TurboQuant (S3Gen's weights in voice mode). With 16K (max_seq 16384)
        // positions 8192.. of each layer are in kv_hi at the same offset (tq_split bytes from the first block):
        // lending it shortens the context to 8192 (lend_ctx); with 8K, lending it leaves the context whole.
        static const uint32_t tab[40] = {TQ_TABLE};
        uint64_t kb = (uint64_t)e->n_kv * 8192 * (e->head_dim / 2 + 4), n_full = 0;
        for (uint32_t i = 0; i < e->n_layers; i++) n_full += e->full_mask >> i & 1;
        uint64_t all = gpu_alloc(e, n_full * 2 * kb);
        e->kv_split = n_full * (4096ull * e->n_kv * e->head_dim * 2 + 4096ull * e->n_kv * (e->head_dim + 4));
        e->kv_hi = e->kv_split ? gpu_alloc(e, e->kv_split) : 0;
        e->lend_ctx = e->max_seq;
        if (e->max_seq > 8192) {
            if (all && e->kv_hi && e->kv_split >= n_full * 2 * kb) {
                e->tq_split = e->kv_hi - all;
                e->lend_ctx = 8192;
            } else {
                e->max_seq = 8192;   // (cannot happen: the block is 96.5 MiB, the second 8K takes 66)
            }
        }
        e->ctx_limit = e->max_seq;
        e->tq_tab = gpu_alloc(e, 256);
        if (e->tq_tab) memcpy((void*)(uintptr_t)e->tq_tab, tab, sizeof(tab));
        for (uint32_t i = 0, j = 0; all && e->tq_tab && i < e->n_layers; i++) {
            if (e->full_mask >> i & 1) {
                e->kc[i] = all + j * 2 * kb;
                e->vc[i] = e->kc[i] + kb;
                j++;
            } else {
                e->kc[i] = gpu_alloc(e, e->conv_n * 12);
                e->vc[i] = gpu_alloc(e, e->state_bytes);
            }
        }
        char line[160];
        snprintf(line, sizeof(line), "[ps4ai] TurboQuant KV cache: %u MiB for 8192 positions (+%u MiB lendable; "
                 "context %u, %u while lent)\n", (unsigned)(n_full * 2 * kb >> 20), (unsigned)(e->kv_split >> 20),
                 e->max_seq, e->lend_ctx);
        sceKernelDebugOutText(0, line);
    } else if (e->qwen35) {
        // values in int8 + a float scale per (position, kv head) (V_INT8 in shaders/qwen35.cl); each full layer's
        // k and v in two halves of max_seq / 2 positions (ATTN_SPLIT): the first halves of all the layers in one
        // block, the second halves in the next, kv_split bytes further
        uint64_t khalf = e->layer_cache / 2, vhalf = (uint64_t)e->max_seq / 2 * e->n_kv * (e->head_dim + 4), n_full = 0;
        for (uint32_t i = 0; i < e->n_layers; i++) n_full += e->full_mask >> i & 1;
        e->kv_split = n_full * (khalf + vhalf);
        uint64_t lo = gpu_alloc(e, e->kv_split);
        e->kv_hi = gpu_alloc(e, e->kv_split);
        if (lo && e->kv_hi && e->kv_hi != lo + e->kv_split) e->kv_hi = 0;   // (gpu_alloc rounds to 256 bytes)
        for (uint32_t i = 0, j = 0; lo && e->kv_hi && i < e->n_layers; i++) {
            if (e->full_mask >> i & 1) {
                e->kc[i] = lo + j * (khalf + vhalf);
                e->vc[i] = e->kc[i] + khalf;
                j++;
            } else {
                e->kc[i] = gpu_alloc(e, e->conv_n * 12);
                e->vc[i] = gpu_alloc(e, e->state_bytes);
            }
        }
    } else {
        for (uint32_t i = 0; i < e->n_layers; i++) {
            int full = e->full_mask >> i & 1;
            e->kc[i] = gpu_alloc(e, full ? e->layer_cache : e->conv_n * 12);
            e->vc[i] = gpu_alloc(e, full ? e->layer_cache : e->state_bytes);
        }
    }
    for (uint32_t i = 0; i < e->n_layers; i++)
        if (!e->kc[i] || !e->vc[i]) {
            fail(e, "caches do not fit in %u MiB", (unsigned)(e->mem_size >> 20));
            return -1;
        }
    e->x = gpu_alloc(e, e->hidden * 4);
    e->xn = gpu_alloc(e, e->hidden * 4);
    e->attn = gpu_alloc(e, (uint64_t)(e->n_heads * e->head_dim > e->n_vh * e->dv ? e->n_heads * e->head_dim
                                                                                  : e->n_vh * e->dv) * 4);
    // (the TurboQuant kernels of a kernels.bin with the batched prompt always lay it out for 16K, 256 blocks)
    uint32_t part_seq = e->max_seq;
    if (e->tq16 && part_seq < 16384) part_seq = 16384;
    e->attn_part = gpu_alloc(e, (uint64_t)e->n_heads * (part_seq / ATT_BLOCK) * (e->head_dim + 2) * 4);
    e->hmlp = gpu_alloc(e, e->inter * 4);
    e->bsum = gpu_alloc(e, (uint64_t)(e->inter > e->hidden ? e->inter : e->hidden) / 32 * 4 + 4096);
    uint32_t stage_rows = STEPS_PER_SUBMIT;
    if (e->pf && e->cpu_embed)
        stage_rows = 2 * PF_ROWS > STEPS_PER_SUBMIT ? 2 * PF_ROWS : STEPS_PER_SUBMIT;   // two batches of the prompt
    e->staging = gpu_alloc(e, (uint64_t)stage_rows * e->hidden * 4);
    // q | k | v (| output gate, Qwen3.5); the DeltaNet input projection: q | k | v | z | b | a
    e->qkv = gpu_alloc(e, (uint64_t)(e->n_heads + 2 * e->n_kv + (e->qwen35 ? e->n_heads : 0)) * e->head_dim * 4);
    e->lin = gpu_alloc(e, (e->conv_n + (uint64_t)e->n_vh * e->dv + 2 * e->n_vh + 64) * 4);
    e->o = gpu_alloc(e, e->hidden * 4);
    e->gu = gpu_alloc(e, 2 * e->inter * 4);
    e->down = gpu_alloc(e, e->hidden * 4);
    e->logits = gpu_alloc(e, (uint64_t)(e->head_idx ? e->t[e->head_idx].rows : e->vocab) * 4);
    if (e->swar) {   // int8 activations for up to PF_ROWS rows of the widest input, and the split-K partial sums
        uint32_t mc = 0, mr = 0;
        for (uint32_t i = 0; i < e->n_tensors; i++)
            if (e->t[i].fmt == 6 || e->t[i].fmt == 8) {
                if (e->t[i].cols > mc) mc = e->t[i].cols;
                if (e->t[i].rows > mr && e->t[i].rows <= SWAR_SPLIT_ROWS) mr = e->t[i].rows;
            }
        e->x8 = gpu_alloc(e, (uint64_t)PF_ROWS * mc * 4);
        e->dx8 = gpu_alloc(e, (uint64_t)PF_ROWS * mc / 32 * 4);
        e->init8 = gpu_alloc(e, (uint64_t)PF_ROWS * mc / 32 * 4);
        e->yp_rows = mr;
        e->yp = gpu_alloc(e, 16ull * mr * 4);
    }
    e->last = gpu_alloc(e, 256);
    e->history = gpu_alloc(e, 4096 * 4);
    // the batched prompt: q4i / q3i matrices (rows in tiles of 64), and PF_ROWS rows of each buffer (5.8 MiB for the 9B)
    uint64_t pf_wa = 0, pf_wh = 0;
    for (uint32_t i = 0; e->pf && i < e->n_layers; i++) {
        uint32_t b = e->layer_base[i], full = e->full_mask >> i & 1;
        uint32_t m[4] = {b + (full ? 4 : 2), b + (full ? 5 : 7), b + (full ? 6 : 8), b + (full ? 7 : 9)};
        for (int j = 0; j < 4; j++) {
            const Tensor* t = &e->t[m[j]];
            int sw = t->fmt == 6 || t->fmt == 8;
            if ((t->fmt != 3 && t->fmt != 5 && !sw) || t->rows % (sw ? 128 : 64) || t->cols % 32)
                e->pf = 0;
            if (!(j & 1) && t->rows > pf_wa) pf_wa = t->rows;   // input projection, gate|up: rows out
            if ((j & 1) && t->cols > pf_wh) pf_wh = t->cols;    // o / out projection, down: columns in
        }
    }
    pf_a_floats = (uint64_t)PF_ROWS * pf_wa, pf_h_floats = (uint64_t)PF_ROWS * pf_wh;
    if (e->pf) {
        e->pf_x = gpu_alloc(e, (uint64_t)PF_ROWS * e->hidden * 4);
        e->pf_xn = gpu_alloc(e, (uint64_t)PF_ROWS * e->hidden * 4);
        e->pf_a = gpu_alloc(e, PF_ROWS * pf_wa * 4);
        e->pf_h = gpu_alloc(e, PF_ROWS * pf_wh * 4);
        if (!e->pf_h) e->pf = 0;
    }
    e->model = gpu_alloc(e, e->weights_size);
    if (!e->model) {
        fail(e, "model does not fit in %u MiB", (unsigned)(e->mem_size >> 20));
        return -1;
    }

    int f = open(e->weights_path, O_RDONLY);
    if (f < 0) {
        fail(e, "cannot open %s", e->weights_path);
        return -1;
    }
    uint64_t done = 0, chunk = 32ull << 20;
    while (done < e->weights_size) {
        uint64_t n = e->weights_size - done < chunk ? e->weights_size - done : chunk;
        ssize_t r = read(f, (uint8_t*)(uintptr_t)e->model + done, n);
        if (r <= 0) break;
        done += r;
        if (progress) progress(user, (float)done / e->weights_size, "Cargando el modelo");
    }
    close(f);
    if (done != e->weights_size) {
        fail(e, "weights file incomplete (%llu of %llu bytes)", (unsigned long long)done,
             (unsigned long long)e->weights_size);
        return -1;
    }
    sceKernelDebugOutText(0, "[ps4ai] model loaded\n");
    if (e->head_idx && e->head_idx != e->n_tensors - 1) {   // the row -> token id table (trimmed lm_head)
        const Tensor* t = &e->t[e->n_tensors - 1];
        e->ids = malloc((size_t)t->rows * 4);
        int fi = open(e->weights_path, O_RDONLY);
        int ok = e->ids && fi >= 0 && pread(fi, e->ids, (size_t)t->rows * 4, (off_t)t->off) == (ssize_t)t->rows * 4;
        if (fi >= 0) close(fi);
        if (!ok) {
            fail(e, "cannot read the vocabulary table");
            return -1;
        }
    }
    if (e->cpu_embed) {   // the input embedding stays in the file; keep it open and cache rows
        const Tensor* t = &e->t[0];
        e->fd = open(e->weights_path, O_RDONLY);
        e->row_cache = malloc((size_t)ROW_CACHE * (t->cols / 2 + t->cols / 16));
        e->row_tag = malloc(ROW_CACHE * sizeof(int32_t));
        if (e->fd < 0 || !e->row_cache || !e->row_tag) {
            fail(e, "cannot set up the embedding rows");
            return -1;
        }
        for (int i = 0; i < ROW_CACHE; i++) e->row_tag[i] = -1;
    }
    mm_tune(e);   // the batched prompt's product kernels, fastest per shape
    swar_tune(e);
    engine_reset(e);
    return 0;
}

// Dequantized embedding row of token `tok` ("q4row": nibbles as q4, then fp16 scales) into dst.
static int fetch_row(Engine* e, uint32_t tok, float* dst) {
    const Tensor* t = &e->t[0];
    uint32_t cols = t->cols, rb = cols / 2 + cols / 16, slot = tok & (ROW_CACHE - 1);
    uint8_t* row = e->row_cache + (size_t)slot * rb;
    if (e->row_tag[slot] != (int32_t)tok) {
        if (lseek(e->fd, (off_t)(t->off + (uint64_t)tok * rb), SEEK_SET) < 0 || read(e->fd, row, rb) != (ssize_t)rb) {
            fail(e, "cannot read the embedding row of token %u", tok);
            return -1;
        }
        e->row_tag[slot] = (int32_t)tok;
    }
    const uint16_t* sc = (const uint16_t*)(row + cols / 2);
    for (uint32_t b = 0; b < cols / 32; b++) {
        uint32_t h = sc[b], ex = (h >> 10) & 31;   // fp16 -> fp32 (normal numbers)
        uint32_t f = ex ? ((h & 0x8000u) << 16) | ((ex + 112u) << 23) | ((h & 0x3FFu) << 13) : (h & 0x8000u) << 16;
        float d;
        memcpy(&d, &f, 4);
        const uint8_t* q = row + b * 16;
        for (int j = 0; j < 16; j++) {
            dst[b * 32 + j] = ((float)(q[j] & 15) - 8.0f) * d;
            dst[b * 32 + 16 + j] = ((float)(q[j] >> 4) - 8.0f) * d;
        }
    }
    return 0;
}

uint64_t engine_kv_lend(Engine* e, uint64_t* size) {
    if (!e->kv_hi) return *size = 0, 0;
    e->ctx_limit = e->lend_ctx;
    *size = e->kv_split;
    return e->kv_hi;
}

void engine_kv_return(Engine* e) { e->ctx_limit = e->max_seq; }

void engine_reset(Engine* e) {
    for (uint32_t i = 0; i < e->n_layers; i++)
        if (!(e->full_mask >> i & 1)) {
            memset((void*)(uintptr_t)e->kc[i], 0, e->conv_n * 12);
            memset((void*)(uintptr_t)e->vc[i], 0, e->state_bytes);
        }
    __asm__ volatile("sfence" ::: "memory");
#ifdef GARLIC_SPEC
    spec_reset(e);
#endif
}

// ---- command buffer ----------------------------------------------------------------------------------------------

static void put(Engine* e, uint32_t w) {
    if (e->ndw < e->dcb_cap) e->dcb[e->ndw] = w;
    e->ndw++;
}

static void pkt3(Engine* e, uint32_t op, int n, int compute) {
    put(e, (3u << 30) | ((uint32_t)(n - 1) << 16) | (op << 8) | (compute ? 2 : 0));
}

static void set_sh(Engine* e, uint32_t reg, int n, const uint32_t* v) {
    pkt3(e, 0x76, n + 1, 0);
    put(e, reg);
    for (int i = 0; i < n; i++) put(e, v[i]);
}

static void set_sh1(Engine* e, uint32_t reg, uint32_t a) { set_sh(e, reg, 1, &a); }

// ACQUIRE_MEM over all memory: the cache actions in cntl (CP_COHER_CNTL), waiting for them to finish.
static void acquire(Engine* e, uint32_t cntl) {
    pkt3(e, 0x58, 6, 0);
    put(e, cntl);
    put(e, 0xFFFFFFFF);
    put(e, 0);
    put(e, 0);
    put(e, 0);
    put(e, 0x0A);
}

static void eop(Engine* e, uint32_t event, uint64_t addr, uint32_t data_sel, uint32_t value) {
    pkt3(e, 0x47, 5, 0);
    put(e, event);
    put(e, (uint32_t)addr);
    put(e, (uint32_t)((addr >> 32) & 0xFFFF) | (data_sel << 29));
    put(e, value);
    put(e, 0);
}

// Kernel arguments as the compiler lays out the C struct: 'p' pointer (8 bytes), 'u' uint, 'f' float.
static uint64_t args_v(Engine* e, const char* spec, ...) {
    uint32_t off = (e->kargs_used + 63) & ~63u;
    uint8_t* p = e->kargs + off;
    uint32_t pos = 0;
    va_list ap;
    va_start(ap, spec);
    for (const char* s = spec; *s; s++) {
        if (*s == 'p') {
            pos = (pos + 7) & ~7u;
            uint64_t v = va_arg(ap, uint64_t);
            memcpy(p + pos, &v, 8);
            pos += 8;
        } else if (*s == 'u') {
            uint32_t v = va_arg(ap, uint32_t);
            memcpy(p + pos, &v, 4);
            pos += 4;
        } else {
            float v = (float)va_arg(ap, double);
            memcpy(p + pos, &v, 4);
            pos += 4;
        }
    }
    va_end(ap);
    e->kargs_used = off + pos;
    return (uint64_t)(uintptr_t)p;
}

// args(e, "spec", values...): the spec must name every value passed, checked when compiling. A spec with one letter
// too many would make va_arg read stack garbage into a kernel's arguments.
#define NARGS_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, N, ...) N
#define NARGS(...) NARGS_(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define args(e, spec, ...) \
    (sizeof(char[sizeof(spec) - 1 == NARGS(__VA_ARGS__) ? 1 : -1]) ? args_v(e, spec, __VA_ARGS__) : 0)

static void dispatch(Engine* e, const GpuKernel* k, uint64_t kargs, uint32_t groups, uint32_t threads) {
    uint32_t pgm[2] = {(uint32_t)(k->addr >> 8), (uint32_t)((k->addr >> 40) & 0xFF)};
    uint32_t rsrc[2] = {k->rsrc1, k->rsrc2};
    uint32_t nthr[3] = {threads, 1, 1};
    uint32_t user[6] = {0, 0, 0, 0, (uint32_t)kargs, (uint32_t)(kargs >> 32)};
    set_sh(e, SH_PGM_LO, 2, pgm);
    set_sh(e, SH_PGM_RSRC1, 2, rsrc);
    set_sh(e, SH_NUM_THREAD_X, 3, nthr);
    set_sh(e, SH_USER_DATA_0, (int)k->user_sgprs, user);
    pkt3(e, 0x15, 4, 1);   // DISPATCH_DIRECT
    put(e, groups);
    put(e, 1);
    put(e, 1);
    put(e, 1);
    pkt3(e, 0x46, 1, 0);   // EVENT_WRITE CS_PARTIAL_FLUSH: wait for the dispatch to finish...
    put(e, EV_CS_PARTIAL_FLUSH);
    acquire(e, COHER_TCL1_KCACHE);   // ...and drop stale L1 / scalar cache lines before the next one
}

static uint64_t W(Engine* e, uint32_t idx) { return (uint64_t)(uintptr_t)e->model + e->t[idx].off; }

uint32_t engine_token_of(const Engine* e, uint32_t row) { return e->ids && row < e->vocab ? e->ids[row] : row; }

int32_t engine_row_of(const Engine* e, uint32_t token) {
    if (!e->ids) return token < e->vocab ? (int32_t)token : -1;
    for (uint32_t r = 0; r < e->vocab; r++)
        if (e->ids[r] == token) return (int32_t)r;
    return -1;
}

#ifdef GARLIC_SPEC
// (the token engine_run or engine_spec_step drew last has its row at hand)
static int32_t engine_row_of_fast(const Engine* e, uint32_t token) {
    if (e->ids && e->last_row < e->vocab && e->ids[e->last_row] == token) return (int32_t)e->last_row;
    return engine_row_of(e, token);
}
#endif

// ---- SWAR products on q4s weights (shaders/spec.h) ----------------------------------------------------------------
// Y[t] (+)= W X[t] for n rows of X (row-major, cols floats each) into Y rows of t->rows: the activations go to int8
// per block of 32 once (quant_x8), then the weights are read once per group of up to 4 rows (swar_mv<n>), with the
// variant swar_tune found fastest for this shape and count (split-K ones add their parts with sum_split).
// gate (0 = none): a GPU word; when it is not 0 the kernels do nothing (the draft chain after it stopped).
static const uint32_t swar_waves[5] = {4, 8, 16, 8, 8}, swar_split[5] = {1, 1, 1, 2, 4};

int swar_fixed = -1;   // (tests: >= 0 forces one variant for every shape and token count, so n tokens sum like 1)

static int swar_variant(Engine* e, const Tensor* t, uint32_t n) {
    if (swar_fixed >= 0) return swar_fixed;
    for (int i = 0; i < e->swar_n; i++)
        if (e->swar_tab[i].rows == t->rows && e->swar_tab[i].cols == t->cols && e->swar_tab[i].fmt == t->fmt)
            return e->swar_tab[i].v[n - 1];
    return t->rows >= 16384 ? 0 : 1;   // (before swar_tune: w4 for tall matrices, else w8)
}

static void swar_chunk(Engine* e, const Tensor* t, uint64_t w, int v, uint32_t c, uint32_t t0, uint64_t Y,
                       uint32_t acc, uint64_t gate) {
    uint32_t nb = t->cols / 32, S = swar_split[v];
    if (S > 1 && t->rows > e->yp_rows) S = 1, v = 1;
    uint64_t y = Y + (uint64_t)t0 * t->rows * 4;
    dispatch(e, (t->fmt == 8 ? e->k_swar3 : e->k_swar)[c - 1][v],
             args(e, "ppppppuuup", w, w + t->scales, e->x8 + (uint64_t)t0 * t->cols * 4, e->dx8 + (uint64_t)t0 * nb * 4,
                  e->init8 + (uint64_t)t0 * nb * 4, S > 1 ? e->yp : y, t->cols, t->rows, acc, gate),
             t->rows / 128 * S, 64 * swar_waves[v]);
    if (S > 1)
        dispatch(e, e->k_sumsplit, args(e, "ppuuup", e->yp, y, c * t->rows, S, acc, gate), (c * t->rows + 255) / 256, 256);
}

static void swar_mm(Engine* e, uint32_t idx, uint64_t X, uint64_t Y, uint32_t n, uint32_t acc, uint64_t gate) {
    const Tensor* t = &e->t[idx];
    uint64_t w = W(e, idx);
    uint32_t nb = t->cols / 32;
    dispatch(e, e->k_qx8, args(e, "ppppup", X, e->x8, e->dx8, e->init8, n * nb, gate), (n * nb + 255) / 256, 256);
    for (uint32_t t0 = 0; t0 < n; t0 += 4) {
        uint32_t c = n - t0 < 4 ? n - t0 : 4;
        swar_chunk(e, t, w, swar_variant(e, t, c), c, t0, Y, acc, gate);
    }
}

static void matvec(Engine* e, uint32_t idx, uint64_t x, uint64_t y) {
    const Tensor* t = &e->t[idx];
    uint64_t w = W(e, idx);
    if (t->fmt == 6 || t->fmt == 8) {
        swar_mm(e, idx, x, y, 1, 0, 0);
        return;
    }
    if (t->fmt == 3 || t->fmt == 5) {   // q4i / q3i: block sums of x, then one lane per row (4 waves per tile
        uint32_t waves = t->rows < 4096 || t->cols < 4096 ? 8 : 4;   // of 64 rows, 8 for small matrices)
        const GpuKernel* k = t->fmt == 5 ? (waves == 8 ? e->k_mv38 : e->k_mv3) : (waves == 8 ? e->k_mvi8 : e->k_mvi);
        dispatch(e, e->k_bsum, args(e, "ppu", x, e->bsum, t->cols), (t->cols / 32 + 255) / 256, 256);
        dispatch(e, k, args(e, "pppppu", w, w + t->scales, x, e->bsum, y, t->cols), t->rows / 64, 64 * waves);
        return;
    }
    dispatch(e, e->k_mv, args(e, "ppppu", w, w + t->scales, x, y, t->cols), t->rows / 4, 64);
}

static void step(Engine* e, uint32_t pos, uint32_t forced, uint32_t slot, int sample, float inv_temp,
                 uint32_t seed) {
    if (e->cpu_embed) {   // the CPU put this token's row in staging slot `slot`
        uint64_t src = e->staging + (uint64_t)slot * e->hidden * 4;
        dispatch(e, e->k_copy, args(e, "ppu", src, e->x, e->hidden), e->hidden / 256, 256);
    } else {
        uint64_t emb = W(e, 0);
        dispatch(e, e->k_embed, args(e, "pppupu", emb, emb + e->t[0].scales, e->last, forced, e->x, e->hidden),
                 1, 256);
    }
    for (uint32_t i = 0; i < e->n_layers; i++) {
        // full attention: attn_norm, mlp_norm, q_norm, k_norm, wqkv, wo, wgu, wdown
        // Gated DeltaNet:  attn_norm, mlp_norm, w_in, conv, A, dt_bias, gnorm, wout, wgu, wdown
        uint32_t b = e->layer_base[i];
        int full = e->full_mask >> i & 1;
        uint32_t mlp = b + (full ? 6 : 8);
        uint64_t kc = e->kc[i], vc = e->vc[i];
        dispatch(e, e->k_rmsnorm, args(e, "pppuf", e->x, W(e, b), e->xn, e->hidden, (double)e->eps), 1, 256);
        if (!full) {
            uint32_t vd = e->n_vh * e->dv;
            matvec(e, b + 2, e->xn, e->lin);
            dispatch(e, e->k_conv, args(e, "pppu", e->lin, W(e, b + 3), kc, (uint32_t)e->conv_n),
                     (uint32_t)(e->conv_n + 255) / 256, 256);
            dispatch(e, e->k_delta,
                     args(e, "ppppppuuf", e->lin, W(e, b + 4), W(e, b + 5), vc, W(e, b + 6), e->attn, e->n_kh,
                          e->n_vh, (double)e->eps),
                     e->n_vh, 256);
            matvec(e, b + 7, e->attn, e->o);
            (void)vd;
            goto mlp_block;
        }
        matvec(e, b + 4, e->xn, e->qkv);
        // attention in blocks of 64 positions (one group per block and kv head), then merge the blocks
        uint32_t blocks = pos / ATT_BLOCK + 1;
        double sc = 1.0 / sqrt((double)e->head_dim);
        if (e->tq) {   // TurboQuant cache: q rotated, k and v in 4 bits; the merge (always) rotates the output back
            // (tq_split: where positions 8192.. are; an older kernels.bin ignores the last argument)
            dispatch(e, e->k_qk_tq,
                     args(e, "pppppppuuufp", e->qkv, W(e, b + 2), W(e, b + 3), e->rope, kc, vc, e->tq_tab, pos,
                          e->n_heads, e->n_kv, (double)e->eps, e->tq_split),
                     e->n_heads + 2 * e->n_kv, 64);
            dispatch(e, e->k_attn_tq,
                     args(e, "pppppuuufp", e->qkv, kc, vc, e->attn_part, e->tq_tab, pos, e->n_heads, e->n_kv, sc,
                          e->tq_split),
                     blocks * e->n_kv, 256);
            dispatch(e, e->k_comb_tq, args(e, "pppuu", e->attn_part, e->attn, e->tq_tab, e->n_heads, blocks),
                     e->n_heads, 256);
        } else {
            // (Qwen3.5: one more argument, where the second half of the cache is)
            dispatch(e, e->k_qk,
                     e->qwen35 ? args(e, "ppppppuuufp", e->qkv, W(e, b + 2), W(e, b + 3), e->rope, kc, vc, pos,
                                      e->n_heads, e->n_kv, (double)e->eps, e->kv_split)
                               : args(e, "ppppppuuuf", e->qkv, W(e, b + 2), W(e, b + 3), e->rope, kc, vc, pos,
                                      e->n_heads, e->n_kv, (double)e->eps),
                     e->n_heads + 2 * e->n_kv, 64);
            // a context of one block is finished by attn_part itself
            dispatch(e, e->k_attn,
                     e->qwen35 ? args(e, "pppppuuufp", e->qkv, kc, vc, e->attn_part, e->attn, pos, e->n_heads, e->n_kv,
                                      sc, e->kv_split)
                               : args(e, "pppppuuuf", e->qkv, kc, vc, e->attn_part, e->attn, pos, e->n_heads, e->n_kv,
                                      sc),
                     blocks * e->n_kv, 256);
            if (blocks > 1)
                dispatch(e, e->k_attn_comb, args(e, "ppuu", e->attn_part, e->attn, e->n_heads, blocks), e->n_heads,
                         e->head_dim);
        }
        if (e->qwen35) {   // output gate: attn *= sigmoid(gate), the gate rows follow q, k, v in the projection
            uint32_t n = e->n_heads * e->head_dim;
            uint64_t gate = e->qkv + (uint64_t)(e->n_heads + 2 * e->n_kv) * e->head_dim * 4;
            dispatch(e, e->k_sig, args(e, "ppu", e->attn, gate, n), (n + 255) / 256, 256);
        }
        matvec(e, b + 5, e->attn, e->o);
    mlp_block:
        dispatch(e, e->k_add, args(e, "ppu", e->x, e->o, e->hidden), e->hidden / 256, 256);
        dispatch(e, e->k_rmsnorm, args(e, "pppuf", e->x, W(e, b + 1), e->xn, e->hidden, (double)e->eps), 1, 256);
        matvec(e, mlp, e->xn, e->gu);
        dispatch(e, e->k_silu, args(e, "ppu", e->gu, e->hmlp, e->inter), (e->inter + 255) / 256, 256);
        matvec(e, mlp + 1, e->hmlp, e->down);
        dispatch(e, e->k_add, args(e, "ppu", e->x, e->down, e->hidden), e->hidden / 256, 256);
    }
    if (sample) {
        dispatch(e, e->k_rmsnorm, args(e, "pppuf", e->x, W(e, 1), e->xn, e->hidden, (double)e->eps), 1, 256);
        matvec(e, e->head_idx, e->xn, e->logits);
        // min-p 0.05: drop tokens under 5 % of the top token's probability (gap T * ln 20 in logits)
        double min_gap = inv_temp > 0.0f ? 2.9957 / inv_temp : 0.0;
        dispatch(e, e->k_sample, args(e, "pufuppuf", e->logits, e->vocab, (double)inv_temp, seed, e->last, e->history, 0,
                                      min_gap),
                 1, 256);
    }
}

// ---- the prompt in batches (shaders/prefill.h) -------------------------------------------------------------------
// The same operations as n calls of step(), layer by layer for all the tokens: each matrix is read and unpacked once
// for up to 32 tokens (matrix x matrix), each DeltaNet state is loaded and stored once per batch, and the attention
// runs token by token over the cache with the decode kernels. Residual adds are folded into the products.

// Y (+)= W X for n rows of X (q4i or q3i), with the kernel mm_tune found fastest for this shape (else the plain one).
static void mm_with(Engine* e, const GpuKernel* k, uint32_t r, uint32_t idx, uint64_t X, uint64_t Y, uint32_t n,
                    uint32_t accumulate) {
    const Tensor* t = &e->t[idx];
    uint64_t w = W(e, idx);
    dispatch(e, k, args(e, "ppppuuuu", w, w + t->scales, X, Y, t->cols, t->rows, n, accumulate),
             t->rows / 64 * ((n + r - 1) / r), 256);
}

static void mmr(Engine* e, uint32_t idx, uint64_t X, uint64_t Y, uint32_t n, uint32_t accumulate) {
    const Tensor* t = &e->t[idx];
    if (t->fmt == 6 || t->fmt == 8) {
        swar_mm(e, idx, X, Y, n, accumulate, 0);
        return;
    }
    for (int i = 0; i < e->pf_nmm; i++)
        if (e->pf_mm[i].fmt == t->fmt && e->pf_mm[i].rows == t->rows && e->pf_mm[i].cols == t->cols) {
            int s = n <= 16;
            mm_with(e, e->pf_mm[i].k[s], e->pf_mm[i].r[s], idx, X, Y, n, accumulate);
            return;
        }
    mm_with(e, t->fmt == 5 ? e->k_mm3r : e->k_mm4r, 16, idx, X, Y, n, accumulate);
}

// n (<= PF_ROWS) tokens at positions pos0.., their embedding rows at `stage` (the CPU put them there, cpu_embed) or
// dequantized from the table in GPU memory (the 4B: one embed dispatch per token, as step() does); with `sample`, the
// token after the last one is drawn as in step().
static void prefill(Engine* e, uint64_t stage, const uint32_t* tokens, uint32_t n, uint32_t pos0, int sample,
                    float inv_temp, uint32_t seed) {
    uint32_t H = e->hidden;
    if (e->cpu_embed) {
        dispatch(e, e->k_copy, args(e, "ppu", stage, e->pf_x, n * H), n * H / 256, 256);
    } else {
        uint64_t emb = W(e, 0);
        for (uint32_t r = 0; r < n; r++)
            dispatch(e, e->k_embed, args(e, "pppupu", emb, emb + e->t[0].scales, e->last, tokens[r],
                                         e->pf_x + (uint64_t)r * H * 4, H), 1, 256);
    }
    for (uint32_t i = 0; i < e->n_layers; i++) {
        uint32_t b = e->layer_base[i];
        int full = e->full_mask >> i & 1;
        uint32_t mlp = b + (full ? 6 : 8);
        uint64_t kc = e->kc[i], vc = e->vc[i];
        dispatch(e, e->k_rmsr, args(e, "pppuf", e->pf_x, W(e, b), e->pf_xn, H, (double)e->eps), n, 256);
        if (!full) {   // Gated DeltaNet: input projection, conv + delta rule over the tokens in order, out projection
            uint32_t ld = e->t[b + 2].rows;
            mmr(e, b + 2, e->pf_xn, e->pf_a, n, 0);
            dispatch(e, e->k_convr, args(e, "pppuuu", e->pf_a, W(e, b + 3), kc, (uint32_t)e->conv_n, ld, n),
                     (uint32_t)(e->conv_n + 255) / 256, 256);
            dispatch(e, e->k_deltar,
                     args(e, "ppppppuufuu", e->pf_a, W(e, b + 4), W(e, b + 5), vc, W(e, b + 6), e->pf_h, e->n_kh,
                          e->n_vh, (double)e->eps, ld, n),
                     e->n_vh, 256);
            mmr(e, b + 7, e->pf_h, e->pf_x, n, 1);
        } else {       // full attention: q | k | v | gate for all, then norms + RoPE + cache and attention per token
            uint32_t ld = e->t[b + 4].rows, nq = e->n_heads * e->head_dim, G = e->n_heads + 2 * e->n_kv;
            double sc = 1.0 / sqrt((double)e->head_dim);
            mmr(e, b + 4, e->pf_xn, e->pf_a, n, 0);
            for (uint32_t r = 0; r < n; r++) {
                uint64_t q = e->pf_a + (uint64_t)r * ld * 4;
                if (e->tq)
                    dispatch(e, e->k_qk_tq,
                             args(e, "pppppppuuufp", q, W(e, b + 2), W(e, b + 3), e->rope, kc, vc, e->tq_tab, pos0 + r,
                                  e->n_heads, e->n_kv, (double)e->eps, e->tq_split),
                             G, 64);
                else
                    dispatch(e, e->k_qk,
                             args(e, "ppppppuuufp", q, W(e, b + 2), W(e, b + 3), e->rope, kc, vc, pos0 + r, e->n_heads,
                                  e->n_kv, (double)e->eps, e->kv_split),
                             G, 64);
            }
            for (uint32_t r = 0; r < n; r++) {   // token r sees positions up to its own (all written above)
                uint64_t q = e->pf_a + (uint64_t)r * ld * 4, out = e->pf_h + (uint64_t)r * nq * 4;
                uint32_t pos = pos0 + r, blocks = pos / ATT_BLOCK + 1;
                if (e->tq) {
                    dispatch(e, e->k_attn_tq,
                             args(e, "pppppuuufp", q, kc, vc, e->attn_part, e->tq_tab, pos, e->n_heads, e->n_kv, sc,
                                  e->tq_split),
                             blocks * e->n_kv, 256);
                    dispatch(e, e->k_comb_tq, args(e, "pppuu", e->attn_part, out, e->tq_tab, e->n_heads, blocks),
                             e->n_heads, 256);
                } else {
                    dispatch(e, e->k_attn,
                             args(e, "pppppuuufp", q, kc, vc, e->attn_part, out, pos, e->n_heads, e->n_kv, sc,
                                  e->kv_split),
                             blocks * e->n_kv, 256);
                    if (blocks > 1)
                        dispatch(e, e->k_attn_comb, args(e, "ppuu", e->attn_part, out, e->n_heads, blocks), e->n_heads,
                                 e->head_dim);
                }
            }
            uint64_t gate = e->pf_a + (uint64_t)(e->n_heads + 2 * e->n_kv) * e->head_dim * 4;
            dispatch(e, e->k_sigr, args(e, "ppuuu", e->pf_h, gate, nq, ld, n), (n * nq + 255) / 256, 256);
            mmr(e, b + 5, e->pf_h, e->pf_x, n, 1);
        }
        dispatch(e, e->k_rmsr, args(e, "pppuf", e->pf_x, W(e, b + 1), e->pf_xn, H, (double)e->eps), n, 256);
        mmr(e, mlp, e->pf_xn, e->pf_a, n, 0);
        dispatch(e, e->k_silur, args(e, "ppuu", e->pf_a, e->pf_h, e->inter, n), (n * e->inter + 255) / 256, 256);
        mmr(e, mlp + 1, e->pf_h, e->pf_x, n, 1);
    }
    if (sample) {   // as step(): the last token's final norm, the logits and the pick
        uint64_t xl = e->pf_x + (uint64_t)(n - 1) * H * 4;
        dispatch(e, e->k_rmsnorm, args(e, "pppuf", xl, W(e, 1), e->xn, H, (double)e->eps), 1, 256);
        matvec(e, e->head_idx, e->xn, e->logits);
        double min_gap = inv_temp > 0.0f ? 2.9957 / inv_temp : 0.0;
        dispatch(e, e->k_sample, args(e, "pufuppuf", e->logits, e->vocab, (double)inv_temp, seed, e->last, e->history, 0,
                                      min_gap),
                 1, 256);
    }
}

// submit = submit_go (the command buffer to the GPU) + submit_wait (until its end-of-pipe label); between the two the
// CPU may prepare the next submission's inputs (not the command buffer, which the GPU is reading).
static int submit_go(Engine* e) {
    eop(e, EV_CACHE_FLUSH_AND_INV_TS, (uint64_t)(uintptr_t)(e->stamps + 1), 3, 0);
    eop(e, EV_CACHE_FLUSH_AND_INV_TS | EOP_WB_INV_L1_L2, (uint64_t)(uintptr_t)e->label, 1, LABEL_DONE);
    while (e->ndw % 8) put(e, 0x80000000);
    if (e->ndw > e->dcb_cap || e->kargs_used > e->kargs_cap) {
        fail(e, "command buffer overflow (%u dwords, %u argument bytes)", e->ndw, e->kargs_used);
        return -1;
    }
    *e->label = 0;
    __asm__ volatile("sfence" ::: "memory");   // the GPU must see the command buffer the CPU just wrote
    void* dcbs[1] = {e->dcb};
    uint32_t sizes[1] = {e->ndw * 4};
    int r = sceGnmSubmitCommandBuffers(1, dcbs, sizes, NULL, NULL);
    if (r) {
        fail(e, "sceGnmSubmitCommandBuffers -> 0x%08x", (unsigned)r);
        return -1;
    }
    sceGnmSubmitDone();
    e->submit_tsc = sceKernelReadTsc();
    return 0;
}

static int submit_wait(Engine* e) {
    uint64_t t0 = e->submit_tsc;
    // Spin for the first 20 ms (T3 and the voice need the answer at once), then sleep between checks. Reading the
    // results as soon as the label flips relies on the L2 write-back before it (EOP_WB_INV_L1_L2): without it the
    // GPU's last writes could still sit in its L2.
    while (*e->label != LABEL_DONE) {
        uint64_t dt = sceKernelReadTsc() - t0;
        if (dt > e->tsc_freq * 10) {
            fail(e, "GPU did not finish within 10 s");
            return -1;
        }
        if (dt > e->tsc_freq / 50) sceKernelUsleep(200);
        else __asm__ volatile("pause" ::: "memory");
    }
    __asm__ volatile("mfence" ::: "memory");
    __asm__ volatile("lfence" ::: "memory");
    e->last_gpu_seconds = (double)(e->stamps[1] - e->stamps[0]) / 100.0e6;   // GPU timestamps tick at 100 MHz
    return 0;
}

static int submit(Engine* e) { return submit_go(e) || submit_wait(e) ? -1 : 0; }

static void begin(Engine* e) {
    e->ndw = 0;
    e->kargs_used = 0;
    for (int i = 0; i < 256; i++) put(e, e->init_state[i]);
    uint32_t all[2] = {0xFFFFFFFF, 0xFFFFFFFF}, zero3[3] = {0, 0, 0};
    set_sh1(e, SH_RESOURCE_LIMITS, 0);
    set_sh(e, SH_STATIC_THREAD_MGMT_SE0, 2, all);
    set_sh1(e, SH_TMPRING_SIZE, 0);
    set_sh(e, SH_START_X, 3, zero3);
    acquire(e, COHER_L2_TCL1_KCACHE);   // see COHER_L2_TCL1_KCACHE
    eop(e, EV_BOTTOM_OF_PIPE_TS, (uint64_t)(uintptr_t)e->stamps, 3, 0);
}

// Start-up check of what the engine relies on, with the "add" kernel every kernels.bin has: dispatches that read what
// the previous one wrote (L1 / scalar cache dropped in between), results written back for the CPU before the label, and
// a second submission after the CPU rewrote the inputs at the same addresses (L2 invalidated when a submission starts).
// Only tested on one console (FW 11.52); on another model or firmware a difference there would give nonsense answers
// instead of an error. Takes well under a millisecond; the buffers are given back.
// /data/garlic_selftest.txt = 0 skips it.
static int gpu_selftest(Engine* e) {
    size_t fs;
    char* off = (char*)read_file("/data/garlic_selftest.txt", &fs);
    int skip = off && fs && off[0] == '0';
    free(off);
    if (skip) {
        say("GPU self-test: skipped (/data/garlic_selftest.txt)");
        return 0;
    }
    enum { N = 16384 };   // 64 groups of 256: every compute unit gets some
    uint64_t mark = e->next;
    float* a = (float*)(uintptr_t)gpu_alloc(e, N * 4);
    float* b = (float*)(uintptr_t)gpu_alloc(e, N * 4);
    if (!a || !b) {
        fail(e, "GPU self-test: no memory");
        return -1;
    }
    uint64_t pa = (uint64_t)(uintptr_t)a, pb = (uint64_t)(uintptr_t)b;
    const char* what = NULL;
    uint32_t bad = 0, first = 0;
    float got = 0.0f, want = 0.0f;
    double gpu = 0.0;
    // 1: a += b; b += a; a += b  ->  a = 2 a0 + 3 b0, b = a0 + 2 b0 (small halves and integers: exact in fp32)
    for (uint32_t i = 0; i < N; i++) a[i] = 0.5f * i, b[i] = 1.0f + (i & 7);
    __asm__ volatile("sfence" ::: "memory");
    begin(e);
    dispatch(e, e->k_add, args(e, "ppu", pa, pb, (uint32_t)N), N / 256, 256);
    dispatch(e, e->k_add, args(e, "ppu", pb, pa, (uint32_t)N), N / 256, 256);
    dispatch(e, e->k_add, args(e, "ppu", pa, pb, (uint32_t)N), N / 256, 256);
    if (submit(e)) {
        e->next = mark;
        return -1;   // (fail() said why: GPU did not finish, or the submission was refused)
    }
    gpu += e->last_gpu_seconds;
    for (uint32_t i = 0; i < N && !what; i++) {
        float a0 = 0.5f * i, b0 = 1.0f + (i & 7);
        if (a[i] != 2 * a0 + 3 * b0) what = "chained dispatches", first = i, got = a[i], want = 2 * a0 + 3 * b0;
        else if (b[i] != a0 + 2 * b0) what = "chained dispatches", first = i, got = b[i], want = a0 + 2 * b0;
    }
    for (uint32_t i = 0; i < N; i++) {
        float a0 = 0.5f * i, b0 = 1.0f + (i & 7);
        bad += a[i] != 2 * a0 + 3 * b0 || b[i] != a0 + 2 * b0;
    }
    // 2: new inputs at the same addresses, one more submission: a += b  ->  a = a0 + b0
    if (!what) {
        for (uint32_t i = 0; i < N; i++) a[i] = 3.0f * (i & 1023), b[i] = -0.25f * (i & 15);
        __asm__ volatile("sfence" ::: "memory");
        begin(e);
        dispatch(e, e->k_add, args(e, "ppu", pa, pb, (uint32_t)N), N / 256, 256);
        if (submit(e)) {
            e->next = mark;
            return -1;
        }
        gpu += e->last_gpu_seconds;
        for (uint32_t i = 0; i < N; i++) {
            float w = 3.0f * (i & 1023) - 0.25f * (i & 15);
            if (a[i] != w) {
                if (!what) what = "stale data in a new submission", first = i, got = a[i], want = w;
                bad++;
            }
        }
    }
    e->next = mark;   // give the two buffers back
    if (what) {
        // (e->error holds 160 bytes: the longest case is ~120)
        fail(e, "GPU self-test failed (%s): %u of %u values wrong, first [%u] = %g, expected %g", what, bad,
             (unsigned)N, first, (double)got, (double)want);
        return -1;
    }
    say("GPU self-test: ok (%u values, 4 dispatches in 2 submissions, %.3f ms GPU)", (unsigned)N, gpu * 1e3);
    return 0;
}

// The Gated DeltaNet layers' state (conv and delta rule), copied on the GPU to buf (save 1) or back from it (save 0).
// With the attention cache, which is written by position, that is all a conversation needs to go back to an earlier
// position: main.c takes a turn's reasoning out of the history this way ("Pensar"). 50.25 MiB for the 9B.
uint64_t engine_state_size(const Engine* e) {
    uint64_t n = 0;
    for (uint32_t i = 0; i < e->n_layers; i++)
        if (!(e->full_mask >> i & 1)) n += ((e->conv_n * 12 + 255) & ~255ull) + ((e->state_bytes + 255) & ~255ull);
#ifdef GARLIC_SPEC
    if (spec_carry()) n += (uint64_t)e->hidden * 4;   // the MTP head's carry (h before the next token)
#endif
    return n;
}

int engine_state_copy(Engine* e, uint64_t buf, int save) {
    if (!e->qwen35 || !e->k_copy) return -1;
    begin(e);
    uint64_t p = buf;
    for (uint32_t i = 0; i < e->n_layers; i++) {
        if (e->full_mask >> i & 1) continue;
        const uint64_t part[2][2] = {{e->kc[i], e->conv_n * 12}, {e->vc[i], e->state_bytes}};
        for (int j = 0; j < 2; j++) {
            uint32_t n = (uint32_t)(part[j][1] / 4);
            dispatch(e, e->k_copy, args(e, "ppu", save ? part[j][0] : p, save ? p : part[j][0], n), (n + 255) / 256, 256);
            p += (part[j][1] + 255) & ~255ull;
        }
    }
#ifdef GARLIC_SPEC
    if (spec_carry())
        dispatch(e, e->k_copy, args(e, "ppu", save ? spec_carry() : p, save ? p : spec_carry(), e->hidden), e->hidden / 256,
                 256);
#endif
    return submit(e);
}

int engine_run(Engine* e, const uint32_t* tokens, int n, int pos0, float temperature, uint32_t seed, uint32_t* next) {
    if (pos0 + n > (int)e->max_seq || pos0 + n > (int)e->ctx_limit) {
        fail(e, "context full (%d tokens)", pos0 + n);
        return -1;
    }
    float inv_temp = temperature > 0.0f ? 1.0f / temperature : 0.0f;
    double gpu = 0.0;
    // 16K: positions past lend_ctx are in kv_hi, so S3Gen's weights, if they were kept there after voice mode, are
    // about to be overwritten (voice mode reads them again when it starts)
    if (e->tq_split && pos0 + n > (int)e->lend_ctx && e->s3.loaded) e->s3.loaded = 0;
    int batched = e->pf && !e->pf_off && n >= 4;
#ifdef GARLIC_SPEC
    if (spec_active(e)) batched = 1;   // (the MTP head needs the pairs of every position: engine_spec.inc)
#endif
    if (batched) {   // the prompt in batches of PF_ROWS tokens, one submission each
        // two halves of staging: the CPU reads the next batch's embedding rows (from the disk when they are not
        // cached, about as long per row as the GPU takes for a token) while the GPU runs this batch. The first batch
        // is of 8 tokens, so that even a short prompt (voice mode: 20-30 tokens) has most of its rows read meanwhile;
        // with the r8 products a batch of 8 costs about the same per token as one of 32.
#define PF_FIRST 8
#define PF_SIZE(i) ((i) == 0 && n > PF_FIRST ? PF_FIRST : n - (i) < PF_ROWS ? n - (i) : PF_ROWS)
        uint64_t rowb = (uint64_t)e->hidden * 4;
        for (int i = 0; e->cpu_embed && i < PF_SIZE(0); i++)
            if (fetch_row(e, tokens[i], (float*)(uintptr_t)(e->staging + (uint64_t)i * rowb))) return -1;
        for (int i0 = 0, c = 0, k; i0 < n; i0 += k, c ^= 1) {
            k = PF_SIZE(i0);
            int i1 = i0 + k, k1 = i1 < n ? PF_SIZE(i1) : 0;
            begin(e);
            prefill(e, e->staging + (uint64_t)c * PF_ROWS * rowb, tokens + i0, (uint32_t)k, (uint32_t)(pos0 + i0),
                    i0 + k == n, inv_temp, seed);
#ifdef GARLIC_SPEC
            if (spec_active(e)) spec_prefill(e, e->staging + (uint64_t)c * PF_ROWS * rowb, (uint32_t)k, (uint32_t)(pos0 + i0));
#endif
            if (submit_go(e)) return -1;
            uint64_t next_stage = e->staging + (uint64_t)(c ^ 1) * PF_ROWS * rowb;
            for (int i = 0; e->cpu_embed && i < k1; i++)
                if (fetch_row(e, tokens[i1 + i], (float*)(uintptr_t)(next_stage + (uint64_t)i * rowb))) {
                    submit_wait(e);
                    return -1;
                }
            if (submit_wait(e)) return -1;
            gpu += e->last_gpu_seconds;
        }
#undef PF_SIZE
#undef PF_FIRST
        e->last_gpu_seconds = e->pf_seconds = gpu;
        e->last_row = *(volatile uint32_t*)(uintptr_t)e->last;
        *next = engine_token_of(e, e->last_row);
        return 0;
    }
    for (int i0 = 0; i0 < n; i0 += STEPS_PER_SUBMIT) {
        int i1 = i0 + STEPS_PER_SUBMIT < n ? i0 + STEPS_PER_SUBMIT : n;
        begin(e);
        if (e->cpu_embed)
            for (int i = i0; i < i1; i++)
                if (fetch_row(e, tokens[i], (float*)(uintptr_t)(e->staging + (uint64_t)(i - i0) * e->hidden * 4)))
                    return -1;
        for (int i = i0; i < i1; i++) step(e, pos0 + i, tokens[i], (uint32_t)(i - i0), i == n - 1, inv_temp, seed);
        if (submit(e)) return -1;
        gpu += e->last_gpu_seconds;
    }
    e->last_gpu_seconds = gpu;
    e->last_row = *(volatile uint32_t*)(uintptr_t)e->last;
    *next = engine_token_of(e, e->last_row);
    return 0;
}

// For each matrix shape of the model, time the product kernels of shaders/prefill.h for batches of 32 and of 16
// tokens (3 runs each, the input a fixed pattern) and keep the fastest whose result matches mm_q*i_rows (sampled,
// relative difference under 1e-4). ~1 s at start-up; one log line per shape.
static void mm_tune(Engine* e) {
    static const char* names[2][5] = {{"mm_q4i_rows", "mm_q4i_r8", "mm_q4i_r32", "mm_q4i_w16", "mm_q4i_w32"},
                                      {"mm_q3i_rows", "mm_q3i_r8", "mm_q3i_r32", "mm_q3i_w16", "mm_q3i_w32"}};
    static const uint32_t rr[5] = {16, 8, 32, 16, 32};
    e->pf_nmm = 0;
    if (!e->pf) return;
    for (uint32_t i = 0; i < e->n_layers && e->pf_nmm < 8; i++) {
        uint32_t b = e->layer_base[i], full = e->full_mask >> i & 1;
        uint32_t m[4] = {b + (full ? 4 : 2), b + (full ? 5 : 7), b + (full ? 6 : 8), b + (full ? 7 : 9)};
        for (int j = 0; j < 4 && e->pf_nmm < 8; j++) {
            const Tensor* t = &e->t[m[j]];
            if (t->fmt == 6 || t->fmt == 8) continue;   // (q4s / q3s: swar_tune)
            int seen = 0;
            for (int s = 0; s < e->pf_nmm; s++)
                seen |= e->pf_mm[s].fmt == t->fmt && e->pf_mm[s].rows == t->rows && e->pf_mm[s].cols == t->cols;
            if (seen) continue;
            int q3 = t->fmt == 5;
            float* X = (float*)(uintptr_t)e->pf_h;
            for (uint32_t k = 0; k < PF_ROWS * t->cols; k++) X[k] = (float)(((k * 2654435761u) >> 16) & 1023) / 512.0f - 1.0f;
            __asm__ volatile("sfence" ::: "memory");
            char line[400];
            int o = snprintf(line, sizeof(line), "mm tune %ux%u %s:", t->rows, t->cols, q3 ? "q3i" : "q4i");
            int ent = e->pf_nmm++;
            e->pf_mm[ent].fmt = t->fmt, e->pf_mm[ent].rows = t->rows, e->pf_mm[ent].cols = t->cols;
            for (int s = 0; s < 2; s++) {
                uint32_t n = s ? 16 : PF_ROWS, nsamp = 0;
                static float ref[2048];
                volatile float* Y = (volatile float*)(uintptr_t)e->pf_a;
                uint64_t ny = (uint64_t)n * t->rows, stride = ny / 2048 + 1;
                double best = 1e9;
                e->pf_mm[ent].k[s] = q3 ? e->k_mm3r : e->k_mm4r, e->pf_mm[ent].r[s] = 16;
                o += snprintf(line + o, sizeof(line) - o, " n=%u", n);
                for (int v = 0; v < 5; v++) {
                    const GpuKernel* k = NULL;
                    for (int q = 0; q < e->nk; q++)
                        if (!strcmp(e->k[q].name, names[q3][v])) k = &e->k[q];
                    if (!k) continue;
                    begin(e);
                    for (int rep = 0; rep < 3; rep++) mm_with(e, k, rr[v], m[j], e->pf_h, e->pf_a, n, 0);
                    if (submit(e)) return;
                    double dt = e->last_gpu_seconds / 3;
                    float md = 0, mx = 0;
                    uint32_t c = 0;
                    for (uint64_t y = 0; y < ny && c < 2048; y += stride, c++) {
                        float a = Y[y];
                        if (v == 0) ref[c] = a;
                        float d = a - ref[c] < 0 ? ref[c] - a : a - ref[c], r = ref[c] < 0 ? -ref[c] : ref[c];
                        if (!(d <= md)) md = d;   // (a NaN counts as wrong)
                        if (r > mx) mx = r;
                    }
                    if (v == 0) nsamp = c;
                    int ok = c == nsamp && md <= 1e-4f * (mx > 0 ? mx : 1);
                    o += snprintf(line + o, sizeof(line) - o, " %s %.2f%s", names[q3][v] + 7, dt * 1e3, ok ? "" : " WRONG");
                    if (ok && dt < best) best = dt, e->pf_mm[ent].k[s] = k, e->pf_mm[ent].r[s] = rr[v];
                }
                o += snprintf(line + o, sizeof(line) - o, " ms -> %s;", e->pf_mm[ent].k[s]->name + 7);
            }
            say("%s", line);
        }
    }
}

// For each q4s matrix shape (and the extra shapes spec_load adds) and 1..4 tokens, time the SWAR variants
// (4 / 8 / 16 waves per tile, split-K in 2 or 4 parts; 3 runs each, the input a fixed pattern) and keep the fastest
// whose result matches the first one (sampled, relative difference under 1e-4). One log line per shape.
// t: the shape (rows may be the first rows of a tensor: the draft's 32K rows of lm_head), w its weights. A shape too
// tall for the test buffers (4 rows of output in pf_a: lm_head) is timed on its first rows.
static void swar_tune_shape(Engine* e, const Tensor* shape, uint64_t w) {
    static const char* names[5] = {"w4", "w8", "w16", "s2", "s4"};
    Tensor t = *shape;
    for (int i = 0; i < e->swar_n; i++)
        if (e->swar_tab[i].rows == t.rows && e->swar_tab[i].cols == t.cols && e->swar_tab[i].fmt == t.fmt) return;
    if (e->swar_n >= (int)(sizeof(e->swar_tab) / sizeof(e->swar_tab[0])) || !e->pf_h || !e->pf_a) return;
    if ((uint64_t)4 * t.cols > pf_h_floats) return;
    if (t.fmt == 8 && !e->k_swar3[0][0]) return;
    int ent = e->swar_n++;
    e->swar_tab[ent].rows = t.rows, e->swar_tab[ent].cols = t.cols, e->swar_tab[ent].fmt = t.fmt;
    while ((uint64_t)4 * t.rows > pf_a_floats) t.rows = (t.rows / 2) & ~127u;   // (timed on the first rows)
    float* X = (float*)(uintptr_t)e->pf_h;
    for (uint32_t k = 0; k < 4 * t.cols; k++) X[k] = (float)(((k * 2654435761u) >> 16) & 1023) / 512.0f - 1.0f;
    __asm__ volatile("sfence" ::: "memory");
    char line[400];
    int o = snprintf(line, sizeof(line), "swar tune %s %ux%u%s:", t.fmt == 8 ? "q3s" : "q4s", e->swar_tab[ent].rows, t.cols,
                     t.rows != e->swar_tab[ent].rows ? " (timed on the first rows)" : "");
    for (uint32_t n = 1; n <= 4; n++) {
        static float ref[2048];
        volatile float* Y = (volatile float*)(uintptr_t)e->pf_a;
        uint64_t ny = (uint64_t)n * t.rows, stride = ny / 2048 + 1;
        double best = 1e9;
        uint32_t nsamp = 0;
        e->swar_tab[ent].v[n - 1] = 1;
        o += snprintf(line + o, sizeof(line) - o, " n=%u", n);
        for (int v = 0; v < 5; v++) {
            if (swar_split[v] > 1 && t.rows > e->yp_rows) continue;
            if (!(t.fmt == 8 ? e->k_swar3 : e->k_swar)[n - 1][v]) continue;   // (q3s has no 16-wave variant)
            begin(e);
            dispatch(e, e->k_qx8, args(e, "ppppup", e->pf_h, e->x8, e->dx8, e->init8, n * (t.cols / 32), (uint64_t)0),
                     (n * (t.cols / 32) + 255) / 256, 256);
            for (int rep = 0; rep < 3; rep++) swar_chunk(e, &t, w, v, n, 0, e->pf_a, 0, 0);
            if (submit(e)) return;
            double dt = e->last_gpu_seconds / 3;
            float md = 0, mx = 0;
            uint32_t c = 0;
            for (uint64_t y = 0; y < ny && c < 2048; y += stride, c++) {
                float a = Y[y];
                if (v == 0) ref[c] = a;
                float d = a - ref[c] < 0 ? ref[c] - a : a - ref[c], r = ref[c] < 0 ? -ref[c] : ref[c];
                if (!(d <= md)) md = d;
                if (r > mx) mx = r;
            }
            if (v == 0) nsamp = c;
            int ok = c == nsamp && md <= 1e-4f * (mx > 0 ? mx : 1);
            o += snprintf(line + o, sizeof(line) - o, " %s %.3f%s", names[v], dt * 1e3, ok ? "" : " WRONG");
            if (ok && dt < best) best = dt, e->swar_tab[ent].v[n - 1] = (uint8_t)v;
        }
        o += snprintf(line + o, sizeof(line) - o, " -> %s;", names[e->swar_tab[ent].v[n - 1]]);
    }
    say("%s", line);
}

static void swar_tune(Engine* e) {
    if (!e->swar) return;
    for (uint32_t i = 0; i < e->n_tensors; i++)
        if (e->t[i].fmt == 6 || e->t[i].fmt == 8) swar_tune_shape(e, &e->t[i], W(e, i));
}



// ---- speech to text: Parakeet TDT 0.6B v3 -----------------------------------------------------------------------
// Mel features on the CPU, encoder on the GPU in one submission, then the TDT greedy loop, one GPU step per decision
// (prediction network when a token was emitted, joint, argmax).

static const AsrTensor* at(Engine* e, const char* name) {
    for (uint32_t i = 0; i < e->asr.nt; i++)
        if (!strcmp(e->asr.t[i].name, name)) return &e->asr.t[i];
    return NULL;
}

static uint64_t AW(Engine* e, const char* name) {
    const AsrTensor* t = at(e, name);
    return t ? e->asr.model + t->off : 0;
}

static uint64_t AWf(Engine* e, const char* fmt, int i) {
    char n[32];
    snprintf(n, sizeof(n), fmt, i);
    return AW(e, n);
}

int asr_init(Engine* e, const char* dir) {
    Asr* a = &e->asr;
    memset(a, 0, sizeof(*a));
    char path[256];
    size_t size;
    snprintf(path, sizeof(path), "%s/asr.idx", dir);
    uint8_t* d = read_file(path, &size);
    if (!d || memcmp(d, "PASR", 4)) {
        free(d);
        return -1;
    }
    memcpy(&a->nt, d + 4, 4);
    memcpy(a->path, d + 8, 128);
    a->path[128] = 0;
    weights_locate(a->path);
    memcpy(&a->size, d + 136, 8);
    a->t = calloc(a->nt, sizeof(AsrTensor));
    for (uint32_t i = 0; i < a->nt; i++) {
        const uint8_t* r = d + 144 + i * 64;
        memcpy(a->t[i].name, r, 32);
        a->t[i].name[31] = 0;
        memcpy(&a->t[i].off, r + 32, 8);
        memcpy(&a->t[i].size, r + 40, 8);
        memcpy(&a->t[i].rows, r + 48, 4);
        memcpy(&a->t[i].cols, r + 52, 4);
        memcpy(&a->t[i].scales, r + 56, 8);
    }
    free(d);
    snprintf(path, sizeof(path), "%s/asr_vocab.bin", dir);
    a->vocab = read_file(path, &size);
    if (!a->vocab || memcmp(a->vocab, "PVOC", 4)) return -1;
    a->piece = malloc(8192 * sizeof(uint32_t));
    for (uint32_t i = 0, o = 8; i < 8192; i++) {
        a->piece[i] = o;
        o += 2 + (a->vocab[o] | a->vocab[o + 1] << 8);
    }

    // memory: weights, then a scratch arena sized to what is left (143360 bytes per encoder frame)
    uint64_t in_gpu = a->size;
    uint64_t room = e->mem_size - ((e->next + 255) & ~255ull);
    if (room < a->size + (256 << 10) + 400 * 143360ull) {
        // no room in GPU memory (the 9B with T3): the weights in flexible (system) memory, where the GPU reads them
        // almost as fast; Python borrows that memory (asr_release)
        void* fm = NULL;
        uint64_t len = (a->size + 0x3FFF) & ~0x3FFFull;
        int r = sceKernelMapFlexibleMemory(&fm, len, 0x33, 0);
        say("asr: weights (%u MiB) in flexible memory -> 0x%08x", (unsigned)(len >> 20), (unsigned)r);
        if (r == 0) a->model = (uint64_t)(uintptr_t)fm, in_gpu = 0, a->flex_len = len;
    }
    uint64_t free_bytes = e->mem_size - ((e->next + 255) & ~255ull);
    uint64_t small = 256 << 10;
    if (free_bytes < in_gpu + small + 50 * 143360ull) return -1;   // not even 4 s of audio
    uint64_t mf = (free_bytes - in_gpu - small) / 143360ull;
    a->max_frames = (uint32_t)(mf > 400 ? 400 : mf);
    if (in_gpu) a->model = gpu_alloc(e, a->size);
    int f = open(a->path, O_RDONLY);
    if (f < 0) return -1;
    uint64_t done = 0;
    while (done < a->size) {
        uint64_t n = a->size - done < (32ull << 20) ? a->size - done : 32ull << 20;
        ssize_t r = read(f, (uint8_t*)(uintptr_t)a->model + done, n);
        if (r <= 0) break;
        done += r;
    }
    close(f);
    if (done != a->size) return -1;
    a->mel = malloc(128 * 257 * 4);
    memcpy(a->mel, (void*)(uintptr_t)AW(e, "mel"), 128 * 257 * 4);

    uint32_t MF = a->max_frames;
    uint64_t T2max = 2ull * MF + 2;
    a->sub_a = gpu_alloc(e, T2max * 32 * 256 * 4);      // conv1+dw1 out, then dw2 out; then the encoder buffers
    a->sub_b = gpu_alloc(e, T2max * 32 * 256 * 4);      // pw1 out, then pw2 out
    a->mel_in = gpu_alloc(e, (8ull * MF + 16) * 128 * 4);
    a->pe = gpu_alloc(e, 2ull * MF * 1024 * 4);
    a->ws_begin = a->sub_a, a->ws_end = a->pe + 2ull * MF * 1024 * 4;
    uint64_t p = a->sub_a;
    a->x = p, p += MF * 1024 * 4;
    a->h = p, p += MF * 1024 * 4;
    a->f = p, p += MF * 4096 * 4;
    a->qkv = p, p += MF * 3072 * 4;
    a->r = p, p += 2ull * MF * 1024 * 4;
    a->att = p, p += MF * 1024 * 4;
    a->g2 = p, p += MF * 2048 * 4;
    a->g = p, p += MF * 1024 * 4;
    a->encp = p;
    a->xh0 = gpu_alloc(e, 1280 * 4);
    a->xh1 = gpu_alloc(e, 1280 * 4);
    a->c0 = gpu_alloc(e, 640 * 4);
    a->c1 = gpu_alloc(e, 640 * 4);
    a->gates = gpu_alloc(e, 2560 * 4);
    a->dec = gpu_alloc(e, 640 * 4);
    a->jin = gpu_alloc(e, 640 * 4);
    a->logits = gpu_alloc(e, 8256 * 4);
    a->res = gpu_alloc(e, 256);
    a->dummy = gpu_alloc(e, 640 * 4);
    if (!a->dummy) return -1;
    a->ready = 1;
    return 0;
}

static void fft512(double* re, double* im) {   // in place, radix 2
    const int n = 512;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double t = re[i];
            re[i] = re[j], re[j] = t;
            t = im[i];
            im[i] = im[j], im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len, wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < len / 2; k++) {
                double ur = re[i + k], ui = im[i + k];
                double vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                double vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr, im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr, im[i + k + len / 2] = ui - vi;
                double t = cr * wr - ci * wi;
                ci = cr * wi + ci * wr, cr = t;
            }
        }
    }
}

// log-mel features as pc/parakeet_ref.py: features() -> out [T = n / 160][128]
static int asr_features(Engine* e, const float* audio, int n, float* out) {
    int T = n / 160;
    float* x = malloc((n + 512) * sizeof(float));
    double* mel = malloc((size_t)T * 128 * sizeof(double));
    if (!x || !mel) {
        free(x), free(mel);
        return -1;
    }
    memset(x, 0, (n + 512) * sizeof(float));
    for (int i = 0; i < n; i++) x[256 + i] = i ? audio[i] - 0.97f * audio[i - 1] : audio[0];
    double win[512] = {0}, re[512], im[512], spec[257];
    for (int k = 0; k < 400; k++) win[56 + k] = 0.5 - 0.5 * cos(2.0 * M_PI * k / 399.0);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < 512; k++) re[k] = x[t * 160 + k] * win[k], im[k] = 0.0;
        fft512(re, im);
        for (int k = 0; k < 257; k++) spec[k] = re[k] * re[k] + im[k] * im[k];
        for (int m = 0; m < 128; m++) {
            const float* fb = e->asr.mel + m * 257;
            double s = 0.0;
            for (int k = 0; k < 257; k++) s += fb[k] * spec[k];
            mel[t * 128 + m] = log(s + 5.9604644775390625e-8);
        }
    }
    for (int m = 0; m < 128; m++) {
        double mean = 0.0, var = 0.0;
        for (int t = 0; t < T; t++) mean += mel[t * 128 + m];
        mean /= T;
        for (int t = 0; t < T; t++) var += (mel[t * 128 + m] - mean) * (mel[t * 128 + m] - mean);
        double sd = sqrt(var / (T - 1)) + 1e-5;
        for (int t = 0; t < T; t++) out[t * 128 + m] = (float)((mel[t * 128 + m] - mean) / sd);
    }
    free(x), free(mel);
    return 0;
}

static void amm(Engine* e, const char* key, uint64_t X, uint64_t Y, uint32_t n, const char* bias, uint32_t act,
                double alpha, double beta) {
    const AsrTensor* t = at(e, key);
    uint64_t w = e->asr.model + t->off;
    dispatch(e, kernel(e, "mm_q4i"),
             args(e, "pppppuuuuuuff", w, w + t->scales, bias ? AW(e, bias) : (uint64_t)0, X, Y, t->cols, t->rows, n,
                  t->cols, t->rows, act, alpha, beta),
             t->rows / 64 * ((n + 7) / 8), 256);
}

static void ammf(Engine* e, const char* fmt, int i, uint64_t X, uint64_t Y, uint32_t n, uint32_t act, double alpha,
                 double beta) {
    char k[32];
    snprintf(k, sizeof(k), fmt, i);
    amm(e, k, X, Y, n, NULL, act, alpha, beta);
}

static void dec_update(Engine* e, uint32_t token) {
    Asr* a = &e->asr;
    const AsrTensor* em = at(e, "dec_emb");
    uint64_t w = a->model + em->off;
    dispatch(e, kernel(e, "embed_q4i"), args(e, "pppupu", w, w + em->scales, a->res, token, a->xh0, 640u), 1, 256);
    amm(e, "lstm0", a->xh0, a->gates, 1, "lstm0_b", 0, 1.0, 0.0);
    dispatch(e, kernel(e, "lstm_cell"), args(e, "pppp", a->gates, a->c0, a->xh0 + 640 * 4, a->xh1), 10, 64);
    amm(e, "lstm1", a->xh1, a->gates, 1, "lstm1_b", 0, 1.0, 0.0);
    dispatch(e, kernel(e, "lstm_cell"), args(e, "pppp", a->gates, a->c1, a->xh1 + 640 * 4, a->dummy), 10, 64);
    amm(e, "dec_proj", a->xh1 + 640 * 4, a->dec, 1, "dec_proj_b", 0, 1.0, 0.0);
}

int asr_transcribe(Engine* e, const float* audio, int n, char* text, int max) {
    Asr* a = &e->asr;
    text[0] = 0;
    if (!a->ready) {
        fail(e, "speech to text not loaded");
        return -1;
    }
    int Tm = n / 160;
    uint32_t T1 = (Tm - 1) / 2 + 1, T2 = (T1 - 1) / 2 + 1, T3 = (T2 - 1) / 2 + 1, T = T3;
    if (Tm < 16 || T3 > a->max_frames) {
        fail(e, "audio too short or too long (%d ms, max %u ms)", n / 16, a->max_frames * 80);
        return -1;
    }
    if (asr_features(e, audio, n, (float*)(uintptr_t)a->mel_in)) {
        fail(e, "out of memory for the audio features");
        return -1;
    }
    float* pe = (float*)(uintptr_t)a->pe;   // relative positions T-1 .. -(T-1), interleaved sin / cos
    for (uint32_t p = 0; p < 2 * T - 1; p++) {
        float pos = (float)((int)T - 1 - (int)p);
        for (int k = 0; k < 512; k++) {
            float ang = pos * (float)pow(10000.0, -2.0 * k / 1024.0);
            pe[p * 1024 + 2 * k] = sinf(ang);
            pe[p * 1024 + 2 * k + 1] = cosf(ang);
        }
    }

    begin(e);
    dispatch(e, kernel(e, "sub_conv1_dw1"),
             args(e, "ppppppuuu", a->mel_in, AW(e, "sub0_w"), AW(e, "sub0_b"), AW(e, "sub1_dw"), AW(e, "sub1_dwb"),
                  a->sub_a, (uint32_t)Tm, T1, T2),
             (T2 * 32 * 256 + 255) / 256, 256);
    dispatch(e, kernel(e, "mm_f32t"),
             args(e, "ppppuuuu", AW(e, "sub1_pwt"), AW(e, "sub1_pwb"), a->sub_a, a->sub_b, 256u, 256u, T2 * 32, 2u),
             T2 * 32 * 4, 64);
    dispatch(e, kernel(e, "sub_dwconv"),
             args(e, "ppppuuuu", a->sub_b, AW(e, "sub2_dw"), AW(e, "sub2_dwb"), a->sub_a, T2, 32u, T3, 16u),
             (T3 * 16 * 256 + 255) / 256, 256);
    dispatch(e, kernel(e, "mm_f32t"),
             args(e, "ppppuuuu", AW(e, "sub2_pwt"), AW(e, "sub2_pwb"), a->sub_a, a->sub_b, 256u, 256u, T3 * 16, 2u),
             T3 * 16 * 4, 64);
    amm(e, "sub_lin", a->sub_b, a->x, T, "sub_lin_b", 0, 1.0, 0.0);
    const GpuKernel* k_ln = kernel(e, "layernorm_rows");
    for (int i = 0; i < 24; i++) {
#define LN(nm, src, dst) dispatch(e, k_ln, args(e, "ppppu", src, AWf(e, "%d." nm ".w", i), AWf(e, "%d." nm ".b", i), \
                                                 dst, 1024u), T, 256)
        LN("norm_feed_forward1", a->x, a->h);
        ammf(e, "%d.feed_forward1.1", i, a->h, a->f, T, 1, 1.0, 0.0);
        ammf(e, "%d.feed_forward1.2", i, a->f, a->x, T, 0, 0.5, 1.0);
        LN("norm_self_att", a->x, a->h);
        ammf(e, "%d.wqkv", i, a->h, a->qkv, T, 0, 1.0, 0.0);
        ammf(e, "%d.wpos", i, a->pe, a->r, 2 * T - 1, 0, 1.0, 0.0);
        dispatch(e, kernel(e, "relattn"),
                 args(e, "pppppu", a->qkv, a->r, AWf(e, "%d.bias_u", i), AWf(e, "%d.bias_v", i), a->att, T), T * 8, 128);
        ammf(e, "%d.wo", i, a->att, a->x, T, 0, 1.0, 1.0);
        LN("norm_conv", a->x, a->h);
        ammf(e, "%d.pw1", i, a->h, a->g2, T, 0, 1.0, 0.0);
        dispatch(e, kernel(e, "glu_rows"), args(e, "ppu", a->g2, a->g, T), (T * 1024 + 255) / 256, 256);
        dispatch(e, kernel(e, "dwconv9_bn_silu"),
                 args(e, "pppppu", a->g, AWf(e, "%d.dw", i), AWf(e, "%d.bn_scale", i), AWf(e, "%d.bn_shift", i), a->h, T),
                 (T * 1024 + 255) / 256, 256);
        ammf(e, "%d.pw2", i, a->h, a->x, T, 0, 1.0, 1.0);
        LN("norm_feed_forward2", a->x, a->h);
        ammf(e, "%d.feed_forward2.1", i, a->h, a->f, T, 1, 1.0, 0.0);
        ammf(e, "%d.feed_forward2.2", i, a->f, a->x, T, 0, 0.5, 1.0);
        LN("norm_out", a->x, a->x);
#undef LN
    }
    amm(e, "enc_proj", a->x, a->encp, T, "enc_proj_b", 0, 1.0, 0.0);
    if (submit(e)) return -1;
    double gpu = e->last_gpu_seconds;

    memset((void*)(uintptr_t)a->xh0, 0, 1280 * 4);
    memset((void*)(uintptr_t)a->xh1, 0, 1280 * 4);
    memset((void*)(uintptr_t)a->c0, 0, 640 * 4);
    memset((void*)(uintptr_t)a->c1, 0, 640 * 4);
    int len = 0, at_frame = 0, steps = 0, update = 1;
    uint32_t token = 8192, t = 0;
    while (t < T && steps < 10 * (int)T) {
        begin(e);
        if (update) dec_update(e, token);
        dispatch(e, kernel(e, "joint_in"), args(e, "pppu", a->encp, a->dec, a->jin, t), 10, 64);
        amm(e, "joint", a->jin, a->logits, 1, "joint_b", 0, 1.0, 0.0);
        dispatch(e, kernel(e, "argmax_tdt"), args(e, "ppuu", a->logits, a->res, 8193u, 5u), 1, 256);
        if (submit(e)) return -1;
        gpu += e->last_gpu_seconds;
        steps++;
        volatile uint32_t* res = (volatile uint32_t*)(uintptr_t)a->res;
        uint32_t tok = res[0], dur = res[1];
        update = tok != 8192;
        if (update) {
            token = tok;
            const uint8_t* p = a->vocab + a->piece[tok];
            int l = p[0] | p[1] << 8;
            if (len + l < max - 1) {
                memcpy(text + len, p + 2, l);
                len += l;
            }
            at_frame++;
        }
        if (!update && dur == 0) dur = 1;
        if (dur == 0 && at_frame >= 10) dur = 1;
        if (dur > 0) at_frame = 0;
        t += dur;
    }
    text[len] = 0;
    if (text[0] == ' ') memmove(text, text + 1, len);   // Metaspace: the first piece starts with a space
    e->last_gpu_seconds = gpu;
    return 0;
}

// ---- test: can the GPU use the app's flexible memory (the system RAM that malloc uses)? -------------------------
// Maps 64 MiB with GPU permissions, copies through it with copy_f32 both ways and checks the data, then times a
// 32 MiB copy inside it. Logs everything with say() (main.c).
void say(const char* fmt, ...);

int engine_flex_test(Engine* e) {
    size_t avail = 0;
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&avail);
    say("flex: available %u MiB; GPU memory used %u of %u MiB", (unsigned)(avail >> 20), (unsigned)(e->next >> 20),
        (unsigned)(e->mem_size >> 20));
    const size_t sz = 64ull << 20;
    static const int prots[] = {0x33, 0x13};
    for (int pi = 0; pi < 2; pi++) {
        void* p = NULL;
        int r = sceKernelMapFlexibleMemory(&p, sz, prots[pi], 0);
        say("flex: map 64 MiB prot 0x%02x -> 0x%08x at %p", prots[pi], (unsigned)r, p);
        if (r) continue;
        float* f = p;
        uint32_t n = e->hidden;
        for (uint32_t i = 0; i < n; i++) f[i] = i * 0.5f + 1;
        float* gx = (float*)(uintptr_t)e->x;
        memset(gx, 0, n * 4);
        begin(e);
        dispatch(e, e->k_copy, args(e, "ppu", (uint64_t)(uintptr_t)f, e->x, n), n / 256, 256);
        int ok_read = submit(e) == 0;
        for (uint32_t i = 0; ok_read && i < n; i++)
            if (gx[i] != i * 0.5f + 1) ok_read = 0;
        float* f2 = f + (16u << 20) / 4;
        memset(f2, 0, n * 4);
        int ok_write = 0;
        if (prots[pi] & 0x20) {
            begin(e);
            dispatch(e, e->k_copy, args(e, "ppu", e->x, (uint64_t)(uintptr_t)f2, n), n / 256, 256);
            ok_write = submit(e) == 0;
            for (uint32_t i = 0; ok_write && i < n; i++)
                if (f2[i] != i * 0.5f + 1) ok_write = 0;
        }
        say("flex: GPU read %s, GPU write %s", ok_read ? "OK" : "FAILED", prots[pi] & 0x20 ? (ok_write ? "OK" : "FAILED") : "not tried");
        if (ok_read && ok_write) {   // bandwidth: 32 MiB from the first half to the second
            uint32_t m = (32u << 20) / 4;
            begin(e);
            dispatch(e, e->k_copy, args(e, "ppu", (uint64_t)(uintptr_t)f, (uint64_t)(uintptr_t)(f + m), m), m / 256, 256);
            if (submit(e) == 0)
                say("flex: 32 MiB copy in %.2f ms (%.1f GB/s read+write)", e->last_gpu_seconds * 1e3,
                    64.0 * 1.048576e6 / 1e9 / e->last_gpu_seconds / 1e3 * 1e3);
            begin(e);   // the same inside Garlic memory, for comparison: model weights as the source
            dispatch(e, e->k_copy, args(e, "ppu", e->model, (uint64_t)(uintptr_t)f, m), m / 256, 256);
            if (submit(e) == 0)
                say("flex: 32 MiB Garlic -> flexible in %.2f ms", e->last_gpu_seconds * 1e3);
        }
        sceKernelMunmap(p, sz);
        return ok_read;
    }
    return 0;
}

// Parakeet's weights in system memory (the 9B) are released while Python runs, and come back after (see engine.h).
// With the weights in GPU memory (the 4B, flex_len 0): nothing to do.
// The memory itself stays mapped: Python borrows it (asr_lent: its heap and its thread's stack) and the weights are
// read back into it afterwards. Given back to the system, it would have to be mapped again, and the 338 MiB might no
// longer be free.
void asr_release(Engine* e) {
    Asr* a = &e->asr;
    if (!a->flex_len || a->unloaded) return;
    a->ready = 0;
    a->unloaded = 1;
}

void* asr_lent(Engine* e, uint64_t* size) {
    Asr* a = &e->asr;
    *size = a->flex_len && a->unloaded && a->model ? a->flex_len : 0;
    return *size ? (void*)(uintptr_t)a->model : NULL;
}

int asr_reload(Engine* e) {
    Asr* a = &e->asr;
    if (!a->flex_len || !a->unloaded) return a->ready ? 0 : -1;
    void* fm = (void*)(uintptr_t)a->model;   // (still mapped: see asr_release)
    if (!fm) {
        int r = sceKernelMapFlexibleMemory(&fm, a->flex_len, 0x33, 0);
        if (r) {
            fail(e, "no system memory for the dictation weights (0x%08x)", (unsigned)r);
            return -1;
        }
    }
    int f = open(a->path, O_RDONLY);
    uint64_t done = 0;
    while (f >= 0 && done < a->size) {
        uint64_t n = a->size - done < (32ull << 20) ? a->size - done : 32ull << 20;
        ssize_t got = read(f, (uint8_t*)fm + done, n);
        if (got <= 0) break;
        done += got;
    }
    if (f >= 0) close(f);
    a->model = (uint64_t)(uintptr_t)fm;   // (kept mapped even if the read fails: the next dictation tries again)
    if (done != a->size) {
        fail(e, "cannot read %s", a->path);
        return -1;
    }
    a->unloaded = 0;
    a->ready = 1;
    return 0;
}

#include "engine_t3.inc"
#include "engine_s3g.inc"

#ifdef GARLIC_SPEC
#include "engine_spec.inc"
#endif
