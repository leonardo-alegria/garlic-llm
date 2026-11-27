// Maple-Preview (deepgrove, 20B-A1B ternary MoE) on the PS4 GPU. Included at the end of engine.c: it uses the same
// GPU plumbing (Garlic memory block, command buffer, dispatch, submit) as the Qwen paths. A model folder with
// model.idx "PMAP" (pc side: maple/convert_maple.py) selects this path in engine_init.
//
// Per token (shaders/maple.cl): x = embedding row (the CPU reads it from maple-embed.bin into a staging row);
// per layer: xn = rms(x); qkv = Wqkv xn (ternary); q/k rms norm (+RoPE on 64 of 128 dims on sliding layers) and the
// cache; attention (sliding: window 512 in a ring of 576 positions; global: all positions, no RoPE); x += Wo attn;
// xn = rms(x); router logits (f16); top-8; gate|up for the 8 experts; h = w * silu(min(g, 7)) * clamp(u, 7);
// down for the 8 experts; x += their sum. Then the head (Q4_K, full vocabulary) and the sampler.
// Up to MP_BATCH tokens go in one submission (the prompt); only the last one is sampled.

#define MP_BATCH 128
#define MP_ECACHE_ROWS 8192   // embedding rows kept in RAM (f16, 32 MiB): maple-embed.bin is read from the hard disk
#define MP_SEQ_BATCH 32   // tokens per submission one at a time (the command buffer: ~12 dispatches per layer each)
#define MP_SWA_RING 576          // MAX_SEQ of the swa_ kernels (shaders/maple.cl)
#define MP_MAX_LAYERS 32

static struct {
    uint32_t E, K, FF, window, rot, swa_mask, n_tok;
    float clamp;
    uint64_t wsize, erow;
    uint64_t tab[MP_MAX_LAYERS][9];   // attn_norm, ffn_norm, q_norm, k_norm, wqkv, wo, router, gate|up, down
    uint64_t out_norm, head;
    char embed_path[129];
    int efd;
    uint64_t base;                    // the weights in GPU memory
    uint64_t kc[MP_MAX_LAYERS], vc[MP_MAX_LAYERS], kv_bytes[MP_MAX_LAYERS];
    uint64_t staging, qkv, att, part, rlog, ids, ids0, wts, gu, h, y8;
    uint32_t layout;                  // model.idx version: 1 row-major t1p, 2 rows interleaved in tiles of 64
    const GpuKernel *k_mv_t1i, *k_mv_q4ki;
    const GpuKernel *k_q4kh, *k_spart, *k_sfin;   // (optional) head with per-group maxima, the two-step sampler
    uint64_t hmax, pairs;
    // batched prompt reading (mp_prefill): up to MP_BATCH tokens per pass
    const GpuKernel *k_rms_b, *k_f16b, *k_lists, *k_sum_b, *k_t1g;
    uint64_t xb, xnb, qkvb, attb, rlogb, idsb, wtsb, gub, hb, y8b, counts, lists, dlist, dcount;
    int batch_ok;
    uint32_t nl;                      // (checks) layers to run, 0: all
    int prof_force;                   // (checks) profile this run even with n > 1
    int fast_tail;                    // head + sampler through k_q4kh / k_spart / k_sfin (checked at start)
    const GpuKernel *k_mv_t1, *k_mv_q4k, *k_mv_f16, *k_topk, *k_act, *k_sum, *k_rms, *k_copy, *k_sample;
    const GpuKernel *k_swa_qk, *k_swa_attn, *k_swa_comb, *k_glb_qk, *k_glb_attn, *k_glb_comb;
    uint16_t* erow_buf;
    uint8_t* ecache;                  // MP_ECACHE_ROWS rows, row s holds token etag[s] - 1 (0: empty)
    uint32_t* etag;
    uint32_t ehit, emiss;             // rows from RAM / from the disk since the last report
    // profiler: a GPU timestamp after each stage of a profiled token (EV_BOTTOM_OF_PIPE_TS, 100 MHz)
    uint64_t prof;                    // GPU buffer of MP_PROF_MAX stamps
    int prof_on, prof_i;
    uint8_t prof_tag[1024];
    uint32_t decode_n;                // one-token runs since start (every MP_PROF_EVERY-th is profiled)
    double dec_wall, dec_gpu;         // their wall and GPU seconds since the last summary
    uint32_t dec_count;
} mp;

#define MP_PROF_MAX 1024
#define MP_PROF_EVERY 64
enum { PT_START, PT_NORM, PT_QKV, PT_SWA, PT_GLB, PT_WO, PT_NORM2, PT_ROUTER, PT_TOPK, PT_GU, PT_ACT, PT_DOWN, PT_SUM,
       PT_HNORM, PT_HEAD, PT_SAMPLE, PT_N };
static const char* const mp_prof_names[PT_N] = {"embed", "norm", "qkv", "attn-swa", "attn-glb", "wo", "norm2", "router",
                                                "top8", "gate/up", "act", "down", "sum", "hnorm", "head", "sample"};

// a timestamp once everything before it has finished (dispatch() waits for each dispatch: CS_PARTIAL_FLUSH)
static void mp_mark(Engine* e, int tag) {
    if (!mp.prof_on || mp.prof_i >= MP_PROF_MAX) return;
    eop(e, EV_BOTTOM_OF_PIPE_TS, mp.prof + 8ull * mp.prof_i, 3, 0);
    mp.prof_tag[mp.prof_i++] = (uint8_t)tag;
}

// After a profiled submission: time per stage (all layers), and the bandwidth of the matrix products.
static void mp_prof_report(Engine* e, uint32_t pos) {
    double t[PT_N] = {0};
    volatile uint64_t* st = (volatile uint64_t*)(uintptr_t)mp.prof;
    for (int i = 1; i < mp.prof_i; i++) t[mp.prof_tag[i]] += (double)(st[i] - st[i - 1]) / 100.0e6;
    double total = (double)(st[mp.prof_i - 1] - st[0]) / 100.0e6;
    uint32_t L = e->n_layers, D = e->hidden, FF = mp.FF, K = mp.K, rs = D / 256 * 54, rs_dn = FF / 256 * 54;
    double b_qkv = (double)L * (e->n_heads + 2 * e->n_kv) * e->head_dim * rs, b_wo = (double)L * D * rs;
    double b_gu = (double)L * K * 2 * FF * rs, b_dn = (double)L * K * D * rs_dn, b_head = (double)e->vocab * (D / 256 * 144);
    char line[480];
    int n = snprintf(line, sizeof(line), "[maple] profile pos %u: GPU %.2f ms =", pos, total * 1e3);
    for (int k = 1; k < PT_N && n < (int)sizeof(line) - 40; k++)
        if (t[k] > 0) n += snprintf(line + n, sizeof(line) - n, " %s %.2f", mp_prof_names[k], t[k] * 1e3);
    say("%s", line);
    say("[maple] profile pos %u: bandwidth qkv %.0f GB/s, wo %.0f, gate/up %.0f, down %.0f, head %.0f; %d stamps", pos,
        t[PT_QKV] > 0 ? b_qkv / t[PT_QKV] / 1e9 : 0, t[PT_WO] > 0 ? b_wo / t[PT_WO] / 1e9 : 0,
        t[PT_GU] > 0 ? b_gu / t[PT_GU] / 1e9 : 0, t[PT_DOWN] > 0 ? b_dn / t[PT_DOWN] / 1e9 : 0,
        t[PT_HEAD] > 0 ? b_head / t[PT_HEAD] / 1e9 : 0, mp.prof_i);
}

static void mp_mv(Engine* e, uint64_t w, uint32_t rs, uint64_t es, uint64_t ids, uint64_t x, uint32_t xstride, uint64_t y,
                  uint32_t R, uint32_t cols, uint32_t add, uint32_t total);
static void mp_head(Engine* e, uint64_t x, uint64_t y, uint32_t rows);
static void mp_sample(Engine* e, float inv_temp, uint32_t seed, double min_gap);

// Row r of a matrix as bytes (the CPU's view for the self-checks): direct for layout 1, gathered from the tiles for 2
static const uint8_t* mp_row(uint64_t mat, uint32_t r, uint32_t rb, uint8_t* tmp) {
    if (mp.layout != 2) return (const uint8_t*)(uintptr_t)(mat + (uint64_t)r * rb);
    uint32_t rd = rb / 4, t = r / 64, l = r % 64;
    const uint32_t* src = (const uint32_t*)(uintptr_t)mat;
    for (uint32_t d = 0; d < rd; d++) ((uint32_t*)tmp)[d] = src[((uint64_t)t * rd + d) * 64 + l];
    return tmp;
}

static float mp_half(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u, f;
    if (e == 0) {
        if (!m) f = s;
        else {   // subnormal
            e = 127 - 15 + 1;
            while (!(m & 1024u)) m <<= 1, e--;
            f = s | (e << 23) | ((m & 1023u) << 13);
        }
    } else if (e == 31) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e + 127 - 15) << 23) | (m << 13);
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static int mp_is_swa(uint32_t layer) { return (mp.swa_mask >> layer) & 1u; }

// ggml TQ1_0 trit n of a byte: ((b * 3^n) & 255) * 3 >> 8, minus 1
static float mp_trit(uint32_t b, uint32_t n) {
    static const uint32_t p3[5] = {1, 3, 9, 27, 81};
    return (float)((((b * p3[n]) & 255u) * 3u) >> 8) - 1.0f;
}

// one t1p row on the CPU (the self-check)
static float mp_cpu_row(const uint8_t* row, uint32_t cols, const float* x) {
    uint32_t nb = cols / 256;
    const uint8_t *qs = row, *qh = row + nb * 48;
    const uint8_t* dd = row + nb * 52;
    double acc = 0;
    for (uint32_t b = 0; b < nb; b++) {
        uint16_t dh;
        memcpy(&dh, dd + 2 * b, 2);
        float d = mp_half(dh), s = 0;
        for (uint32_t k = 0; k < 48; k++) {
            uint32_t byte = qs[b * 48 + k], base = k < 32 ? k : 128 + k, step = k < 32 ? 32 : 16;
            for (uint32_t n = 0; n < 5; n++) s += mp_trit(byte, n) * x[b * 256 + base + n * step];
        }
        for (uint32_t i = 0; i < 4; i++)
            for (uint32_t n = 0; n < 4; n++) s += mp_trit(qh[b * 4 + i], n) * x[b * 256 + 240 + n * 4 + i];
        acc += d * s;
    }
    return (float)acc;
}

// one Q4_K row on the CPU (the self-checks): w = d * sc * q - dmin * m per sub-block of 32
static float mp_cpu_q4k_row(const uint8_t* row, uint32_t cols, const float* x) {
    double acc = 0;
    for (uint32_t b = 0; b < cols / 256; b++) {
        const uint8_t* bp = row + b * 144;
        uint16_t h0, h1;
        memcpy(&h0, bp, 2), memcpy(&h1, bp + 2, 2);
        float d = mp_half(h0), dmin = mp_half(h1);
        const uint8_t *sc = bp + 4, *qs = bp + 16;
        for (uint32_t j = 0; j < 8; j++) {
            uint32_t s, m;
            if (j < 4) s = sc[j] & 63, m = sc[j + 4] & 63;
            else s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4), m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
            for (uint32_t i = 0; i < 32; i++) {
                uint32_t byte = qs[(j / 2) * 32 + i], q = j & 1 ? byte >> 4 : byte & 15;
                acc += (double)(d * s * q - dmin * m) * x[b * 256 + 64 * (j / 2) + 32 * (j & 1) + i];
            }
        }
    }
    return (float)acc;
}

static float mp_cpu_f16_row(const uint16_t* row, uint32_t cols, const float* x) {
    double acc = 0;
    for (uint32_t c = 0; c < cols; c++) acc += (double)mp_half(row[c]) * x[c];
    return (float)acc;
}

// Compares n GPU outputs with the CPU's; logs and returns 1 if they agree (relative to the largest value).
static int mp_cmp(const char* what, const float* cpu, volatile const float* gpu, uint32_t n) {
    float err = 0, mag = 0;
    uint32_t worst = 0;
    for (uint32_t i = 0; i < n; i++) {
        float dlt = fabsf(cpu[i] - gpu[i]);
        if (dlt > err) err = dlt, worst = i;
        if (fabsf(cpu[i]) > mag) mag = fabsf(cpu[i]);
    }
    int ok = err <= 2e-3f * (mag > 1 ? mag : 1);
    say("[maple] check %-22s %s: max |GPU - CPU| %.3g at %u (GPU %.4g, CPU %.4g), values up to %.3g", what,
        ok ? "ok" : "MISMATCH", err, worst, (double)gpu[worst], (double)cpu[worst], mag);
    return ok;
}

static int mp_index(Engine* e, const char* dir) {
    char path[256];
    size_t size;
    asset_path(path, sizeof(path), dir, "model.idx");
    uint8_t* d = read_file(path, &size);
    if (!d || size < 8 + 68 + 12 + 256 + 16 || memcmp(d, "PMAP", 4)) {
        free(d);
        fail(e, "cannot read %s", path);
        return -1;
    }
    memcpy(&mp.layout, d + 4, 4);
    if (mp.layout != 1 && mp.layout != 2) {
        free(d);
        fail(e, "model.idx: unknown Maple layout %u", mp.layout);
        return -1;
    }
    uint32_t v[17];
    memcpy(v, d + 8, sizeof(v));
    e->n_layers = v[0], e->hidden = v[1], e->n_heads = v[2], e->n_kv = v[3], e->head_dim = v[4];
    mp.E = v[5], mp.K = v[6], mp.FF = v[7];
    e->vocab = v[8], e->max_seq = v[9];
    mp.window = v[10], mp.rot = v[11];
    e->eos = v[12], e->im_end = v[13], e->eos2 = v[13], e->im_start = v[14];
    mp.swa_mask = v[15], mp.n_tok = v[16];
    float f3[3];
    memcpy(f3, d + 8 + 68, 12);
    e->eps = f3[0], e->rope_theta = f3[1], mp.clamp = f3[2];
    uint32_t p = 8 + 68 + 12;
    memcpy(e->weights_path, d + p, 128);
    e->weights_path[128] = 0;
    memcpy(mp.embed_path, d + p + 128, 128);
    mp.embed_path[128] = 0;
    p += 256;
    memcpy(&mp.wsize, d + p, 8);
    memcpy(&mp.erow, d + p + 8, 8);
    p += 16;
    if (e->n_layers > MP_MAX_LAYERS || e->head_dim != 128 || e->hidden > 2048 || mp.K > 8 || mp.E > 256 ||
        e->n_heads / e->n_kv > 4 || mp.FF * 2 > 2048 || e->max_seq != 16384 || mp.rot != 64 || mp.window > 512 ||
        size < p + 72ull * e->n_layers + 24) {
        free(d);
        fail(e, "model.idx: a shape the Maple kernels do not support");
        return -1;
    }
    for (uint32_t i = 0; i < e->n_layers; i++) memcpy(mp.tab[i], d + p + 72 * i, 72);
    memcpy(&mp.out_norm, d + p + 72 * e->n_layers, 8);
    memcpy(&mp.head, d + p + 72 * e->n_layers + 8, 8);
    free(d);
    weights_locate(e->weights_path);
    weights_locate(mp.embed_path);
    return 0;
}

// kernels.bin of the model folder (Garlic's load_kernels without its Qwen-specific lookups, which read e->t)
static int mp_load_kernels(Engine* e, const char* dir) {
    char path[256];
    size_t size;
    asset_path(path, sizeof(path), dir, "kernels.bin");
    uint8_t* d = read_file(path, &size);
    if (!d || size < 12 || memcmp(d, "PKRN", 4)) {
        free(d);
        fail(e, "cannot read %s", path);
        return -1;
    }
    uint32_t n, text_size;
    memcpy(&n, d + 4, 4);
    memcpy(&text_size, d + 8, 4);
    if (size < 12ull + n * 52ull + text_size) {
        free(d);
        fail(e, "%s is cut short", path);
        return -1;
    }
    uint64_t text = gpu_alloc(e, text_size);
    if (!text) {
        free(d);
        fail(e, "no GPU memory for the kernels");
        return -1;
    }
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
    say("[maple] %d kernels from %s", e->nk, path);
    return 0;
}

static const GpuKernel* mp_kernel_opt(Engine* e, const char* name) {
    for (int i = 0; i < e->nk; i++)
        if (!strcmp(e->k[i].name, name)) return &e->k[i];
    return NULL;
}

// Kernel lab: the expert gate|up product of layer 0 (8 experts, 3.5 MB) with parts of mv_t1i taken out, each 4 times,
// timed on the GPU. Which variant costs what the real kernel costs says what limits it (memory, math, set-up).
static void mp_lab(Engine* e) {
    static const char* const names[5] = {"mv_t1i", "lab_load", "lab_alu", "lab_stage", "lab_stream"};
    const GpuKernel* k[5];
    for (int i = 0; i < 5; i++)
        if (!(k[i] = mp_kernel_opt(e, names[i]))) {
            say("[maple] lab: kernel %s missing, skipped", names[i]);
            return;
        }
    uint32_t D = e->hidden, FF = mp.FF, rs = D / 256 * 54, total = mp.K * 2 * FF;
    uint64_t w = mp.base + mp.tab[0][7], es = (uint64_t)2 * FF * rs, bytes = (uint64_t)mp.K * es;
    uint32_t* ids = (uint32_t*)(uintptr_t)mp.ids;
    for (uint32_t i = 0; i < mp.K; i++) ids[i] = i;   // experts 0..7: contiguous, as lab_stream reads them
    __asm__ volatile("sfence" ::: "memory");
    mp.prof_on = 1, mp.prof_i = 0;
    begin(e);
    mp_mark(e, 0);
    for (int v = 0; v < 5; v++) {
        for (int r = 0; r < 4; r++) {
            if (v < 4)
                dispatch(e, k[v], args(e, "pupppupuuu", w, rs / 4, es, mp.ids, e->xn, 0, mp.gu, 2 * FF, D, 0), total / 64,
                         256);
            else
                dispatch(e, k[v], args(e, "pup", w, (uint32_t)(bytes / 16), mp.gu), 128, 256);
        }
        mp_mark(e, v + 1);
    }
    int bad = submit(e);
    mp.prof_on = 0;
    if (bad) return;
    volatile uint64_t* st = (volatile uint64_t*)(uintptr_t)mp.prof;
    double us[5];
    for (int v = 0; v < 5; v++) us[v] = (double)(st[v + 1] - st[v]) / 100.0 / 4.0;   // 100 MHz ticks -> us per dispatch
    say("[maple] lab gate/up %.2f MB, %u groups: real %.1f us (%.1f GB/s) | loads only %.1f (%.1f GB/s) | math only %.1f | "
        "setup only %.1f | 16-byte stream %.1f us (%.1f GB/s)", bytes / 1e6, total / 64, us[0], bytes / us[0] / 1e3, us[1],
        bytes / us[1] / 1e3, us[2], us[3], us[4], bytes / us[4] / 1e3);
}

// What do the small kernels cost? Each 4 times: an empty one-group dispatch, top-8, rmsnorm, the router, the head
// (old, new) and the sampler (old, new; greedy).
static void mp_lab2(Engine* e) {
    const GpuKernel* empty = mp_kernel_opt(e, "lab_empty");
    if (!empty || !mp.fast_tail) return;
    uint32_t D = e->hidden;
    mp.prof_on = 1, mp.prof_i = 0;
    begin(e);
    mp_mark(e, 0);
    for (int v = 0; v < 8; v++) {
        for (int r = 0; r < 4; r++) {
            switch (v) {
            case 0: dispatch(e, empty, args(e, "p", mp.gu), 1, 256); break;
            case 1: dispatch(e, mp.k_topk, args(e, "puupp", mp.rlog, mp.E, mp.K, mp.pairs, mp.pairs + 64), 1, 256); break;
            case 2: dispatch(e, mp.k_rms, args(e, "pppuf", e->x, mp.base + mp.out_norm, mp.gu, D, (double)e->eps), 1, 256);
                    break;
            case 3: dispatch(e, mp.k_mv_f16, args(e, "pppu", mp.base + mp.tab[0][6], e->xn, mp.gu, D), mp.E / 4, 256); break;
            case 4: dispatch(e, mp.k_mv_q4ki, args(e, "pppuu", mp.base + mp.head, e->xn, e->logits, D, e->vocab),
                             (e->vocab + 63) / 64, 256);
                    break;
            case 5: mp_head(e, e->xn, e->logits, e->vocab); break;
            case 6: mp.fast_tail = 0, mp_sample(e, 0.0f, 1, 0.0), mp.fast_tail = 1; break;
            case 7: mp_sample(e, 0.0f, 1, 0.0); break;
            }
        }
        mp_mark(e, v + 1);
    }
    int bad = submit(e);
    mp.prof_on = 0;
    if (bad) return;
    volatile uint64_t* st = (volatile uint64_t*)(uintptr_t)mp.prof;
    double us[8];
    for (int v = 0; v < 8; v++) us[v] = (double)(st[v + 1] - st[v]) / 100.0 / 4.0;
    double hb = (double)e->vocab * (D / 256 * 144);
    say("[maple] lab small: empty %.1f us | top8 %.1f | rmsnorm %.1f | router %.1f | head %.0f -> %.0f us (%.0f -> %.0f GB/s) | "
        "sample %.0f -> %.0f us", us[0], us[1], us[2], us[3], us[4], us[5], hb / us[4] / 1e3, hb / us[5] / 1e3, us[6], us[7]);
}

static int maple_init(Engine* e, const char* dir, ProgressFn progress, void* user) {
    e->maple = 1;
    if (mp_index(e, dir)) return -1;
    if (reserve_garlic(e)) {
        fail(e, "could not reserve GPU memory");
        return -1;
    }
    say("[maple] %u MiB of GPU memory; %u layers, hidden %u, %u experts (top %u), vocab %u, context %u",
        (unsigned)(e->mem_size >> 20), e->n_layers, e->hidden, mp.E, mp.K, e->vocab, e->max_seq);
    e->dcb = (uint32_t*)(uintptr_t)gpu_alloc(e, DCB_BYTES);
    e->dcb_cap = DCB_BYTES / 4;
    e->kargs = (uint8_t*)(uintptr_t)gpu_alloc(e, KARGS_BYTES);
    e->kargs_cap = KARGS_BYTES;
    e->label = (volatile uint32_t*)(uintptr_t)gpu_alloc(e, 256);
    e->stamps = (volatile uint64_t*)(uintptr_t)gpu_alloc(e, 256);
    if (mp_load_kernels(e, dir)) return -1;
    uint32_t* init = (uint32_t*)(uintptr_t)gpu_alloc(e, 1024);
    sceGnmDrawInitDefaultHardwareState350(init, 256);
    memcpy(e->init_state, init, sizeof(e->init_state));
    e->k_add = kernel(e, "add");
    mp.k_copy = e->k_copy = kernel(e, "copy_f32");
    mp.k_rms = kernel(e, "rmsnorm");
    mp.k_sample = kernel(e, "sample");
    mp.k_mv_t1 = kernel(e, "mv_t1"), mp.k_mv_q4k = kernel(e, "mv_q4k"), mp.k_mv_f16 = kernel(e, "mv_f16");
    if (mp.layout == 2) {
        mp.k_mv_t1i = kernel(e, "mv_t1i"), mp.k_mv_q4ki = kernel(e, "mv_q4ki");
        if (!mp.k_mv_t1i || !mp.k_mv_q4ki) return -1;
        mp.k_q4kh = mp_kernel_opt(e, "mv_q4kh"), mp.k_spart = mp_kernel_opt(e, "sample_part");
        mp.k_sfin = mp_kernel_opt(e, "sample_fin");
        mp.k_rms_b = mp_kernel_opt(e, "rms_b"), mp.k_f16b = mp_kernel_opt(e, "mv_f16b");
        mp.k_lists = mp_kernel_opt(e, "moe_lists"), mp.k_sum_b = mp_kernel_opt(e, "moe_sum_b");
        mp.k_t1g = mp_kernel_opt(e, "mv_t1g");
    }
    say("[maple] weight layout %u (%s)", mp.layout, mp.layout == 2 ? "tiles of 64 rows, interleaved" : "row-major");
    mp.k_topk = kernel(e, "moe_topk"), mp.k_act = kernel(e, "moe_act"), mp.k_sum = kernel(e, "moe_sum");
    mp.k_swa_qk = kernel(e, "swa_qk_rope_cache"), mp.k_swa_attn = kernel(e, "swa_attn_part");
    mp.k_swa_comb = kernel(e, "swa_attn_combine");
    mp.k_glb_qk = kernel(e, "glb_qk_rope_cache"), mp.k_glb_attn = kernel(e, "glb_attn_part");
    mp.k_glb_comb = kernel(e, "glb_attn_combine");
    if (!e->k_add || !mp.k_copy || !mp.k_rms || !mp.k_sample || !mp.k_mv_t1 || !mp.k_mv_q4k || !mp.k_mv_f16 ||
        !mp.k_topk || !mp.k_act || !mp.k_sum || !mp.k_swa_qk || !mp.k_swa_attn || !mp.k_swa_comb || !mp.k_glb_qk ||
        !mp.k_glb_attn || !mp.k_glb_comb)
        return -1;   // (kernel() said which)
    if (gpu_selftest(e)) return -1;

    // the weights, whole, into GPU memory
    mp.base = gpu_alloc(e, mp.wsize);
    if (!mp.base) {
        fail(e, "the weights (%u MiB) do not fit in GPU memory", (unsigned)(mp.wsize >> 20));
        return -1;
    }
    int f = open(e->weights_path, O_RDONLY);
    if (f < 0) {
        fail(e, "cannot open %s", e->weights_path);
        return -1;
    }
    uint64_t done = 0;
    while (done < mp.wsize) {
        uint64_t want = mp.wsize - done > (32u << 20) ? (32u << 20) : mp.wsize - done;
        ssize_t r = read(f, (void*)(uintptr_t)(mp.base + done), want);
        if (r <= 0) break;
        done += (uint64_t)r;
        if (progress) progress(user, (float)done / (float)mp.wsize * 0.97f, "weights");
    }
    close(f);
    if (done != mp.wsize) {
        fail(e, "%s is shorter than model.idx says (%u of %u MiB)", e->weights_path, (unsigned)(done >> 20),
             (unsigned)(mp.wsize >> 20));
        return -1;
    }
    mp.efd = open(mp.embed_path, O_RDONLY);
    if (mp.efd < 0) {
        fail(e, "cannot open %s", mp.embed_path);
        return -1;
    }
    mp.erow_buf = malloc(mp.erow);
    mp.ecache = malloc((size_t)MP_ECACHE_ROWS * mp.erow);
    mp.etag = calloc(MP_ECACHE_ROWS, 4);
    if (!mp.ecache || !mp.etag) {   // (works without: one read per token)
        free(mp.ecache), free(mp.etag);
        mp.ecache = NULL, mp.etag = NULL;
    }

    // caches and work buffers
    uint32_t H = e->n_heads, KV = e->n_kv, HD = e->head_dim, D = e->hidden;
    for (uint32_t i = 0; i < e->n_layers; i++) {
        uint64_t pos = mp_is_swa(i) ? MP_SWA_RING : e->max_seq;
        mp.kv_bytes[i] = (uint64_t)KV * HD * pos * 2;
        mp.kc[i] = gpu_alloc(e, mp.kv_bytes[i]);
        mp.vc[i] = gpu_alloc(e, mp.kv_bytes[i]);
        if (!mp.kc[i] || !mp.vc[i]) {
            fail(e, "no GPU memory for the attention cache of layer %u", i);
            return -1;
        }
    }
    e->rope = gpu_alloc(e, (uint64_t)e->max_seq * mp.rot * 4);
    mp.part = gpu_alloc(e, (uint64_t)H * ((e->max_seq / 64) * HD + 2 * (e->max_seq / 64)) * 4);
    mp.staging = gpu_alloc(e, (uint64_t)2 * MP_BATCH * D * 4);   // (two halves: the next pass's rows are written while the GPU reads)
    e->x = gpu_alloc(e, D * 4), e->xn = gpu_alloc(e, D * 4);
    mp.qkv = gpu_alloc(e, (uint64_t)(H + 2 * KV) * HD * 4), mp.att = gpu_alloc(e, (uint64_t)H * HD * 4);
    mp.rlog = gpu_alloc(e, mp.E * 4), mp.ids = gpu_alloc(e, 64), mp.ids0 = gpu_alloc(e, 64), mp.wts = gpu_alloc(e, 64);
    mp.gu = gpu_alloc(e, (uint64_t)mp.K * 2 * mp.FF * 4), mp.h = gpu_alloc(e, (uint64_t)mp.K * mp.FF * 4);
    mp.y8 = gpu_alloc(e, (uint64_t)mp.K * D * 4);
    e->logits = gpu_alloc(e, (uint64_t)e->vocab * 4);
    e->last = gpu_alloc(e, 256), e->history = gpu_alloc(e, 4096 * 4);
    mp.prof = gpu_alloc(e, 8ull * MP_PROF_MAX);
    mp.hmax = gpu_alloc(e, ((uint64_t)e->vocab / 64 + 64) * 4), mp.pairs = gpu_alloc(e, ((uint64_t)e->vocab / 1024 + 64) * 8);
    mp.fast_tail = mp.k_q4kh && mp.k_spart && mp.k_sfin && D == 2048 && mp.hmax && mp.pairs;
    mp.batch_ok = mp.k_rms_b && mp.k_f16b && mp.k_lists && mp.k_sum_b && mp.k_t1g && D == 2048 && mp.FF == 512 &&
                  H * HD == 2048 && mp.E == 256 && mp.K <= 8;
    if (mp.batch_ok) {
        uint64_t B = MP_BATCH;
        mp.xb = gpu_alloc(e, B * D * 4), mp.xnb = gpu_alloc(e, B * D * 4);
        mp.qkvb = gpu_alloc(e, B * (H + 2 * KV) * HD * 4), mp.attb = gpu_alloc(e, B * H * HD * 4);
        mp.rlogb = gpu_alloc(e, B * mp.E * 4), mp.idsb = gpu_alloc(e, B * 8 * 4), mp.wtsb = gpu_alloc(e, B * 8 * 4);
        mp.gub = gpu_alloc(e, B * mp.K * 2 * mp.FF * 4), mp.hb = gpu_alloc(e, B * mp.K * mp.FF * 4);
        mp.y8b = gpu_alloc(e, B * mp.K * D * 4);
        mp.counts = gpu_alloc(e, 256 * 4), mp.lists = gpu_alloc(e, 256 * B * 4);
        mp.dlist = gpu_alloc(e, B * 4), mp.dcount = gpu_alloc(e, 256);
        mp.batch_ok = mp.xb && mp.xnb && mp.qkvb && mp.attb && mp.rlogb && mp.idsb && mp.wtsb && mp.gub && mp.hb &&
                      mp.y8b && mp.counts && mp.lists && mp.dlist && mp.dcount;
        if (mp.batch_ok)
            for (uint32_t i = 0; i < B; i++) ((uint32_t*)(uintptr_t)mp.dlist)[i] = i;
    }
    if (!e->rope || !mp.part || !mp.staging || !e->x || !e->xn || !mp.qkv || !mp.att || !mp.rlog || !mp.ids ||
        !mp.ids0 || !mp.wts || !mp.gu || !mp.h || !mp.y8 || !e->logits || !e->last || !e->history) {
        fail(e, "no GPU memory for the work buffers");
        return -1;
    }
    memset((void*)(uintptr_t)mp.ids0, 0, 64);
    // RoPE table [max_seq][rot]: cos of the first rot/2 frequencies, then sin (shaders/attn.h)
    float* rope = (float*)(uintptr_t)e->rope;
    uint32_t half = mp.rot / 2;
    for (uint32_t p = 0; p < e->max_seq; p++)
        for (uint32_t i = 0; i < half; i++) {
            double ang = p * pow((double)e->rope_theta, -2.0 * i / mp.rot);
            rope[(uint64_t)p * mp.rot + i] = (float)cos(ang);
            rope[(uint64_t)p * mp.rot + half + i] = (float)sin(ang);
        }
    __asm__ volatile("sfence" ::: "memory");
    e->ctx_limit = e->lend_ctx = e->max_seq;
    e->qwen35 = 0, e->tq = 0, e->pf = 0;

    // self-checks on the console: every product kernel, in the configurations the model uses, against the CPU
    {
        float* xs = (float*)(uintptr_t)e->xn;
        static float cpu[256];
        static uint8_t tmp[2048];   // (one gathered row)
        volatile float* out = (volatile float*)(uintptr_t)mp.gu;   // (8 x 1024 floats: room for all the tests)
        for (uint32_t i = 0; i < D; i++) xs[i] = (float)((int)((i * 2654435761u) >> 24) - 128) / 128.0f;
        uint32_t rs = D / 256 * 54, ok = 1;
        uint64_t wq = mp.base + mp.tab[0][4];
        // ternary, 4, 16 and 64 rows per group (dense, layer 0 qkv rows 0..63)
        static const uint32_t rpgs[1] = {4};
        for (int k = 0; k < 1; k++) {
            memset((void*)out, 0, 64 * 4);
            __asm__ volatile("sfence" ::: "memory");
            begin(e);
            mp_mv(e, wq, rs, 0, mp.ids0, e->xn, 0, mp.gu, 64, D, 0, 64);
            if (submit(e)) return -1;
            (void)rpgs;
            for (uint32_t r = 0; r < 64; r++) cpu[r] = mp_cpu_row(mp_row(wq, r, rs, tmp), D, xs);
            char nm[40];
            snprintf(nm, sizeof(nm), "ternary rpg %u", rpgs[k]);
            ok &= mp_cmp(nm, cpu, out, 64);
        }
        // ternary experts: 2 slots of 64 rows, experts 3 and 77 of layer 0's gate|up
        {
            uint32_t* ids = (uint32_t*)(uintptr_t)mp.ids;
            ids[0] = 3, ids[1] = 77;
            uint64_t wg = mp.base + mp.tab[0][7], es = (uint64_t)2 * mp.FF * rs;
            memset((void*)out, 0, 128 * 4);
            __asm__ volatile("sfence" ::: "memory");
            begin(e);
            mp_mv(e, wg, rs, es, mp.ids, e->xn, 0, mp.gu, 64, D, 0, 128);
            if (submit(e)) return -1;
            for (uint32_t r = 0; r < 64; r++) {
                cpu[r] = mp_cpu_row(mp_row(wg + 3 * es, r, rs, tmp), D, xs);
                cpu[64 + r] = mp_cpu_row(mp_row(wg + 77 * es, r, rs, tmp), D, xs);
            }
            ok &= mp_cmp("ternary experts", cpu, out, 128);
        }
        // head (Q4_K), rows 0..63, 64 per group
        {
            uint32_t rb = D / 256 * 144;
            memset((void*)out, 0, 64 * 4);
            __asm__ volatile("sfence" ::: "memory");
            begin(e);
            mp_head(e, e->xn, mp.gu, 64);
            if (submit(e)) return -1;
            for (uint32_t r = 0; r < 64; r++) cpu[r] = mp_cpu_q4k_row(mp_row(mp.base + mp.head, r, rb, tmp), D, xs);
            ok &= mp_cmp("head q4k", cpu, out, 64);
        }
        // router (f16), layer 0, all experts
        {
            memset((void*)out, 0, mp.E * 4);
            __asm__ volatile("sfence" ::: "memory");
            begin(e);
            dispatch(e, mp.k_mv_f16, args(e, "pppu", mp.base + mp.tab[0][6], e->xn, mp.gu, D), mp.E / 4, 256);
            if (submit(e)) return -1;
            for (uint32_t r = 0; r < mp.E; r++)
                cpu[r] = mp_cpu_f16_row((const uint16_t*)(uintptr_t)(mp.base + mp.tab[0][6]) + (uint64_t)r * D, D, xs);
            ok &= mp_cmp("router f16", cpu, out, mp.E);
        }
        // head over the whole vocabulary, then the sampler both ways (greedy and T = 1): the same token
        if (mp.fast_tail) {
            volatile uint32_t* last = (volatile uint32_t*)(uintptr_t)e->last;
            uint32_t got[4];
            for (int v = 0; v < 4; v++) {
                float it = v & 1 ? 1.0f : 0.0f;
                int fast = v >> 1;
                mp.fast_tail = 1;   // (the head kernel that also writes the maxima)
                *last = 0xFFFFFFFFu;
                __asm__ volatile("sfence" ::: "memory");
                begin(e);
                mp_head(e, e->xn, e->logits, e->vocab);
                mp.fast_tail = fast;
                mp_sample(e, it, 12345u, it > 0.0f ? 2.9957 : 0.0);
                if (submit(e)) return -1;
                got[v] = *last;
            }
            mp.fast_tail = got[0] == got[2] && got[1] == got[3] && got[0] < e->vocab && got[1] < e->vocab;
            say("[maple] check sampler two-step   %s: greedy %u / %u, T=1 %u / %u (one-group / two-step)",
                mp.fast_tail ? "ok" : "MISMATCH (using the one-group sampler)", got[0], got[2], got[1], got[3]);
            if (!mp.fast_tail) say("[maple] head: back to mv_q4ki");
        }
        say("[maple] kernel checks: %s", ok ? "all ok" : "SOME FAILED (answers will be wrong)");
        if (mp.batch_ok) {   // a prompt of 48 tokens one at a time, then batched: after 1 layer, and after all of them
            static uint32_t tk[48];
            float* ref = (float*)malloc((size_t)e->vocab * 4);
            static float xr[2048];
            for (int i = 0; i < 48; i++) tk[i] = 100u + (uint32_t)((i * 7919u) % 30000u);
            int all_ok = ref != NULL;
            for (int pass = 0; pass < 2 && all_ok; pass++) {
                mp.nl = pass == 0 ? 1 : 0;
                uint32_t nx1 = 0, nx2 = 0;
                mp.batch_ok = 0;
                int bad = maple_run(e, tk, 48, 0, 0.0f, 1, &nx1);
                if (!bad) {
                    memcpy(ref, (const void*)(uintptr_t)e->logits, (size_t)e->vocab * 4);
                    memcpy(xr, (const void*)(uintptr_t)e->x, D * 4);
                }
                mp.batch_ok = 1;
                double t0 = (double)sceKernelReadTsc() / (double)e->tsc_freq;
                mp.prof_force = pass == 1;
                bad = bad || maple_run(e, tk, 48, 0, 0.0f, 1, &nx2);
                mp.prof_force = 0;
                double dt = (double)sceKernelReadTsc() / (double)e->tsc_freq - t0;
                float md = 0.0f, mx = 0.0f, xd = 0.0f, xm = 0.0f;
                if (!bad) {
                    const float* lg = (const float*)(uintptr_t)e->logits;
                    const float* xx = (const float*)(uintptr_t)e->x;
                    for (uint32_t i = 0; i < e->vocab; i++) {
                        float dd = fabsf(lg[i] - ref[i]), aa = fabsf(ref[i]);
                        if (dd > md) md = dd;
                        if (aa > mx) mx = aa;
                    }
                    for (uint32_t i = 0; i < D; i++) {
                        float dd = fabsf(xx[i] - xr[i]), aa = fabsf(xr[i]);
                        if (dd > xd) xd = dd;
                        if (aa > xm) xm = aa;
                    }
                }
                // after 1 layer the two ways must agree to rounding; after 24, an expert choice that was a near-tie
                // may go the other way (both are right), so: the same next token
                int ok1 = !bad && (pass == 0 ? xd <= 1e-3f * (xm > 1.0f ? xm : 1.0f) : nx1 == nx2);
                say("[maple] check batched prompt %2u layer%s %s: x max |diff| %.3g of %.3g, logits %.3g of %.3g, next %u / %u; "
                    "48 tokens in %.1f ms", pass == 0 ? 1u : e->n_layers, pass == 0 ? " " : "s", ok1 ? "ok" : "MISMATCH",
                    xd, xm, md, mx, nx1, nx2, dt * 1e3);
                all_ok &= ok1;
            }
            mp.nl = 0;
            mp.batch_ok = all_ok;
            if (!all_ok) say("[maple] batched prompt reading off (one token at a time)");
            free(ref);
        }
    }
    if (mp.layout == 2) mp_lab(e), mp_lab2(e);
    {   // a profiled token at start-up (position 0; the chat overwrites it), so the log has the breakdown at once
        uint32_t tok = e->im_start, nx;
        if (maple_run(e, &tok, 1, 0, 0.0f, 1, &nx)) return -1;
    }
    say("[maple] ready: GPU memory used %u of %u MiB", (unsigned)(e->next >> 20), (unsigned)(e->mem_size >> 20));
    if (progress) progress(user, 1.0f, "ready");
    return 0;
}

static int mp_read_row(Engine* e, uint32_t tok, void* dst) {
    if (pread(mp.efd, dst, mp.erow, (off_t)tok * (off_t)mp.erow) != (ssize_t)mp.erow) {
        fail(e, "cannot read the embedding of token %u", tok);
        return -1;
    }
    return 0;
}

static int mp_cmp_u32(const void* a, const void* b) {
    uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
    return x < y ? -1 : x > y;
}

// k tokens' embedding rows (f16 in maple-embed.bin) -> f32 staging rows slot0.. . Rows come from the RAM cache; the
// missing ones are read in file order first (fewer seeks on the hard disk). maple_run calls this while the GPU runs.
static int mp_embed_chunk(Engine* e, const uint32_t* tokens, uint32_t k, uint32_t slot0) {
    uint32_t miss[MP_BATCH], nm = 0;
    for (uint32_t j = 0; j < k; j++) {
        uint32_t tok = tokens[j];
        if (tok >= e->vocab) {
            fail(e, "token %u outside the vocabulary", tok);
            return -1;
        }
        if (!mp.ecache || mp.etag[tok % MP_ECACHE_ROWS] == tok + 1) continue;
        uint32_t m = 0;
        while (m < nm && miss[m] != tok) m++;
        if (m == nm) miss[nm++] = tok;
    }
    if (nm > 1) qsort(miss, nm, 4, mp_cmp_u32);
    for (uint32_t m = 0; m < nm; m++) {
        uint32_t s = miss[m] % MP_ECACHE_ROWS;
        mp.etag[s] = 0;
        if (mp_read_row(e, miss[m], mp.ecache + (size_t)s * mp.erow)) return -1;
        mp.etag[s] = miss[m] + 1;
    }
    uint32_t direct = 0;
    for (uint32_t j = 0; j < k; j++) {
        uint32_t tok = tokens[j], s = tok % MP_ECACHE_ROWS;
        const uint16_t* src;
        if (mp.ecache && mp.etag[s] == tok + 1) {
            src = (const uint16_t*)(mp.ecache + (size_t)s * mp.erow);
        } else {   // (no cache, or two tokens of this pass share a cache row)
            if (mp_read_row(e, tok, mp.erow_buf)) return -1;
            src = mp.erow_buf;
            direct++;
        }
        float* dst = (float*)(uintptr_t)(mp.staging + (uint64_t)(slot0 + j) * e->hidden * 4);
        for (uint32_t i = 0; i < e->hidden; i++) dst[i] = mp_half(src[i]);
    }
    mp.emiss += nm + direct;
    mp.ehit += k > nm + direct ? k - nm - direct : 0;
    return 0;
}

// mv_t1 over `total` rows (slots of R rows): rows per group 64 for big products, 16 for small ones, 4 for tests
static void mp_mv(Engine* e, uint64_t w, uint32_t rs, uint64_t es, uint64_t ids, uint64_t x, uint32_t xstride, uint64_t y,
                  uint32_t R, uint32_t cols, uint32_t add, uint32_t total) {
    if (mp.layout == 2) {   // lane = row, a tile of 64 rows per group (R % 64 == 0)
        dispatch(e, mp.k_mv_t1i, args(e, "pupppupuuu", w, rs / 4, es, ids, x, xstride, y, R, cols, add), total / 64, 256);
        return;
    }
    dispatch(e, mp.k_mv_t1, args(e, "pupppupuuuu", w, rs, es, ids, x, xstride, y, R, cols, add, 4u), total / 4, 256);
}

static void mp_head(Engine* e, uint64_t x, uint64_t y, uint32_t rows) {
    uint32_t D = e->hidden;
    if (mp.layout == 2 && mp.fast_tail) {
        dispatch(e, mp.k_q4kh, args(e, "pppuup", mp.base + mp.head, x, y, D, rows, mp.hmax), (rows + 63) / 64, 256);
        return;
    }
    if (mp.layout == 2) {
        dispatch(e, mp.k_mv_q4ki, args(e, "pppuu", mp.base + mp.head, x, y, D, rows), (rows + 63) / 64, 256);
        return;
    }
    dispatch(e, mp.k_mv_q4k, args(e, "pppuuu", mp.base + mp.head, x, y, D, rows, 4u), (rows + 3) / 4, 256);
}

// the sampler over e->logits (after mp_head): two steps over many groups, or the one-group kernel
static void mp_sample(Engine* e, float inv_temp, uint32_t seed, double min_gap) {
    if (mp.fast_tail) {
        uint32_t ng = (e->vocab + 1023) / 1024, nh = (e->vocab + 63) / 64;
        dispatch(e, mp.k_spart, args(e, "pufupupf", e->logits, e->vocab, (double)inv_temp, seed, mp.hmax, nh, mp.pairs,
                                     min_gap), ng, 256);
        dispatch(e, mp.k_sfin, args(e, "puppu", mp.pairs, ng, e->last, e->history, 0u), 1, 256);
        return;
    }
    dispatch(e, mp.k_sample, args(e, "pufuppuf", e->logits, e->vocab, (double)inv_temp, seed, e->last, e->history, 0u,
                                  min_gap), 1, 256);
}

// layer i's attention for one position: q/k norms (+RoPE), the caches, attention of qkv into att. Returns the stage.
static int mp_attn(Engine* e, uint32_t i, uint64_t qkv, uint64_t att, uint32_t pos) {
    uint32_t H = e->n_heads, KV = e->n_kv, HD = e->head_dim;
    double scale = 1.0 / sqrt((double)HD);
    const uint64_t* t = mp.tab[i];
    uint64_t b = mp.base;
    if (mp_is_swa(i)) {
        uint32_t lo = pos >= mp.window - 1 ? pos - (mp.window - 1) : 0, blk0 = lo / 64, nblk = pos / 64 - blk0 + 1;
        dispatch(e, mp.k_swa_qk, args(e, "ppppppuuuf", qkv, b + t[2], b + t[3], e->rope, mp.kc[i], mp.vc[i], pos, H, KV,
                                      (double)e->eps), H + 2 * KV, 64);
        dispatch(e, mp.k_swa_attn, args(e, "pppppuuufuu", qkv, mp.kc[i], mp.vc[i], mp.part, att, pos, H, KV, scale, blk0,
                                        lo), nblk * KV, 256);
        if (pos + 1 > 64) dispatch(e, mp.k_swa_comb, args(e, "ppuu", mp.part, att, H, nblk), H, HD);
        return PT_SWA;
    }
    uint32_t nblk = pos / 64 + 1;
    dispatch(e, mp.k_glb_qk, args(e, "ppppppuuuf", qkv, b + t[2], b + t[3], e->rope, mp.kc[i], mp.vc[i], pos, H, KV,
                                  (double)e->eps), H + 2 * KV, 64);
    dispatch(e, mp.k_glb_attn, args(e, "pppppuuuf", qkv, mp.kc[i], mp.vc[i], mp.part, att, pos, H, KV, scale),
             nblk * KV, 256);
    if (pos + 1 > 64) dispatch(e, mp.k_glb_comb, args(e, "ppuu", mp.part, att, H, nblk), H, HD);
    return PT_GLB;
}

// mv_t1g: lists/counts give each expert's slots (dense: mp.dlist / mp.dcount, one "expert")
// nsplit: groups per (expert, tile), each taking every nsplit-th four of the slots (dense: one per four tokens)
static void mp_mvg(Engine* e, uint64_t w, uint32_t rs, uint64_t es, uint64_t counts, uint64_t lists, uint32_t ngrid,
                   uint32_t K, uint64_t x, uint32_t xstride, uint32_t by_slot, uint64_t y, uint32_t R, uint32_t cols,
                   uint32_t add, uint32_t nsplit) {
    dispatch(e, mp.k_t1g, args(e, "pupppuupuupuuuu", w, rs / 4, es, counts, lists, (uint32_t)MP_BATCH, K, x, xstride,
                               by_slot, y, R, cols, add, nsplit), ngrid * (R / 64) * nsplit, 256);
}

// nt prompt tokens (embeddings at stage, nt rows) through every layer at once; leaves the last one's x in e->x
static void mp_prefill(Engine* e, uint32_t nt, uint32_t pos0, uint64_t stage) {
    uint32_t H = e->n_heads, KV = e->n_kv, HD = e->head_dim, D = e->hidden, FF = mp.FF, K = mp.K, E = mp.E;
    uint32_t QKV = (H + 2 * KV) * HD, rs = D / 256 * 54, rs_dn = FF / 256 * 54, ns = (nt + 3) / 4;
    dispatch(e, mp.k_copy, args(e, "ppu", stage, mp.xb, nt * D), nt * D / 256, 256);
    mp_mark(e, PT_START);
    uint32_t nl = mp.nl ? mp.nl : e->n_layers;
    for (uint32_t i = 0; i < nl; i++) {
        const uint64_t* t = mp.tab[i];
        uint64_t b = mp.base;
        dispatch(e, mp.k_rms_b, args(e, "pppuf", mp.xb, b + t[0], mp.xnb, D, (double)e->eps), nt, 256);
        mp_mark(e, PT_NORM);
        mp_mvg(e, b + t[4], rs, 0, mp.dcount, mp.dlist, 1, 1, mp.xnb, D, 0, mp.qkvb, QKV, D, 0, ns);
        mp_mark(e, PT_QKV);
        int st = PT_SWA;
        for (uint32_t j = 0; j < nt; j++)
            st = mp_attn(e, i, mp.qkvb + (uint64_t)j * QKV * 4, mp.attb + (uint64_t)j * H * HD * 4, pos0 + j);
        mp_mark(e, st);
        mp_mvg(e, b + t[5], rs, 0, mp.dcount, mp.dlist, 1, 1, mp.attb, H * HD, 0, mp.xb, D, H * HD, 1, ns);
        mp_mark(e, PT_WO);
        dispatch(e, mp.k_rms_b, args(e, "pppuf", mp.xb, b + t[1], mp.xnb, D, (double)e->eps), nt, 256);
        mp_mark(e, PT_NORM2);
        dispatch(e, mp.k_f16b, args(e, "pppuuu", b + t[6], mp.xnb, mp.rlogb, D, E, nt), E / 4, 256);
        mp_mark(e, PT_ROUTER);
        dispatch(e, mp.k_topk, args(e, "puupp", mp.rlogb, E, K, mp.idsb, mp.wtsb), nt, 256);
        dispatch(e, mp.k_lists, args(e, "puuupp", mp.idsb, nt * K, E, (uint32_t)MP_BATCH, mp.counts, mp.lists), 1, 256);
        mp_mark(e, PT_TOPK);
        mp_mvg(e, b + t[7], rs, (uint64_t)2 * FF * rs, mp.counts, mp.lists, E, K, mp.xnb, D, 0, mp.gub, 2 * FF, D, 0, 1);
        mp_mark(e, PT_GU);
        dispatch(e, mp.k_act, args(e, "pppuuf", mp.gub, mp.wtsb, mp.hb, FF, nt * K * FF, (double)mp.clamp),
                 (nt * K * FF + 255) / 256, 256);
        mp_mark(e, PT_ACT);
        mp_mvg(e, b + t[8], rs_dn, (uint64_t)D * rs_dn, mp.counts, mp.lists, E, K, mp.hb, FF, 1, mp.y8b, D, FF, 0, 1);
        mp_mark(e, PT_DOWN);
        dispatch(e, mp.k_sum_b, args(e, "ppuuu", mp.xb, mp.y8b, D, K, nt), nt * D / 256, 256);
        mp_mark(e, PT_SUM);
    }
    dispatch(e, mp.k_copy, args(e, "ppu", mp.xb + (uint64_t)(nt - 1) * D * 4, e->x, D), D / 256, 256);
}

static void mp_token(Engine* e, uint32_t slot, uint32_t pos) {
    uint32_t H = e->n_heads, KV = e->n_kv, HD = e->head_dim, D = e->hidden, FF = mp.FF, K = mp.K;
    uint32_t rs = D / 256 * 54, rs_dn = FF / 256 * 54;
    dispatch(e, mp.k_copy, args(e, "ppu", mp.staging + (uint64_t)slot * D * 4, e->x, D), D / 256, 256);
    mp_mark(e, PT_START);
    uint32_t nl = mp.nl ? mp.nl : e->n_layers;
    for (uint32_t i = 0; i < nl; i++) {
        const uint64_t* t = mp.tab[i];
        uint64_t b = mp.base;
        dispatch(e, mp.k_rms, args(e, "pppuf", e->x, b + t[0], e->xn, D, (double)e->eps), 1, 256);
        mp_mark(e, PT_NORM);
        mp_mv(e, b + t[4], rs, 0, mp.ids0, e->xn, 0, mp.qkv, (H + 2 * KV) * HD, D, 0, (H + 2 * KV) * HD);
        mp_mark(e, PT_QKV);
        mp_mark(e, mp_attn(e, i, mp.qkv, mp.att, pos));
        mp_mv(e, b + t[5], rs, 0, mp.ids0, mp.att, 0, e->x, D, H * HD, 1, D);
        mp_mark(e, PT_WO);
        dispatch(e, mp.k_rms, args(e, "pppuf", e->x, b + t[1], e->xn, D, (double)e->eps), 1, 256);
        mp_mark(e, PT_NORM2);
        dispatch(e, mp.k_mv_f16, args(e, "pppu", b + t[6], e->xn, mp.rlog, D), mp.E / 4, 256);
        mp_mark(e, PT_ROUTER);
        dispatch(e, mp.k_topk, args(e, "puupp", mp.rlog, mp.E, K, mp.ids, mp.wts), 1, 256);
        mp_mark(e, PT_TOPK);
        mp_mv(e, b + t[7], rs, (uint64_t)2 * FF * rs, mp.ids, e->xn, 0, mp.gu, 2 * FF, D, 0, K * 2 * FF);
        mp_mark(e, PT_GU);
        dispatch(e, mp.k_act, args(e, "pppuuf", mp.gu, mp.wts, mp.h, FF, K * FF, (double)mp.clamp), K * FF / 256, 256);
        mp_mark(e, PT_ACT);
        mp_mv(e, b + t[8], rs_dn, (uint64_t)D * rs_dn, mp.ids, mp.h, FF, mp.y8, D, FF, 0, K * D);
        mp_mark(e, PT_DOWN);
        dispatch(e, mp.k_sum, args(e, "ppuu", e->x, mp.y8, D, K), D / 256, 256);
        mp_mark(e, PT_SUM);
    }
}

static int maple_run(Engine* e, const uint32_t* tokens, int n, int pos0, float temperature, uint32_t seed,
                     uint32_t* next) {
    if (n <= 0 || pos0 + n > (int)e->max_seq) {
        fail(e, "positions %d..%d are outside the context (%u)", pos0, pos0 + n - 1, e->max_seq);
        return -1;
    }
    double w0 = (double)sceKernelReadTsc() / (double)e->tsc_freq, gpu = 0.0;
    mp.prof_on = mp.prof_force || (n == 1 && (mp.decode_n++ % MP_PROF_EVERY) == 0);   // (the first token, then every 64th)
    mp.prof_i = 0;
    int chunk = mp.batch_ok ? MP_BATCH : MP_SEQ_BATCH;
    uint32_t eh0 = mp.ehit, em0 = mp.emiss;
    if (mp_embed_chunk(e, tokens, (uint32_t)(n < chunk ? n : chunk), 0)) return -1;
    for (int i0 = 0, half = 0; i0 < n; i0 += chunk, half ^= 1) {
        int k = n - i0 < chunk ? n - i0 : chunk;
        uint32_t s0 = half ? MP_BATCH : 0;   // this pass's staging rows (the next pass's go to the other half)
        __asm__ volatile("sfence" ::: "memory");
        if (mp.batch_ok && k >= 2) *(volatile uint32_t*)(uintptr_t)mp.dcount = (uint32_t)k;
        __asm__ volatile("sfence" ::: "memory");
        begin(e);
        if (mp.batch_ok && k >= 2) mp_prefill(e, (uint32_t)k, (uint32_t)(pos0 + i0), mp.staging + (uint64_t)s0 * e->hidden * 4);
        else
            for (int j = 0; j < k; j++) mp_token(e, s0 + (uint32_t)j, (uint32_t)(pos0 + i0 + j));
        if (i0 + k == n) {   // the last token: head and sampler
            uint32_t D = e->hidden;
            float inv_temp = temperature > 0.0f ? 1.0f / temperature : 0.0f;
            double min_gap = inv_temp > 0.0f ? 2.9957 / inv_temp : 0.0;
            dispatch(e, mp.k_rms, args(e, "pppuf", e->x, mp.base + mp.out_norm, e->xn, D, (double)e->eps), 1, 256);
            mp_mark(e, PT_HNORM);
            mp_head(e, e->xn, e->logits, e->vocab);
            mp_mark(e, PT_HEAD);
            mp_sample(e, inv_temp, seed, min_gap);
            mp_mark(e, PT_SAMPLE);
        }
        if (submit_go(e)) return -1;
        int bad = 0;
        if (i0 + k < n) {   // the next pass's embeddings while the GPU works on this one
            int k2 = n - i0 - k < chunk ? n - i0 - k : chunk;
            bad = mp_embed_chunk(e, tokens + i0 + k, (uint32_t)k2, half ? 0 : MP_BATCH);
        }
        if (submit_wait(e) || bad) return -1;
        gpu += e->last_gpu_seconds;
    }
    double wall = (double)sceKernelReadTsc() / (double)e->tsc_freq - w0;
    if (mp.prof_on) {
        mp_prof_report(e, (uint32_t)pos0);
        mp.prof_on = 0;
    } else if (n == 1) {   // decode: averages every 64 tokens
        mp.dec_wall += wall, mp.dec_gpu += gpu, mp.dec_count++;
        if (mp.dec_count == MP_PROF_EVERY - 1) {
            say("[maple] decode: %u tokens, per token wall %.2f ms = GPU %.2f + CPU/submit %.2f (%.1f tok/s)", mp.dec_count,
                mp.dec_wall / mp.dec_count * 1e3, mp.dec_gpu / mp.dec_count * 1e3,
                (mp.dec_wall - mp.dec_gpu) / mp.dec_count * 1e3, mp.dec_count / mp.dec_wall);
            mp.dec_wall = mp.dec_gpu = 0.0, mp.dec_count = 0;
        }
    } else {
        say("[maple] prompt: %d tokens at %d in %.2f s (GPU %.2f s): %.1f tok/s; embeddings %u from RAM, %u from disk", n,
            pos0, wall, gpu, n / wall, mp.ehit - eh0, mp.emiss - em0);
    }
    e->last_row = *(volatile uint32_t*)(uintptr_t)e->last;
    *next = e->last_row;
    return 0;
}

// The prompt cache (server mode): what a position depends on besides the global caches (which are written by
// position) is the sliding layers' rings. Their copy lets a conversation go back to an earlier position.
static uint64_t maple_state_size(const Engine* e) {
    uint64_t n = 0;
    for (uint32_t i = 0; i < e->n_layers; i++)
        if (mp_is_swa(i)) n += 2 * ((mp.kv_bytes[i] + 255) & ~255ull);
    return n;
}

static int maple_state_copy(Engine* e, uint64_t buf, int save) {
    begin(e);
    uint64_t p = buf;
    for (uint32_t i = 0; i < e->n_layers; i++) {
        if (!mp_is_swa(i)) continue;
        const uint64_t part[2] = {mp.kc[i], mp.vc[i]};
        for (int j = 0; j < 2; j++) {
            uint32_t n = (uint32_t)(mp.kv_bytes[i] / 4);
            dispatch(e, mp.k_copy, args(e, "ppu", save ? part[j] : p, save ? p : part[j], n), (n + 255) / 256, 256);
            p += (mp.kv_bytes[i] + 255) & ~255ull;
        }
    }
    return submit(e);
}

static void maple_unload(void) {
    if (mp.efd > 0) close(mp.efd);
    free(mp.erow_buf);
    free(mp.ecache);
    free(mp.etag);
    memset(&mp, 0, sizeof(mp));
    mp.efd = -1;
}
