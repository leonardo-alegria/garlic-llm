// ---- text to speech, first half: Chatterbox T3 (text tokens -> speech tokens) ------------------------------------
// Included at the end of engine.c (uses its dispatch / submit / gpu_alloc). Same building blocks as the language
// model: rmsnorm, q4i matrix-vector products, RoPE + KV cache, flash-decoding attention (the *_t3 kernels of
// shaders/t3.cl, heads of 64), SwiGLU. The input rows (voice prefix, text and speech embeddings + learned positions)
// are read from the weights file by the CPU; sampling is on the CPU too. Reference: pc/t3_ref.py.

static const AsrTensor* t3t(T3* t, const char* name) {
    for (uint32_t i = 0; i < t->nt; i++)
        if (!strcmp(t->t[i].name, name)) return &t->t[i];
    return NULL;
}

static void t3_tune(Engine* e, int verbose);

int t3_init(Engine* e, const char* dir) {
    T3* t = &e->t3;
    memset(t, 0, sizeof(*t));
    char path[256];
    size_t size;
    snprintf(path, sizeof(path), "%s/t3.idx", dir);
    uint8_t* d = read_file(path, &size);
    if (!d || size < 152 || memcmp(d, "PT3I", 4)) {
        free(d);
        return -1;
    }
    memcpy(&t->nt, d + 4, 4);
    memcpy(t->path, d + 8, 128);
    t->path[128] = 0;
    weights_locate(t->path);
    memcpy(&t->size, d + 136, 8);
    memcpy(&t->gpu_size, d + 144, 8);
    t->t = calloc(t->nt, sizeof(AsrTensor));
    for (uint32_t i = 0; i < t->nt; i++) {
        const uint8_t* r = d + 152 + i * 64;
        memcpy(t->t[i].name, r, 32);
        t->t[i].name[31] = 0;
        memcpy(&t->t[i].off, r + 32, 8);
        memcpy(&t->t[i].size, r + 40, 8);
        memcpy(&t->t[i].rows, r + 48, 4);
        memcpy(&t->t[i].cols, r + 52, 4);
        memcpy(&t->t[i].scales, r + 56, 8);
    }
    memcpy(t->inv_freq, d + 152 + t->nt * 64, 32 * 4);
    free(d);
    char key[32];
    for (int i = 0; i < T3_LAYERS; i++) {
        snprintf(key, sizeof(key), "%d.qkv", i), t->qkv_w[i] = t3t(t, key);
        snprintf(key, sizeof(key), "%d.o", i), t->o_w[i] = t3t(t, key);
        snprintf(key, sizeof(key), "%d.gu", i), t->gu_w[i] = t3t(t, key);
        snprintf(key, sizeof(key), "%d.down", i), t->down_w[i] = t3t(t, key);
        snprintf(key, sizeof(key), "%d.in_norm", i), t->in_n[i] = t3t(t, key);
        snprintf(key, sizeof(key), "%d.post_norm", i), t->post_n[i] = t3t(t, key);
        if (!t->qkv_w[i] || !t->o_w[i] || !t->gu_w[i] || !t->down_w[i] || !t->in_n[i] || !t->post_n[i]) return -1;
    }
    t->norm = t3t(t, "norm"), t->head = t3t(t, "head");
    t->text_emb = t3t(t, "text_emb"), t->text_pos = t3t(t, "text_pos");
    t->speech_emb = t3t(t, "speech_emb"), t->speech_pos = t3t(t, "speech_pos"), t->prefix = t3t(t, "prefix");
    t->k_qk = kernel(e, "qk_rope_cache_t3"), t->k_attn = kernel(e, "attn_part_t3"), t->k_comb = kernel(e, "attn_combine_t3");
    t->k_rms = kernel(e, "t3_rmsnorm_bs"), t->k_mv = kernel(e, "t3_mv"), t->k_att = kernel(e, "t3_attn");
    t->k_silu = kernel(e, "t3_silu_bs"), t->k_mvk = kernel(e, "t3_mvk"), t->k_sample = kernel(e, "t3_sample");
    if (!t->norm || !t->head || !t->text_emb || !t->text_pos || !t->speech_emb || !t->speech_pos || !t->prefix ||
        !t->k_qk || !t->k_attn || !t->k_comb || !t->k_rms || !t->k_mv || !t->k_att || !t->k_silu || !t->k_mvk ||
        !t->k_sample)
        return -1;
    // the batched prefill (optional: without these kernels, one position per pass)
    t->k_mvb = kernel(e, "t3_mv_b"), t->k_mvkb = kernel(e, "t3_mvk_b"), t->k_rope = kernel(e, "t3_rope_b");
    t->k_attb = kernel(e, "t3_attn_b");
    e->error[0] = 0;

    // GPU memory: weights, RoPE table, KV cache (K fp16, V int8 + scales), activations, input rows
    const uint64_t kv_k = 16ull * 64 * T3_MAX_SEQ * 2, kv_v = (uint64_t)T3_MAX_SEQ * 16 * 64 + T3_MAX_SEQ * 16 * 4;
    uint64_t need = t->gpu_size + T3_MAX_SEQ * 64 * 4 + T3_LAYERS * (kv_k + kv_v) + (1u << 20);
    uint64_t free_bytes = e->mem_size - ((e->next + 255) & ~255ull);
    if (free_bytes < need) {
        snprintf(e->error, sizeof(e->error), "T3 needs %u MiB, %u MiB free", (unsigned)(need >> 20),
                 (unsigned)(free_bytes >> 20));
        return -1;
    }
    t->model = gpu_alloc(e, t->gpu_size);
    t->fd = open(t->path, O_RDONLY);
    if (t->fd < 0) return -1;
    uint64_t done = 0;
    while (done < t->gpu_size) {
        uint64_t n = t->gpu_size - done < (32ull << 20) ? t->gpu_size - done : 32ull << 20;
        ssize_t r = pread(t->fd, (uint8_t*)(uintptr_t)t->model + done, n, (off_t)done);
        if (r <= 0) break;
        done += r;
    }
    if (done != t->gpu_size) return -1;
    t->rope = gpu_alloc(e, T3_MAX_SEQ * 64 * 4);
    float* rp = (float*)(uintptr_t)t->rope;
    for (int p = 0; p < T3_MAX_SEQ; p++)
        for (int j = 0; j < 32; j++) {
            double a = p * (double)t->inv_freq[j];
            rp[p * 64 + j] = (float)cos(a), rp[p * 64 + 32 + j] = (float)sin(a);
        }
    for (int i = 0; i < T3_LAYERS; i++) t->kc[i] = gpu_alloc(e, kv_k), t->vc[i] = gpu_alloc(e, kv_v);
    t->x = gpu_alloc(e, 1024 * 4), t->xn = gpu_alloc(e, 1024 * 4), t->qkv = gpu_alloc(e, 3072 * 4);
    t->att = gpu_alloc(e, 1024 * 4), t->o = gpu_alloc(e, 1024 * 4), t->gu = gpu_alloc(e, 8192 * 4);
    t->hm = gpu_alloc(e, 4096 * 4), t->logits = gpu_alloc(e, 6592 * 4);
    t->part = gpu_alloc(e, (16ull * (T3_MAX_SEQ / 64) * 64 + 2 * 16 * (T3_MAX_SEQ / 64)) * 4);
    t->staging = gpu_alloc(e, 32ull * 1024 * 4);
    if (!t->staging) return -1;
    t->parts = gpu_alloc(e, 4 * 1024 * 4);
    t->g_seen = gpu_alloc(e, 6656);
    t->g_res = gpu_alloc(e, 256);
    if (!t->parts || !t->g_seen || !t->g_res) return -1;
    // rows of the speech tables kept in memory (the file is read once per row): 1024 token rows, 512 position rows
    t->emb_cache = malloc(1024 * 2048);
    t->emb_tag = malloc(1024 * sizeof(int32_t));
    t->pos_rows = malloc(512 * 2048);
    if (!t->emb_cache || !t->emb_tag || !t->pos_rows) return -1;
    for (int i = 0; i < 1024; i++) t->emb_tag[i] = -1;
    if (pread(t->fd, t->pos_rows, 512 * 2048, (off_t)t->speech_pos->off) != 512 * 2048) return -1;
    // voice rows per language (optional)
    char pp[256];
    size_t psz;
    snprintf(pp, sizeof(pp), "%s/t3_prefix.bin", dir);
    uint8_t* pd = read_file(pp, &psz);
    if (pd && psz >= 8 && !memcmp(pd, "PT3P", 4)) {
        uint32_t n;
        memcpy(&n, pd + 4, 4);
        if (n >= 1 && n <= 16 && psz >= 8 + n * 4 + (size_t)n * 34 * 2048) {
            for (uint32_t i = 0; i < n; i++) memcpy(t->pre_lang[i], pd + 8 + i * 4, 4), t->pre_lang[i][3] = 0;
            t->pre = malloc((size_t)n * 34 * 2048);
            if (t->pre) memcpy(t->pre, pd + 8 + n * 4, (size_t)n * 34 * 2048), t->npre = (int)n;
        }
    }
    free(pd);
    t->ready = 1;
    t3_tune(e, 1);
    return 0;
}

// Row r of an fp16 table in the weights file, added to dst (1024 floats).
static void t3_add_row(T3* t, const AsrTensor* a, uint32_t r, float* dst) {
    uint16_t buf[1024];
    const uint16_t* h = buf;
    if (a == t->prefix && t->pre && r < 34) h = t->pre + ((size_t)t->pre_sel * 34 + r) * 1024;
    else if (a == t->speech_pos && t->pos_rows && r < 512) h = t->pos_rows + r * 1024;
    else if (a == t->speech_emb && t->emb_cache) {
        uint16_t* slot = t->emb_cache + (r & 1023) * 1024;
        if (t->emb_tag[r & 1023] != (int32_t)r) {
            if (pread(t->fd, slot, 2048, (off_t)(a->off + (uint64_t)r * 2048)) != 2048) return;
            t->emb_tag[r & 1023] = (int32_t)r;
        }
        h = slot;
    } else if (pread(t->fd, buf, sizeof(buf), (off_t)(a->off + (uint64_t)r * 2048)) != (ssize_t)sizeof(buf))
        return;
    for (int i = 0; i < 1024; i++) {   // fp16 -> fp32
        uint32_t s = (h[i] & 0x8000u) << 16, ex = h[i] >> 10 & 31, m = h[i] & 1023, bits;
        if (ex == 0) {
            if (!m) bits = s;
            else {
                ex = 113;
                while (!(m & 1024)) m <<= 1, ex--;
                bits = s | ex << 23 | (m & 1023) << 13;
            }
        } else if (ex == 31) bits = s | 0x7F800000u | m << 13;
        else bits = s | (ex + 112) << 23 | m << 13;
        float f;
        memcpy(&f, &bits, 4);
        dst[i] += f;
    }
}

// One position through the 30 layers with the fused kernels of shaders/t3.cl (8 dispatches per layer); the logits
// too when head is set. x holds the input row; the block sums for each matrix come from the kernel before it.
static void t3fmv(Engine* e, const AsrTensor* a, uint64_t x, uint64_t y, int accumulate) {
    uint64_t w = e->t3.model + a->off;
    dispatch(e, e->t3.k_mv, args(e, "pppppuu", w, w + a->scales, x, e->bsum, y, a->cols, (uint32_t)accumulate),
             a->rows / 64, 512);
}

static uint32_t t3_rng(uint32_t* s) {   // xorshift32
    *s ^= *s << 13, *s ^= *s >> 17, *s ^= *s << 5;
    return *s;
}

// o and down (1024 rows): plain t3_mv adding into x, or split in nks K slices whose sums the next t3_rmsnorm_bs adds.
static uint32_t t3_res(Engine* e, const AsrTensor* a, uint64_t in) {
    T3* t = &e->t3;
    if (t->nks <= 1) {
        t3fmv(e, a, in, t->x, 1);
        return 0;
    }
    uint64_t w = t->model + a->off;
    dispatch(e, t->k_mvk, args(e, "pppppuuu", w, w + a->scales, in, e->bsum, t->parts, a->cols, a->rows, t->nks),
             a->rows / 64 * t->nks, 512);
    return t->nks;
}

static void t3_norm(Engine* e, const AsrTensor* w, uint32_t nparts) {
    T3* t = &e->t3;
    dispatch(e, t->k_rms, args(e, "ppupppf", t->x, t->parts, nparts, t->model + w->off, t->xn, e->bsum, 1e-5), 1, 256);
}

static void t3_pass(Engine* e, uint32_t pos, int head) {
    T3* t = &e->t3;
    uint32_t pend = 0;   // partial sums of the last split product, not yet added into x
    for (int i = 0; i < T3_LAYERS; i++) {
        t3_norm(e, t->in_n[i], pend);
        t3fmv(e, t->qkv_w[i], t->xn, t->qkv, 0);
        dispatch(e, t->k_att, args(e, "ppppppu", t->qkv, t->rope, t->kc[i], t->vc[i], t->att, e->bsum, pos), 16, 256);
        pend = t3_res(e, t->o_w[i], t->att);
        t3_norm(e, t->post_n[i], pend);
        t3fmv(e, t->gu_w[i], t->xn, t->gu, 0);
        dispatch(e, t->k_silu, args(e, "ppp", t->gu, t->hm, e->bsum), 16, 256);
        pend = t3_res(e, t->down_w[i], t->hm);
    }
    if (head || pend) t3_norm(e, t->norm, pend);   // (also folds the last partial sums into x)
    if (head) {
        t3fmv(e, t->head, t->xn, t->logits, 0);
        dispatch(e, t->k_sample, args(e, "pppuu", t->logits, t->g_seen, t->g_res, t3_rng(&t->rng), (uint32_t)t->greedy),
                 1, 256);
    }
}

// ---- batched prefill: up to T3_PF positions per pass (the weights are read once for 16 of them) --------------
// The same kernels' arithmetic per position as t3_pass (shaders/t3.cl *_b), so the same bits; the work buffers live
// in pf_scratch (Parakeet's, idle while T3 runs). Rows of x at pf.x: [n][1024].
#define T3_PF 32
#define T3_NB 16   // positions per group of t3_mv_b (as in shaders/t3.cl)

typedef struct {
    uint64_t x, xn, bs, qkv, qb, att, gu, hm, parts;
} T3Pf;

static T3Pf t3_pf(T3* t) {
    T3Pf b;
    uint64_t a = (t->pf_scratch + 255) & ~255ull;
    b.x = a, a += T3_PF * 4096ull;
    b.xn = a, a += T3_PF * 4096ull;
    b.bs = a, a += T3_PF * 512ull;
    b.qkv = a, a += T3_PF * 12288ull;
    b.qb = a, a += T3_PF * 4096ull;
    b.att = a, a += T3_PF * 4096ull;
    b.gu = a, a += T3_PF * 32768ull;
    b.hm = a, a += T3_PF * 16384ull;
    b.parts = a, a += T3_PF * 16384ull;
    return b;
}

static uint64_t t3_pf_bytes(void) { return 256 + T3_PF * (4096ull * 4 + 512 + 12288 + 4096 + 32768 + 16384 * 2); }

void t3_scratch(Engine* e, uint64_t base, uint64_t size) {
    T3* t = &e->t3;
    int ok = t->k_mvb && t->k_mvkb && t->k_rope && t->k_attb && base && size >= t3_pf_bytes();
    t->pf_scratch = ok ? base : 0, t->pf_size = ok ? size : 0;
}

static void t3fmv_b(Engine* e, const AsrTensor* a, uint64_t x, uint64_t bs, uint64_t y, uint32_t n, int accumulate) {
    uint64_t w = e->t3.model + a->off;
    dispatch(e, e->t3.k_mvb, args(e, "pppppuuuu", w, w + a->scales, x, bs, y, a->cols, a->rows, n, (uint32_t)accumulate),
             a->rows / 64 * ((n + T3_NB - 1) / T3_NB), 512);
}

static uint32_t t3_res_b(Engine* e, const T3Pf* b, const AsrTensor* a, uint64_t in, uint32_t n) {   // t3_res
    T3* t = &e->t3;
    if (t->nks <= 1) {
        t3fmv_b(e, a, in, b->bs, b->x, n, 1);
        return 0;
    }
    uint64_t w = t->model + a->off;
    dispatch(e, t->k_mvkb, args(e, "pppppuuuu", w, w + a->scales, in, b->bs, b->parts, a->cols, a->rows, t->nks, n),
             a->rows / 64 * t->nks * ((n + T3_NB - 1) / T3_NB), 512);
    return t->nks;
}

// Positions p0 .. p0 + n - 1 through the 30 layers (their input rows in b->x); with head, the logits and the draw
// of the last one (into t->xn / logits / g_res, as t3_pass).
static void t3_pass_b(Engine* e, const T3Pf* b, uint32_t p0, uint32_t n, int head) {
    T3* t = &e->t3;
    uint32_t pend = 0;
    for (int i = 0; i < T3_LAYERS; i++) {
        dispatch(e, t->k_rms, args(e, "ppupppf", b->x, b->parts, pend, t->model + t->in_n[i]->off, b->xn, b->bs, 1e-5),
                 n, 256);
        t3fmv_b(e, t->qkv_w[i], b->xn, b->bs, b->qkv, n, 0);
        dispatch(e, t->k_rope, args(e, "pppppu", b->qkv, t->rope, t->kc[i], t->vc[i], b->qb, p0), 16 * n, 64);
        dispatch(e, t->k_attb, args(e, "pppppu", b->qb, t->kc[i], t->vc[i], b->att, b->bs, p0), 16 * n, 256);
        pend = t3_res_b(e, b, t->o_w[i], b->att, n);
        dispatch(e, t->k_rms, args(e, "ppupppf", b->x, b->parts, pend, t->model + t->post_n[i]->off, b->xn, b->bs,
                                   1e-5), n, 256);
        t3fmv_b(e, t->gu_w[i], b->xn, b->bs, b->gu, n, 0);
        dispatch(e, t->k_silu, args(e, "ppp", b->gu, b->hm, b->bs), 16 * n, 256);
        pend = t3_res_b(e, b, t->down_w[i], b->hm, n);
    }
    if (head) {   // the last position: final norm (folding its partial sums into its x), head, draw
        uint64_t last = n - 1;
        dispatch(e, t->k_rms, args(e, "ppupppf", b->x + last * 4096, b->parts + last * 16384, pend,
                                   t->model + t->norm->off, t->xn, e->bsum, 1e-5), 1, 256);
        t3fmv(e, t->head, t->xn, t->logits, 0);
        dispatch(e, t->k_sample, args(e, "pppuu", t->logits, t->g_seen, t->g_res, t3_rng(&t->rng), (uint32_t)t->greedy),
                 1, 256);
    }
}

// GPU milliseconds per token of one kind of step over the 30 layers, at position pos (start-up benchmark).
static double t3_time(Engine* e, int what, uint32_t pos) {
    T3* t = &e->t3;
    begin(e);
    for (int i = 0; i < T3_LAYERS; i++) {
        switch (what) {
        case 0: t3_norm(e, t->in_n[i], 0); break;
        case 1: t3fmv(e, t->qkv_w[i], t->xn, t->qkv, 0); break;
        case 2:
            dispatch(e, t->k_att, args(e, "ppppppu", t->qkv, t->rope, t->kc[i], t->vc[i], t->att, e->bsum, pos), 16, 256);
            break;
        case 3: t3_res(e, t->o_w[i], t->att); break;
        case 4: t3fmv(e, t->gu_w[i], t->xn, t->gu, 0); break;
        case 5: dispatch(e, t->k_silu, args(e, "ppp", t->gu, t->hm, e->bsum), 16, 256); break;
        case 6: t3_res(e, t->down_w[i], t->hm); break;
        default: t3_pass(e, pos, 0); break;
        }
    }
    if (submit(e)) return -1;
    return e->last_gpu_seconds * 1000.0 / (what == 7 ? T3_LAYERS : 1);
}

// Plain or split products for o and down, whichever is faster here; verbose logs the time of each kind of step.
static void t3_tune(Engine* e, int verbose) {
    T3* t = &e->t3;
    double best = 1e9;
    uint32_t pick = 1;
    for (uint32_t k = 1; k <= 4; k *= 2) {
        t->nks = k;
        double ms = t3_time(e, 3, 100) + t3_time(e, 6, 100);
        if (verbose) say("T3 o+down with %u K slices: %.2f ms per token", k, ms);
        if (ms > 0 && ms < best) best = ms, pick = k;
    }
    t->nks = pick;
    if (verbose) {
        static const char* names[8] = {"rmsnorm", "qkv", "attention", "o", "gate|up", "silu", "down", "whole pass"};
        for (int w = 0; w < 8; w++) say("T3 %s: %.2f ms per token (position 100)", names[w], t3_time(e, w, 100));
    }
}


void t3_voice(Engine* e, const char* lang) {
    T3* t = &e->t3;
    t->pre_sel = 0;
    for (int i = 1; i < t->npre; i++)
        if (!strncmp(t->pre_lang[i], lang, 2)) t->pre_sel = i;
}

int t3_start(Engine* e, const uint16_t* text, int n_text, int max_out, uint32_t seed, int greedy) {
    T3* t = &e->t3;
    if (!t->ready) return -1;
    e->error[0] = 0;
    int n_in = 34 + n_text + 2;
    if (n_in + max_out > T3_MAX_SEQ) max_out = T3_MAX_SEQ - n_in;
    if (max_out <= 0) return -1;
    float row[1024];
    float* stage = (float*)(uintptr_t)t->staging;
    memset((void*)(uintptr_t)t->g_seen, 0, 6563);
    ((volatile uint8_t*)(uintptr_t)t->g_seen)[6561] = 1;   // start of speech: never a real output
    t->rng = seed * 2654435761u + 12345;
    if (!t->rng) t->rng = 1;
    t->greedy = greedy;
    // prefill: voice prefix, text (+ positions), start-of-speech twice; 32 positions per submission, in one batched
    // pass when there are work buffers for it (t3_scratch), else one pass per position
    uint64_t tp = sceKernelReadTsc();
    T3Pf pf = t3_pf(t);
    t->prefill_batched = t->pf_scratch ? T3_PF : 1;
    for (int p0 = 0; p0 < n_in; p0 += 32) {
        int n = n_in - p0 < 32 ? n_in - p0 : 32;
        begin(e);
        for (int k = 0; k < n; k++) {
            int p = p0 + k;
            memset(row, 0, sizeof(row));
            if (p < 34) t3_add_row(t, t->prefix, p, row);
            else if (p < 34 + n_text) t3_add_row(t, t->text_emb, text[p - 34], row), t3_add_row(t, t->text_pos, p - 34, row);
            else t3_add_row(t, t->speech_emb, 6561, row), t3_add_row(t, t->speech_pos, 0, row);
            memcpy(stage + k * 1024, row, sizeof(row));
            if (!t->pf_scratch) {
                dispatch(e, e->k_copy, args(e, "ppu", t->staging + k * 4096ull, t->x, 1024), 4, 256);
                t3_pass(e, p, p == n_in - 1);
            }
        }
        if (t->pf_scratch) {
            dispatch(e, e->k_copy, args(e, "ppu", t->staging, pf.x, 1024u * n), 4 * n, 256);
            t3_pass_b(e, &pf, p0, n, p0 + n == n_in);
        }
        if (submit(e)) return -1;
    }
    t->s_prefill = (double)(sceKernelReadTsc() - tp) / e->tsc_freq;
    t->n_in = n_in, t->k = 0, t->max_out = max_out;
    t->s_rows = t->s_sample = t->s_submit = t->s_gpu = 0;
    if (t->debug) {   // the state right after the prefill
        const volatile float* lg = (const volatile float*)(uintptr_t)t->logits;
        const volatile float* xx = (const volatile float*)(uintptr_t)t->x;
        int nan = 0, xnan = 0, best = 0;
        float mx = -1e30f;
        for (int i = 0; i < 6563; i++) {
            float v = lg[i];
            if (v != v) nan++;
            else if (v > mx) mx = v, best = i;
        }
        for (int i = 0; i < 1024; i++) xnan += xx[i] != xx[i];
        say("T3 start (%s, n_in %d, max_out %d, nks %u, sel %d): drawn %u, argmax %d max %.3f, logits nan %d, x nan %d"
            " x0 %.4f",
            greedy ? "greedy" : "sampled", n_in, max_out, t->nks, t->pre_sel,
            *(volatile uint32_t*)(uintptr_t)t->g_res, best, mx, nan, xnan, xx[0]);
    }
    return 0;
}

int t3_next(Engine* e) {
    T3* t = &e->t3;
    if (t->k >= t->max_out) return -1;
    uint64_t c0 = sceKernelReadTsc();
    int tok = (int)*(volatile uint32_t*)(uintptr_t)t->g_res;   // drawn on the GPU after the last pass (t3_sample)
    uint64_t c1 = sceKernelReadTsc();
    t->s_sample += (double)(c1 - c0) / e->tsc_freq;
    if (tok >= 6561) {   // stop (6562); 6561 is never valid output
        t->k = t->max_out;
        return -1;
    }
    float row[1024];
    memset(row, 0, sizeof(row));
    t3_add_row(t, t->speech_emb, tok, row);
    t3_add_row(t, t->speech_pos, t->k + 1, row);
    uint64_t c2 = sceKernelReadTsc();
    t->s_rows += (double)(c2 - c1) / e->tsc_freq;
    begin(e);
    memcpy((float*)(uintptr_t)t->staging, row, sizeof(row));
    dispatch(e, e->k_copy, args(e, "ppu", t->staging, t->x, 1024), 4, 256);
    t3_pass(e, t->n_in + t->k, 1);
    t->k++;
    uint64_t c3 = sceKernelReadTsc();
    if (submit(e)) return -2;
    t->s_submit += (double)(sceKernelReadTsc() - c3) / e->tsc_freq;
    t->s_gpu += e->last_gpu_seconds;
    if (t->debug && t->k <= 3) {
        const volatile float* lg = (const volatile float*)(uintptr_t)t->logits;
        const volatile float* xx = (const volatile float*)(uintptr_t)t->x;
        float mx = -1e30f;
        int nan = 0;
        for (int i = 0; i < 6563; i++) {
            float v = lg[i];
            if (v != v) nan++;
            else if (v > mx) mx = v;
        }
        say("T3 debug k %d: tok %d row %.4f %.4f x %.4f %.4f logits max %.3f nan %d -> next %u", t->k, tok, row[0],
            row[1], xx[0], xx[1], mx, nan, *(volatile uint32_t*)(uintptr_t)t->g_res);
    }
    return tok;
}

int t3_generate(Engine* e, const uint16_t* text, int n_text, uint16_t* out, int max_out, uint32_t seed, int greedy) {
    uint64_t t0 = sceKernelReadTsc();
    if (t3_start(e, text, n_text, max_out, seed, greedy)) return -1;
    int n_out = 0, tok;
    while ((tok = t3_next(e)) >= 0) out[n_out++] = (uint16_t)tok;
    e->t3.last_seconds = (double)(sceKernelReadTsc() - t0) / e->tsc_freq;
    return tok == -2 ? -1 : n_out;
}
