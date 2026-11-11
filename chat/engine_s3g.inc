// ---- text to speech, second half: S3Gen (meanflow) + HiFT (speech tokens -> 24 kHz waveform) ---------------------
// Included at the end of engine.c. Reference: pc/s3gen_ref.py; kernels in shaders/s3gen.cl.
// Memory: the weights (95 MiB, /data/chatterbox-s3gen-mf.bin) go where s3g_load is told (voice mode: the second
// half of the language model's context, engine_kv_lend); the work buffers in a scratch block (Parakeet's, which is
// idle while Garlic speaks). A sentence runs as steps (s3g_begin, then s3g_step until it returns -1), so the
// screen can be redrawn between them and the first samples can play before the last are made.

#define S3_PAD 32       // zero frames before and after each buffer (the widest conv reaches 25 frames out)
#define S3_HALO 20      // HiFT chunks: mel frames of context each side (its receptive field is ~19)

enum { K3_MMQ, K3_MMF16, K3_MMF32, K3_LN, K3_EMB, K3_RELPOS, K3_RELBIAS, K3_RELATTN, K3_ATTN, K3_MIX, K3_SNAKE,
       K3_CFMINIT, K3_F0, K3_NSFPH, K3_NSFSRC, K3_STFT, K3_ISTFT, K3_N, K3_ATTNS = K3_N, K3_ATTNS2 };
static const char* s3_kname[K3_N] = {"s3_mm_q4i", "s3_mm_f16i", "s3_mm_f32t", "s3_ln", "s3_embed_q4i", "s3_relpos",
                                     "s3_relbias", "s3_relattn2", "s3_attn", "s3_mix", "s3_snake", "s3_cfm_init",
                                     "s3_f0", "s3_nsf_phase", "s3_nsf_source", "s3_stft", "s3_istft"};
static int s3_attn_k = K3_ATTN;   // the CFM attention kernel in use (s3_attn_tune) and its threads per group
static uint32_t s3_attn_thr = 64;
static const int s3_ups[3][3] = {{8, 16, 256}, {5, 11, 128}, {3, 7, 64}};   // HiFT upsampling: stride, kernel, out
static const int s3_src_res_k[3] = {7, 7, 11}, s3_res_k[3] = {3, 7, 11}, s3_dils[3] = {1, 3, 5};

static int s3_cmp(const void* a, const void* b) { return strcmp(((const S3Tensor*)a)->name, ((const S3Tensor*)b)->name); }

static const S3Tensor* s3t(Engine* e, const char* fmt, ...) {
    S3Tensor key;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(key.name, sizeof(key.name), fmt, ap);
    va_end(ap);
    const S3Tensor* t = bsearch(&key, e->s3.t, e->s3.nt, sizeof(S3Tensor), s3_cmp);
    if (!t && !e->error[0]) fail(e, "S3Gen tensor %s missing", key.name);
    return t;
}

static uint64_t s3w(Engine* e, const S3Tensor* t) { return t ? e->s3.model + t->off : 0; }

// A voice file ("PS3V": u32 prompt tokens P, u32 mel frames, P x i32, frames x 80 f32, 80 f32, then optionally one
// f32: the gain for its audio, pc/export_ru_voice.py). 0 if it is fine.
static int s3_read_voice(const char* path, S3Voice* v) {
    size_t size;
    uint8_t* d = read_file(path, &size);
    if (!d || size < 12 || memcmp(d, "PS3V", 4)) {
        free(d);
        return -1;
    }
    memcpy(&v->np, d + 4, 4);
    memcpy(&v->nmel, d + 8, 4);
    if (size < 12 + v->np * 4ull + v->nmel * 320ull + 320 || v->nmel != 2 * v->np) {
        free(d);
        return -1;
    }
    v->ptok = malloc(v->np * 4);
    v->pmel = malloc(v->nmel * 320);
    memcpy(v->ptok, d + 12, v->np * 4);
    memcpy(v->pmel, d + 12 + v->np * 4, v->nmel * 320);
    memcpy(v->spk, d + 12 + v->np * 4 + v->nmel * 320, 320);
    size_t end = 12 + v->np * 4ull + v->nmel * 320ull + 320;
    v->gain = 1.0f;
    if (size >= end + 4) memcpy(&v->gain, d + end, 4);
    if (!(v->gain > 0.1f && v->gain < 10.0f)) v->gain = 1.0f;
    free(d);
    return 0;
}

void s3g_voice(Engine* e, const char* lang) {
    S3G* g = &e->s3;
    if (!g->nvoice) return;
    const S3Voice* v = &g->voice[0];
    for (int i = 1; i < g->nvoice; i++)
        if (!strncmp(g->voice[i].lang, lang, 2)) v = &g->voice[i];
    g->np = v->np, g->nmel = v->nmel, g->ptok = v->ptok, g->pmel = v->pmel, g->gain = v->gain;
    memcpy(g->spk, v->spk, sizeof(g->spk));
}

int s3g_init(Engine* e, const char* dir) {
    S3G* g = &e->s3;
    memset(g, 0, sizeof(*g));
    char path[256];
    size_t size;
    snprintf(path, sizeof(path), "%s/s3g.idx", dir);
    uint8_t* d = read_file(path, &size);
    if (!d || size < 148 || memcmp(d, "PS3G", 4)) {
        free(d);
        return -1;
    }
    memcpy(&g->nt, d + 4, 4);
    memcpy(g->path, d + 8, 128);
    g->path[128] = 0;
    weights_locate(g->path);
    memcpy(&g->size, d + 136, 8);
    if (size < 144 + (size_t)g->nt * 76) {
        free(d);
        return -1;
    }
    g->t = calloc(g->nt, sizeof(S3Tensor));
    for (uint32_t i = 0; i < g->nt; i++) {
        const uint8_t* r = d + 144 + i * 76;
        S3Tensor* t = &g->t[i];
        memcpy(t->name, r, 32);
        t->name[31] = 0;
        memcpy(&t->off, r + 32, 8);
        memcpy(&t->size, r + 40, 8);
        memcpy(&t->scales, r + 48, 8);
        memcpy(&t->rows, r + 56, 4);
        memcpy(&t->cols, r + 60, 4);
        memcpy(&t->fmt, r + 64, 4);
        memcpy(&t->cin, r + 68, 4);
        memcpy(&t->nout, r + 72, 4);
    }
    free(d);
    qsort(g->t, g->nt, sizeof(S3Tensor), s3_cmp);
    // the voices: the built-in one, then the languages that have their own (the same speaker for the rest)
    snprintf(path, sizeof(path), "%s/s3g_voice.bin", dir);
    if (s3_read_voice(path, &g->voice[0])) return -1;
    g->nvoice = 1;
    static const char* langs[] = {"es", "pt", "ru", "it", "fr", "de", "en"};
    for (int i = 0; i < 7 && g->nvoice < 8; i++) {
        snprintf(path, sizeof(path), "%s/s3g_voice_%s.bin", dir, langs[i]);
        S3Voice* v = &g->voice[g->nvoice];
        if (s3_read_voice(path, v)) continue;
        snprintf(v->lang, sizeof(v->lang), "%s", langs[i]);
        g->nvoice++;
    }
    s3g_voice(e, "");
    for (int i = 0; i < K3_N; i++)
        if (!(g->k[i] = kernel(e, s3_kname[i]))) return -1;
    g->k[K3_ATTNS] = kernel(e, "s3_attn_s");   // (optional: an older kernels.bin has only s3_attn)
    g->k[K3_ATTNS2] = kernel(e, "s3_attn_s2");
    e->error[0] = 0;
    g->ready = 1;
    return 0;
}

// The CFM attention: s3_attn (K / V through LDS), s3_attn_s (K / V as scalars) or s3_attn_s2 (the same split
// between 2 waves), whichever is fastest on this console, timed once on made-up q, k, v of a 3 s sentence (464
// frames) in the scratch block; another kernel than s3_attn only if it gives the same output (relative difference
// < 1e-5: they add in a different order).
static void s3_attn_tune(Engine* e) {
    S3G* g = &e->s3;
    static int done;
    if (done) return;
    done = 1;
    const uint32_t T = 464;
    static const int ks[3] = {K3_ATTN, K3_ATTNS, K3_ATTNS2};
    static const uint32_t thr[3] = {64, 64, 128};
    static const char* names[3] = {"s3_attn", "s3_attn_s", "s3_attn_s2"};
    uint64_t qkv = (g->scratch + 255) & ~255ull, out[3];
    out[0] = qkv + T * 1536 * 4ull + 4096;
    for (int v = 1; v < 3; v++) out[v] = out[v - 1] + T * 512 * 4ull + 4096;
    if (out[2] + T * 512 * 4ull > g->scratch + g->scratch_size) return;
    float* x = (float*)(uintptr_t)qkv;
    uint32_t st = 12345;
    for (uint32_t i = 0; i < T * 1536; i++) {
        st ^= st << 13, st ^= st >> 17, st ^= st << 5;
        x[i] = ((float)(st >> 8) / 16777216.0f - 0.5f) * 6.0f;   // scores up to ~30 / 8: a peaked softmax, as real
    }
    __asm__ volatile("sfence" ::: "memory");
    double best[3] = {1e9, 1e9, 1e9};
    float rel[3] = {0, 1, 1};
    for (int rep = 0; rep < 2; rep++)
        for (int v = 0; v < 3; v++) {
            if (!g->k[ks[v]]) continue;
            begin(e);
            for (int r = 0; r < 8; r++)
                dispatch(e, g->k[ks[v]], args(e, "ppu", qkv, out[v], T), (T + 63) / 64 * 8, thr[v]);
            if (submit(e)) return;
            if (e->last_gpu_seconds / 8 < best[v]) best[v] = e->last_gpu_seconds / 8;
        }
    const volatile float* a = (const volatile float*)(uintptr_t)out[0];
    for (int v = 1; v < 3; v++) {
        if (!g->k[ks[v]]) continue;
        const volatile float* b = (const volatile float*)(uintptr_t)out[v];
        float dmax = 0, amax = 0;
        for (uint32_t i = 0; i < T * 512; i++) {
            float d = fabsf(a[i] - b[i]), m = fabsf(a[i]);
            if (!(d <= dmax)) dmax = d;   // (NaN counts as a difference)
            if (m > amax) amax = m;
        }
        rel[v] = amax > 0 ? dmax / amax : 1.0f;
    }
    int pick = 0;
    for (int v = 1; v < 3; v++)
        if (g->k[ks[v]] && rel[v] < 1e-5f && best[v] < best[pick]) pick = v;
    s3_attn_k = ks[pick], s3_attn_thr = thr[pick];
    say("S3Gen CFM attention (%u frames): s3_attn %.2f ms, s3_attn_s %.2f ms (difference %.1e), s3_attn_s2 %.2f ms "
        "(difference %.1e); using %s", T, 1000 * best[0], 1000 * best[1], (double)rel[1], 1000 * best[2],
        (double)rel[2], names[pick]);
}

int s3g_load(Engine* e, uint64_t weights, uint64_t wsize, uint64_t scratch, uint64_t ssize) {
    S3G* g = &e->s3;
    g->loaded = 0;
    if (!g->ready || wsize < g->size) {
        fail(e, "S3Gen needs %u MiB for its weights (%u given)", (unsigned)(g->size >> 20), (unsigned)(wsize >> 20));
        return -1;
    }
    int f = open(g->path, O_RDONLY);
    if (f < 0) {
        fail(e, "%s missing", g->path);
        return -1;
    }
    uint64_t done = 0;
    while (done < g->size) {
        ssize_t r = read(f, (uint8_t*)(uintptr_t)weights + done,
                         g->size - done < (8u << 20) ? g->size - done : (8u << 20));
        if (r <= 0) break;
        done += r;
    }
    close(f);
    if (done != g->size) {
        fail(e, "%s incomplete", g->path);
        return -1;
    }
    __asm__ volatile("sfence" ::: "memory");
    g->model = weights;
    g->scratch = (scratch + 255) & ~255ull;
    g->scratch_size = ssize - (g->scratch - scratch);
    g->loaded = 1;
    s3_attn_tune(e);
    return 0;
}

// ---- buffers and dispatch helpers ----
static uint64_t s3raw(S3G* g, uint64_t bytes) {
    uint64_t base = (g->top + 255) & ~255ull;
    g->top = base + bytes;
    return base;
}

static uint64_t s3buf(S3G* g, uint32_t T, uint32_t C) {   // [T][C] floats between S3_PAD zero frames; frame 0
    return s3raw(g, ((uint64_t)T + 2 * S3_PAD) * C * 4 + 2048) + (uint64_t)S3_PAD * C * 4;
}

static void s3mix(Engine* e, uint64_t Y, uint64_t X, uint32_t T, uint32_t C, uint32_t ldy, uint32_t ldx, float alpha,
                  float beta, float slope) {
    dispatch(e, e->s3.k[K3_MIX], args(e, "ppuuuufff", Y, X, T, C, ldy, ldx, (double)alpha, (double)beta, (double)slope),
             (uint32_t)(((uint64_t)T * C + 255) / 256), 256);
}

static void s3zero(Engine* e, uint64_t from, uint64_t to) {   // whole buffers, pads included
    uint64_t n = (to - from) / 4;
    for (uint64_t o = 0; o < n; o += 1u << 26) {
        uint64_t k = n - o < (1u << 26) ? n - o : (1u << 26);
        s3mix(e, from + o * 4, from + o * 4, 1, (uint32_t)k, 0, 0, 0.0f, 0.0f, 1.0f);
    }
}

// Y = act(X W^T + b) for n rows (a conv: X at its first input frame, ldx = stride * C).
static void s3mm(Engine* e, const S3Tensor* w, uint64_t X, uint64_t Y, uint32_t n, uint32_t ldx, uint32_t ldy,
                 const S3Tensor* b, uint32_t act, float alpha, float beta, uint32_t dil) {
    S3G* g = &e->s3;
    if (!w) return;
    uint64_t wa = s3w(e, w), ba = s3w(e, b);
    if (!ldx) ldx = w->cols;
    if (!ldy) ldy = w->rows;
    uint32_t groups = w->rows / 64 * ((n + 15) / 16);
    if (w->fmt == 1)
        dispatch(e, g->k[K3_MMQ], args(e, "pppppuuuuuuff", wa, wa + w->scales, ba, X, Y, w->cols, w->rows, n, ldx, ldy,
                                       act, (double)alpha, (double)beta), groups, 256);
    else if (w->fmt == 2)
        dispatch(e, g->k[K3_MMF16], args(e, "ppppuuuuuuuff", wa, ba, X, Y, w->cols, w->rows, n, ldx, ldy,
                                         w->cin ? w->cin : w->cols, dil, (double)alpha, (double)beta), groups, 256);
    else
        dispatch(e, g->k[K3_MMF32], args(e, "ppppuuuuuuu", wa, ba, X, Y, w->cols, w->rows, n, ldx, ldy, w->nout, act),
                 w->rows / 64 * n, 64);
}

static void s3ln(Engine* e, uint64_t X, uint64_t Y, uint32_t T, uint32_t cols, const char* p, const char* wn,
                 const char* bn, float eps, uint32_t act, const S3Tensor* tb, uint32_t ldx, uint32_t ldy) {
    const S3Tensor *w = s3t(e, "%s%s", p, wn), *b = s3t(e, "%s%s", p, bn);
    if (!w || !b) return;
    dispatch(e, e->s3.k[K3_LN], args(e, "pppppuuufu", X, s3w(e, w), s3w(e, b), s3w(e, tb), Y, cols, ldx ? ldx : cols,
                                     ldy ? ldy : cols, (double)eps, act), T, 256);
}

// ---- encoder: tokens (prompt | new) -> mu in columns 80..159 of xin ----
static void s3_conformer(Engine* e, int i, uint64_t x, uint64_t h, uint64_t qkv, uint64_t r, uint64_t att, uint64_t f,
                         uint64_t pe, uint64_t rb, uint32_t T) {
    char p[16];
    snprintf(p, sizeof(p), "enc.%d.", i);
    S3G* g = &e->s3;
    s3ln(e, x, h, T, 512, p, "ln1_w", "ln1_b", 1e-12f, 0, NULL, 0, 0);
    s3mm(e, s3t(e, "%swqkv", p), h, qkv, T, 0, 0, s3t(e, "%sbqkv", p), 0, 1, 0, 1);
    s3mm(e, s3t(e, "%swpos", p), pe, r, 2 * T - 1, 0, 0, NULL, 0, 1, 0, 1);
    uint64_t bu = s3w(e, s3t(e, "%sbias_u", p)), bv = s3w(e, s3t(e, "%sbias_v", p));
    uint32_t rows = 2 * T - 1;
    dispatch(e, g->k[K3_RELBIAS], args(e, "ppppu", r, bu, bv, rb, T), (rows * 8 + 63) / 64, 64);
    dispatch(e, g->k[K3_RELATTN], args(e, "pppppu", qkv, r, bu, rb, att, T), (T + 63) / 64 * 8, 64);
    s3mm(e, s3t(e, "%swo", p), att, x, T, 0, 0, s3t(e, "%sbo", p), 0, 1, 1, 1);
    s3ln(e, x, h, T, 512, p, "ln2_w", "ln2_b", 1e-12f, 0, NULL, 0, 0);
    s3mm(e, s3t(e, "%sff1_w", p), h, f, T, 0, 0, s3t(e, "%sff1_b", p), 1, 1, 0, 1);
    s3mm(e, s3t(e, "%sff2_w", p), f, x, T, 0, 0, s3t(e, "%sff2_b", p), 0, 1, 1, 1);
}

static void s3_encoder(Engine* e) {
    S3G* g = &e->s3;
    uint32_t T = g->T, T2 = g->T2;
    g->top = g->phase_base;
    uint64_t e0 = s3buf(g, T, 512), x = s3buf(g, T2, 512), h = s3buf(g, T2, 512), h1 = s3buf(g, T, 512);
    uint64_t qkv = s3buf(g, T2, 1536), r = s3buf(g, 2 * T2, 512), att = s3buf(g, T2, 512), f = s3buf(g, T2, 2048);
    uint64_t up = s3buf(g, T, 1024), pe1 = s3buf(g, 2 * T, 512), pe2 = s3buf(g, 2 * T2, 512);
    uint64_t rb = s3raw(g, 8ull * 2 * T2 * 4);
    s3zero(e, g->phase_base, g->top);
    dispatch(e, g->k[K3_RELPOS], args(e, "ppu", g->div, pe1, T), ((2 * T - 1) * 256 + 255) / 256, 256);
    dispatch(e, g->k[K3_RELPOS], args(e, "ppu", g->div, pe2, T2), ((2 * T2 - 1) * 256 + 255) / 256, 256);
    const S3Tensor* em = s3t(e, "enc.emb");
    if (!em) return;
    dispatch(e, g->k[K3_EMB], args(e, "ppppu", s3w(e, em), s3w(e, em) + em->scales, g->tok, e0, 512), T, 256);
    s3mm(e, s3t(e, "enc.in_w"), e0, h, T, 0, 0, s3t(e, "enc.in_b"), 0, 1, 0, 1);
    s3ln(e, h, x, T, 512, "enc.", "in_ln_w", "in_ln_b", 1e-5f, 0, NULL, 0, 0);
    s3mm(e, s3t(e, "enc.la1_w"), x, h1, T, 512, 0, s3t(e, "enc.la1_b"), 4, 1, 0, 1);              // frames t .. t+3
    s3mm(e, s3t(e, "enc.la2_w"), h1 - 2 * 512 * 4, x, T, 512, 0, s3t(e, "enc.la2_b"), 0, 1, 1, 1);   // t-2 .. t
    for (int i = 0; i < 6; i++) s3_conformer(e, i, x, h, qkv, r, att, f, pe1, rb, T);
    s3mm(e, s3t(e, "enc.up_w"), x - 2 * 512 * 4, up, T, 512, 0, s3t(e, "enc.up_b"), 0, 1, 0, 1);    // [T][2 x 512]
    s3mm(e, s3t(e, "enc.up_in_w"), up, h, T2, 0, 0, s3t(e, "enc.up_in_b"), 0, 1, 0, 1);
    s3ln(e, h, x, T2, 512, "enc.", "up_ln_w", "up_ln_b", 1e-5f, 0, NULL, 0, 0);
    for (int i = 6; i < 10; i++) s3_conformer(e, i, x, h, qkv, r, att, f, pe2, rb, T2);
    s3ln(e, x, h, T2, 512, "enc.", "out_ln_w", "out_ln_b", 1e-5f, 0, NULL, 0, 0);
    s3mm(e, s3t(e, "enc.proj_w"), h, g->xin + 80 * 4, T2, 0, 320, s3t(e, "enc.proj_b"), 0, 1, 0, 1);
}

// ---- CFM: one Euler step of the meanflow estimator on xin [T2][320] (columns 0..79 = x) ----
static void s3_resnet(Engine* e, int n, int step, uint64_t src, uint32_t cin, uint64_t dst, uint64_t a, uint64_t a2,
                      uint32_t T) {
    char p[16];
    snprintf(p, sizeof(p), "cfm.res%d.", n);
    s3mm(e, s3t(e, "%sc1_w", p), src - 2ull * cin * 4, a, T, cin, 0, s3t(e, "%sc1_b", p), 0, 1, 0, 1);
    s3ln(e, a, a2, T, 256, p, "c1_ln_w", "c1_ln_b", 1e-5f, 1, s3t(e, "cfm.tb%d.%d", step, n), 0, 0);
    s3mm(e, s3t(e, "%sc2_w", p), a2 - 2 * 256 * 4, a, T, 256, 0, s3t(e, "%sc2_b", p), 0, 1, 0, 1);
    s3ln(e, a, dst, T, 256, p, "c2_ln_w", "c2_ln_b", 1e-5f, 1, NULL, 0, 0);
    s3mm(e, s3t(e, "%sres_w", p), src, dst, T, cin, 0, s3t(e, "%sres_b", p), 0, 1, 1, 1);
}

static void s3_transformer(Engine* e, int m, uint64_t x, uint64_t h, uint64_t qkv, uint64_t att, uint64_t f, uint32_t T) {
    char p[16];
    snprintf(p, sizeof(p), "cfm.tf%d.", m);
    s3ln(e, x, h, T, 256, p, "ln1_w", "ln1_b", 1e-5f, 0, NULL, 0, 0);
    s3mm(e, s3t(e, "%swqkv", p), h, qkv, T, 0, 0, NULL, 0, 1, 0, 1);
    dispatch(e, e->s3.k[s3_attn_k], args(e, "ppu", qkv, att, T), (T + 63) / 64 * 8, s3_attn_thr);
    s3mm(e, s3t(e, "%swo", p), att, x, T, 0, 0, s3t(e, "%sbo", p), 0, 1, 1, 1);
    s3ln(e, x, h, T, 256, p, "ln3_w", "ln3_b", 1e-5f, 0, NULL, 0, 0);
    s3mm(e, s3t(e, "%sff1_w", p), h, f, T, 0, 0, s3t(e, "%sff1_b", p), 3, 1, 0, 1);
    s3mm(e, s3t(e, "%sff2_w", p), f, x, T, 0, 0, s3t(e, "%sff2_b", p), 0, 1, 1, 1);
}

static void s3_cfm_step(Engine* e, int step) {
    S3G* g = &e->s3;
    uint32_t T = g->T2;
    g->top = g->phase_base;
    uint64_t x = s3buf(g, T, 256), y = s3buf(g, T, 256), a = s3buf(g, T, 256), a2 = s3buf(g, T, 256);
    uint64_t h = s3buf(g, T, 256), qkv = s3buf(g, T, 1536), att = s3buf(g, T, 512), f = s3buf(g, T, 1024);
    uint64_t cat = s3buf(g, T, 512), v = s3buf(g, T, 80);
    s3zero(e, g->phase_base, g->top);
    s3_resnet(e, 0, step, g->xin, 320, x, a, a2, T);
    for (int m = 0; m < 4; m++) s3_transformer(e, m, x, h, qkv, att, f, T);
    s3mix(e, cat + 256 * 4, x, T, 256, 512, 256, 0, 1, 1);                       // skip -> cat[:, 256:]
    s3mm(e, s3t(e, "cfm.down_w"), x - 2 * 256 * 4, y, T, 256, 0, s3t(e, "cfm.down_b"), 0, 1, 0, 1);
    for (int n = 1; n <= 12; n++) {
        uint64_t src = n % 2 ? y : x, dst = n % 2 ? x : y;
        s3_resnet(e, n, step, src, 256, dst, a, a2, T);
        for (int m = 4 * n; m < 4 * n + 4; m++) s3_transformer(e, m, dst, h, qkv, att, f, T);
    }
    s3mix(e, cat, y, T, 256, 512, 256, 0, 1, 1);                                  // x -> cat[:, :256]
    s3_resnet(e, 13, step, cat, 512, x, a, a2, T);
    for (int m = 52; m < 56; m++) s3_transformer(e, m, x, h, qkv, att, f, T);
    s3mm(e, s3t(e, "cfm.up_w"), x - 2 * 256 * 4, y, T, 256, 0, s3t(e, "cfm.up_b"), 0, 1, 0, 1);
    s3mm(e, s3t(e, "cfm.fin_w"), y - 2 * 256 * 4, a, T, 256, 0, s3t(e, "cfm.fin_b"), 0, 1, 0, 1);
    s3ln(e, a, a2, T, 256, "cfm.", "fin_ln_w", "fin_ln_b", 1e-5f, 1, NULL, 0, 0);
    s3mm(e, s3t(e, "cfm.proj_w"), a2, v, T, 0, 80, s3t(e, "cfm.proj_b"), 0, 1, 0, 1);
    s3mix(e, g->xin, v, T, 80, 320, 80, 1, 0.5f, 1);                             // x += (r - t) v
}

// ---- HiFT ----
static void s3_resblock(Engine* e, uint64_t x, uint32_t T, uint32_t C, const char* p, int k, uint64_t t1, uint64_t t2) {
    S3G* g = &e->s3;
    for (int l = 0; l < 3; l++) {
        int d = s3_dils[l], pd = d * (k - 1) / 2;
        uint64_t a1 = s3w(e, s3t(e, "%s.%d.a1", p, l)), ia1 = s3w(e, s3t(e, "%s.%d.ia1", p, l));
        uint64_t a2 = s3w(e, s3t(e, "%s.%d.a2", p, l)), ia2 = s3w(e, s3t(e, "%s.%d.ia2", p, l));
        uint32_t groups = (uint32_t)(((uint64_t)T * C + 255) / 256);
        dispatch(e, g->k[K3_SNAKE], args(e, "ppppuuuu", x, t1, a1, ia1, T, C, C, C), groups, 256);
        s3mm(e, s3t(e, "%s.%d.c1_w", p, l), t1 - (uint64_t)pd * C * 4, t2, T, C, 0, s3t(e, "%s.%d.c1_b", p, l), 0, 1, 0,
             d);
        dispatch(e, g->k[K3_SNAKE], args(e, "ppppuuuu", t2, t2, a2, ia2, T, C, C, C), groups, 256);
        s3mm(e, s3t(e, "%s.%d.c2_w", p, l), t2 - (uint64_t)(k - 1) / 2 * C * 4, x, T, C, 0, s3t(e, "%s.%d.c2_b", p, l),
             0, 1, 1, 1);
    }
}

// F0, the NSF excitation and its STFT for the whole sentence (and the output buffer).
static void s3_hift_sources(Engine* e) {
    S3G* g = &e->s3;
    uint32_t M = g->M, Tp = g->nmel, L = 480 * M, F = 120 * M + 1;
    g->top = g->phase_base;
    g->mel = s3buf(g, M, 80);
    uint64_t fa = s3buf(g, M, 512), fb = s3buf(g, M, 512);
    g->f0 = s3raw(g, M * 4);
    uint64_t ph = s3raw(g, M * 9 * 4);
    g->s = s3raw(g, (uint64_t)L * 4);
    uint32_t sp = 8;   // the STFT keeps 8 zero frames each side (source conv 0 reads 7 out)
    g->S = s3raw(g, ((uint64_t)F + 2 * sp) * 18 * 4 + 1024) + sp * 18 * 4;
    g->out = s3raw(g, (uint64_t)L * 4 + 4096);
    g->glob_end = g->top;
    s3zero(e, g->phase_base, g->top);
    s3mix(e, g->mel, g->xin + (uint64_t)Tp * 320 * 4, M, 80, 80, 320, 0, 1, 1);
    uint64_t src = g->mel;
    uint32_t C0 = 80;
    for (int i = 0; i < 5; i++) {
        uint64_t dst = i % 2 ? fb : fa;
        s3mm(e, s3t(e, "hift.f0c%d_w", i), src - C0 * 4, dst, M, C0, 0, s3t(e, "hift.f0c%d_b", i), 5, 1, 0, 1);
        src = dst, C0 = 512;
    }
    dispatch(e, g->k[K3_F0], args(e, "ppppu", src, s3w(e, s3t(e, "hift.f0_cls_w")), s3w(e, s3t(e, "hift.f0_cls_b")),
                                  g->f0, 512), M, 64);
    dispatch(e, g->k[K3_NSFPH], args(e, "ppu", g->f0, ph, M), 1, 64);
    dispatch(e, g->k[K3_NSFSRC], args(e, "ppppppuu", g->f0, ph, s3w(e, s3t(e, "hift.src_w")),
                                      s3w(e, s3t(e, "hift.src_b")), g->phi, g->s, L, g->seed), (L + 255) / 256, 256);
    dispatch(e, g->k[K3_STFT], args(e, "pppuu", g->s, s3w(e, s3t(e, "hift.stft_basis")), g->S, L, F),
             (F * 18 + 255) / 256, 256);
}

// Bytes of the two ping-pong regions a chunk of T mel frames uses: stages 0 and 2 (+ the spectrum) in region 0,
// conv_pre's output and stage 1 in region 1 (5 buffers of [frames][channels] per stage).
static void s3_chunk_regions(uint32_t T, uint64_t r[2]) {
    uint64_t pad = 2 * S3_PAD, sl = 2048 + 256;
    uint64_t t0 = 8ull * T, t1 = 40ull * T, t2 = 120ull * T + 1;
    uint64_t r0a = 5 * ((t0 + pad) * 256 * 4 + sl), r0b = 5 * ((t2 + pad) * 64 * 4 + sl) + (t2 + pad) * 18 * 4 + sl;
    uint64_t r1a = 5 * ((t1 + pad) * 128 * 4 + sl), r1b = ((uint64_t)T + pad) * 512 * 4 + sl;
    r[0] = ((r0a > r0b ? r0a : r0b) + 4095) & ~4095ull;
    r[1] = ((r1a > r1b ? r1a : r1b) + 4095) & ~4095ull;
}

static uint64_t s3_chunk_bytes(uint32_t T) {
    uint64_t r[2];
    s3_chunk_regions(T, r);
    return r[0] + r[1];
}

// Input frames q+lo .. q+hi that output phase r of a ConvTranspose1d (stride u, kernel k, padding pad) reads.
static void s3_poly(int r, int u, int k, int pad, int* lo, int* hi) {
    *lo = 1 << 30, *hi = -(1 << 30);
    for (int j = 0; j < k; j++) {
        int d = r + pad - j, q = d >= 0 ? d / u : -((-d + u - 1) / u);   // floor(d / u)
        if (d != q * u) continue;
        if (q < *lo) *lo = q;
        if (q > *hi) *hi = q;
    }
}

// HiFT over mel frames [a, b) as if alone, writing the exact samples of its centre [c0, c1) to out.
static void s3_hift_chunk(Engine* e, uint32_t a, uint32_t b, uint32_t c0, uint32_t c1) {
    S3G* g = &e->s3;
    uint32_t T = b - a, M = g->M;
    uint64_t rsize[2];
    s3_chunk_regions(T, rsize);
    uint64_t region[2] = {(g->glob_end + 4095) & ~4095ull, 0};
    region[1] = region[0] + rsize[0];
    s3zero(e, region[1], region[1] + rsize[1]);
    g->top = region[1];
    uint64_t x = s3buf(g, T, 512);
    s3mm(e, s3t(e, "hift.pre_w"), g->mel + ((int64_t)a - 3) * 80 * 4, x, T, 80, 0, s3t(e, "hift.pre_b"), 0, 1, 0, 1);
    uint32_t C = 512, Tc = T;
    for (int i = 0; i < 3; i++) {
        int u = s3_ups[i][0], kup = s3_ups[i][1], cout = s3_ups[i][2], padc = (kup - u) / 2;
        s3mix(e, x, x, Tc, C, C, C, 1, 0, 0.1f);                                         // leaky 0.1
        uint32_t Tn = u * Tc + (i == 2 ? 1 : 0);
        int ri = i == 1;   // stages 0 and 2 in region 0, stage 1 in region 1 (each over the one before last)
        s3zero(e, region[ri], region[ri] + rsize[ri]);
        g->top = region[ri];
        uint64_t y = s3buf(g, Tn, cout), si = s3buf(g, Tn, cout), t1 = s3buf(g, Tn, cout), t2b = s3buf(g, Tn, cout);
        uint64_t acc = s3buf(g, Tn, cout);
        if (g->top > g->scratch + g->scratch_size) return;
        uint64_t yo = y + (i == 2 ? (uint64_t)cout * 4 : 0);   // stage 2: frame 0 is the reflection pad / 120a
        // polyphase groups of the ConvTranspose (s3gen_ref.poly_groups): consecutive phases reading one window
        for (int r = 0, gidx = 0; r < u; gidx++) {
            int lo, hi, lo2, hi2, r1 = r + 1;
            s3_poly(r, u, kup, padc, &lo, &hi);
            while (r1 < u && (s3_poly(r1, u, kup, padc, &lo2, &hi2), lo2 == lo && hi2 == hi)) r1++;
            s3mm(e, s3t(e, "hift.up%d.%d_w", i, gidx), x + (int64_t)lo * C * 4, yo + (uint64_t)r * cout * 4, Tc, C,
                 (uint32_t)(u * cout), s3t(e, "hift.up%d.%d_b", i, gidx), 0, 1, 0, 1);
            r = r1;
        }
        if (i == 2 && a == 0) s3mix(e, y, y + 2ull * cout * 4, 1, cout, cout, cout, 0, 1, 1);   // ReflectionPad1d((1,0))
        C = cout, Tc = Tn;
        uint32_t g0 = i == 0 ? 8 * a : i == 1 ? 40 * a : 120 * a;   // global stage frame of local frame 0
        if (i == 0)
            s3mm(e, s3t(e, "hift.sd0_w"), g->S + ((int64_t)15 * g0 - 7) * 18 * 4, si, Tc, 15 * 18, 0, s3t(e, "hift.sd0_b"),
                 0, 1, 0, 1);
        else if (i == 1)
            s3mm(e, s3t(e, "hift.sd1_w"), g->S + ((int64_t)3 * g0 - 1) * 18 * 4, si, Tc, 3 * 18, 0, s3t(e, "hift.sd1_b"),
                 0, 1, 0, 1);
        else
            s3mm(e, s3t(e, "hift.sd2_w"), g->S + (uint64_t)g0 * 18 * 4, si, Tc, 18, 0, s3t(e, "hift.sd2_b"), 0, 1, 0, 1);
        char p[16];
        snprintf(p, sizeof(p), "hift.sr%d", i);
        s3_resblock(e, si, Tc, C, p, s3_src_res_k[i], t1, t2b);
        s3mix(e, y, si, Tc, C, C, C, 1, 1, 1);                                           // x += source branch
        for (int j = 0; j < 3; j++) {   // mean of the 3 ResBlocks: acc, then si as the second / third
            uint64_t dst = j == 0 ? acc : si;
            s3mix(e, dst, y, Tc, C, C, C, 0, 1, 1);
            snprintf(p, sizeof(p), "hift.rb%d.%d", i, j);
            s3_resblock(e, dst, Tc, C, p, s3_res_k[j], t1, t2b);
            if (j) s3mix(e, acc, si, Tc, C, C, C, 1, 1, 1);
        }
        x = t1;
        s3mix(e, x, acc, Tc, C, C, C, 0, 1.0f / 3.0f, 1);
    }
    s3mix(e, x, x, Tc, C, C, C, 1, 0, 0.01f);
    uint64_t spec = s3buf(g, Tc, 18);
    if (g->top > g->scratch + g->scratch_size) return;
    s3mm(e, s3t(e, "hift.post_w"), x - 3 * 64 * 4, spec, Tc, 64, 18, s3t(e, "hift.post_b"), 0, 1, 0, 1);
    uint32_t n = 480 * (c1 - c0), F = 120 * M + 1;
    dispatch(e, g->k[K3_ISTFT], args(e, "pppuuuuu", spec, s3w(e, s3t(e, "hift.istft_basis")),
                                     g->out + (uint64_t)480 * c0 * 4, 18, 120 * a, 480 * c0, n, F), (n + 255) / 256, 256);
}

// ---- one sentence ----
static float s3_gauss_cpu(uint32_t* st) {   // N(0, 1) for the CFM start noise (xorshift + Box-Muller)
    float u[2];
    for (int k = 0; k < 2; k++) {
        uint32_t x = *st;
        x ^= x << 13, x ^= x >> 17, x ^= x << 5;
        *st = x;
        u[k] = ((float)(x >> 8) + 0.5f) * (1.0f / 16777216.0f);
    }
    return sqrtf(-2.0f * logf(u[0])) * cosf(6.2831853f * u[1]);
}

int s3g_begin(Engine* e, const uint16_t* tokens, int n, uint32_t seed) {
    S3G* g = &e->s3;
    if (!g->loaded || n < 2) return -1;
    e->error[0] = 0;
    g->t0 = sceKernelReadTsc();
    // the longest sentence the scratch block takes (each buffer grows with it)
    for (;; n = n * 9 / 10) {
        uint64_t T = g->np + n, T2 = 2 * T, M = T2 - g->nmel, pd = 2 * S3_PAD;
        uint64_t fixed = (T2 + pd) * 320 * 4 + 1024 + T * 4 + T2 * 320 + 320 + g->nmel * 320 + 64 + 16 * 4096;
        uint64_t enc = (T + pd) * 4 * 2048 + (2 * T + pd) * 2048 + (T2 + pd) * 4 * (512 * 3 + 1536 + 2048) +
                       (2 * T2 + pd) * 4096 + 16 * T2 * 4 + 16 * 4096;
        uint64_t cfm = (T2 + pd) * 4 * (256 * 5 + 1536 + 512 + 1024 + 512 + 80) + 16 * 2048;
        uint64_t hift = (M + pd) * 4 * (80 + 1024) + M * 40 + 480 * M * 8 + (120 * M + 17) * 72 + 16384;
        uint64_t big = enc > cfm ? enc : cfm;
        g->chunk = 0;
        for (uint32_t c = 96; c >= 16; c -= 16)
            if (fixed + hift + s3_chunk_bytes(c + 2 * S3_HALO) + 65536 <= g->scratch_size) {
                g->chunk = c;
                break;
            }
        if (fixed + big + 65536 <= g->scratch_size && g->chunk) break;
        if (n < 8) {
            fail(e, "S3Gen scratch too small (%u MiB)", (unsigned)(g->scratch_size >> 20));
            return -1;
        }
    }
    g->n_tok = n;
    g->T = g->np + n, g->T2 = 2 * g->T, g->M = g->T2 - g->nmel;
    g->seed = seed;
    g->top = g->scratch;
    g->xin = s3buf(g, g->T2, 320);
    uint64_t xin_end = g->top;
    g->div = s3raw(g, 1024);
    g->tok = s3raw(g, g->T * 4ull);
    uint64_t noise = s3raw(g, (uint64_t)g->T2 * 320), spk = s3raw(g, 320), pmel = s3raw(g, (uint64_t)g->nmel * 320);
    g->phi = s3raw(g, 64);
    g->phase_base = (g->top + 4095) & ~4095ull;
    uint32_t* tk = (uint32_t*)(uintptr_t)g->tok;
    for (uint32_t i = 0; i < g->np; i++) tk[i] = (uint32_t)g->ptok[i];
    for (int i = 0; i < n; i++) tk[g->np + i] = tokens[i] < 6561 ? tokens[i] : 0;
    float* dv = (float*)(uintptr_t)g->div;
    float lnf = (float)(-(log(10000.0) / 512));
    for (int k = 0; k < 256; k++) dv[k] = (float)exp((double)((float)(2 * k) * lnf));
    uint32_t st = seed * 2654435761u + 0x9E3779B9u;
    if (!st) st = 1;
    float* nz = (float*)(uintptr_t)noise;
    for (uint32_t i = 0; i < g->T2 * 80; i++) nz[i] = s3_gauss_cpu(&st);
    memcpy((void*)(uintptr_t)spk, g->spk, 320);
    memcpy((void*)(uintptr_t)pmel, g->pmel, (size_t)g->nmel * 320);
    float* ph = (float*)(uintptr_t)g->phi;
    ph[0] = 0;
    for (int h = 1; h < 9; h++) {   // uniform phases, in cycles
        uint32_t x = st;
        x ^= x << 13, x ^= x >> 17, x ^= x << 5;
        st = x;
        ph[h] = (float)(x >> 8) / 16777216.0f - 0.5f;
    }
    begin(e);
    s3zero(e, g->scratch, xin_end);
    dispatch(e, g->k[K3_CFMINIT], args(e, "ppppuu", g->xin, noise, spk, pmel, g->T2, g->nmel), (g->T2 * 80 + 255) / 256,
             256);
    s3_encoder(e);
    g->step = 0;
    if (e->error[0] || submit(e)) return -1;
    g->s_enc = e->last_gpu_seconds, g->s_cfm = g->s_src = g->s_hift = 0;
    return n;
}

int s3g_step(Engine* e, float* out, int max) {
    S3G* g = &e->s3;
    if (!g->loaded || g->step < 0) return -1;
    int s = g->step++;
    if (s < 2) {   // the two Euler steps
        begin(e);
        s3_cfm_step(e, s);
        if (e->error[0] || submit(e)) return -2;
        g->s_cfm += e->last_gpu_seconds;
        return 0;
    }
    if (s == 2) {   // F0, excitation, STFT
        begin(e);
        s3_hift_sources(e);
        g->c0 = 0;
        if (e->error[0] || submit(e)) return -2;
        g->s_src = e->last_gpu_seconds;
        return 0;
    }
    uint32_t M = g->M;
    if (g->c0 >= M) {
        g->step = -1;
        g->last_seconds = (double)(sceKernelReadTsc() - g->t0) / e->tsc_freq;
        return -1;
    }
    uint32_t c0 = g->c0, c1 = c0 + g->chunk < M ? c0 + g->chunk : M;
    uint32_t a = c0 > S3_HALO ? c0 - S3_HALO : 0, b = c1 + S3_HALO < M ? c1 + S3_HALO : M;
    begin(e);
    s3_hift_chunk(e, a, b, c0, c1);
    if (g->top > g->scratch + g->scratch_size) {
        fail(e, "S3Gen chunk does not fit");
        return -2;
    }
    if (e->error[0] || submit(e)) return -2;
    g->s_hift += e->last_gpu_seconds;
    g->c0 = c1;
    // samples [480 c0, 480 c1), cut to the official length (the last token's 40 ms dropped)
    uint32_t total = (uint32_t)(g->n_tok - 1) * 960, i0 = 480 * c0, i1 = 480 * c1 < total ? 480 * c1 : total;
    int k = i1 > i0 ? (int)(i1 - i0) : 0;
    if (k > max) k = max;
    memcpy(out, (const float*)(uintptr_t)g->out + i0, (size_t)k * 4);
    return k;
}
