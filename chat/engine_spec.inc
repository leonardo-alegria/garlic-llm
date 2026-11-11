// Garlic LLM (the 9B): speculative decoding with Qwen3.5's own MTP head.
// Included at the end of engine.c with GARLIC_SPEC (uses its dispatch / submit / swar_mm). Text chat only: the head,
// its table of draft rows and its cache live in T3's weights memory, which only voice mode uses (spec_load lends it;
// t3 weights are read again when voice mode starts, main.c).
//
// The MTP head (as vLLM's Qwen3_5MultiTokenPredictor): index i of its own sequence takes (h_i, x_{i+1}) - the 9B's
// hidden state after the final norm at position i and the embedding of the next token - as
// fc([rmsnorm_e(emb), rmsnorm_h(h)]), one full-attention layer (its own TurboQuant cache, RoPE position i) and a
// norm; lm_head of that is the draft for x_{i+2}, and the output is the h of the next chained step.
//
// A cycle (engine_spec_step), two submissions:
//   V: the 9B over [tok, d_1..d_K] at pos..pos+K (SWAR products for n = K + 1 rows; the Gated DeltaNet does not store
//      its state), logits of every row, spec_accept on the GPU (exact: the output follows the same distribution as
//      one token at a time), then the DeltaNet state moved forward by the j + 1 tokens kept (conv_commit, delta_rows_v).
//   D: the MTP head over the j + 1 new pairs (h_{pos+r}, x_{pos+r+1}) - the last one with the token just drawn - and
//      the draft chain from its last output: up to K_MAX steps, each proposed only if the draft's top probability is
//      at least theta (draft_sample stops the chain and the gated kernels skip the rest).
// The prompt (engine_run, batched) also feeds the head with its pairs, so the draft sees the whole conversation.

#define SPEC_KMAX 3
#define SPEC_ROWS PF_ROWS            // rows of the MTP buffers (a prompt batch)
#define ST_STOP 0
#define ST_K 1
#define ST_D 2
#define ST_J 8
#define ST_NEXT 9
#define ST_COMMIT 10
#define ST_REJ 11
#define ST_STOPROWS 12
#define MT_PRE_E 0
#define MT_PRE_H 1
#define MT_FC 2
#define MT_ATTN_NORM 3
#define MT_MLP_NORM 4
#define MT_Q_NORM 5
#define MT_K_NORM 6
#define MT_QKV 7
#define MT_O 8
#define MT_GU 9
#define MT_DOWN 10
#define MT_NORM 11
#define MT_N 12

typedef struct {
    int idx_ok, ready, on;       // mtp.idx read; weights in memory; the conversation's pairs are all in its cache
    int drafted;                  // st holds a draft for the token main.c holds (engine_spec_step can verify it)
    char path[129];
    uint64_t wsize, tsize;        // tensors, draft table (nd rows of q4row) in the weights file
    uint32_t nd;                  // draft rows: the first nd rows of lm_head (the most frequent tokens)
    Tensor t[MT_N];
    uint64_t base, size;          // the lent memory (T3's weights)
    uint64_t model, table, kc, vc, tq_split;
    uint64_t E, Hc, X, x, xn, O, att, Q, L, raw, xs, st;
    const GpuKernel *k_emb, *k_draft, *k_accept, *k_convv, *k_commit, *k_deltav, *k_copyg, *k_rmsg;
    float theta;
    int kmax;
    int trace;                    // (unused)
    // The head's own index of conversation position p is p - pbase: after voice mode it starts again from where the
    // conversation is (pbase), seeing only what comes after. RoPE attention depends only on the distance between
    // positions, so for the head this is the same as a conversation that began there.
    uint32_t pbase;
    uint32_t stop_rows[4];
    // statistics since the last spec_stats_reset
    uint64_t cycles, proposed, kept, tokens;
    double gpu_v, gpu_d, wall;
} Spec;

static Spec g_spec;

static uint64_t SW(Engine* e, int k) { (void)e; return g_spec.model + g_spec.t[k].off; }

static const GpuKernel* kern_opt(Engine* e, const char* name) {
    for (int i = 0; i < e->nk; i++)
        if (!strcmp(e->k[i].name, name)) return &e->k[i];
    return NULL;
}

// mtp.idx (pc/convert_9b_q4s.py --mtp): "PMTP", nd, path[128], wsize (Q), tsize (Q), then MT_N records "<QQIIQI".
int spec_init(Engine* e, const char* dir) {
    Spec* s = &g_spec;
    memset(s, 0, sizeof(*s));
    char p[256];
    size_t sz;
    snprintf(p, sizeof(p), "%s/mtp.idx", dir);
    uint8_t* d = read_file(p, &sz);
    if (!d || sz < 156 + MT_N * 36 || memcmp(d, "PMTP", 4) || !e->swar || !e->ids) {
        free(d);
        return -1;
    }
    memcpy(&s->nd, d + 4, 4);
    memcpy(s->path, d + 8, 128);
    s->path[128] = 0;
    weights_locate(s->path);
    memcpy(&s->wsize, d + 136, 8);
    memcpy(&s->tsize, d + 144, 8);
    for (int i = 0; i < MT_N; i++) {
        const uint8_t* r = d + 156 + i * 36;
        memcpy(&s->t[i].off, r, 8);
        memcpy(&s->t[i].size, r + 8, 8);
        memcpy(&s->t[i].rows, r + 16, 4);
        memcpy(&s->t[i].cols, r + 20, 4);
        memcpy(&s->t[i].scales, r + 24, 8);
        memcpy(&s->t[i].fmt, r + 32, 4);
    }
    free(d);
    s->k_emb = kern_opt(e, "embed_q4row"), s->k_draft = kern_opt(e, "draft_sample");
    s->k_accept = kern_opt(e, "spec_accept"), s->k_convv = kern_opt(e, "conv_silu_rows_v");
    s->k_commit = kern_opt(e, "conv_commit"), s->k_deltav = kern_opt(e, "delta_rows_v");
    s->k_copyg = kern_opt(e, "copy_f32_g"), s->k_rmsg = kern_opt(e, "rmsnorm_g");
    if (!s->k_emb || !s->k_draft || !s->k_accept || !s->k_convv || !s->k_commit || !s->k_deltav || !s->k_copyg ||
        !s->k_rmsg || !e->tq || !e->pf || s->nd > e->vocab || s->nd % 128)
        return -1;
    s->theta = 0.5f;   // the defaults: theta 0.5, K 2
    s->kmax = 2;
    {   // /data/garlic_spec.txt: "<kmax> <theta>" (kmax 0 = off)
        char* f = (char*)read_file("/data/garlic_spec.txt", &sz);
        if (f && sz) {
            char b[64] = {0};
            memcpy(b, f, sz < 63 ? sz : 63);
            int k = atoi(b);
            char* sp = strchr(b, ' ');
            if (k >= 0 && k <= SPEC_KMAX) s->kmax = k;
            if (sp) {
                float th = (float)atof(sp + 1);
                if (th >= 0.0f && th <= 2.0f) s->theta = th;   // (above 1: never propose; tests)
            }
        }
        free(f);
    }
    s->idx_ok = 1;
    return 0;
}

// Memory the head needs in the lent block: weights + draft table + cache (16K, TurboQuant) + buffers.
static uint64_t spec_layout(Engine* e, uint64_t base) {
    Spec* s = &g_spec;
    uint64_t p = base;
#define TAKE(field, bytes) (s->field = (p = (p + 255) & ~255ull), p += (bytes), s->field)
    TAKE(model, s->wsize);
    TAKE(table, s->tsize);
    uint64_t kb = (uint64_t)e->n_kv * 8192 * (e->head_dim / 2 + 4);
    TAKE(kc, 4 * kb);
    s->vc = s->kc + kb;
    s->tq_split = 2 * kb;
    uint64_t H = e->hidden, R = SPEC_ROWS;
    uint32_t nl = 0, ld = 0, npad = e->t[e->head_idx].rows;
    for (uint32_t i = 0; i < e->n_layers; i++)
        if (!(e->full_mask >> i & 1)) nl++, ld = e->t[e->layer_base[i] + 2].rows;
    TAKE(E, R * H * 4);
    TAKE(Hc, (R + 1) * H * 4);
    TAKE(X, R * 2 * H * 4);
    TAKE(x, R * H * 4);
    TAKE(xn, R * H * 4);
    TAKE(O, R * H * 4);
    TAKE(att, R * (uint64_t)e->n_heads * e->head_dim * 4);
    TAKE(Q, (uint64_t)SPEC_KMAX * s->nd * 4);
    TAKE(L, (uint64_t)(SPEC_KMAX + 1) * npad * 4);
    TAKE(raw, (uint64_t)nl * (SPEC_KMAX + 1) * e->conv_n * 4);
    TAKE(xs, (uint64_t)nl * (SPEC_KMAX + 1) * ld * 4);
    TAKE(st, 256);
#undef TAKE
    return p - base;
}

// The head into the lent block [base, base + size): reads the weights and the draft table (~200 MiB, ~2.5 s).
// start: the conversation position the head follows from (0 for an empty one; after voice mode, where it is).
int spec_load(Engine* e, uint64_t base, uint64_t size, int start) {
    Spec* s = &g_spec;
    if (!s->idx_ok) return -1;
    s->ready = s->on = s->drafted = 0;
    uint64_t need = spec_layout(e, base);
    if (need > size) {
        say("spec: the MTP head needs %u MiB, the block has %u", (unsigned)(need >> 20), (unsigned)(size >> 20));
        return -1;
    }
    int f = open(s->path, O_RDONLY);
    if (f < 0) {
        say("spec: cannot open %s", s->path);
        return -1;
    }
    uint64_t total = s->wsize + s->tsize, done = 0;
    while (done < total) {
        uint64_t n = total - done < (32ull << 20) ? total - done : 32ull << 20;
        uint64_t dst = done < s->wsize ? s->model + done : s->table + (done - s->wsize);
        if (done < s->wsize && done + n > s->wsize) n = s->wsize - done;
        ssize_t r = pread(f, (uint8_t*)(uintptr_t)dst, n, (off_t)done);
        if (r <= 0) break;
        done += r;
    }
    close(f);
    if (done != total) {
        say("spec: %s incomplete", s->path);
        return -1;
    }
    s->base = base, s->size = size;
    {   // the head's matrix shapes and the draft's rows of lm_head, for the SWAR variants (once)
        Tensor d = e->t[e->head_idx];
        d.rows = s->nd;
        swar_tune_shape(e, &d, W(e, e->head_idx));
        for (int i = 0; i < MT_N; i++)
            if (s->t[i].fmt == 6) swar_tune_shape(e, &s->t[i], SW(e, i));
    }
    uint32_t* st = (uint32_t*)(uintptr_t)s->st;
    memset(st, 0, 256);
    uint32_t stops[4] = {e->im_end, e->eos2, 248069u /* </think> */, NO_TOKEN};
    for (int i = 0; i < 4; i++) {
        int32_t r = stops[i] == NO_TOKEN ? -1 : engine_row_of(e, stops[i]);
        s->stop_rows[i] = st[ST_STOPROWS + i] = r < 0 ? NO_TOKEN : (uint32_t)r;
    }
    __asm__ volatile("sfence" ::: "memory");
    s->ready = 1;
    s->on = 1;
    s->pbase = start > 0 ? (uint32_t)start : 0;
    say("spec: MTP head ready (%u MiB of the %u lent; draft rows %u, K up to %d, theta %.2f)", (unsigned)(need >> 20),
        (unsigned)(size >> 20), s->nd, s->kmax, (double)s->theta);
    return 0;
}

void spec_unload(Engine* e) {
    (void)e;
    g_spec.ready = g_spec.on = g_spec.drafted = 0;
}

int spec_active(const Engine* e) { (void)e; return g_spec.ready && g_spec.on && g_spec.kmax > 0; }

// A new conversation (engine_reset): from position 0 every pair goes into the head's cache.
static void spec_reset(Engine* e) {
    (void)e;
    g_spec.on = g_spec.ready;
    g_spec.pbase = 0;
    g_spec.drafted = 0;
}

// The head over c rows: embeddings E (rows of hidden floats), hidden states Hs (after the 9B's final norm), MTP
// indices i0..i0 + c - 1; output O. gate: a GPU word that skips the heavy kernels when set (the draft chain).
static void mtp_rows(Engine* e, uint64_t E, uint64_t Hs, uint32_t c, uint32_t i0, uint64_t gate) {
    Spec* s = &g_spec;
    uint32_t H = e->hidden, nq = e->n_heads * e->head_dim, ld = s->t[MT_QKV].rows, G = e->n_heads + 2 * e->n_kv;
    double sc = 1.0 / sqrt((double)e->head_dim);
    // X = [rmsnorm_e(E) | rmsnorm_h(Hs)], then x = fc X
    dispatch(e, s->k_rmsg, args(e, "pppufuup", E, SW(e, MT_PRE_E), s->X, H, (double)e->eps, H, 2 * H, gate), c, 256);
    dispatch(e, s->k_rmsg, args(e, "pppufuup", Hs, SW(e, MT_PRE_H), s->X + H * 4, H, (double)e->eps, H, 2 * H, gate),
             c, 256);
    Tensor* t = s->t;
#define HEAD_MM(k, Xa, Ya, n, acc)                                                                               \
    do {                                                                                                         \
        const Tensor* tt = &t[k];                                                                                \
        uint64_t w_ = SW(e, k);                                                                                  \
        uint32_t nb_ = tt->cols / 32;                                                                            \
        dispatch(e, e->k_qx8, args(e, "ppppup", (Xa), e->x8, e->dx8, e->init8, (n) * nb_, gate),                \
                 ((n) * nb_ + 255) / 256, 256);                                                                  \
        for (uint32_t t0_ = 0; t0_ < (n); t0_ += 4) {                                                            \
            uint32_t c_ = (n) - t0_ < 4 ? (n) - t0_ : 4;                                                         \
            swar_chunk(e, tt, w_, swar_variant(e, tt, c_), c_, t0_, (Ya), (acc), gate);                          \
        }                                                                                                        \
    } while (0)
    HEAD_MM(MT_FC, s->X, s->x, c, 0);
    dispatch(e, s->k_rmsg, args(e, "pppufuup", s->x, SW(e, MT_ATTN_NORM), s->xn, H, (double)e->eps, H, H, gate), c, 256);
    HEAD_MM(MT_QKV, s->xn, e->pf_a, c, 0);
    for (uint32_t r = 0; r < c; r++) {
        uint64_t q = e->pf_a + (uint64_t)r * ld * 4;
        dispatch(e, e->k_qk_tq,
                 args(e, "pppppppuuufp", q, SW(e, MT_Q_NORM), SW(e, MT_K_NORM), e->rope, s->kc, s->vc, e->tq_tab, i0 + r,
                      e->n_heads, e->n_kv, (double)e->eps, s->tq_split),
                 G, 64);
    }
    for (uint32_t r = 0; r < c; r++) {
        uint64_t q = e->pf_a + (uint64_t)r * ld * 4, out = s->att + (uint64_t)r * nq * 4;
        uint32_t pos = i0 + r, blocks = pos / ATT_BLOCK + 1;
        dispatch(e, e->k_attn_tq,
                 args(e, "pppppuuufp", q, s->kc, s->vc, e->attn_part, e->tq_tab, pos, e->n_heads, e->n_kv, sc, s->tq_split),
                 blocks * e->n_kv, 256);
        dispatch(e, e->k_comb_tq, args(e, "pppuu", e->attn_part, out, e->tq_tab, e->n_heads, blocks), e->n_heads, 256);
    }
    uint64_t gatep = e->pf_a + (uint64_t)(e->n_heads + 2 * e->n_kv) * e->head_dim * 4;
    dispatch(e, e->k_sigr, args(e, "ppuuu", s->att, gatep, nq, ld, c), (c * nq + 255) / 256, 256);
    HEAD_MM(MT_O, s->att, s->x, c, 1);
    dispatch(e, s->k_rmsg, args(e, "pppufuup", s->x, SW(e, MT_MLP_NORM), s->xn, H, (double)e->eps, H, H, gate), c, 256);
    HEAD_MM(MT_GU, s->xn, e->pf_a, c, 0);
    dispatch(e, e->k_silur, args(e, "ppuu", e->pf_a, e->pf_h, e->inter, c), (c * e->inter + 255) / 256, 256);
    HEAD_MM(MT_DOWN, e->pf_h, s->x, c, 1);
    dispatch(e, s->k_rmsg, args(e, "pppufuup", s->x, SW(e, MT_NORM), s->O, H, (double)e->eps, H, H, gate), c, 256);
#undef HEAD_MM
}

// Draft step k (0-based) from the head's output row `orow`: the draft's logits over the first nd rows of lm_head, and
// draft_sample (which may stop the chain).
static void draft_logits(Engine* e, uint64_t orow, uint32_t k, float inv_temp, uint32_t seed, double min_gap) {
    Spec* s = &g_spec;
    Tensor t = e->t[e->head_idx];
    t.rows = s->nd;
    uint64_t w = W(e, e->head_idx), q = s->Q + (uint64_t)k * s->nd * 4;
    uint32_t nb = t.cols / 32;
    dispatch(e, e->k_qx8, args(e, "ppppup", orow, e->x8, e->dx8, e->init8, nb, s->st), (nb + 255) / 256, 256);
    swar_chunk(e, &t, w, swar_variant(e, &t, 1), 1, 0, q, 0, s->st);
    dispatch(e, s->k_draft, args(e, "pufuuffp", q, s->nd, (double)inv_temp, seed, k, min_gap, (double)s->theta, s->st),
             1, 256);
}

// The chain: from the head's output row `last` at index i_last, up to kmax steps (the first from that row).
static void draft_chain(Engine* e, uint32_t last, uint32_t i_last, float inv_temp, uint32_t seed, double min_gap,
                        int kmax) {
    Spec* s = &g_spec;
    uint64_t H = e->hidden;
    uint64_t orow = s->O + (uint64_t)last * H * 4;
    draft_logits(e, orow, 0, inv_temp, seed, min_gap);
    for (int k = 1; k < kmax; k++) {
        // the next pair: (output of the previous step, the token it proposed), at the next index; rows 0 of E / Hc+1
        dispatch(e, s->k_emb, args(e, "ppuupup", s->table, s->st, (uint32_t)(ST_D + k - 1), 0u, s->E, (uint32_t)H, s->st),
                 1, 256);
        dispatch(e, s->k_copyg, args(e, "ppup", orow, s->Hc + H * 4, (uint32_t)H, s->st), (uint32_t)(H / 256), 256);
        mtp_rows(e, s->E, s->Hc + H * 4, 1, i_last + k, s->st);
        orow = s->O;
        draft_logits(e, orow, (uint32_t)k, inv_temp, seed, min_gap);
    }
}

// Hook of the batched prompt (engine_run): after the 9B's layers for n tokens at pos0 (their embedding rows at
// `stage`), the head gets the pairs of these positions: (carry, x_pos0), (h_pos0, x_pos0+1), ..., and the last h
// becomes the carry (its pair is completed by the next token: the next batch, or the token drawn now).
static void spec_prefill(Engine* e, uint64_t stage, uint32_t n, uint32_t pos0) {
    Spec* s = &g_spec;
    uint64_t H = e->hidden;
    s->drafted = 0;
    dispatch(e, e->k_rmsr, args(e, "pppuf", e->pf_x, W(e, 1), s->Hc + H * 4, (uint32_t)H, (double)e->eps), n, 256);
    if (pos0 < s->pbase) {   // (cannot happen: the conversation went back before the head started following it)
        s->on = 0;
        return;
    }
    pos0 -= s->pbase;   // (the head's index)
    uint32_t first = pos0 == 0 ? 1 : 0, c = n - first;
    if (c) {
        dispatch(e, e->k_copy, args(e, "ppu", stage + first * H * 4, s->E, (uint32_t)(c * H)), (uint32_t)(c * H / 256), 256);
        mtp_rows(e, s->E, s->Hc + first * H * 4, c, pos0 - 1 + first, 0);
    }
    dispatch(e, e->k_copy, args(e, "ppu", s->Hc + (uint64_t)n * H * 4, s->Hc, (uint32_t)H), (uint32_t)(H / 256), 256);
}

// The state block as the CPU sees it (after a submission)
static volatile uint32_t* ST(void) { return (volatile uint32_t*)(uintptr_t)g_spec.st; }

// Submission D: the head over the pairs (Hc rows 0..c-1 with the tokens of rows `rows` / the CPU's staging row for
// the last one when it is not a draft row), then the chain. i0 = index of the first pair.
static int spec_draft(Engine* e, const uint32_t* rows, uint32_t c, uint32_t i0, int stage_last, float inv_temp,
                      uint32_t seed, int kmax) {
    Spec* s = &g_spec;
    uint64_t H = e->hidden;
    double min_gap = inv_temp > 0.0f ? 2.9957 / inv_temp : 0.0;
    volatile uint32_t* st = ST();
    st[ST_STOP] = 0, st[ST_K] = 0;
    __asm__ volatile("sfence" ::: "memory");
    begin(e);
    for (uint32_t r = 0; r < c; r++) {
        if (r == c - 1 && stage_last)
            dispatch(e, e->k_copy, args(e, "ppu", e->staging, s->E + (uint64_t)r * H * 4, (uint32_t)H), (uint32_t)(H / 256),
                     256);
        else
            dispatch(e, s->k_emb, args(e, "ppuupup", s->table, s->st, 0xFFFFu, rows[r], s->E + (uint64_t)r * H * 4,
                                       (uint32_t)H, (uint64_t)0), 1, 256);
    }
    mtp_rows(e, s->E, s->Hc, c, i0, 0);
    // the carry for a prompt that follows: h of the last pair
    dispatch(e, e->k_copy, args(e, "ppu", s->Hc + (uint64_t)(c - 1) * H * 4, s->Hc, (uint32_t)H), (uint32_t)(H / 256),
             256);
    if (kmax > 0) {
        // (draft_chain uses Hc row 1 for the chained h: the carry in row 0 stays)
        draft_chain(e, c - 1, i0 + c - 1, inv_temp, seed, min_gap, kmax);
    } else {
        st[ST_STOP] = 1;
    }
    if (submit(e)) return -1;
    s->gpu_d += e->last_gpu_seconds;
    s->drafted = 1;
    return 0;
}

// Submission V: the 9B over [tok, d_1..d_K] at pos..pos+K, acceptance, commit. Hc rows 1..K+1 get the final-norm
// hidden states (Hc row 0 keeps the carry: h at pos - 1).
static int spec_verify(Engine* e, uint32_t tok, uint32_t pos, uint32_t K, float inv_temp, uint32_t seed) {
    Spec* s = &g_spec;
    uint32_t H = e->hidden, n = K + 1;
    uint64_t rowb = (uint64_t)H * 4;
    volatile uint32_t* st = ST();
    st[ST_K] = K;
    __asm__ volatile("sfence" ::: "memory");
    int32_t r0 = engine_row_of_fast(e, tok);
    int staged = r0 < 0 || (uint32_t)r0 >= s->nd;
    if (staged && fetch_row(e, tok, (float*)(uintptr_t)e->staging)) return -1;
    __asm__ volatile("sfence" ::: "memory");
    begin(e);
    if (staged)
        dispatch(e, e->k_copy, args(e, "ppu", e->staging, e->pf_x, H), H / 256, 256);
    else
        dispatch(e, s->k_emb, args(e, "ppuupup", s->table, s->st, 0xFFFFu, (uint32_t)r0, e->pf_x, H, (uint64_t)0), 1, 256);
    for (uint32_t i = 0; i < K; i++)
        dispatch(e, s->k_emb, args(e, "ppuupup", s->table, s->st, (uint32_t)(ST_D + i), 0u, e->pf_x + (i + 1) * rowb, H,
                                   (uint64_t)0), 1, 256);
    uint32_t li = 0;
    for (uint32_t i = 0; i < e->n_layers; i++) {
        uint32_t b = e->layer_base[i];
        int full = e->full_mask >> i & 1;
        uint32_t mlp = b + (full ? 6 : 8);
        uint64_t kc = e->kc[i], vc = e->vc[i];
        dispatch(e, e->k_rmsr, args(e, "pppuf", e->pf_x, W(e, b), e->pf_xn, H, (double)e->eps), n, 256);
        if (!full) {
            uint32_t ld = e->t[b + 2].rows;
            uint64_t raw = s->raw + (uint64_t)li * (SPEC_KMAX + 1) * e->conv_n * 4;
            uint64_t xs = s->xs + (uint64_t)li * (SPEC_KMAX + 1) * ld * 4;
            mmr(e, b + 2, e->pf_xn, e->pf_a, n, 0);
            dispatch(e, s->k_convv, args(e, "pppuuup", e->pf_a, W(e, b + 3), kc, (uint32_t)e->conv_n, ld, n, raw),
                     (uint32_t)(e->conv_n + 255) / 256, 256);
            dispatch(e, e->k_copy, args(e, "ppu", e->pf_a, xs, n * ld), (n * ld + 255) / 256, 256);
            dispatch(e, s->k_deltav,
                     args(e, "ppppppuufuupu", e->pf_a, W(e, b + 4), W(e, b + 5), vc, W(e, b + 6), e->pf_h, e->n_kh,
                          e->n_vh, (double)e->eps, ld, n, (uint64_t)0, 0u),
                     e->n_vh, 256);
            mmr(e, b + 7, e->pf_h, e->pf_x, n, 1);
            li++;
        } else {
            uint32_t ld = e->t[b + 4].rows, nq = e->n_heads * e->head_dim, G = e->n_heads + 2 * e->n_kv;
            double sc = 1.0 / sqrt((double)e->head_dim);
            mmr(e, b + 4, e->pf_xn, e->pf_a, n, 0);
            for (uint32_t r = 0; r < n; r++)
                dispatch(e, e->k_qk_tq,
                         args(e, "pppppppuuufp", e->pf_a + (uint64_t)r * ld * 4, W(e, b + 2), W(e, b + 3), e->rope, kc, vc,
                              e->tq_tab, pos + r, e->n_heads, e->n_kv, (double)e->eps, e->tq_split),
                         G, 64);
            for (uint32_t r = 0; r < n; r++) {
                uint64_t q = e->pf_a + (uint64_t)r * ld * 4, out = e->pf_h + (uint64_t)r * nq * 4;
                uint32_t p = pos + r, blocks = p / ATT_BLOCK + 1;
                dispatch(e, e->k_attn_tq,
                         args(e, "pppppuuufp", q, kc, vc, e->attn_part, e->tq_tab, p, e->n_heads, e->n_kv, sc, e->tq_split),
                         blocks * e->n_kv, 256);
                dispatch(e, e->k_comb_tq, args(e, "pppuu", e->attn_part, out, e->tq_tab, e->n_heads, blocks), e->n_heads,
                         256);
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
    // final norm of every row (the head's h), logits of every row, acceptance
    dispatch(e, e->k_rmsr, args(e, "pppuf", e->pf_x, W(e, 1), s->Hc + rowb, H, (double)e->eps), n, 256);
    uint32_t npad = e->t[e->head_idx].rows;
    swar_mm(e, e->head_idx, s->Hc + rowb, s->L, n, 0, 0);
    double min_gap = inv_temp > 0.0f ? 2.9957 / inv_temp : 0.0;
    dispatch(e, s->k_accept,
             args(e, "puupuufufp", s->L, npad, e->vocab, s->Q, s->nd, s->nd, (double)inv_temp, seed, min_gap, s->st), 1,
             256);
    // commit: each DeltaNet state moved forward by the tokens kept
    li = 0;
    for (uint32_t i = 0; i < e->n_layers; i++) {
        if (e->full_mask >> i & 1) continue;
        uint32_t b = e->layer_base[i], ld = e->t[b + 2].rows;
        uint64_t raw = s->raw + (uint64_t)li * (SPEC_KMAX + 1) * e->conv_n * 4;
        uint64_t xs = s->xs + (uint64_t)li * (SPEC_KMAX + 1) * ld * 4;
        uint64_t cnt = s->st + ST_COMMIT * 4;
        dispatch(e, s->k_commit, args(e, "ppup", e->kc[i], raw, (uint32_t)e->conv_n, cnt), (uint32_t)(e->conv_n + 255) / 256,
                 256);
        dispatch(e, s->k_deltav,
                 args(e, "ppppppuufuupu", xs, W(e, b + 4), W(e, b + 5), e->vc[i], W(e, b + 6), (uint64_t)0, e->n_kh, e->n_vh,
                      (double)e->eps, ld, n, cnt, 1u),
                 e->n_vh, 256);
        li++;
    }
    if (submit(e)) return -1;
    s->gpu_v += e->last_gpu_seconds;
    return 0;
}

// One cycle. tok (at position pos) is the token drawn last; the tokens kept after it go to out[0..*n_out), the token
// drawn after them to *next. max_new caps the tokens kept (out + next). Returns 0, or -1 with e->error set.
int engine_spec_step(Engine* e, uint32_t tok, int pos, float temperature, uint32_t seed, int max_new, uint32_t* out,
                     int* n_out, uint32_t* next) {
    Spec* s = &g_spec;
    float inv_temp = temperature > 0.0f ? 1.0f / temperature : 0.0f;
    double t0 = (double)sceKernelReadTsc() / e->tsc_freq;
    *n_out = 0;
    if (pos + 1 > (int)e->ctx_limit) {
        fail(e, "context full (%d tokens)", pos + 1);
        return -1;
    }
    uint32_t H = e->hidden;
    if ((uint32_t)pos <= s->pbase) {   // (nothing of the conversation in the head yet: one token the usual way)
        *n_out = 0;
        return engine_run(e, &tok, 1, pos, temperature, seed, next);
    }
    uint32_t hp = (uint32_t)pos - s->pbase;   // the head's index of pos
    if (!s->drafted) {   // the pair (carry, tok) first: the head and a draft
        int32_t r = engine_row_of_fast(e, tok);
        int stage_last = r < 0 || (uint32_t)r >= s->nd;
        if (stage_last && fetch_row(e, tok, (float*)(uintptr_t)e->staging)) return -1;
        uint32_t rr = (uint32_t)(r < 0 ? 0 : r);
        if (spec_draft(e, &rr, 1, hp - 1, stage_last, inv_temp, seed * 2654435761u + 17, s->kmax)) return -1;
    }
    volatile uint32_t* st = ST();
    uint32_t K = st[ST_STOP] && st[ST_K] == 0 ? 0 : st[ST_K];
    int room = (int)e->ctx_limit - pos - 1;
    if ((int)K > room) K = room < 0 ? 0 : (uint32_t)room;
    if ((int)K > max_new - 1) K = max_new > 1 ? (uint32_t)(max_new - 1) : 0;
    uint32_t d[SPEC_KMAX];
    for (uint32_t i = 0; i < K; i++) d[i] = st[ST_D + i];
    if (spec_verify(e, tok, (uint32_t)pos, K, inv_temp, seed)) return -1;
    uint32_t j = st[ST_J], nx = st[ST_NEXT];
    if (j > K || nx >= e->vocab) {
        fail(e, "spec: acceptance gave j %u next %u", j, nx);
        return -1;
    }
    for (uint32_t i = 0; i < j; i++) out[i] = engine_token_of(e, d[i]);
    *n_out = (int)j;
    *next = engine_token_of(e, nx);
    e->last_row = nx;
    s->cycles++, s->proposed += K, s->kept += j, s->tokens += j + 1;
    // the head over the new pairs (h_pos+r, x_pos+r+1), r = 0..j (the last with the token just drawn), and a draft
    int stop = 0;
    for (int i = 0; i < 4; i++) stop |= s->stop_rows[i] == nx;
    if (stop) {
        // (the turn or the reasoning ends: no draft. The head still gets the pairs of the drafts kept, (h_pos+r,
        // d_r+1) at index pos + r, so that its cache has no hole there; the carry is h at pos + j, for what comes next)
        begin(e);
        for (uint32_t r = 0; r < j; r++)
            dispatch(e, s->k_emb, args(e, "ppuupup", s->table, s->st, 0xFFFFu, d[r], s->E + (uint64_t)r * H * 4, H,
                                       (uint64_t)0), 1, 256);
        if (j) mtp_rows(e, s->E, s->Hc + (uint64_t)H * 4, j, hp, 0);
        dispatch(e, e->k_copy, args(e, "ppu", s->Hc + (uint64_t)(j + 1) * H * 4, s->Hc, H), H / 256, 256);
        if (submit(e)) return -1;
        s->drafted = 0;
    } else {
        uint32_t rows[SPEC_KMAX + 1];
        for (uint32_t i = 0; i < j; i++) rows[i] = d[i];
        rows[j] = nx;
        int stage_last = nx >= s->nd;
        if (stage_last && fetch_row(e, *next, (float*)(uintptr_t)e->staging)) return -1;
        // Hc rows 1..j+1 are the h's of positions pos..pos+j: shift them to rows 0..j for spec_draft
        begin(e);
        dispatch(e, e->k_copy, args(e, "ppu", s->Hc + (uint64_t)H * 4, s->X, (j + 1) * H), (j + 1) * H / 256, 256);
        dispatch(e, e->k_copy, args(e, "ppu", s->X, s->Hc, (j + 1) * H), (j + 1) * H / 256, 256);
        if (submit(e)) return -1;
        if (spec_draft(e, rows, j + 1, hp, stage_last, inv_temp, seed * 2654435761u + 17, s->kmax)) return -1;
    }
    s->wall += (double)sceKernelReadTsc() / e->tsc_freq - t0;
    e->last_gpu_seconds = 0;
    return 0;
}

void spec_stats(const Engine* e, char* buf, int n) {
    (void)e;
    const Spec* s = &g_spec;
    snprintf(buf, n, "spec: %llu cycles, %llu tokens (%.2f per cycle), drafts kept %llu of %llu (%.0f %%), GPU %.1f + %.1f "
             "ms per cycle, %.1f tokens/s in the engine",
             (unsigned long long)s->cycles, (unsigned long long)s->tokens, s->cycles ? (double)s->tokens / s->cycles : 0.0,
             (unsigned long long)s->kept, (unsigned long long)s->proposed,
             s->proposed ? 100.0 * s->kept / s->proposed : 0.0, s->cycles ? s->gpu_v * 1e3 / s->cycles : 0.0,
             s->cycles ? s->gpu_d * 1e3 / s->cycles : 0.0, s->wall > 0 ? s->tokens / s->wall : 0.0);
}

void spec_stats_reset(void) {
    Spec* s = &g_spec;
    s->cycles = s->proposed = s->kept = s->tokens = 0;
    s->gpu_v = s->gpu_d = s->wall = 0;
}

// The head's carry (h at the position before the next token) goes with the DeltaNet state ("Pensar"): 16 KiB.
uint64_t spec_carry(void) { return g_spec.ready ? g_spec.Hc : 0; }

// T3's weights read again into their memory (the head had borrowed it): ~280 MiB from the disk.
int t3_reload(Engine* e) {
    T3* t = &e->t3;
    if (!t->ready || t->fd < 0) return -1;
    uint64_t done = 0;
    while (done < t->gpu_size) {
        uint64_t n = t->gpu_size - done < (32ull << 20) ? t->gpu_size - done : 32ull << 20;
        ssize_t r = pread(t->fd, (uint8_t*)(uintptr_t)t->model + done, n, (off_t)done);
        if (r <= 0) break;
        done += r;
    }
    __asm__ volatile("sfence" ::: "memory");
    return done == t->gpu_size ? 0 : -1;
}

// The draft length cap (0 = the head off); k < 0 leaves it. Returns the previous one.
int spec_kmax(int k) {
    int old = g_spec.kmax;
    if (k >= 0 && k <= SPEC_KMAX) g_spec.kmax = k;
    return old;
}

// The draft's stop threshold (theta < 0 leaves it). Returns the previous one.
float spec_theta(float theta) {
    float old = g_spec.theta;
    if (theta >= 0.0f) g_spec.theta = theta;
    return old;
}
