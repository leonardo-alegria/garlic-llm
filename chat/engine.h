// Qwen3 / Qwen3.5 inference on the PS4 GPU (the same command packets as pc/gpu.py).
#ifndef CHAT_ENGINE_H
#define CHAT_ENGINE_H

#include <stdint.h>

// Speculative decoding with the MTP head (engine_spec.inc): the 9B app only. Not the 4B (its own MTP head is not
// converted). Without mtp.idx or the q4s model it stays off (spec_init fails).
#if !defined(GARLIC_4B)
#define GARLIC_SPEC 1
#endif

typedef struct {
    char name[32];
    uint64_t addr;
    uint32_t rsrc1, rsrc2, user_sgprs, kernarg_sgpr;
} GpuKernel;

typedef struct {
    uint64_t off, size;
    uint32_t rows, cols;
    uint64_t scales;
    uint32_t fmt;
} Tensor;

typedef void (*ProgressFn)(void* user, float fraction, const char* what);

// Speech to text: Parakeet TDT 0.6B v3 (assets/asr.idx, asr_vocab.bin; weights in /data), in the GPU memory left
// after the language model.
typedef struct {
    char name[32];
    uint64_t off, size;
    uint32_t rows, cols;
    uint64_t scales;
} AsrTensor;

typedef struct {
    int ready;
    AsrTensor* t;
    uint32_t nt;
    char path[129];
    uint64_t size, model;
    float* mel;                 // [128][257] filter bank (CPU copy)
    uint8_t* vocab;             // "PVOC" blob
    uint32_t* piece;            // offset of each token's piece in vocab
    uint32_t max_frames;        // encoder frames (80 ms each) that fit
    uint64_t mel_in, pe, sub_a, sub_b, x, h, f, qkv, r, att, g2, g, encp;
    uint64_t xh0, xh1, c0, c1, gates, dec, jin, logits, res, dummy;
    uint64_t ws_begin, ws_end;  // its GPU work buffers (free for others while it is not transcribing)
    uint64_t flex_len;          // the weights are in flexible (system) memory, this long; 0 if in GPU memory
    volatile int unloaded;      // the weights were released (asr_release); asr_reload brings them back
} Asr;

// Text to speech, first half: Chatterbox Multilingual's T3 (assets/t3.idx; weights in /data): the text tokens of one
// sentence -> speech tokens (25 per second of audio). A Llama of 30 layers, 16 heads of 64, q4i weights.
#define T3_LAYERS 30
#define T3_MAX_SEQ 512   // as MAX_SEQ in shaders/t3.cl: 34 voice rows + text + speech of one sentence

typedef struct {
    int ready;
    AsrTensor* t;
    uint32_t nt;
    char path[129];
    uint64_t size, gpu_size, model;
    int fd;                     // the weights file: embedding rows are read from it
    float inv_freq[32];
    uint64_t rope, kc[T3_LAYERS], vc[T3_LAYERS];
    uint64_t x, xn, qkv, att, part, o, gu, hm, logits, staging;
    const AsrTensor *qkv_w[T3_LAYERS], *o_w[T3_LAYERS], *gu_w[T3_LAYERS], *down_w[T3_LAYERS];
    const AsrTensor *in_n[T3_LAYERS], *post_n[T3_LAYERS], *norm, *head;
    const AsrTensor *text_emb, *text_pos, *speech_emb, *speech_pos, *prefix;
    const GpuKernel *k_qk, *k_attn, *k_comb;
    const GpuKernel *k_rms, *k_mv, *k_att, *k_silu, *k_mvk;   // fused decode step (shaders/t3.cl)
    uint64_t parts;             // partial sums of the split products (o, down): [4][1024]
    const GpuKernel *k_mvb, *k_mvkb, *k_rope, *k_attb;      // batched prefill (several positions per pass)
    uint64_t pf_scratch, pf_size;   // work buffers for the batched prefill (0: one position per pass); t3_scratch
    double s_prefill;           // seconds of the last t3_start's prefill
    int prefill_batched;        // positions per pass in the last prefill
    uint32_t nks;               // K slices for o and down (1 = plain t3_mv; picked by timing at start-up)
    double s_rows, s_sample, s_submit, s_gpu;   // seconds spent per part since the last t3_start (diagnostics)
    const GpuKernel* k_sample;
    uint64_t g_seen, g_res;     // GPU sampling: tokens said so far (bytes), the token drawn
    uint16_t* emb_cache;        // speech_emb rows already read (direct-mapped by token), speech_pos rows 0..511
    int32_t* emb_tag;
    uint16_t* pos_rows;
    int debug;
    // T3's 34 voice rows per language (assets/t3_prefix.bin: native pronunciation, the same speaker row; block 0 =
    // the built-in voice, used for English and any other language), and the one in use
    uint16_t* pre;
    char pre_lang[16][4];
    int npre, pre_sel;
    double last_seconds;        // time of the last t3_generate
    // the sentence in progress (t3_start / t3_next)
    uint8_t seen[6563];
    uint32_t rng;
    int n_in, k, max_out, greedy;
} T3;

// Text to speech, second half: S3Gen (meanflow) + HiFT (assets/s3g.idx, s3g_voice.bin; weights in /data): speech
// tokens -> 24 kHz audio. See engine_s3g.inc.
typedef struct {
    char name[32];
    uint64_t off, size, scales;
    uint32_t rows, cols, fmt, cin, nout;   // fmt: 0 f32, 1 q4i, 2 f16i, 3 f32t; cin: a conv's input channels
} S3Tensor;

typedef struct {               // a voice: prompt speech tokens, prompt mel frames (2 per token), speaker vector
    char lang[4];               // "" = the built-in voice (assets/s3g_voice.bin), else assets/s3g_voice_<lang>.bin
    uint32_t np, nmel;
    int32_t* ptok;
    float* pmel;
    float spk[80];
    float gain;                 // on its audio, so every voice sounds as loud as Garlic's (1 if the file has none)
} S3Voice;

typedef struct {
    int ready, loaded;
    S3Tensor* t;                // sorted by name
    uint32_t nt;
    char path[129];
    uint64_t size;
    S3Voice voice[8];           // [0] the built-in voice, then the languages that have their own (Russian)
    int nvoice;
    uint32_t np, nmel;          // the voice in use (s3g_voice): prompt speech tokens, prompt mel frames, speaker vector
    int32_t* ptok;
    float* pmel;
    float spk[80];
    float gain;                 // and the gain for its audio (S3Voice.gain, applied by audio_out.c)
    uint64_t model, scratch, scratch_size;
    const GpuKernel* k[20];
    double s_enc, s_cfm, s_src, s_hift;   // GPU seconds of the last sentence per stage (diagnostics)
    // the sentence in progress
    uint64_t top, phase_base, xin, div, tok, phi, mel, f0, s, S, out, glob_end, t0;
    uint32_t n_tok, T, T2, M, chunk, c0, seed;
    int step;
    double last_seconds;        // time of the last whole sentence
} S3G;

typedef struct {
    // model shape (assets/model.idx)
    uint32_t n_layers, hidden, inter, n_heads, n_kv, head_dim, vocab, max_seq, eos, eos2;
    float eps, rope_theta;
    // Qwen3.5 (model.idx "PMQ5"): Gated DeltaNet shape, partial RoPE, which layers are full attention
    int qwen35;
    uint32_t n_kh, n_vh, dk, dv, rope_dim, full_mask, im_start, im_end;
    uint32_t* layer_base;   // index in t[] of each layer's first tensor (attn_norm)
    int has_head;           // untied output head: the last tensor (Qwen3.5-9B), or the one before "ids"
    uint32_t head_idx;      // index of lm_head in t[] (0 = tied to the embedding)
    // The 9B's q4s model (pc/convert_9b_q4s.py): lm_head trimmed to `vocab` rows ordered by frequency; ids[row] is
    // the token id of each row (NULL: rows are ids). The sampled index is a row; engine_run returns the id.
    uint32_t* ids;
    uint32_t last_row;      // the row engine_run drew last (its token: engine_token_of)
    int cpu_embed;          // input embedding rows stay in the file (fmt 4); the CPU reads each token's row
    int fd;                 // the weights file, kept open for those rows
    uint8_t* row_cache;     // direct-mapped cache of embedding rows
    int32_t* row_tag;
    uint64_t staging;       // one dequantized row per step of a submission (GPU memory)
    char weights_path[129];
    uint64_t weights_size;
    Tensor* t;
    uint32_t n_tensors;

    // GPU-visible memory (Garlic): command buffer, kernel arguments, code, model, caches, activations
    uint8_t* mem;
    uint64_t mem_size, next;
    uint32_t* dcb;
    uint32_t ndw, dcb_cap;
    uint8_t* kargs;
    uint32_t kargs_used, kargs_cap;
    volatile uint32_t* label;
    volatile uint64_t* stamps;
    uint64_t model, rope, layer_cache, attn_part;
    uint64_t *kc, *vc;            // per layer: KV cache (full attention) or conv state / delta-rule state (DeltaNet)
    uint64_t conv_n, state_bytes;
    // Qwen3.5: each full layer's cache is two halves (ATTN_SPLIT in shaders/attn.h); the second halves of all the
    // layers form one block [kv_hi, kv_hi + kv_split) that may be lent out while the context stays under max_seq / 2
    uint64_t kv_hi, kv_split;
    uint32_t ctx_limit;           // positions the conversation may use: max_seq, or max_seq / 2 while kv_hi is lent
    uint32_t lend_ctx;            // ctx_limit while kv_hi is lent (max_seq with the TurboQuant cache)
    // TurboQuant 4-bit KV cache (shaders/attn_tq.h). The whole context fits in a third of the memory, so kv_hi is a
    // block of its own: lending it does not shorten an 8K context.
    int tq;
    uint64_t tq_tab;              // 40 words for the kernels: levels, boundaries, sign bits (pc/turboquant.py)
    const GpuKernel *k_qk_tq, *k_attn_tq, *k_comb_tq;
    // With TurboQuant the context is 16K: positions 0..8191 in the cache block, 8192..16383 at tq_split bytes
    // further, inside kv_hi (so while kv_hi is lent the context is 8K, lend_ctx)
    uint64_t tq_split;
    int tq16;                     // kernels.bin's TurboQuant kernels take tq_split (16K of context)
    // The prompt in batches of up to PF_ROWS tokens (shaders/prefill.h): row buffers for the residual (pf_x),
    // its norm (pf_xn), the projections (pf_a: DeltaNet input, q|k|v|gate or gate|up) and their products (pf_h:
    // attention or delta-rule output, SiLU(gate) * up)
    int pf, pf_off;               // batched prompt available; pf_off = 1: token by token (tests)
    uint64_t pf_x, pf_xn, pf_a, pf_h;
    const GpuKernel *k_mm4r, *k_mm3r, *k_rmsr, *k_silur, *k_sigr, *k_convr, *k_deltar;
    double pf_seconds;            // GPU time of the last batched prompt (diagnostics)
    uint64_t submit_tsc;          // when the last submission went to the GPU (submit_go)
    // the fastest product kernel for each matrix shape, timed at start-up (engine.c mm_tune): [0] for batches of
    // more than 16 tokens, [1] for up to 16; r = tokens per group of that kernel
    struct { uint32_t fmt, rows, cols, r[2]; const GpuKernel* k[2]; } pf_mm[8];
    int pf_nmm;
    uint64_t x, xn, attn, hmlp, qkv, lin, o, gu, down, logits, last, history;
    GpuKernel k[200];             // the kernels of kernels.bin
    int nk;
    const GpuKernel *k_embed, *k_rmsnorm, *k_qk, *k_attn, *k_attn_comb, *k_silu, *k_add, *k_mv, *k_sample;
    const GpuKernel *k_sig, *k_conv, *k_delta;   // Qwen3.5 only
    const GpuKernel *k_bsum, *k_mvi, *k_mvi8;    // interleaved Q4 ("q4i", fmt 3) matrices
    const GpuKernel *k_mv3, *k_mv38, *k_copy;    // interleaved Q3 ("q3i", fmt 5); staging row -> x
    uint64_t bsum;                               // block sums of the current matrix-vector input
    // SWAR int4 x int8 products ("q4s", fmt 6; shaders/spec.h): the activations in int8 per block of 32 (x8, dx,
    // init) for up to PF_ROWS rows, the split-K partial sums, and per matrix shape and number of tokens (1..4) the
    // fastest kernel variant (swar_tune at start-up)
    int swar;
    const GpuKernel *k_qx8, *k_sumsplit, *k_swar[4][5], *k_swar3[4][5];   // q4s (fmt 6), q3s (fmt 8)
    uint64_t x8, dx8, init8, yp;
    uint64_t yp_rows;                            // the largest rows x 4 tokens x 4 parts the partial sums take
    struct { uint32_t rows, cols, fmt; uint8_t v[4]; } swar_tab[24];
    int swar_n;
    uint32_t init_state[256];
    uint64_t tsc_freq;
    double last_gpu_seconds;
    char error[160];
    Asr asr;
    T3 t3;
    S3G s3;
} Engine;

// Direct memory engine_init leaves to the system (MiB; main.c sets it before engine_init).
extern int eng_margin_mb;
// Reads model.idx and kernels.bin from assets_dir, reserves GPU memory and loads the weights file.
int engine_init(Engine* e, const char* assets_dir, ProgressFn progress, void* user);
// Runs n tokens through the model starting at position pos0 and samples the token after the last one.
// temperature 0 = greedy. Returns 0 and the sampled token in *next, or -1 with e->error set.
// (n >= 4 goes through the layers in batches of up to 32 tokens, shaders/prefill.h, unless e->pf_off.)
int engine_run(Engine* e, const uint32_t* tokens, int n, int pos0, float temperature, uint32_t seed, uint32_t* next);
// The token id of an lm_head row (rows are ids unless lm_head is trimmed: Engine.ids), and the row of a token id
// (-1 if that token is not in lm_head: it can be read but not written).
uint32_t engine_token_of(const Engine* e, uint32_t row);
int32_t engine_row_of(const Engine* e, uint32_t token);
// Forget the conversation: clears the Gated DeltaNet states (the KV cache is simply overwritten from position 0).
void engine_reset(Engine* e);
// The Gated DeltaNet states saved to GPU memory at buf (engine_state_size bytes; save 1) or put back from it
// (save 0), so that the conversation can go back to the position where they were saved. 0 = done.
uint64_t engine_state_size(const Engine* e);
int engine_state_copy(Engine* e, uint64_t buf, int save);
// Load Parakeet into the GPU memory still free (after engine_init). 0 if ready, -1 if missing or too big.
int asr_init(Engine* e, const char* assets_dir);
int engine_flex_test(Engine* e);
// Load T3 into the GPU memory still free. 0 if ready.
int t3_init(Engine* e, const char* assets_dir);
// Speech tokens for one sentence (text tokens with start 255 and stop 0): sampled as the official code
// (repetition penalty 1.2, temperature 0.8, min_p 0.05), or greedy for tests. Returns how many, -1 on error.
int t3_generate(Engine* e, const uint16_t* text, int n_text, uint16_t* out, int max_out, uint32_t seed, int greedy);
// The same one token at a time: t3_start runs the prefill, then each t3_next gives the next speech token (-1 when
// the sentence is finished, -2 on error).
int t3_start(Engine* e, const uint16_t* text, int n_text, int max_out, uint32_t seed, int greedy);
int t3_next(Engine* e);
// The voice rows for a language ("es", "pt", ...): its native ones if there are, else the built-in voice's.
void t3_voice(Engine* e, const char* lang);
// Work memory for the batched prefill (up to T3_PF positions per pass, the same results bit for bit as one at a
// time): Parakeet's work buffers in voice mode, which are idle while T3 runs. 0 = one position per pass.
void t3_scratch(Engine* e, uint64_t base, uint64_t size);
// S3Gen: index, voice and kernels (no GPU memory). s3g_load reads the weights into `weights` and takes `scratch`
// as its work memory. Then per sentence: s3g_begin (returns how many tokens it takes, -1 on error) and s3g_step
// until it returns -1: each call runs one part on the GPU and returns how many new samples it put in out (0 for
// the parts before the vocoder), -2 on error.
int s3g_init(Engine* e, const char* assets_dir);
// S3Gen's voice for a language ("ru", ...): its own if assets has one, else the built-in voice. Takes effect at the
// next s3g_begin.
void s3g_voice(Engine* e, const char* lang);
int s3g_load(Engine* e, uint64_t weights, uint64_t wsize, uint64_t scratch, uint64_t ssize);
int s3g_begin(Engine* e, const uint16_t* tokens, int n, uint32_t seed);
int s3g_step(Engine* e, float* out, int max);
// Lend the second half of the language model's context (kv_hi, kv_split bytes): from then on ctx_limit is
// lend_ctx (max_seq / 2; with the TurboQuant cache and 8K of context, kv_hi is spare memory and the context stays
// max_seq), so positions past it must not be in use. engine_kv_return gives it back (contents are not kept).
uint64_t engine_kv_lend(Engine* e, uint64_t* size);
void engine_kv_return(Engine* e);
// 16 kHz mono audio -> UTF-8 text. Returns 0, or -1 with e->error set.
int asr_transcribe(Engine* e, const float* audio, int n, char* text, int max);
// With the 9B, Parakeet's weights live in system memory (338 MiB), which leaves ~20 MiB for everything else. asr_release
// gives them back while Python runs (it needs a heap of up to 64 MiB); asr_reload maps and reads them again (~4 s,
// safe to call from another thread; 0 when ready). Both do nothing if the weights are in GPU memory.
void asr_release(Engine* e);
// While released (the 9B): the weights' memory, still mapped, for Python's heap and stack (NULL and 0 otherwise).
void* asr_lent(Engine* e, uint64_t* size);
int asr_reload(Engine* e);

// Speculative decoding with Qwen3.5's MTP head (GARLIC_SPEC; engine_spec.inc). spec_init reads mtp.idx from the
// engine's assets; spec_load puts the head, its draft rows and its cache into [base, base + size) (T3's weights
// memory, outside voice mode); start = the conversation position it follows from (0 = empty; after voice mode,
// where the conversation is).
int spec_init(Engine* e, const char* dir);
int spec_load(Engine* e, uint64_t base, uint64_t size, int start);
void spec_unload(Engine* e);
int spec_active(const Engine* e);
// One cycle: feeds tok (drawn last, at pos) and the draft's tokens, keeps what the exact acceptance keeps (out[0..*n_out),
// at most max_new - 1 of them) and draws the next token. Same distribution as engine_run one token at a time.
int engine_spec_step(Engine* e, uint32_t tok, int pos, float temperature, uint32_t seed, int max_new, uint32_t* out,
                     int* n_out, uint32_t* next);
void spec_stats(const Engine* e, char* buf, int n);
void spec_stats_reset(void);
// T3's weights read again into their memory (after the head borrowed it).
int t3_reload(Engine* e);
// The draft length cap (0 = the head off; k < 0 keeps it); returns the old one.
int spec_kmax(int k);
float spec_theta(float theta);   // the draft's stop threshold (theta < 0 keeps it); returns the old one

#endif
