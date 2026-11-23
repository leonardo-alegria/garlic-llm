// ---- Garlic's voice: its replies spoken with Chatterbox Multilingual, T3 + S3Gen on the GPU ---------------------
// Included in main.c. While Garlic writes, its reply is cut into pieces (speech_text.c); each piece goes through
// T3 (text tokens -> speech tokens, one per GPU submission) and S3Gen + HiFT (-> 24 kHz audio, a few submissions),
// and its samples are queued for playback (audio_out.c). The GPU does one unit of work at a time: a token of the
// language model, or a step of the voice, whichever the audio needs first. On screen, only the words already
// spoken (as the audio plays), drawn by the voice screen's own thread (main.c). Memory: S3Gen's weights borrow the
// second half of the language model's context (4096 tokens in voice mode; with the TurboQuant cache, a spare block
// and the context stays at 8192) and its work buffers borrow Parakeet's,
// which is idle while Garlic speaks.

#include "audio_out.h"
#include "speech_text.h"
#include "tts_tok.h"

#define TTS_AHEAD (24000 * 6)       // the voice gets the GPU when less than 6 s of audio is left to play

typedef struct {
    int64_t s0, s1;   // samples [s0, s1) of the playback queue ...
    int b0, b1;       // ... say reply bytes [b0, b1)
} Spoken;

static TtsTok tts_tok;
static int tts_ready;       // tokenizer, T3 and S3Gen set up (the voice can be loaded)
static int tts_loaded;      // S3Gen weights in the lent context, audio port open
static struct {
    int msg;                // the reply being spoken (index in msgs), -1 none
    int queued;             // its bytes handed to the voice so far
    int stage;              // 0 waiting for a piece, 1 T3, 2 S3Gen
    int b0, b1;             // bytes of the piece in progress
    uint16_t speech[512];
    int n_speech;
    Spoken sp[256];
    int nsp;
    uint32_t seed;
    char lang[4];           // language of the reply so far
    char user_lang[4];      // language the user spoke in (the first guess for the reply)
    double t_piece;         // when the piece in progress started
    double lv_sum;          // the piece's loudness as played (gain included): energy of its 20 ms frames with speech
    int lv_n;
    int64_t lim0;           // the limiter's count when the piece started
    int clause_end;         // the piece ends at a comma (or a cut), not at the end of a sentence
    int n_pieces, n_gaps;   // this reply's pieces, and the times its audio ran dry
    double s_audio, s_gaps;
} tts;
static float tts_pcm[24000 * 13];
#define tts_fluent 3   // how replies are cut into pieces: always mode 3 (see tts_piece_end_punct)
static void tts_gaps_log(void);
static void tts_reply_summary(void);

static const char* lang_code(void) {
    static const char* c[N_LANGS] = {"es", "en", "pt", "ru"};
    return c[ui_lang];
}

static void tts_boot(void) {
    size_t n;
    uint8_t* d = read_all(ASSETS "/tts_tok.bin", &n);
    if (!d || ttstok_load(&tts_tok, d, n)) {
        say("voice: tts_tok.bin missing");
        return;
    }
    // Russian stress marks (as the official tokenizer's RussianTextStresser), used in place
    static RuStress ru_stress;
    uint8_t* rs = read_all(ASSETS "/ru_stress.bin", &n);
    if (rs && !rustress_load(&ru_stress, rs, n)) tts_tok.ru = &ru_stress;
    else say("voice: ru_stress.bin missing, Russian without stress marks");
    if (!eng.t3.ready || s3g_init(&eng, ASSETS)) {
        say("voice: T3 %d, S3Gen %s", eng.t3.ready, eng.s3.ready ? "ok" : eng.error);
        return;
    }
    say("voice: %d S3Gen voices, Russian stress dictionary %u forms", eng.s3.nvoice, ru_stress.n_keys);
    // T3 reads each piece's prompt in batches of up to 32 positions, in Parakeet's work buffers (idle while T3
    // runs, as for S3Gen)
    t3_scratch(&eng, eng.asr.ws_begin, eng.asr.ws_end - eng.asr.ws_begin);
    tts_ready = 1;
    tts.msg = -1;
}

static void tts_stop(void) {   // drop what is being made and what is queued
    aout_clear();
    tts.stage = 0;
    tts.nsp = 0;
    if (tts.msg >= 0 && tts.msg < n_msgs) tts.queued = (int)msgs[tts.msg].len;   // the rest stays unspoken
}

// S3Gen stays loaded after voice mode with the TurboQuant cache (eng.tq), so entering voice mode again is instant.
// kv_hi is then either a block of its own (8K) or the context's positions 8192.. (16K): the engine clears
// eng.s3.loaded when a text conversation writes there (engine_run), and tts_enter reads the weights again. Without
// TurboQuant (garlic_tq.txt = 0) the block goes back to the context on leaving and the weights are
// read again on entering. Nothing of S3Gen lives in its work buffers between sentences (Parakeet's, which dictation
// uses meanwhile): s3g_begin writes all its inputs anew.
static int tts_keep(void) { return eng.tq; }

// Voice mode on: lend the second half of the context to S3Gen's weights (a long conversation starts over, as its
// tokens past the middle are about to be overwritten). With the TurboQuant cache (eng.tq) the lent block is spare
// memory and the context stays whole (eng.lend_ctx = max_seq). Returns 1 if Garlic can speak.
static int tts_enter(void) {
    tts.msg = -1, tts.stage = 0, tts.nsp = 0;
    if (!tts_ready) return 0;
    if (pos > (int)eng.lend_ctx - 1024) {
        pos = 0, need_close = 0;
        Msg* note = msg_new(MSG_NOTE);
        msg_append(note, tr(S_CONTEXT_FULL), strlen(tr(S_CONTEXT_FULL)));
    }
    if (tts_loaded && eng.s3.loaded) {   // (kept from the last time: tts_keep)
        engine_kv_lend(&eng, &(uint64_t){0});
        say("voice: S3Gen still loaded; context %u tokens", eng.ctx_limit);
        return 1;
    }
    if (tts_loaded) say("voice: S3Gen's weights were overwritten by the conversation; reading them again");
    tts_loaded = 0;
    draw_loading(1.0f, tr(S_VOICE_LOADING));
    present();
    uint64_t wsize, w = engine_kv_lend(&eng, &wsize);
    double t0 = now();
    if (!w || aout_init(user_id) ||
        s3g_load(&eng, w, wsize, eng.asr.ws_begin, eng.asr.ws_end - eng.asr.ws_begin)) {
        say("voice: cannot load S3Gen (%s)", eng.error);
        engine_kv_return(&eng);
        return 0;
    }
    say("voice: S3Gen loaded in %.2f s; context now %u tokens", now() - t0, eng.ctx_limit);
    tts_loaded = 1;
    return 1;
}

static void tts_leave(void) {
    if (!tts_loaded) return;
    tts_gaps_log();
    tts_reply_summary();
    tts_stop();
    if (tts_keep() && eng.s3.loaded) {   // the weights stay in kv_hi for next time (ctx_limit is max_seq either way)
        engine_kv_return(&eng);
        return;
    }
    eng.s3.loaded = 0;
    engine_kv_return(&eng);
    tts_loaded = 0;
}

static int last_ai(void) {
    for (int i = n_msgs - 1; i >= 0; i--)
        if (msgs[i].who == MSG_AI) return i;
    return -1;
}

static void tts_gaps_log(void) {
}

static void tts_reply_summary(void) {
    if (tts.n_pieces)
        say("voice: reply spoken in %d pieces, %.1f s of audio (one piece every %.1f s), %d gaps (%.2f s)", tts.n_pieces,
            tts.s_audio, tts.s_audio / tts.n_pieces, tts.n_gaps, tts.s_gaps);
    tts.n_pieces = tts.n_gaps = 0, tts.s_audio = tts.s_gaps = 0;
}

// A new reply started (start_turn made its message): speak it from the beginning.
static void tts_new_reply(void) {
    tts_gaps_log();
    tts_reply_summary();
    tts_stop();
    tts.msg = last_ai();
    tts.queued = 0;
    tts.lang[0] = 0;
    snprintf(tts.user_lang, sizeof(tts.user_lang), "%s", lang_code());
    for (int i = tts.msg - 1; i >= 0; i--)
        if (msgs[i].who == MSG_USER) {
            snprintf(tts.user_lang, sizeof(tts.user_lang), "%s", st_lang(msgs[i].text, lang_code()));
            break;
        }
}

// End of the next piece to speak from tts.queued, or 0. The reply's first piece is short so the voice starts soon:
// it ends at its first comma (after 15 bytes) or, past 60 bytes without one, between words near byte 45 (*cut set:
// said with a comma's pause), as soon as that much is written; later pieces are whole sentences (speech_text.c).
static int tts_piece_end_fluent(const Msg* m, int* cut);
static int tts_piece_end_punct(const Msg* m, int* cut);

static int tts_piece_end(const Msg* m, int* cut) {
    if (tts_fluent >= 2) return tts_piece_end_punct(m, cut);
    if (tts_fluent) return tts_piece_end_fluent(m, cut);
    int len = (int)m->len, end = st_next_piece(m->text, len, tts.queued, !generating);
    *cut = 0;
    if (tts.queued == 0) {
        // a first sentence that short ("¡Claro! ") goes with what follows: alone, T3 tends to keep talking after it
        while (end > 0 && end < 16) {
            int e2 = st_next_piece(m->text, len, end, !generating);
            if (e2 > end) {
                end = e2;
                continue;
            }
            if (generating) end = 0;   // (the next sentence is not written yet)
            break;
        }
        int c = 0, k = 0, lim = len;
        for (int i = 0; i < lim; i++)   // (never inside code: a cut there made the next piece read the code aloud)
            if (m->text[i] == '`') lim = i;
        for (int i = 15; i + 1 < lim && i < 70 && !c; i++)
            if (m->text[i] == ',' && m->text[i + 1] == ' ') c = i + 2;
        if (!c && lim >= 60)
            for (int i = 45; i > 20 && !c; i--)
                if (m->text[i] == ' ') c = i + 1, k = 1;
        if (c && (end <= 0 || c < end)) end = c, *cut = k;
    } else {
        int64_t left = aout_queued() - aout_played();
        int lim = len < tts.queued + 200 ? len : tts.queued + 200, c = 0, k = 0;
        for (int i = tts.queued; i < lim; i++)   // (never inside code: st_next_piece ends a piece before it)
            if (m->text[i] == '`') lim = i;
        if (left < 24000 * 3)   // the audio is about to run out: a clause is enough (a comma at least 25 bytes in)
            for (int i = tts.queued + 25; i + 1 < lim && !c; i++)
                if (m->text[i] == ',' && m->text[i + 1] == ' ') c = i + 2;
        // about to fall silent (a short first piece plays out before the next sentence is written and through T3):
        // with under 2 s left, a comma 12 bytes in will do or, once 40 bytes are written without one, a cut between
        // words near byte 36 (said with a comma's pause)
        // (lengths as said: the bytes count with each digit as the words it becomes, st_spoken_len)
        if (left < 24000 * 2 && !c && end <= 0) {
            for (int i = tts.queued + 12; i + 1 < lim && !c && st_spoken_len(m->text + tts.queued, i - tts.queued) < 60; i++)
                if (m->text[i] == ',' && m->text[i + 1] == ' ') c = i + 2;
            if (!c && st_spoken_len(m->text + tts.queued, lim - tts.queued) >= 40) {
                int i = tts.queued;
                while (i < lim && st_spoken_len(m->text + tts.queued, i - tts.queued) < 36) i++;
                for (; i > tts.queued + 12 && !c; i--)
                    if (m->text[i] == ' ' && m->text[i - 1] != '\n') c = i + 1, k = 1;
            }
        }
        if (c && (end <= 0 || c < end)) end = c, *cut = k;
    }
    return end > tts.queued ? end : 0;
}

// Mode 1: fewer and longer pieces. Every piece boundary brings a short quiet and a fresh start of T3's prosody, and
// a boundary at a comma that st_clean drops is said as the end of a sentence, so:
// - the first piece is as short as in mode 0 (the voice starts as soon), but a cut between words falls at a natural
//   point (st_natural_cut: before "que", "y", "porque"..., never after an article or a preposition);
// - later pieces are as long as the audio still queued allows: whole sentences joined (tts_join_fluent) up to
//   min(10 s, (queued - 1 s) / 0.25) of speech, so the next one is ready before the queue runs dry; only when less
//   than TTS_CLAUSE_S is left a clause (to a comma) will do, and under 2 s a natural cut;
// - a sentence too long for one piece (st_next_piece's HARD_MAX cut at any space) is cut at a natural point instead;
// - a piece that ends at a comma keeps it (tts_work), so T3 says it as a clause that goes on.
#define TTS_CLAUSE_S 3.0
#define TTS_WAIT_S 4.0
#define TTS_MIN_S 6.0
static int tts_join_fluent(const Msg* m, int start, int end);

static int hard_cut_at(const char* t, int q, int end) {   // st_next_piece ended the piece at a space, not after punctuation
    return end > q + 1 && t[end - 1] == ' ' && !strchr(".!?,;:\"')]*", t[end - 2]) && (uint8_t)t[end - 2] != 0xA6;
}

static int tts_piece_end_fluent(const Msg* m, int* cut) {
    const char* t = m->text;
    int len = (int)m->len, q = tts.queued, end = st_next_piece(t, len, q, !generating);
    *cut = 0;
    int lim = len;
    for (int i = q; i < lim; i++)   // (never inside code)
        if (t[i] == '`') lim = i;
    if (q == 0) {
        while (end > 0 && end < 16) {   // "¡Claro! " goes with what follows (see above)
            int e2 = st_next_piece(t, len, end, !generating);
            if (e2 > end) {
                end = e2;
                continue;
            }
            if (generating) end = 0;
            break;
        }
        int c = 0, k = 0;
        for (int i = 15; i + 1 < lim && i < 70 && !c; i++)
            if (t[i] == ',' && t[i + 1] == ' ') c = i + 2;
        if (!c && lim >= 60) {
            c = st_natural_cut(t, lim, 0, 25, 60), k = 1;
            for (int i = 45; i > 20 && !c; i--)   // (no natural point: as in mode 0)
                if (t[i] == ' ') c = i + 1;
        }
        if (c && (end <= 0 || c < end)) end = c, *cut = k;
        return end > 0 ? end : 0;
    }
    if (end > 0 && hard_cut_at(t, q, end)) {   // a sentence too long for one piece: at a natural point
        int c = st_natural_cut(t, len, q, 90, 160);
        if (c > q) end = c, *cut = 1;
    }
    double left = (double)(aout_queued() - aout_played()) / 24000.0;
    int c = 0, k = 0;
    if (left < TTS_CLAUSE_S)   // a clause is enough: a comma 25 bytes in
        for (int i = q + 25; i + 1 < lim && !c && (end <= 0 || i < end); i++)
            if (t[i] == ',' && t[i + 1] == ' ') c = i + 2;
    if (left < 2.0 && !c && end <= 0) {   // about to fall silent (see above): a comma 12 bytes in, or a natural cut
        for (int i = q + 12; i + 1 < lim && !c && st_spoken_len(t + q, i - q) < 60; i++)
            if (t[i] == ',' && t[i + 1] == ' ') c = i + 2;
        if (!c && st_spoken_len(t + q, lim - q) >= 40) c = st_natural_cut(t, lim, q, 15, 50), k = 1;
    }
    if (c && (end <= 0 || c < end)) end = c, *cut = k;
    if (end <= q) return 0;
    // with TTS_WAIT_S of audio queued or more and the reply still being written, a piece shorter than TTS_MIN_S
    // (after tts_join_fluent) waits for more sentences: the 9B writes meanwhile, and the piece comes out longer
    if (generating && left >= TTS_WAIT_S && !*cut) {
        int e = tts_join_fluent(m, q, end);
        if (st_speech_tokens(t + q, e - q, tts.lang[0] ? tts.lang : tts.user_lang) < TTS_MIN_S * 25) return 0;
    }
    return end;
}

// How far tts_work joins complete sentences after `end` into the piece [start, end) (mode 1): while
// the piece stays under min(10 s, (queued - 1 s) / 0.25) of speech and 250 speech tokens (T3's cap is 300), in one
// language. With less than TTS_CLAUSE_S queued, short pieces only, as in mode 0.
static int tts_join_fluent(const Msg* m, int start, int end) {
    double left = (double)(aout_queued() - aout_played()) / 24000.0;
    int len = (int)m->len;
    if (left < TTS_CLAUSE_S) {
        while (end - start < 50) {
            int e2 = st_next_piece(m->text, len, end, !generating);
            if (e2 <= end || st_spoken_len(m->text + start, e2 - start) > 150) break;
            if (strcmp(st_text_lang(m->text + start, end - start, tts.lang), st_text_lang(m->text + end, e2 - end, tts.lang)))
                break;
            end = e2;
        }
        return end;
    }
    double cap = (left - 1.0) / 0.25;
    if (cap > 10.0) cap = 10.0;
    if (cap < 3.0) cap = 3.0;
    const char* lang = tts.lang[0] ? tts.lang : tts.user_lang;
    for (;;) {
        int e2 = st_next_piece(m->text, len, end, !generating);
        if (e2 <= end || hard_cut_at(m->text, end, e2)) break;   // (not into the middle of a long sentence)
        int n = st_speech_tokens(m->text + start, e2 - start, lang);
        if (n > 250 || n / 25.0 > cap) break;
        if (strcmp(st_text_lang(m->text + start, end - start, tts.lang), st_text_lang(m->text + end, e2 - end, tts.lang)))
            break;
        end = e2;
    }
    return end;
}

// Modes 2 and 3: a piece never ends between words, so the voice does not pause in the middle of a phrase ("fue uno
// de los, | primeros pensadores"). A piece ends only where the text has a period or a line break, or (mode 3) a
// comma; if no such point is written yet the voice waits, so a gap falls where a pause is natural anyway. Pieces
// join sentences up to TTS_PUNCT_S of speech
// (T3 sees whole sentences, and its cap is raised to 450 tokens), waiting for text while TTS_WAIT_PUNCT_S or more is
// queued. Only a sentence longer than one piece without any comma is cut between words (st_natural_cut).
#define TTS_PUNCT_S 11.0
#define TTS_WAIT_PUNCT_S 3.0

static double tts_est_s(const Msg* m, int a, int e) {
    return st_speech_tokens(m->text + a, e - a, tts.lang[0] ? tts.lang : tts.user_lang) / 25.0;
}

// sentences after `end` joined into [start, end) while under TTS_PUNCT_S and in one language; *more: the next
// sentence is written (it did not fit, or is in another language)
static int tts_join_punct(const Msg* m, int start, int end, int* more) {
    int len = (int)m->len;
    if (more) *more = 0;
    for (;;) {
        int e2 = st_next_piece(m->text, len, end, !generating);
        if (e2 <= end) return end;
        if (tts_est_s(m, start, e2) > TTS_PUNCT_S || hard_cut_at(m->text, end, e2) ||
            strcmp(st_text_lang(m->text + start, end - start, tts.lang), st_text_lang(m->text + end, e2 - end, tts.lang))) {
            if (more) *more = 1;
            return end;
        }
        end = e2;
    }
}

static int tts_piece_end_punct(const Msg* m, int* cut) {
    const char* t = m->text;
    int len = (int)m->len, q = tts.queued, end = st_next_piece(t, len, q, !generating), commas = tts_fluent == 3;
    *cut = 0;
    int lim = len;
    for (int i = q; i < lim; i++)   // (never inside code)
        if (t[i] == '`') lim = i;
    if (q == 0) {
        while (end > 0 && end < 16) {   // "¡Claro! " goes with what follows (see above)
            int e2 = st_next_piece(t, len, end, !generating);
            if (e2 > end) {
                end = e2;
                continue;
            }
            if (generating) end = 0;
            break;
        }
        if (commas)   // the first comma 15 bytes in, if it comes before the sentence's end
            for (int i = 15; i + 1 < lim && (end <= 0 || i < end); i++)
                if (t[i] == ',' && t[i + 1] == ' ') return i + 2;
        if (end > 0 && st_spoken_len(t, end) <= 100 && !hard_cut_at(t, 0, end)) return end;   // a short sentence
        // a long first sentence without a comma (waiting for all of it would delay the first sound): cut before a
        // word that opens a phrase ("el día, | porque las..."), said as a clause: the first one 25-60 bytes in once
        // 60 are written, else 25-100 once 100 are; else at the sentence's end, or a natural point
        int sl = st_spoken_len(t, lim);
        if (sl < 60 && end <= 0) return 0;
        int c = st_opener_cut(t, lim, 0, 25, 60);
        if (!c && sl >= 100) c = st_opener_cut(t, lim, 0, 25, 100);
        if (!c && end > 0 && hard_cut_at(t, 0, end)) c = st_natural_cut(t, lim, 0, 60, 160);
        if (c) {
            *cut = 1;
            return c;
        }
        return end > 0 ? end : 0;
    }
    if (end > 0 && hard_cut_at(t, q, end)) {   // a sentence longer than a piece with no comma: a natural point
        int c = st_natural_cut(t, len, q, 90, 160);
        if (c > q) {
            *cut = 1;
            return c;
        }
    }
    double left = (double)(aout_queued() - aout_played()) / 24000.0;
    if (commas && left < TTS_CLAUSE_S)   // the audio is running out: a clause to a comma 25 bytes in will do
        for (int i = q + 25; i + 1 < lim && (end <= 0 || i < end); i++)
            if (t[i] == ',' && t[i + 1] == ' ') return i + 2;
    // about to fall silent with no comma written: cut before a word that opens a phrase ("más intensamente, | que los
    // otros colores"), so the audio does not run dry
    if (commas && left < 2.0 && end <= q && st_spoken_len(t + q, lim - q) >= 60) {
        int c = st_opener_cut(t, lim, q, 40, 100);
        if (c > q) {
            *cut = 1;
            return c;
        }
    }
    if (end <= q) return 0;
    int more;
    tts_join_punct(m, q, end, &more);
    if (generating && !more && left >= TTS_WAIT_PUNCT_S) return 0;   // (wait: the next sentence may still fit)
    return end;
}

// Is there voice work to do now (a piece in progress, or a complete one waiting)?
static int tts_pending(void) {
    if (!tts_loaded || tts.msg < 0 || tts.msg >= n_msgs) return 0;
    if (tts.stage) return 1;
    int cut;
    return tts_piece_end(&msgs[tts.msg], &cut) > 0;
}

// One unit of voice work on the GPU. Errors switch the voice off for this reply.
static void tts_work(void) {
    const Msg* m = &msgs[tts.msg];
    tts_gaps_log();
    if (tts.stage == 0) {
        int cut, len = (int)m->len, end = tts_piece_end(m, &cut);
        if (!end) return;
        int start = tts.queued;
        // after the first piece, short pieces that are already complete are joined (fewer, longer sentences sound
        // better)
        // (not a sentence in another language: "En inglés se dice así: Good morning!" keeps its two voices)
        if (tts_fluent >= 2 && start > 0) end = tts_join_punct(m, start, end, NULL);
        else if (tts_fluent && start > 0) end = tts_join_fluent(m, start, end);
        else while (start > 0 && end - start < 50) {
            int e2 = st_next_piece(m->text, len, end, !generating);
            if (e2 <= end || st_spoken_len(m->text + start, e2 - start) > 150) break;
            if (strcmp(st_text_lang(m->text + start, end - start, tts.lang), st_text_lang(m->text + end, e2 - end, tts.lang)))
                break;
            end = e2;
        }
        tts.queued = end;
        static char words[1024];
        int wl = st_clean(m->text + start, end - start, words, sizeof(words) - 2);
        if (!wl) return;   // nothing to say (code, emoji)
        // the reply's language so far (guessed from the user's words first), then this piece's own words
        snprintf(tts.lang, sizeof(tts.lang), "%s", st_lang(m->text, tts.user_lang[0] ? tts.user_lang : lang_code()));
        const char* lang = st_lang(words, tts.lang);
        wl = st_speak(words, wl, (int)sizeof(words) - 2, lang);   // numbers, abbreviations, acronyms as words
        if (!wl) return;
        if (cut && !strchr(".,;:!?", words[wl - 1])) words[wl] = ',', words[wl + 1] = 0;
        if (tts_fluent && !cut && !strchr(".,;:!?", words[wl - 1])) {   // the piece's final comma, which st_clean drops
            int e = end;
            while (e > start && (m->text[e - 1] == ' ' || m->text[e - 1] == '\n')) e--;
            if (e > start && m->text[e - 1] == ',') words[wl] = ',', words[wl + 1] = 0;
        }
        // a colon as a comma: T3 can read a colon aloud ("понять:" in Russian); between digits (10:30) it stays
        for (char* c = words; *c; c++)
            if (*c == ':' && !(c > words && c[-1] >= '0' && c[-1] <= '9' && c[1] >= '0' && c[1] <= '9')) *c = ',';
        static uint16_t toks[512];
        int n = ttstok_encode(&tts_tok, words, lang, toks, 400);
        int max_out = (int)strlen(words) * 3 + 40;   // T3 ends the sentence itself; this only bounds a runaway
        // (modes 2-3 make pieces of up to ~11 s, which can need about 400 tokens: the cap is 450, which S3Gen's
        // work area still takes, ~52 of 54 MiB; t3_start keeps it within T3_MAX_SEQ)
        int cap = tts_fluent >= 2 ? 450 : 300;
        if (max_out > cap) max_out = cap;
        t3_voice(&eng, lang);    // native pronunciation for this language
        s3g_voice(&eng, lang);   // and its own voice where it has one (Russian), else Garlic's
        if (t3_start(&eng, toks, n, max_out, ++tts.seed, 0)) {
            say("voice: T3 failed (%s)", eng.error);
            return;
        }
        tts.stage = 1, tts.b0 = start, tts.b1 = end, tts.n_speech = 0;
        tts.t_piece = now();
        {
            int e = (int)strlen(words);
            while (e > 0 && (words[e - 1] == ' ' || words[e - 1] == '\n')) e--;
            tts.clause_end = e > 0 && strchr(",;:", words[e - 1]) != NULL;
        }
        say("voice [%s] %.120s", lang, words);
        return;
    }
    if (tts.stage == 1) {   // a few speech tokens per frame
        for (int k = 0; k < 6; k++) {
            int t = t3_next(&eng);
            if (t >= 0 && tts.n_speech < 512) {
                tts.speech[tts.n_speech++] = (uint16_t)t;
                continue;
            }
            if (t == -2) say("voice: T3 failed (%s)", eng.error);
            if (t == -1) {
                int nn = tts.n_speech ? tts.n_speech : 1;
                say("voice: T3 %d speech tokens in %.2f s after a prefill of %d positions in %.3f s (%d per pass); per "
                    "token: GPU %.2f ms, submit+wait %.2f ms, rows %.2f ms, sampling %.2f ms", tts.n_speech,
                    now() - tts.t_piece, eng.t3.n_in, eng.t3.s_prefill, eng.t3.prefill_batched,
                    1000 * eng.t3.s_gpu / nn, 1000 * eng.t3.s_submit / nn, 1000 * eng.t3.s_rows / nn,
                    1000 * eng.t3.s_sample / nn);
                // T3 ends every piece with 8-12 silence tokens (0.3-0.5 s), which with S3Gen's lead-in make ~0.6 s of
                // quiet at every boundary, commas included: keep 3 of them (0.12 s) after a clause, 7 (0.28 s) after
                // a sentence
                static const uint16_t sil[6] = {6324, 6405, 4137, 4218, 6081, 6082};
                int run = 0, keep = tts.clause_end ? 3 : 7;
                for (; run < tts.n_speech; run++) {
                    uint16_t s = tts.speech[tts.n_speech - 1 - run];
                    int is = 0;
                    for (int q = 0; q < 6; q++) is |= s == sil[q];
                    if (!is) break;
                }
                if (run > keep) {
                    tts.n_speech -= run - keep;
                    say("voice: %d of the %d trailing silence tokens dropped (%s)", run - keep, run,
                        tts.clause_end ? "clause" : "sentence");
                }
            }
            if (t == -1 && tts.n_speech >= 2 && s3g_begin(&eng, tts.speech, tts.n_speech, tts.seed) > 0) {
                tts.stage = 2;
                if (tts.nsp < (int)(sizeof(tts.sp) / sizeof(tts.sp[0]))) {
                    int64_t q = aout_queued();
                    tts.sp[tts.nsp++] = (Spoken){q, q, tts.b0, tts.b1};
                }
            } else {
                if (t == -1 && tts.n_speech >= 2) say("voice: S3Gen failed (%s)", eng.error);
                tts.stage = 0;
            }
            return;
        }
        return;
    }
    int r = s3g_step(&eng, tts_pcm, (int)(sizeof(tts_pcm) / sizeof(tts_pcm[0])));
    if (r > 0) {
        if (tts.nsp && tts.sp[tts.nsp - 1].s1 == tts.sp[tts.nsp - 1].s0) {   // the piece's first samples
            say("voice: piece starts playing with %.2f s of audio still queued",
                (double)(aout_queued() - aout_played()) / 24000.0);
            tts.lv_sum = 0, tts.lv_n = 0, tts.lim0 = aout_gain(eng.s3.gain);
        }
        // every voice as loud as Garlic's (S3Voice.gain); the level of each 20 ms frame with speech (above -50 dBFS)
        // goes to the log
        aout_gain(eng.s3.gain);
        double g2 = (double)eng.s3.gain * eng.s3.gain;
        for (int f0 = 0; f0 + 480 <= r; f0 += 480) {
            double ss = 0;
            for (int i = 0; i < 480; i++) ss += (double)tts_pcm[f0 + i] * tts_pcm[f0 + i];
            ss = ss * g2 / 480;
            if (ss > 1e-5) tts.lv_sum += ss, tts.lv_n++;
        }
        aout_push(tts_pcm, r);
        if (tts.nsp) tts.sp[tts.nsp - 1].s1 += r;
    } else if (r == -1) {
        tts.stage = 0;
        tts.n_pieces++, tts.s_audio += (tts.n_speech - 1) * 0.04;
        const S3G* g = &eng.s3;
        say("voice: %d speech tokens -> %.2f s of audio, S3Gen %.2f s (GPU: encoder %.0f ms, CFM %.0f ms, sources %.0f "
            "ms, HiFT %.0f ms); level %.1f dBFS (gain %+.1f dB, %d samples limited)", tts.n_speech,
            (tts.n_speech - 1) * 0.04, g->last_seconds, 1000 * g->s_enc, 1000 * g->s_cfm, 1000 * g->s_src,
            1000 * g->s_hift, tts.lv_n ? 10 * log10(tts.lv_sum / tts.lv_n) : -100.0, 20 * log10(g->gain),
            (int)(aout_gain(g->gain) - tts.lim0));
    } else if (r == -2) {
        say("voice: S3Gen failed (%s)", eng.error);
        tts.stage = 0;
    }
}

// Bytes of a reply to show when `played` samples have been played: those already spoken, the current piece revealed
// word by word as it plays. (The voice screen's thread calls it with its own copy of the text and of sp.)
static int spoken_bytes(const char* text, int len, const Spoken* sp, int nsp, int64_t played) {
    int shown = 0;
    for (int i = 0; i < nsp; i++) {
        const Spoken* s = &sp[i];
        if (played >= s->s1 && s->s1 > s->s0) shown = s->b1;
        else if (played > s->s0 && s->s1 > s->s0) {
            int b = s->b0 + (int)((double)(s->b1 - s->b0) * (double)(played - s->s0) / (double)(s->s1 - s->s0));
            while (b < s->b1 && b < len && text[b] != ' ' && text[b] != '\n') b++;   // whole words
            if (b > shown) shown = b;
        }
    }
    return shown < len ? shown : len;
}
