// Text tokenizer of Chatterbox Multilingual (see tts_tok.h), the C version of the official Python steps:
// punc_norm -> lower() -> NFKD -> "[lang]" + text -> spaces become [SPACE] -> added tokens matched first ->
// Whitespace pre-tokenizer (\w+ | [^\w\s]+) -> BPE with the merge ranks -> ids, between start 255 and stop 0.
// Russian stress marks are added first when their dictionary is loaded (ru_stress.h).
#include "tts_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | p[1] << 8); }

int ttstok_load(TtsTok* t, const uint8_t* d, size_t n) {
    memset(t, 0, sizeof(*t));
    if (n < 8 || memcmp(d, "PTTK", 4)) return -1;
    const uint8_t* p = d + 4;
    t->n_vocab = rd32(p), p += 4;
    t->vocab = malloc(t->n_vocab * sizeof(*t->vocab));
    for (uint32_t i = 0; i < t->n_vocab; i++) t->vocab[i] = p, p += 1 + p[0];
    t->n_merges = rd32(p), p += 4;
    t->merges = p, p += t->n_merges * 6;
    t->n_added = rd32(p), p += 4;
    t->added = malloc(t->n_added * sizeof(*t->added));
    for (uint32_t i = 0; i < t->n_added; i++) t->added[i] = p, p += 3 + p[2];
    t->n_lower = rd32(p), p += 4;
    t->lower = p, p += t->n_lower * 8;
    t->n_nfkd = rd32(p), p += 4;
    t->nfkd = malloc(t->n_nfkd * sizeof(*t->nfkd));
    for (uint32_t i = 0; i < t->n_nfkd; i++) t->nfkd[i] = p, p += 5 + 4 * p[4];
    t->word_bits = p;
    return (size_t)(p + 0x3000 / 8 - d) <= n ? 0 : -1;
}

// ---- UTF-8 ----
static uint32_t next_cp(const char** s) {
    const uint8_t* p = (const uint8_t*)*s;
    uint32_t c = *p++;
    int extra = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : 0;
    if (extra) c &= 0x3F >> extra;
    for (int k = 0; k < extra && (*p & 0xC0) == 0x80; k++) c = c << 6 | (*p++ & 0x3F);
    *s = (const char*)p;
    return c;
}

static int put_cp(uint32_t c, char* o) {
    if (c < 0x80) return o[0] = (char)c, 1;
    if (c < 0x800) return o[0] = (char)(0xC0 | c >> 6), o[1] = (char)(0x80 | (c & 63)), 2;
    if (c < 0x10000) return o[0] = (char)(0xE0 | c >> 12), o[1] = (char)(0x80 | (c >> 6 & 63)), o[2] = (char)(0x80 | (c & 63)), 3;
    o[0] = (char)(0xF0 | c >> 18), o[1] = (char)(0x80 | (c >> 12 & 63)), o[2] = (char)(0x80 | (c >> 6 & 63)), o[3] = (char)(0x80 | (c & 63));
    return 4;
}

static uint32_t lower_cp(const TtsTok* t, uint32_t c) {
    int lo = 0, hi = (int)t->n_lower - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        uint32_t k = rd32(t->lower + m * 8);
        if (k == c) return rd32(t->lower + m * 8 + 4);
        if (k < c) lo = m + 1;
        else hi = m - 1;
    }
    return c;
}

static const uint8_t* nfkd_of(const TtsTok* t, uint32_t c) {
    int lo = 0, hi = (int)t->n_nfkd - 1;
    while (lo <= hi) {
        int m = (lo + hi) / 2;
        uint32_t k = rd32(t->nfkd[m]);
        if (k == c) return t->nfkd[m];
        if (k < c) lo = m + 1;
        else hi = m - 1;
    }
    return NULL;
}

static int is_word(const TtsTok* t, uint32_t c) {   // Oniguruma \w: letters, marks, numbers, '_'
    if (c >= 0x3000) return c < 0x1F000;   // CJK and the like are words; emoji and symbols are not
    return t->word_bits[c >> 3] >> (c & 7) & 1;
}

static int is_space(uint32_t c) {   // Oniguruma \s (the pre-tokenizer)
    return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
static int py_space(uint32_t c) { return is_space(c) || (c >= 0x1C && c <= 0x1F); }   // Python str.split()

// ---- punc_norm (mtl_tts.py) ----
static void replace_all(char* s, size_t cap, const char* from, const char* to) {
    size_t fl = strlen(from), tl = strlen(to);
    for (char* p = strstr(s, from); p; p = strstr(p + tl, from)) {
        size_t rest = strlen(p + fl);
        if ((size_t)(p - s) + tl + rest + 1 > cap) return;
        memmove(p + tl, p + fl, rest + 1);
        memcpy(p, to, tl);
    }
}

static void punc_norm(const char* in, char* out, size_t cap) {
    size_t o = 0;   // collapse runs of white space into one space, trim both ends
    int pending = 0;
    for (const char* s = in; *s && o + 8 < cap;) {
        const char* before = s;
        uint32_t c = next_cp(&s);
        if (py_space(c)) {
            pending = o > 0;
            continue;
        }
        if (pending) out[o++] = ' ', pending = 0;
        memcpy(out + o, before, s - before);
        o += s - before;
    }
    out[o] = 0;
    if (o >= 2 && (uint8_t)in[0] == 0xC3 && (uint8_t)in[1] == 0x9F) out[0] = out[1] = 'S';   // text[0].upper(): "ß" -> "SS"
    if (!*in) {
        strncpy(out, "You need to add some text for me to talk.", cap - 1);
        out[cap - 1] = 0;
        return;
    }
    static const char* rep[][2] = {{"...", ", "}, {"\xE2\x80\xA6", ", "}, {":", ","},      {" - ", ", "},
                                   {";", ", "},   {"\xE2\x80\x94", "-"}, {"\xE2\x80\x93", "-"}, {" ,", ","},
                                   {"\xE2\x80\x9C", "\""}, {"\xE2\x80\x9D", "\""}, {"\xE2\x80\x98", "'"}, {"\xE2\x80\x99", "'"}};
    for (size_t i = 0; i < sizeof(rep) / sizeof(rep[0]); i++) replace_all(out, cap, rep[i][0], rep[i][1]);
    o = strlen(out);
    while (o && out[o - 1] == ' ') out[--o] = 0;
    static const char* enders[] = {".", "!", "?", "-", ",", "\xE3\x80\x81", "\xEF\xBC\x8C", "\xE3\x80\x82", "\xEF\xBC\x9F", "\xEF\xBC\x81"};
    for (size_t i = 0; i < sizeof(enders) / sizeof(enders[0]); i++) {
        size_t l = strlen(enders[i]);
        if (o >= l && !memcmp(out + o - l, enders[i], l)) return;
    }
    if (o + 2 < cap) out[o] = '.', out[o + 1] = 0;
}

// ---- BPE on one pre-token ----
static int vocab_id(const TtsTok* t, const char* s, int len) {
    for (uint32_t i = 0; i < t->n_vocab; i++)   // 2454 entries: a linear scan per character is fine
        if (t->vocab[i][0] == len && !memcmp(t->vocab[i] + 1, s, len)) return (int)i;
    return -1;
}

static int bpe_word(const TtsTok* t, const char* s, int len, uint16_t* out, int max) {
    int ids[512], n = 0;
    for (const char* p = s; p < s + len && n < 512;) {   // characters
        const char* q = p;
        next_cp(&q);
        int id = vocab_id(t, p, (int)(q - p));
        ids[n++] = id >= 0 ? id : 1;   // [UNK]
        p = q;
    }
    for (;;) {   // merge the adjacent pair with the lowest rank (leftmost first), until none applies
        int best = -1, best_rank = 1 << 30, merged = 0;
        for (int i = 0; i + 1 < n; i++)
            for (uint32_t r = 0; r < t->n_merges && (int)r < best_rank; r++) {
                const uint8_t* m = t->merges + r * 6;
                if (rd16(m) == ids[i] && rd16(m + 2) == ids[i + 1]) {
                    best = i, best_rank = (int)r, merged = rd16(m + 4);
                    break;
                }
            }
        if (best < 0) break;
        ids[best] = merged;
        memmove(ids + best + 1, ids + best + 2, (n - best - 2) * sizeof(int));
        n--;
    }
    int k = 0;
    for (int i = 0; i < n && k < max; i++) out[k++] = (uint16_t)ids[i];
    return k;
}

static int pretok_bpe(const TtsTok* t, const char* s, int len, uint16_t* out, int max) {
    int k = 0;
    const char* end = s + len;
    while (s < end && k < max) {
        const char* p = s;
        uint32_t c = next_cp(&p);
        if (is_space(c)) {
            s = p;
            continue;
        }
        int w = is_word(t, c);
        const char* e = p;   // a run of word characters, or of other non-space characters
        while (e < end) {
            const char* q = e;
            uint32_t d = next_cp(&q);
            if (is_space(d) || is_word(t, d) != w) break;
            e = q;
        }
        k += bpe_word(t, s, (int)(e - s), out + k, max - k);
        s = e;
    }
    return k;
}

int ttstok_encode(const TtsTok* t, const char* text, const char* lang, uint16_t* out, int max) {
    static char a[4096], b[8192], st[8192];
    punc_norm(text, a, sizeof(a));
    // Russian: stress marks first (on the text as written, before NFKD splits й and ё), as the official tokenizer
    const char* src = a;
    if (t->ru && !strcmp(lang, "ru")) {
        rustress_text(t->ru, a, st, sizeof(st));
        src = st;
    }
    // lower(), then NFKD, with "[lang]" in front and spaces as [SPACE]
    size_t o = (size_t)snprintf(b, sizeof(b), "[%s]", lang);
    for (const char* s = src; *s && o + 40 < sizeof(b);) {
        uint32_t c = lower_cp(t, next_cp(&s));
        if (c == ' ') {
            memcpy(b + o, "[SPACE]", 7), o += 7;
            continue;
        }
        const uint8_t* d = c < 0x3000 ? nfkd_of(t, c) : NULL;
        if (d)
            for (int i = 0; i < d[4]; i++) o += put_cp(rd32(d + 5 + 4 * i), b + o);
        else o += put_cp(c, b + o);
    }
    b[o] = 0;
    int k = 0;
    if (k < max) out[k++] = 255;   // start of text
    const char* s = b;
    const char* seg = s;
    while (*s && k < max - 1) {   // added tokens first (longest first), the rest between them through BPE
        int hit = -1, hl = 0;
        for (uint32_t i = 0; i < t->n_added; i++) {
            int l = t->added[i][2];
            if (!strncmp(s, (const char*)t->added[i] + 3, l)) {
                hit = rd16(t->added[i]), hl = l;
                break;
            }
        }
        if (hit < 0) {
            next_cp(&s);
            continue;
        }
        if (s > seg) k += pretok_bpe(t, seg, (int)(s - seg), out + k, max - 1 - k);
        if (k < max - 1) out[k++] = (uint16_t)hit;
        s += hl;
        seg = s;
    }
    if (*seg && k < max - 1) k += pretok_bpe(t, seg, (int)strlen(seg), out + k, max - 1 - k);
    out[k++] = 0;   // stop of text
    return k;
}
