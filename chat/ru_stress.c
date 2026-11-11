// See ru_stress.h.
#include "ru_stress.h"

#include <string.h>

static uint32_t rs_rd32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

int rustress_load(RuStress* d, const uint8_t* blob, size_t size) {
    if (size < 16 || memcmp(blob, "RUST", 4) || rs_rd32(blob + 4) != 1) return -1;
    d->n_keys = rs_rd32(blob + 8);
    d->n_blocks = rs_rd32(blob + 12);
    if (16 + (size_t)d->n_blocks * 4 > size) return -1;
    d->index = blob + 16;
    d->data = blob + 16 + (size_t)d->n_blocks * 4;
    d->data_size = size - 16 - (size_t)d->n_blocks * 4;
    return 0;
}

static int rs_cmp(const uint8_t* a, int na, const uint8_t* b, int nb) {
    int n = na < nb ? na : nb, c = memcmp(a, b, n);
    return c ? c : na - nb;
}

// key: codes 1..33 (а..я with ё as е = 1..32, '-' = 33). Fills marks[] (bit 7 more, bit 6 ё, bits 0..5 position).
// Returns the number of marks, 0 if the form is not in the dictionary.
static int rustress_lookup(const RuStress* d, const uint8_t* key, int n, uint8_t* marks, int max_marks) {
    int lo = 0, hi = (int)d->n_blocks - 1, b = -1;
    while (lo <= hi) {                       // last block whose first key <= key
        int m = (lo + hi) / 2;
        const uint8_t* r = d->data + rs_rd32(d->index + 4 * m);
        if (rs_cmp(r + 2, r[1], key, n) <= 0) b = m, lo = m + 1;
        else hi = m - 1;
    }
    if (b < 0) return 0;
    const uint8_t* p = d->data + rs_rd32(d->index + 4 * b);
    const uint8_t* end = b + 1 < (int)d->n_blocks ? d->data + rs_rd32(d->index + 4 * (b + 1)) : d->data + d->data_size;
    uint8_t cur[256];
    int cn = 0;
    while (p < end) {
        int pre = p[0], ln = p[1];
        memcpy(cur + pre, p + 2, ln);
        cn = pre + ln;
        p += 2 + ln;
        const uint8_t* mk = p;
        while (*p & 0x80) p++;
        p++;
        int c = rs_cmp(cur, cn, key, n);
        if (c == 0) {
            int k = 0;
            for (const uint8_t* q = mk; q < p && k < max_marks; q++) marks[k++] = *q;
            return k;
        }
        if (c > 0) return 0;
    }
    return 0;
}

// --- UTF-8 helpers (only Cyrillic letters matter here)
static int rs_code(uint32_t cp) {            // letter code 1..32, 0 if not a Russian letter
    if (cp >= 0x410 && cp <= 0x42F) cp += 0x20;
    if (cp == 0x401 || cp == 0x451) return 6;   // ё -> е
    if (cp >= 0x430 && cp <= 0x44F) return (int)(cp - 0x430) + 1;
    return 0;
}

static uint32_t rs_next(const uint8_t** s) {
    const uint8_t* p = *s;
    uint32_t c = *p++;
    if (c >= 0xC0 && c < 0xE0 && (*p & 0xC0) == 0x80) c = (c & 0x1F) << 6 | (*p++ & 0x3F);
    else if (c >= 0xE0 && c < 0xF0 && (p[0] & 0xC0) == 0x80 && (p[1] & 0xC0) == 0x80)
        c = (c & 0x0F) << 12 | (p[0] & 0x3F) << 6 | (p[1] & 0x3F), p += 2;
    else if (c >= 0xF0 && (p[0] & 0xC0) == 0x80 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80)
        c = (c & 0x07) << 18 | (p[0] & 0x3F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F), p += 3;
    *s = p;
    return c;
}

typedef struct { const uint8_t* src; int len; uint32_t cp; } RsChar;   // one char of the word, as in the input

// Writes word chars [a, b) with the marks of the dictionary applied.
static size_t rs_emit(const RsChar* w, int a, int b, const uint8_t* mk, int nm, char* out, size_t o, size_t cap) {
    for (int i = a; i < b; i++) {
        int yo = 0, ac = 0;
        for (int k = 0; k < nm; k++)
            if ((mk[k] & 0x3F) == i - a) { if (mk[k] & 0x40) yo = 1; else ac = 1; }
        if (yo && (w[i].cp == 0x435 || w[i].cp == 0x415 || w[i].cp == 0x451 || w[i].cp == 0x401)) {
            int up = w[i].cp == 0x415 || w[i].cp == 0x401;
            if (o + 2 < cap) out[o++] = (char)0xD0 + (up ? 0 : 1), out[o++] = (char)(up ? 0x81 : 0x91);
        } else {
            for (int j = 0; j < w[i].len; j++) if (o + 1 < cap) out[o++] = (char)w[i].src[j];
        }
        if (ac && o + 2 < cap) out[o++] = (char)0xCC, out[o++] = (char)0x81;   // U+0301
    }
    return o;
}

static size_t rs_word(const RuStress* d, const RsChar* w, int n, char* out, size_t o, size_t cap) {
    uint8_t key[64], mk[8];
    int nm = 0;
    if (n <= 64) {
        for (int i = 0; i < n; i++) key[i] = w[i].cp == '-' ? 33 : (uint8_t)rs_code(w[i].cp);
        nm = rustress_lookup(d, key, n, mk, 8);
    }
    if (nm) return rs_emit(w, 0, n, mk, nm, out, o, cap);
    int a = 0;                                   // not found: hyphenated parts one by one
    for (int i = 0; i <= n; i++) {
        if (i < n && w[i].cp != '-') continue;
        int m = 0;
        if (i > a && i - a <= 64 && (a > 0 || i < n)) {
            for (int j = a; j < i; j++) key[j - a] = (uint8_t)rs_code(w[j].cp);
            m = rustress_lookup(d, key, i - a, mk, 8);
        }
        o = rs_emit(w, a, i, mk, m, out, o, cap);
        if (i < n && o + 1 < cap) out[o++] = '-';
        a = i + 1;
    }
    return o;
}

// Stresses UTF-8 text into out (cap bytes, NUL-terminated). Each mark adds 2 bytes: cap >= 2 * strlen(in) + 1 is
// always enough. Returns the output length.
size_t rustress_text(const RuStress* d, const char* in, char* out, size_t cap) {
    const uint8_t* s = (const uint8_t*)in;
    size_t o = 0;
    RsChar w[128];
    int n = 0;
    while (1) {
        const uint8_t* st = s;
        uint32_t cp = *s ? rs_next(&s) : 0;
        const uint8_t* t = s;
        uint32_t nx = *t ? rs_next(&t) : 0;
        // a word = Russian letters, with a hyphen only between two letters (кто-то, из-за)
        int in_word = rs_code(cp) || (cp == '-' && n > 0 && rs_code(nx));
        if (in_word && n == 128) {               // absurdly long run: flush it unchanged
            o = rs_emit(w, 0, n, NULL, 0, out, o, cap);
            n = 0;
        }
        if (in_word) {
            w[n].src = st, w[n].len = (int)(s - st), w[n].cp = cp, n++;
            continue;
        }
        if (n) o = rs_word(d, w, n, out, o, cap), n = 0;
        if (!cp) break;
        for (const uint8_t* q = st; q < s; q++) if (o + 1 < cap) out[o++] = (char)*q;
    }
    if (cap) out[o < cap ? o : cap - 1] = 0;
    return o;
}
