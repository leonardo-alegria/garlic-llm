// Qwen byte-level BPE tokenizer.
//
// Encoding follows Hugging Face's Qwen2 tokenizer: the text is first split with Qwen's pre-tokenizer pattern
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+
//   | \s+(?!\S) | \s+
// (tried in that order at each position), then each piece's bytes are merged pairwise by merge rank.
// Unicode classes are approximated for the scripts a chat in Latin languages needs: letters are ASCII letters
// plus everything outside ASCII except punctuation, symbol and space blocks; numbers are ASCII digits and the
// Latin-1 superscripts / fractions.

#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

int tok_load(Tokenizer* t, const uint8_t* data, size_t size) {
    if (size < 12 + 1024 || memcmp(data, "PTOK", 4)) return -1;
    memset(t, 0, sizeof(*t));
    t->n_vocab = rd32(data + 4);
    t->n_merges = rd32(data + 8);
    t->byte_id = (const uint32_t*)(data + 12);
    const uint8_t* p = data + 12 + 1024;
    t->raw = malloc(sizeof(*t->raw) * t->n_vocab);
    t->raw_len = malloc(sizeof(*t->raw_len) * t->n_vocab);
    t->special = malloc(t->n_vocab);
    for (uint32_t i = 0; i < t->n_vocab; i++) {
        t->special[i] = p[0];
        t->raw_len[i] = (uint16_t)(p[1] | p[2] << 8);
        t->raw[i] = p + 3;
        p += 3 + t->raw_len[i];
    }
    uint32_t cap = 1;
    while (cap < t->n_merges * 2) cap <<= 1;
    t->mask = cap - 1;
    t->keys = calloc(cap, sizeof(uint64_t));
    t->rank = malloc(cap * sizeof(uint32_t));
    t->merged = malloc(cap * sizeof(uint32_t));
    for (uint32_t r = 0; r < t->n_merges; r++, p += 12) {
        uint64_t key = ((uint64_t)rd32(p) << 32 | rd32(p + 4)) + 1;
        uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 40) & t->mask;
        while (t->keys[h] && t->keys[h] != key) h = (h + 1) & t->mask;
        if (!t->keys[h]) {
            t->keys[h] = key;
            t->rank[h] = r;
            t->merged[h] = rd32(p + 8);
        }
    }
    return (size_t)(p - data) == size ? 0 : -2;
}

static int find_merge(const Tokenizer* t, uint32_t a, uint32_t b, uint32_t* merged) {
    uint64_t key = ((uint64_t)a << 32 | b) + 1;
    uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 40) & t->mask;
    while (t->keys[h]) {
        if (t->keys[h] == key) {
            *merged = t->merged[h];
            return (int)t->rank[h];
        }
        h = (h + 1) & t->mask;
    }
    return -1;
}

// ---- Unicode helpers -------------------------------------------------------------------------------------------

static uint32_t next_cp(const uint8_t* s, int n, int i, int* len) {
    uint8_t c = s[i];
    int l = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + l > n) l = 1;
    uint32_t cp = l == 1 ? c : l == 2 ? c & 0x1F : l == 3 ? c & 0x0F : c & 0x07;
    for (int k = 1; k < l; k++) cp = cp << 6 | (s[i + k] & 0x3F);
    *len = l;
    return cp;
}

static int is_space(uint32_t c) {
    return c == ' ' || (c >= 9 && c <= 13) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static int is_num(uint32_t c) {
    return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE);
}

static int is_letter(uint32_t c) {
    if (c < 0x80) return (c | 32) >= 'a' && (c | 32) <= 'z';
    if (c < 0xC0) return c == 0xAA || c == 0xB5 || c == 0xBA;
    if (c == 0xD7 || c == 0xF7) return 0;
    if (is_space(c) || is_num(c)) return 0;
    if ((c >= 0x2000 && c <= 0x2BFF) || (c >= 0x3000 && c <= 0x303F) || (c >= 0xFE30 && c <= 0xFE4F) ||
        (c >= 0xFF00 && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65) ||
        (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0xE000 && c <= 0xF8FF) || c == 0xFFFD)
        return 0;
    return 1;
}

// Length in bytes of the pre-token starting at byte i.
static int pretoken(const uint8_t* s, int n, int i) {
    int l0, l1;
    uint32_t c = next_cp(s, n, i, &l0);

    // 1. contractions 's 't 're 've 'm 'll 'd (case-insensitive)
    if (c == '\'' && i + 1 < n) {
        static const char* suf[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (int k = 0; k < 7; k++) {
            int m = (int)strlen(suf[k]);
            if (i + 1 + m <= n) {
                int ok = 1;
                for (int j = 0; j < m; j++)
                    if ((s[i + 1 + j] | 32) != suf[k][j]) ok = 0;
                if (ok) return 1 + m;
            }
        }
    }
    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    {
        int j = i;
        if (!is_letter(c) && c != '\r' && c != '\n' && !is_num(c) && i + l0 < n) {
            uint32_t c2 = next_cp(s, n, i + l0, &l1);
            if (is_letter(c2)) j = i + l0;
        }
        int k = j;
        while (k < n) {
            uint32_t ck = next_cp(s, n, k, &l1);
            if (!is_letter(ck)) break;
            k += l1;
        }
        if (k > j) return k - i;
    }
    // 3. \p{N}
    if (is_num(c)) return l0;
    // 4. ' '?[^\s\p{L}\p{N}]+[\r\n]*
    {
        int j = i;
        if (c == ' ') j += 1;
        int k = j;
        while (k < n) {
            uint32_t ck = next_cp(s, n, k, &l1);
            if (is_space(ck) || is_letter(ck) || is_num(ck)) break;
            k += l1;
        }
        if (k > j) {
            while (k < n && (s[k] == '\r' || s[k] == '\n')) k++;
            return k - i;
        }
    }
    // whitespace run starting here
    int end = i, last_nl = -1;
    while (end < n) {
        uint32_t ce = next_cp(s, n, end, &l1);
        if (!is_space(ce)) break;
        if (ce == '\r' || ce == '\n') last_nl = end + l1;
        end += l1;
    }
    // 5. \s*[\r\n]+
    if (last_nl > i) return last_nl - i;
    // 6. \s+(?!\S): whole run at the end of the text, else all but its last character
    if (end > i) {
        if (end == n) return end - i;
        int k = i, prev = i;
        while (k < end) {
            prev = k;
            next_cp(s, n, k, &l1);
            k += l1;
        }
        if (prev > i) return prev - i;
        return end - i;   // 7. \s+ (a single whitespace character)
    }
    return l0;
}

static int bpe(const Tokenizer* t, const uint8_t* s, int n, uint32_t* out, int max) {
    uint32_t buf[512];
    if (n > 512) n = 512;
    for (int i = 0; i < n; i++) buf[i] = t->byte_id[s[i]];
    int len = n;
    for (;;) {
        int best = -1, best_i = -1;
        uint32_t best_id = 0, m;
        for (int i = 0; i + 1 < len; i++) {
            int r = find_merge(t, buf[i], buf[i + 1], &m);
            if (r >= 0 && (best < 0 || r < best)) {
                best = r;
                best_i = i;
                best_id = m;
            }
        }
        if (best < 0) break;
        buf[best_i] = best_id;
        memmove(buf + best_i + 1, buf + best_i + 2, (len - best_i - 2) * sizeof(uint32_t));
        len--;
    }
    if (len > max) len = max;
    memcpy(out, buf, len * sizeof(uint32_t));
    return len;
}

int tok_encode(const Tokenizer* t, const char* utf8, uint32_t* out, int max) {
    const uint8_t* s = (const uint8_t*)utf8;
    int n = (int)strlen(utf8), i = 0, count = 0;
    while (i < n && count < max) {
        int l = pretoken(s, n, i);
        count += bpe(t, s + i, l, out + count, max - count);
        i += l;
    }
    return count;
}

int tok_piece(const Tokenizer* t, uint32_t id, const uint8_t** bytes) {
    if (id >= t->n_vocab || t->special[id]) return 0;
    *bytes = t->raw[id];
    return t->raw_len[id];
}
