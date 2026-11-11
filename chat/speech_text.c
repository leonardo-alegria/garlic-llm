// See speech_text.h.
#include "speech_text.h"

#include <string.h>

static int is_cont(uint8_t c) { return (c & 0xC0) == 0x80; }

static uint32_t cp_at(const char* s, int* len) {
    const uint8_t* p = (const uint8_t*)s;
    uint32_t c = p[0];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (n > 1) c &= 0x3F >> (n - 1);
    for (int i = 1; i < n; i++) {
        if (!is_cont(p[i])) {
            n = i;
            break;
        }
        c = c << 6 | (p[i] & 0x3F);
    }
    *len = n;
    return c;
}

// ---- where the next piece to speak ends ----
// A piece ends after . ! ? … (and any closing quotes / brackets) followed by white space, at a line break, or at
// the end of a code block's opening fence. Pieces longer than SOFT_MAX bytes end at the next comma, colon or
// semicolon; longer than HARD_MAX at the next space (a piece is spoken in at most ~12 s: T3 stops at 300 tokens).
// Lengths count each digit as 7 bytes, as st_speak says numbers in words ("753" -> "setecientos cincuenta y tres"),
// so a piece with dates in it stays within T3's limit.
#define SOFT_MAX 100
#define HARD_MAX 160
#define DIGIT_EXTRA 6

// Is the period at text[i] (a space follows) part of an abbreviation or a German ordinal rather than a sentence end?
// "p. ej.", "н. э.", "v. Chr. gegründet", "753 a.C. y", "av. J.-C.", "Dr. López", "am 1. Mai", "3. lugar"
static int abbrev_period(const char* text, int len, int from, int i) {
    int a = i;
    while (a > from && text[a - 1] != ' ' && text[a - 1] != '\n' && text[a - 1] != '(') a--;
    char w[16];
    int k = 0, letters = 0, digits = 0, chars = 0;
    for (int p = a; p < i && k < 15; p++) {
        uint8_t c = (uint8_t)text[p];
        w[k++] = (char)(c >= 'A' && c <= 'Z' ? c + 32 : c);
        if ((c | 32) >= 'a' && (c | 32) <= 'z') letters++;
        if (c >= '0' && c <= '9') digits++;
        if (!is_cont(c)) chars++;
    }
    w[k] = 0;
    if (i - a >= 15) return 0;
    int q = i + 1;
    while (q < len && text[q] == ' ') q++;
    uint8_t nx = q < len ? (uint8_t)text[q] : 0;
    int next_lower = (nx >= 'a' && nx <= 'z') || (nx >= '0' && nx <= '9') ||
                     (nx == 0xC3 && q + 1 < len && (uint8_t)text[q + 1] >= 0x9F) ||                      // á é ñ ü ß
                     (nx == 0xD0 && q + 1 < len && (uint8_t)text[q + 1] >= 0xB0) || nx == 0xD1 ||          // а-я
                     nx == '-' || nx == '(' || nx == ')' || nx == ',' || (nx == 0xE2 && q + 1 < len && (uint8_t)text[q + 1] == 0x80);   // – —
    if (chars == 1 && !digits && (letters || (uint8_t)text[a] >= 0xC0)) return 1;   // a lone letter: p. ej., н. э.
    static const char* const always[] = {"av", "apr", "sr", "sra", "srta", "dr", "dra", "prof", "aprox", "ej", "ex",
                                         "ca", "bzw", "vs", "mr", "mrs", "ms", "vol", "cap", "pág", "núm", "art", 0};
    static const char* const before_lower[] = {"etc", "a.c", "d.c", "j.-c", "j-c", "chr", "usw", "ecc", "ee.uu", "u.s",
                                               "т.д", "т.п", "н.э", "э", 0};
    for (const char* const* p = always; *p; p++)
        if (!strcmp(w, *p)) return 1;
    for (const char* const* p = before_lower; *p; p++)
        if (!strcmp(w, *p)) return next_lower;
    if (digits == i - a && digits) {   // a number and its period
        if (next_lower) return 1;
        // a numbered list's "2. " at the start of a line: not a sentence of its own (it would be joined to the
        // sentence before, and the item's words would come in the next piece)
        if (a == from || text[a - 1] == '\n') return 1;
        static const char* const months[] = {"Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
                                             "September", "Oktober", "November", "Dezember", 0};
        for (const char* const* p = months; *p; p++) {
            int l = (int)strlen(*p);
            if (q + l <= len && !strncmp(text + q, *p, l) && !(q + l < len && text[q + l] >= 'a' && text[q + l] <= 'z')) return 1;
        }
        int b = a - 1;   // "der 2. Weltkrieg": a German article before the number
        while (b > from && text[b - 1] == ' ') b--;
        int b0 = b;
        while (b0 > from && text[b0 - 1] != ' ') b0--;
        static const char* const art[] = {"der", "die", "das", "den", "dem", "des", "am", "im", "zum", "vom", "beim", "zur", 0};
        char aw[8];
        int al = b - b0;
        if (al > 0 && al < 8) {
            for (int p = 0; p < al; p++) aw[p] = (char)(text[b0 + p] >= 'A' && text[b0 + p] <= 'Z' ? text[b0 + p] + 32 : text[b0 + p]);
            aw[al] = 0;
            for (const char* const* p = art; *p; p++)
                if (!strcmp(aw, *p)) return 1;
        }
    }
    return 0;
}

int st_next_piece(const char* text, int len, int from, int final) {
    int in_code = 0, extra = 0;   // ``` fences: the code itself is not read aloud, but it still ends a piece
    for (int i = from; i < len; i++) {
        char c = text[i];
        if (c == '`' && i + 2 < len && text[i + 1] == '`' && text[i + 2] == '`') {
            in_code = !in_code;
            if (in_code && i > from) return i;   // speak what came before the code first
            i += 2;
            continue;
        }
        if (in_code) {
            if (c == '\n' && !final && i + 1 >= len) return -1;
            continue;
        }
        if (c == '\n') return i + 1;
        if (c >= '0' && c <= '9') extra += DIGIT_EXTRA;
        int n = i - from + extra;
        if (c == '.' || c == '!' || c == '?' || ((uint8_t)c == 0xE2 && i + 2 < len && (uint8_t)text[i + 1] == 0x80 &&
                                                 (uint8_t)text[i + 2] == 0xA6)) {
            int j = i + ((uint8_t)c == 0xE2 ? 3 : 1);
            while (j < len && strchr(".!?\"')]*", text[j])) j++;
            if (j < len && (text[j] == ' ' || text[j] == '\n')) {
                if (c == '.' && j == i + 1 && text[j] == ' ' && abbrev_period(text, len, from, i)) continue;
                return j + 1;
            }
            if (j >= len) {
                if (final) return len;
                return -1;   // the next token decides
            }
        }
        if (n >= SOFT_MAX && (c == ',' || c == ';' || c == ':') && i + 1 < len && text[i + 1] == ' ') return i + 2;
        if (n >= HARD_MAX && c == ' ') return i + 1;
    }
    return final && len > from ? len : -1;
}

// ---- text to read aloud ----
// Drops markdown marks (* _ # ` > | and list bullets), code blocks, links' URLs, emoji and other symbols; keeps
// letters, digits and punctuation. Returns the length of out (0 = nothing to say).
int st_clean(const char* in, int n, char* out, int max) {
    int o = 0, in_code = 0, line_start = 1, words = 0;
    for (int i = 0; i < n && o < max - 8;) {
        if (in[i] == '`' && i + 2 < n && in[i + 1] == '`' && in[i + 2] == '`') {
            in_code = !in_code;
            i += 3;
            continue;
        }
        if (in_code) {
            i++;
            continue;
        }
        if (line_start) {   // list bullets, headings, quotes, numbered lists "1. "
            while (i < n && (in[i] == ' ' || in[i] == '\t')) i++;
            if (i + 1 < n && (in[i] == '-' || in[i] == '*' || in[i] == '+') && in[i + 1] == ' ') i += 2;
            while (i < n && (in[i] == '#' || in[i] == '>')) i++;
            line_start = 0;
            continue;
        }
        if (!strncmp(in + i, "http://", 7) || !strncmp(in + i, "https://", 8)) {   // URLs: not read
            while (i < n && in[i] != ' ' && in[i] != '\n' && in[i] != ')') i++;
            continue;
        }
        int l;
        uint32_t c = cp_at(in + i, &l);
        if (c == '\n') {
            line_start = 1;
            // a line without final punctuation (a title, a list item): a pause
            int k = o;
            while (k > 0 && out[k - 1] == ' ') k--;
            if (k > 0 && !strchr(".!?,;:", out[k - 1])) out[k++] = ',';
            o = k;
            if (o && out[o - 1] != ' ') out[o++] = ' ';
            i += l;
            continue;
        }
        int keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0xC0 ||
                   strchr(" .,!?;:'\"()-%&+=/$", (int)c) != NULL || c == 0xA1 || c == 0xBF || c == 0xAB || c == 0xBB ||
                   c == 0xB0 || c == 0xBA || c == 0xAA || c == 0xA3;   // ° º ª £ (st_speak reads them)
        if (c >= 0x2000 && c < 0x2070) keep = c == 0x2026 || c == 0x2014 || c == 0x2013 || (c >= 0x2018 && c <= 0x201D);
        if (c >= 0x2190) keep = 0;   // arrows, symbols, emoji, ...
        if (c == '\t') c = ' ', keep = 1, l = 1;
        if (c == '*' || c == '_' || c == '#' || c == '`' || c == '|' || c == '~' || c == '<' || c == '>') keep = 0;
        if (keep) {
            if (c == ' ' && (o == 0 || out[o - 1] == ' ')) {
                i += l;
                continue;
            }
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0xC0) words++;
            memcpy(out + o, in + i, l);
            o += l;
        }
        i += l;
    }
    while (o > 0 && (out[o - 1] == ' ' || out[o - 1] == ',')) o--;
    out[o] = 0;
    return words ? o : 0;
}

// ---- language of a piece ----
// Naive Bayes over hashed character 1-4-grams of each word (chat/lang_id_table.h, trained on Wikipedia)
// for es, pt, en, fr, de, it; Russian and Greek by their script. The fallback (the
// reply's language, or the user's for the reply) is kept unless another language explains the text better by more
// than LID_MARGIN nats, or by more than LID_SHORT nats and LID_RATIO nats per n-gram (short but clear: "Oi, tudo
// bem?", "Bonjour"). The margins let a whole sentence in another language change the language, while short
// fragments in the reply's own language ("O sea, depende.", "No sé.", "Ejecuta el script: python snakegame.py")
// keep it. Code (``` and `) is skipped; ¿ and ¡ count for Spanish.
#include "lang_id_table.h"
#ifndef LID_MARGIN
#define LID_MARGIN 45.0f
#endif
#define LID_SHORT 20.0f
#define LID_RATIO 0.6f
#define LID_SPANISH_MARK 15.0f

static uint32_t lower_cp(uint32_t c) {   // as Python's str.lower() for the letters that matter here
    if ((c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7)) return c + 32;
    if (c >= 0x100 && c < 0x178 && !(c & 1) && c != 0x130 && c != 0x138) return c + 1;
    if (c >= 0x410 && c <= 0x42F) return c + 32;
    if (c >= 0x400 && c <= 0x40F) return c + 80;
    if (c >= 0x391 && c <= 0x3A9) return c + 32;
    return c;
}
static int letter_cp(uint32_t c) {   // Python's [^\W\d_] for Latin, Greek and Cyrillic
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == 0xAA || c == 0xB5 || c == 0xBA ||
           (c >= 0xC0 && c <= 0x24F && c != 0xD7 && c != 0xF7) || (c >= 0x370 && c < 0x530 && c != 0x37E && c != 0x387);
}
static int put_utf8(uint32_t c, char* o) {
    if (c < 0x80) return o[0] = (char)c, 1;
    if (c < 0x800) return o[0] = (char)(0xC0 | c >> 6), o[1] = (char)(0x80 | (c & 63)), 2;
    return o[0] = (char)(0xE0 | c >> 12), o[1] = (char)(0x80 | (c >> 6 & 63)), o[2] = (char)(0x80 | (c & 63)), 3;
}

// adds the n-gram costs of one word (" w ", lower case, UTF-8 in b with character starts at st[0..nc]) to cost
static int lid_word(const char* b, const int* st, int nc, uint32_t* cost) {   // -> n-grams counted
    int count = 0;
    for (int n = 1; n <= 4; n++)
        for (int i = 0; i + n <= nc; i++) {
            int a = st[i], e = st[i + n];
            if (n == 1 && b[a] == ' ') continue;
            uint32_t h = 2166136261u;
            for (int k = a; k < e; k++) h = (h ^ (uint8_t)b[k]) * 16777619u;
            const uint8_t* row = lid_table + (h & ((1u << LID_BITS) - 1)) * LID_NLANG;
            for (int l = 0; l < LID_NLANG; l++) cost[l] += row[l];
            count++;
        }
    return count;
}

const char* st_lang(const char* s, const char* fallback) {
    int cyr = 0, greek = 0, latin = 0, words = 0, in_code = 0, in_tick = 0, grams = 0, marks = 0;
    uint32_t cost[LID_NLANG] = {0};   // in 1/8 nats
    char b[200];
    int st[64], nc = 0, bl = 1;
    b[0] = ' ', st[0] = 0;
    for (int i = 0;; ) {
        if (s[i] == '`') {
            if (s[i + 1] == '`' && s[i + 2] == '`') in_code = !in_code, i += 3;
            else in_tick = !in_tick, i++;
            continue;
        }
        int l = 1;
        uint32_t c = s[i] ? cp_at(s + i, &l) : 0;
        int skip = in_code || in_tick;
        if (!skip && (c == 0xBF || c == 0xA1)) marks++;   // ¿ ¡
        int is_l = !skip && letter_cp(c);
        int apos = !skip && c == '\'' && nc > 0 && letter_cp(s[i + 1] ? (uint8_t)s[i + 1] : 0);
        if ((is_l || apos) && nc < 60 && bl < (int)sizeof(b) - 8) {
            if (c >= 0x400 && c < 0x530) cyr++;
            else if (c >= 0x370 && c < 0x400) greek++;
            else if (is_l) latin++;
            st[1 + nc++] = bl;   // (st[0]: the leading space)
            bl += put_utf8(lower_cp(c), b + bl);
        } else if (nc) {   // the word ends: " word " -> its n-grams
            st[nc + 1] = bl;
            b[bl++] = ' ';
            st[nc + 2] = bl;
            grams += lid_word(b, st, nc + 2, cost);
            words++;
            nc = 0, bl = 1;
        }
        if (!s[i]) break;
        i += l;
    }
    if (cyr > latin) return "ru";
    if (greek > latin) return "el";
    if (!words) return fallback;
    float c[LID_NLANG];
    int best = 0, fb = -1;
    for (int k = 0; k < LID_NLANG; k++) {
        c[k] = (float)cost[k] / 8.0f - (k == 0 ? LID_SPANISH_MARK * (float)marks : 0.0f);   // (lid_langs[0] = "es")
        if (c[k] < c[best]) best = k;
        if (!strcmp(lid_langs[k], fallback)) fb = k;
    }
    if (fb < 0 || best == fb) return fb < 0 ? lid_langs[best] : fallback;
    float d = c[fb] - c[best];
    if (d > LID_MARGIN || (d > LID_SHORT && d > LID_RATIO * (float)grams)) return lid_langs[best];
    return fallback;
}

const char* st_text_lang(const char* text, int n, const char* fallback) {
    static char w[1024];
    int l = st_clean(text, n, w, sizeof(w));
    return l ? st_lang(w, fallback) : fallback;
}

int st_spoken_len(const char* text, int n) {
    int l = n;
    for (int i = 0; i < n; i++)
        if (text[i] >= '0' && text[i] <= '9') l += DIGIT_EXTRA;
    return l;
}

// ---- where a long stretch can be cut between words ----
// (a cut at any space, as HARD_MAX's, can fall in the middle of a phrase: "fue uno de los, | primeros pensadores")
static const char st_no_end[] =   // words that cannot end a piece: articles, prepositions, determiners, auxiliaries
    " el la los las un una unos unas lo al del de en a con por para sin sobre entre hacia desde hasta su sus mi mis"
    " tu tus nuestro nuestra este esta estos estas ese esa que y o e ni pero muy más es son fue era ha han"
    " os as um uma uns umas do da dos das no na nos nas ao aos pelo pela em com seu sua meu minha"
    " the an of to in on at by for with from and or but is are was were be been my your his her its our their this"
    " that le les une des du d l au aux dans par pour sur avec sans et ou mais son sa ses mon ma mes ce cette est"
    " der die das den dem ein eine einen einem einer und oder aber im an am auf mit von zu zum zur für ist sind"
    " il gli i di della dei degli delle alla ai nel nella su sul ma è"
    " и в во на с со к ко по о об от до из за для у не что как это его её их мой твой а но или ";
static const char st_openers[] =   // words that open a new phrase: a cut can come before them
    " que y o pero porque cuando donde como mientras aunque si pues para con sin en"
    " e ou mas quando onde enquanto embora se com sem em"
    " that which who and or but because when where while although if so to with without in"
    " qui et parce quand où comme pendant bien pour avec sans dans"
    " dass und oder aber weil wenn wo wie während obwohl ob um mit ohne"
    " che ma perché dove mentre anche per senza"
    " что и или но потому когда где как пока хотя если чтобы с без в ";

static int word_char(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80 || c == '-' ||
           c == '\'';
}

// lower-case copy of text[a, e) as " word " in out (UTF-8); 0 if it does not fit
static int word_key(const char* text, int a, int e, char* out, int max) {
    int o = 0;
    out[o++] = ' ';
    for (int i = a; i < e;) {
        int l;
        uint32_t c = cp_at(text + i, &l);
        if (o + 4 >= max - 2) return 0;
        o += put_utf8(lower_cp(c), out + o);
        i += l;
    }
    out[o++] = ' ', out[o] = 0;
    return o;
}

// mode 0: the last opener in [lo, hi), else the last allowed space; mode 1: the first opener in [lo, hi), else 0
static int natural_cut(const char* text, int len, int from, int lo, int hi, int mode) {
    int best = 0, fallback = 0, sl = 0;
    for (int i = from; i < len && sl < hi; i++) {
        uint8_t c = (uint8_t)text[i];
        sl += (c >= '0' && c <= '9') ? 1 + DIGIT_EXTRA : 1;
        if (c == '\n' || c == '`') break;   // (a line ends a piece anyway; never into code)
        if (c != ' ' || i == from || sl < lo || text[i - 1] == ' ') continue;
        int a = i;
        while (a > from && word_char((uint8_t)text[a - 1])) a--;
        int e = i + 1;
        while (e < len && word_char((uint8_t)text[e])) e++;
        if (a == i || e == i + 1 || e >= len) continue;   // (the next word must be complete)
        if (text[e] == '.' && e + 1 < len && word_char((uint8_t)text[e + 1])) continue;   // not before "a.C.", "J.-C."
        {   // nor before a number ("siglo | III", "el | 753"): it goes with the word before
            int num = 1;
            for (int j = i + 1; j < e && num; j++) num = strchr("0123456789IVXLCDM", text[j]) != NULL;
            if (num) continue;
        }
        char pk[64], nk[64];
        if (!strchr(".,;:!?)\"'", text[i - 1])) {   // (after punctuation the sentence code cuts already)
            if (!word_key(text, a, i, pk, sizeof(pk)) || strstr(st_no_end, pk)) continue;
        }
        fallback = i + 1;
        if (word_key(text, i + 1, e, nk, sizeof(nk)) && strstr(st_openers, nk)) {
            best = i + 1;
            if (mode == 1) return best;
        }
    }
    return mode == 1 ? 0 : best ? best : fallback;
}

int st_natural_cut(const char* text, int len, int from, int lo, int hi) { return natural_cut(text, len, from, lo, hi, 0); }
int st_opener_cut(const char* text, int len, int from, int lo, int hi) { return natural_cut(text, len, from, lo, hi, 1); }

int st_speech_tokens(const char* w, int n, const char* lang) {
    static const char* const l[] = {"es", "ru", "it", "pt", "fr", "de", "en"};
    static const float k[] = {1.77f, 1.46f, 1.55f, 1.80f, 1.15f, 1.30f, 1.25f};
    float r = 1.6f;
    for (int i = 0; i < 7; i++)
        if (!strcmp(lang, l[i])) r = k[i];
    int chars = 0;
    for (int i = 0; i < n; i++)
        if (!is_cont((uint8_t)w[i])) chars += (w[i] >= '0' && w[i] <= '9') ? 1 + DIGIT_EXTRA : 1;
    return 20 + (int)(r * (float)chars);
}
