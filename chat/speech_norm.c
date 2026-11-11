// Numbers, abbreviations and acronyms as words, in the piece's language, before Chatterbox reads it (st_speak in
// speech_text.h). T3, the text-to-speech model, reads digits and abbreviations ("10:30", "a.C.", "v. Chr.") badly,
// and all-caps acronyms as if they were words ("LLM", "CPU", "PS4"), so they are written out as words first. Only
// exact patterns change: a digit, an abbreviation with its periods, an all-caps token. Everything else is copied as
// it is.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "speech_text.h"

enum { ES, EN, PT, FR, DE, IT, RU, NL_ };
enum { G_ALONE, G_M, G_F };   // a number said alone ("uno"), before a masculine noun ("un"), before a feminine one

typedef struct {
    char* p;
    int n, max;
} Buf;

static void put(Buf* b, const char* s) {
    while (*s && b->n < b->max - 1) b->p[b->n++] = *s++;
    b->p[b->n] = 0;
}
static void putn(Buf* b, const char* s, int n) {
    for (int i = 0; i < n && s[i] && b->n < b->max - 1; i++) b->p[b->n++] = s[i];
    b->p[b->n] = 0;
}
static int is_letter(uint32_t c);
static uint32_t cp(const char* s, int* len);
static void sp(Buf* b) {   // a space before a new word, unless at the start or after an opening mark (¿ ¡ ( « ")
    if (!b->n || b->p[b->n - 1] == ' ') return;
    int k = b->n - 1;
    while (k > 0 && ((uint8_t)b->p[k] & 0xC0) == 0x80) k--;
    int l;
    uint32_t c = cp(b->p + k, &l);
    if (is_letter(c) || (c >= '0' && c <= '9') || strchr(".,;:!?%)'\"", (int)(c < 0x80 ? c : 1)) || c == 0xBB)
        put(b, " ");
}
static void word(Buf* b, const char* w) {
    sp(b);
    put(b, w);
}

// ---------------------------------------------------------------------------------------------------- UTF-8
static uint32_t cp(const char* s, int* len) {
    const uint8_t* p = (const uint8_t*)s;
    uint32_t c = p[0];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (n > 1) c &= 0x3F >> (n - 1);
    for (int i = 1; i < n; i++) {
        if ((p[i] & 0xC0) != 0x80) {
            n = i;
            break;
        }
        c = c << 6 | (p[i] & 0x3F);
    }
    *len = n;
    return c;
}
static int is_upper(uint32_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7) || (c >= 0x410 && c <= 0x42F) || c == 0x401;
}
static int is_lower(uint32_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 0xDF && c <= 0xFF && c != 0xF7) || (c >= 0x430 && c <= 0x44F) || c == 0x451;
}
static int is_letter(uint32_t c) { return is_upper(c) || is_lower(c) || (c >= 0x100 && c < 0x250); }
static int is_digit(char c) { return c >= '0' && c <= '9'; }
static int alnum_at(const char* s, int i, int n) {
    if (i < 0 || i >= n) return 0;
    if (is_digit(s[i])) return 1;
    while (i > 0 && ((uint8_t)s[i] & 0xC0) == 0x80) i--;   // (start of the character)
    int l;
    return is_letter(cp(s + i, &l));
}
static int prev_alnum(const char* s, int i) {   // is the character before byte i a letter or digit?
    if (i <= 0) return 0;
    int k = i - 1;
    while (k > 0 && ((uint8_t)s[k] & 0xC0) == 0x80) k--;
    int l;
    uint32_t c = cp(s + k, &l);
    return is_digit((char)c) || is_letter(c);
}

// the next word after byte i (skipping spaces), lower-cased ASCII / as-is otherwise, into w; returns its start or -1
static int lower_word(const char* s, int n, int i, char* w, int wmax) {   // letters from s[i], lower case; -> end
    int k = 0, j = i;
    while (j < n && k < wmax - 4) {
        int l;
        uint32_t c = cp(s + j, &l);
        if (!is_letter(c)) break;
        if (c >= 'A' && c <= 'Z') w[k++] = (char)(c + 32);
        else if (c >= 0x410 && c <= 0x42F) {   // Cyrillic capitals
            uint32_t d = c + 32;
            w[k++] = (char)(0xC0 | d >> 6), w[k++] = (char)(0x80 | (d & 63));
        } else memcpy(w + k, s + j, l), k += l;
        j += l;
    }
    w[k] = 0;
    return j;
}
static int next_word(const char* s, int n, int i, char* w, int wmax) {
    while (i < n && s[i] == ' ') i++;
    int j = lower_word(s, n, i, w, wmax);
    if (j == i) return w[0] = 0, -1;
    int l;
    cp(s + i, &l);
    if (j == i + l && j < n && s[j] == '.') strcat(w, ".");   // a lone letter keeps its period ("г.")
    return i;
}
static int ends(const char* w, const char* suf) {
    size_t a = strlen(w), b = strlen(suf);
    return a >= b && !strcmp(w + a - b, suf);
}
static int in_list(const char* w, const char* const* list) {
    for (; *list; list++)
        if (!strcmp(w, *list)) return 1;
    return 0;
}

// ---------------------------------------------------------------------------------------------------- gender
// of the noun after a number (only where the number's form changes: un / una, dos / duas, одна / две ...)
static int noun_gender(int L, const char* w, uint64_t n) {
    if (!*w) return G_ALONE;
    static const char* const not_nouns[] = {"y", "o", "e", "de", "del", "a", "al", "en", "por", "para", "con", "u", "ou",
                                            "et", "und", "and", "or", "of", "to", "in", "on", "da", "do", "di", "du", "des",
                                            "и", "в", "на", "с", "и", "es", "is", "son", "sont", "ist", "sind", "è", "é", 0};
    if (in_list(w, not_nouns)) return G_ALONE;
    switch (L) {
    case ES: {
        static const char* const m_a[] = {"día", "días", "mapa", "mapas", "problema", "problemas", "programa", "programas",
                                          "sistema", "sistemas", "tema", "temas", "idioma", "idiomas", "clima", "planeta",
                                          "planetas", "sofá", "papá", "poema", "poemas", "dilema", 0};
        static const char* const f[] = {"vez", "veces", "mujer", "mujeres", "noche", "noches", "gente", "parte", "partes",
                                        "frase", "frases", "clase", "clases", "calle", "calles", "llave", "nube", "imagen",
                                        "imágenes", "red", "redes", "luz", "luces", "flor", "flores", "mano", "manos",
                                        "foto", "fotos", "moto", "motos", "radio", "leche", "fase", "fases", "base", "bases", 0};
        if (in_list(w, m_a)) return G_M;
        if (in_list(w, f) || ends(w, "a") || ends(w, "as") || ends(w, "ción") || ends(w, "ciones") || ends(w, "sión") ||
            ends(w, "siones") || ends(w, "dad") || ends(w, "dades") || ends(w, "tud"))
            return G_F;
        return G_M;
    }
    case PT: {
        static const char* const m_a[] = {"dia", "dias", "mapa", "mapas", "problema", "problemas", "programa", "programas",
                                          "sistema", "sistemas", "tema", "temas", "idioma", "idiomas", "clima", "planeta", "sofá", 0};
        static const char* const f[] = {"vez", "vezes", "mulher", "mulheres", "noite", "noites", "gente", "parte", "partes",
                                        "frase", "frases", "mão", "mãos", "foto", "fotos", "imagem", "imagens", "rede",
                                        "redes", "luz", "flor", "cor", "cores", "fase", "fases", "base", 0};
        if (in_list(w, m_a)) return G_M;
        if (in_list(w, f) || ends(w, "a") || ends(w, "as") || ends(w, "ção") || ends(w, "ções") || ends(w, "dade") ||
            ends(w, "dades") || ends(w, "agem") || ends(w, "agens"))
            return G_F;
        return G_M;
    }
    case IT: {
        static const char* const m_a[] = {"problema", "problemi", "sistema", "programma", "tema", "clima", "pianeta", 0};
        if (in_list(w, m_a)) return G_M;
        return ends(w, "a") || ends(w, "zione") || ends(w, "tà") ? G_F : G_M;
    }
    case FR: {
        static const char* const f[] = {"heure", "heures", "minute", "minutes", "seconde", "secondes", "fois", "semaine",
                                        "semaines", "année", "années", "journée", "personne", "personnes", "voiture",
                                        "ville", "maison", "page", "pages", "chose", "choses", "question", "étape",
                                        "étapes", "partie", "fille", "femme", "nuit", "photo", 0};
        return in_list(w, f) || ends(w, "tion") || ends(w, "sion") || ends(w, "té") ? G_F : G_M;
    }
    case DE:   // the noun is capitalised in the text; w is lower case here
        return ends(w, "ung") || ends(w, "heit") || ends(w, "keit") || ends(w, "schaft") || ends(w, "ion") ||
                       ends(w, "tät") || ends(w, "e")
                   ? G_F
                   : G_M;
    case RU: {
        uint64_t u = n % 10, t = n % 100;
        if (u == 1 && t != 11) return ends(w, "а") || ends(w, "я") ? G_F : G_M;   // одна минута
        if (u == 2 && t != 12) return ends(w, "ы") || ends(w, "и") ? G_F : G_M;   // две минуты
        return G_M;
    }
    }
    return G_M;
}

// ---------------------------------------------------------------------------------------------------- cardinals
static void es_lt1000(Buf* b, int n, int g) {
    static const char* u[30] = {"cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
                                "diez", "once", "doce", "trece", "catorce", "quince", "dieciséis", "diecisiete",
                                "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidós", "veintitrés",
                                "veinticuatro", "veinticinco", "veintiséis", "veintisiete", "veintiocho", "veintinueve"};
    static const char* t[10] = {"", "", "", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa"};
    static const char* h[10] = {"", "ciento", "doscient", "trescient", "cuatrocient", "quinient", "seiscient",
                                "setecient", "ochocient", "novecient"};
    if (n == 100) return word(b, "cien");
    if (n >= 100) {
        word(b, h[n / 100]);
        if (n >= 200) put(b, g == G_F ? "as" : "os");
        n %= 100;
        if (!n) return;
    }
    if (n < 30) {
        if (n == 1) return word(b, g == G_F ? "una" : g == G_M ? "un" : "uno");
        if (n == 21) return word(b, g == G_F ? "veintiuna" : g == G_M ? "veintiún" : "veintiuno");
        return word(b, u[n]);
    }
    word(b, t[n / 10]);
    if (n % 10) {
        word(b, "y");
        es_lt1000(b, n % 10, g);
    }
}
static void es_card(Buf* b, uint64_t n, int g) {
    if (!n) return word(b, "cero");
    uint64_t bil = n / 1000000000000ull, mil = n / 1000000 % 1000000, th = n / 1000 % 1000, r = n % 1000;
    if (bil) {
        if (bil == 1) word(b, "un billón");
        else es_card(b, bil, G_M), word(b, "billones");
    }
    if (mil) {
        if (mil == 1) word(b, "un millón");
        else es_card(b, mil, G_M), word(b, "millones");
    }
    if (th) {
        if (th > 1) es_lt1000(b, (int)th, g == G_F ? G_F : G_M);
        word(b, "mil");
    }
    if (r) es_lt1000(b, (int)r, g);
}

static void pt_lt1000(Buf* b, int n, int g) {
    static const char* u[20] = {"zero", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez",
                                "onze", "doze", "treze", "catorze", "quinze", "dezesseis", "dezessete", "dezoito", "dezenove"};
    static const char* t[10] = {"", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa"};
    static const char* h[10] = {"", "cento", "duzent", "trezent", "quatrocent", "quinhent", "seiscent", "setecent",
                                "oitocent", "novecent"};
    if (n == 100) return word(b, "cem");
    if (n >= 100) {
        word(b, h[n / 100]);
        if (n >= 200) put(b, g == G_F ? "as" : "os");
        n %= 100;
        if (!n) return;
        word(b, "e");
    }
    if (n >= 20) {
        word(b, t[n / 10]);
        n %= 10;
        if (!n) return;
        word(b, "e");
    }
    if (n == 1) return word(b, g == G_F ? "uma" : "um");
    if (n == 2) return word(b, g == G_F ? "duas" : "dois");
    word(b, u[n]);
}
static void pt_card(Buf* b, uint64_t n, int g) {
    if (!n) return word(b, "zero");
    uint64_t grp[4] = {n / 1000000000ull % 1000, n / 1000000 % 1000, n / 1000 % 1000, n % 1000};
    static const char* one[4] = {"um bilhão", "um milhão", "mil", ""};
    static const char* many[4] = {"bilhões", "milhões", "mil", ""};
    int any = 0;
    for (int k = 0; k < 4; k++) {
        if (!grp[k]) continue;
        if (any) {   // "mil e quinhentos", "dois mil e vinte e seis", "mil novecentos e oitenta e cinco"
            uint64_t r = 0;
            for (int j = k; j < 4; j++) r = r * 1000 + grp[j];
            while (r && r % 1000 == 0) r /= 1000;
            if (r < 100 || (r < 1000 && r % 100 == 0)) word(b, "e");
        }
        any = 1;
        if (k == 3) pt_lt1000(b, (int)grp[k], g);
        else if (grp[k] == 1) word(b, one[k]);
        else pt_lt1000(b, (int)grp[k], k == 2 ? g : G_M), word(b, many[k]);
    }
}

static void it_lt1000(Buf* b, int n) {   // one word, n > 0
    static const char* u[20] = {"zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci",
                                "undici", "dodici", "tredici", "quattordici", "quindici", "sedici", "diciassette",
                                "diciotto", "diciannove"};
    static const char* t[10] = {"", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"};
    if (n >= 100) {
        if (n / 100 > 1) put(b, u[n / 100]);
        n %= 100;
        put(b, n / 10 == 8 || n == 8 ? "cent" : "cento");   // centotto, centottanta
    }
    if (n >= 20) {
        const char* w = t[n / 10];
        int k = n % 10;
        if (k == 1 || k == 8) putn(b, w, (int)strlen(w) - 1);   // ventuno, ventotto
        else put(b, w);
        if (k == 3) return put(b, "tré");
        n = k;
    }
    if (n) put(b, u[n]);
}
static void it_card(Buf* b, uint64_t n, int g) {
    if (!n) return word(b, "zero");
    if (n == 1) return word(b, g == G_F ? "una" : g == G_M ? "un" : "uno");
    uint64_t bil = n / 1000000000ull % 1000, mil = n / 1000000 % 1000, th = n / 1000 % 1000, r = n % 1000;
    if (bil) {
        sp(b);
        if (bil == 1) put(b, "un miliardo");
        else it_lt1000(b, (int)bil), put(b, " miliardi");
    }
    if (mil) {
        sp(b);
        if (mil == 1) put(b, "un milione");
        else it_lt1000(b, (int)mil), put(b, " milioni");
    }
    if (th || r) sp(b);
    if (th == 1) put(b, "mille");
    else if (th) it_lt1000(b, (int)th), put(b, "mila");
    if (r) it_lt1000(b, (int)r);
}

static void de_lt100(Buf* b, int n, int alone) {
    static const char* u[20] = {"null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
                                "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"};
    static const char* t[10] = {"", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"};
    if (n < 20) return put(b, n == 1 && !alone ? "ein" : u[n]);
    if (n % 10) put(b, n % 10 == 1 ? "ein" : u[n % 10]), put(b, "und");
    put(b, t[n / 10]);
}
static void de_lt1000(Buf* b, int n, int alone) {
    if (n >= 100) {
        if (n / 100 > 1) de_lt100(b, n / 100, 0);
        put(b, "hundert");
        n %= 100;
        if (!n) return;
    }
    de_lt100(b, n, alone);
}
static void de_card(Buf* b, uint64_t n, int g) {
    if (!n) return word(b, "null");
    if (n == 1) return word(b, g == G_F ? "eine" : g == G_M ? "ein" : "eins");
    uint64_t bil = n / 1000000000ull % 1000, mil = n / 1000000 % 1000, th = n / 1000 % 1000, r = n % 1000;
    if (bil) {
        sp(b);
        if (bil == 1) put(b, "eine Milliarde");
        else de_lt1000(b, (int)bil, 0), put(b, " Milliarden");
    }
    if (mil) {
        sp(b);
        if (mil == 1) put(b, "eine Million");
        else de_lt1000(b, (int)mil, 0), put(b, " Millionen");
    }
    if (th || r) sp(b);
    if (th) {
        if (th > 1) de_lt1000(b, (int)th, 0);
        put(b, "tausend");
    }
    if (r) de_lt1000(b, (int)r, g == G_ALONE);
}

// French writes the compounds below 100 with hyphens (quatre-vingt-cinq): written with spaces, T3 misreads them
static const char* fr_u[17] = {"zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
                               "onze", "douze", "treize", "quatorze", "quinze", "seize"};
static void fr_tail(Buf* b, int k) {   // 1-19 after a hyphen
    if (k < 17) return put(b, fr_u[k]);
    put(b, "dix-"), put(b, fr_u[k - 10]);
}
static void fr_lt100(Buf* b, int n, int g) {
    static const char* t[7] = {"", "", "vingt", "trente", "quarante", "cinquante", "soixante"};
    if (n == 1) return word(b, g == G_F ? "une" : "un");
    if (n < 17) return word(b, fr_u[n]);
    if (n < 20) return word(b, "dix-"), put(b, fr_u[n - 10]);
    if (n < 70) {
        word(b, t[n / 10]);
        if (n % 10 == 1) put(b, " et "), put(b, g == G_F ? "une" : "un");
        else if (n % 10) put(b, "-"), put(b, fr_u[n % 10]);
        return;
    }
    if (n < 80) {
        word(b, "soixante");
        if (n == 71) return put(b, " et onze");
        put(b, "-");
        return fr_tail(b, n - 60);
    }
    word(b, "quatre-vingt");
    if (n == 80) return put(b, "s");
    put(b, "-");
    fr_tail(b, n - 80);
}
static void fr_lt1000(Buf* b, int n, int g, int last) {
    if (n >= 100) {
        if (n / 100 > 1) fr_lt100(b, n / 100, G_M), word(b, n % 100 == 0 && last ? "cents" : "cent");
        else word(b, "cent");
        n %= 100;
        if (!n) return;
    }
    fr_lt100(b, n, g);
}
static void fr_card(Buf* b, uint64_t n, int g) {
    if (!n) return word(b, "zéro");
    uint64_t bil = n / 1000000000ull % 1000, mil = n / 1000000 % 1000, th = n / 1000 % 1000, r = n % 1000;
    if (bil) {
        if (bil == 1) word(b, "un milliard");
        else fr_lt1000(b, (int)bil, G_M, 0), word(b, "milliards");
    }
    if (mil) {
        if (mil == 1) word(b, "un million");
        else fr_lt1000(b, (int)mil, G_M, 0), word(b, "millions");
    }
    if (th) {
        if (th > 1) fr_lt1000(b, (int)th, G_M, 0);
        word(b, "mille");
    }
    if (r) fr_lt1000(b, (int)r, g, 1);
}

static const char* en_u[20] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                               "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
static void en_lt100(Buf* b, int n) {
    static const char* t[10] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
    if (n < 20) return word(b, en_u[n]);
    word(b, t[n / 10]);
    if (n % 10) put(b, "-"), put(b, en_u[n % 10]);
}
static void en_lt1000(Buf* b, int n) {
    if (n >= 100) {
        word(b, en_u[n / 100]), word(b, "hundred");
        n %= 100;
        if (!n) return;
    }
    en_lt100(b, n);
}
static void en_card(Buf* b, uint64_t n) {
    if (!n) return word(b, "zero");
    uint64_t grp[4] = {n / 1000000000ull % 1000, n / 1000000 % 1000, n / 1000 % 1000, n % 1000};
    static const char* name[4] = {"billion", "million", "thousand", ""};
    for (int k = 0; k < 4; k++)
        if (grp[k]) {
            en_lt1000(b, (int)grp[k]);
            if (k < 3) word(b, name[k]);
        }
}
static void en_year(Buf* b, int n) {   // 1985 nineteen eighty-five, 1905 nineteen oh five, 1900 nineteen hundred
    int hi = n / 100, lo = n % 100;
    en_lt100(b, hi);
    if (!lo) return word(b, "hundred");
    if (lo < 10) word(b, "oh");
    en_lt100(b, lo);
}

static const char* ru_u[20] = {"ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
                               "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
                               "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"};
static const char* ru_plural(uint64_t n, const char* one, const char* few, const char* many) {
    uint64_t u = n % 10, t = n % 100;
    if (u == 1 && t != 11) return one;
    if (u >= 2 && u <= 4 && (t < 12 || t > 14)) return few;
    return many;
}
static void ru_lt1000(Buf* b, int n, int g) {
    static const char* h[10] = {"", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"};
    static const char* t[10] = {"", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"};
    if (n >= 100) word(b, h[n / 100]), n %= 100;
    if (n >= 20) word(b, t[n / 10]), n %= 10;
    if (!n) return;
    if (n == 1 && g == G_F) return word(b, "одна");
    if (n == 2 && g == G_F) return word(b, "две");
    word(b, ru_u[n]);
}
static void ru_card(Buf* b, uint64_t n, int g) {
    if (!n) return word(b, "ноль");
    uint64_t bil = n / 1000000000ull % 1000, mil = n / 1000000 % 1000, th = n / 1000 % 1000, r = n % 1000;
    if (bil) ru_lt1000(b, (int)bil, G_M), word(b, ru_plural(bil, "миллиард", "миллиарда", "миллиардов"));
    if (mil) ru_lt1000(b, (int)mil, G_M), word(b, ru_plural(mil, "миллион", "миллиона", "миллионов"));
    if (th) {
        if (th > 1) ru_lt1000(b, (int)th, G_F);
        word(b, ru_plural(th, "тысяча", "тысячи", "тысяч"));
    }
    if (r) ru_lt1000(b, (int)r, g);
}

// Russian genitive (около двадцати пяти тысяч), n < 1 000 000
static void ru_lt1000_gen(Buf* b, int n, int g) {
    static const char* u[20] = {"ноля", "одного", "двух", "трёх", "четырёх", "пяти", "шести", "семи", "восьми", "девяти",
                                "десяти", "одиннадцати", "двенадцати", "тринадцати", "четырнадцати", "пятнадцати",
                                "шестнадцати", "семнадцати", "восемнадцати", "девятнадцати"};
    static const char* t[10] = {"", "", "двадцати", "тридцати", "сорока", "пятидесяти", "шестидесяти", "семидесяти",
                                "восьмидесяти", "девяноста"};
    static const char* h[10] = {"", "ста", "двухсот", "трёхсот", "четырёхсот", "пятисот", "шестисот", "семисот",
                                "восьмисот", "девятисот"};
    if (n >= 100) word(b, h[n / 100]), n %= 100;
    if (n >= 20) word(b, t[n / 10]), n %= 10;
    if (n) word(b, n == 1 && g == G_F ? "одной" : u[n]);
}
static void ru_card_gen(Buf* b, uint64_t n, int g) {
    if (!n) return word(b, "ноля");
    uint64_t th = n / 1000, r = n % 1000;
    if (th) {
        if (th > 1) ru_lt1000_gen(b, (int)th, G_F);
        word(b, th % 10 == 1 && th % 100 != 11 ? "тысячи" : "тысяч");
    }
    if (r) ru_lt1000_gen(b, (int)r, g);
}

static int lang_of(const char* lang) {
    static const char* names[NL_] = {"es", "en", "pt", "fr", "de", "it", "ru"};
    for (int i = 0; i < NL_; i++)
        if (!strncmp(lang, names[i], 2)) return i;
    return -1;
}

static void card(Buf* b, int L, uint64_t n, int g) {
    switch (L) {
    case ES: return es_card(b, n, g);
    case PT: return pt_card(b, n, g);
    case IT: return it_card(b, n, g);
    case DE: return de_card(b, n, g);
    case FR: return fr_card(b, n, g);
    case RU: return ru_card(b, n, g);
    default: return en_card(b, n);
    }
}
static void digits(Buf* b, int L, const char* s, int n) {   // "007" -> zero zero seven
    for (int i = 0; i < n; i++)
        if (is_digit(s[i])) card(b, L, (uint64_t)(s[i] - '0'), G_ALONE);
}

// ---------------------------------------------------------------------------------------------------- ordinals
// Russian: stem + ending; cases 0 nom.m, 1 prep.m, 2 gen.m, 3 instr.m, 4 dat.m, 5 nom.f, 6 nom.n, 7 prep/gen pl, 8 nom pl
static void ru_ord(Buf* b, uint64_t n, int cs) {
    static const char* end[3][9] = {{"ый", "ом", "ого", "ым", "ому", "ая", "ое", "ых", "ые"},
                                    {"ой", "ом", "ого", "ым", "ому", "ая", "ое", "ых", "ые"},
                                    {"ий", "ьем", "ьего", "ьим", "ьему", "ья", "ье", "ьих", "ьи"}};
    static const char* s20[20] = {"", "перв", "втор", "трет", "четвёрт", "пят", "шест", "седьм", "восьм", "девят",
                                  "десят", "одиннадцат", "двенадцат", "тринадцат", "четырнадцат", "пятнадцат",
                                  "шестнадцат", "семнадцат", "восемнадцат", "девятнадцат"};
    static const char c20[20] = {0, 0, 1, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    static const char* st[10] = {"", "", "двадцат", "тридцат", "сороков", "пятидесят", "шестидесят", "семидесят",
                                 "восьмидесят", "девяност"};
    static const char* sh[10] = {"", "сот", "двухсот", "трёхсот", "четырёхсот", "пятисот", "шестисот", "семисот",
                                 "восьмисот", "девятисот"};
    static const char* sk[4] = {"", "тысячн", "двухтысячн", "трёхтысячн"};
    uint64_t last;
    const char* stem;
    int cls = 0;
    if (n % 100) {
        last = n % 100 < 20 ? n % 100 : n % 10 ? n % 10 : n % 100;
        stem = last < 20 ? s20[last] : st[last / 10];
        cls = last < 20 ? c20[last] : last == 40;
    } else if (n % 1000) {
        last = n % 1000, stem = sh[last / 100];
    } else if (n / 1000 < 4) {
        last = n, stem = sk[n / 1000];
    } else {
        return ru_card(b, n, G_M);
    }
    if (n - last) ru_card(b, n - last, G_M);
    word(b, stem);
    put(b, end[cls][cs]);
}

// the ordinal of n (masculine, feminine) in es, pt, it, fr, en, de (endings by the caller)
static void ord(Buf* b, int L, uint64_t n, int fem, int apoc) {
    char tmp[256];
    Buf t = {tmp, 0, sizeof(tmp)};
    tmp[0] = 0;
    const char* o = fem ? "a" : "o";
    if (L == ES || L == PT) {
        static const char* es1[11] = {"", "primer", "segund", "tercer", "cuart", "quint", "sext", "séptim", "octav", "noven", "décim"};
        static const char* pt1[11] = {"", "primeir", "segund", "terceir", "quart", "quint", "sext", "sétim", "oitav", "non", "décim"};
        static const char* es10[10] = {"", "", "vigésim", "trigésim", "cuadragésim", "quincuagésim", "sexagésim", "septuagésim", "octogésim", "nonagésim"};
        static const char* pt10[10] = {"", "", "vigésim", "trigésim", "quadragésim", "quinquagésim", "sexagésim", "septuagésim", "octogésim", "nonagésim"};
        const char** u = L == ES ? es1 : pt1;
        if (n > 99 || n == 0) return card(b, L, n, fem ? G_F : G_M);
        int tens = (int)(n / 10), k = (int)(n % 10);
        if (n == 11 && L == ES) return word(b, fem ? "undécima" : "undécimo");
        if (n == 12 && L == ES) return word(b, fem ? "duodécima" : "duodécimo");
        if (n > 10 && n < 20 && L == ES) {   // decimotercero, decimocuarta: one word
            word(b, "decimo");
            put(b, u[k]);
            if (!((k == 1 || k == 3) && apoc && !fem)) put(b, o);
            return;
        }
        if (n > 10) {
            word(b, tens == 1 ? "décim" : (L == ES ? es10 : pt10)[tens]);
            put(b, o);
            if (!k) return;
            n = (uint64_t)k;
        }
        word(b, u[n]);
        if (!(L == ES && (n == 1 || n == 3) && apoc && !fem)) put(b, o);
        return;
    }
    if (L == IT) {
        static const char* it1[11] = {"", "prim", "second", "terz", "quart", "quint", "sest", "settim", "ottav", "non", "decim"};
        if (n <= 10) return word(b, it1[n]), put(b, o);
        it_card(&t, n, G_ALONE);
        int l = t.n;
        if (l >= 4 && !strcmp(tmp + l - 4, "tré")) t.n = l - 4, tmp[t.n] = 0, put(&t, "tre");   // ventitreesimo
        else if (l >= 3 && !strcmp(tmp + l - 3, "sei")) ;                               // ventiseiesimo
        else tmp[--t.n] = 0;                                                            // undicesimo, ventesimo
        word(b, tmp + (tmp[0] == ' '));
        put(b, "esim");
        put(b, o);
        return;
    }
    if (L == FR) {
        if (n == 1) return word(b, fem ? "première" : "premier");
        fr_card(&t, n, G_M);
        int l = t.n;
        char* w = tmp + l;
        while (w > tmp && w[-1] != ' ' && w[-1] != '-') w--;
        if (!strcmp(w, "un")) strcpy(w, "unième");
        else if (!strcmp(w, "cinq")) strcpy(w, "cinquième");
        else if (!strcmp(w, "neuf")) strcpy(w, "neuvième");
        else {
            l = (int)strlen(tmp);
            if (tmp[l - 1] == 'e' || tmp[l - 1] == 's') tmp[l - 1] = 0;
            strcat(tmp, "ième");
        }
        return word(b, tmp + (tmp[0] == ' '));
    }
    if (L == DE) {   // stem only: erst, zweit, dritt, ... zwanzigst (the caller adds e / en)
        if (n == 1) return word(b, "erst");
        if (n == 3) return word(b, "dritt");
        if (n == 7) return word(b, "siebt");
        if (n == 8) return word(b, "acht");
        de_card(&t, n, G_ALONE);
        word(b, tmp + (tmp[0] == ' '));
        put(b, n < 20 ? "t" : "st");
        return;
    }
    // English
    en_card(&t, n);
    int l = t.n;
    char* w = tmp + l;
    while (w > tmp && w[-1] != ' ' && w[-1] != '-') w--;
    static const char* const irr[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"}, {"five", "fifth"},
                                         {"eight", "eighth"}, {"nine", "ninth"}, {"twelve", "twelfth"}, {0, 0}};
    int done = 0;
    for (int i = 0; irr[i][0]; i++)
        if (!strcmp(w, irr[i][0])) strcpy(w, irr[i][1]), done = 1;
    if (!done) {
        l = (int)strlen(tmp);
        if (tmp[l - 1] == 'y') tmp[l - 1] = 0, strcat(tmp, "ieth");
        else strcat(tmp, "th");
    }
    word(b, tmp + (tmp[0] == ' '));
}

// ---------------------------------------------------------------------------------------------------- Roman numerals
static int roman(const char* s, int n) {   // value of a strict Roman numeral of I, V, X (1-39), else 0
    static const char* units[10] = {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
    int v = 0, i = 0;
    while (i < n && s[i] == 'X' && v < 30) v += 10, i++;
    for (int u = 9; u >= 1; u--) {
        int l = (int)strlen(units[u]);
        if (n - i == l && !strncmp(s + i, units[u], l)) return v + u;
    }
    return i == n ? v : 0;
}

// ---------------------------------------------------------------------------------------------------- acronyms
static const char* letter_name(int L, uint32_t c) {
    static const char* es[26] = {"a", "be", "ce", "de", "e", "efe", "ge", "hache", "i", "jota", "ka", "ele", "eme",
                                 "ene", "o", "pe", "cu", "erre", "ese", "te", "u", "uve", "uve doble", "equis", "ye", "zeta"};
    static const char* pt[26] = {"á", "bê", "cê", "dê", "é", "efe", "gê", "agá", "i", "jota", "cá", "éle", "eme", "ene",
                                 "ó", "pê", "quê", "erre", "esse", "tê", "u", "vê", "dáblio", "xis", "ípsilon", "zê"};
    static const char* it[26] = {"a", "bi", "ci", "di", "e", "effe", "gi", "acca", "i", "i lunga", "cappa", "elle",
                                 "emme", "enne", "o", "pi", "cu", "erre", "esse", "ti", "u", "vu", "doppia vu", "ics",
                                 "ipsilon", "zeta"};
    static const char* fr[26] = {"a", "bé", "cé", "dé", "e", "effe", "gé", "ache", "i", "ji", "ka", "elle", "emme",
                                 "enne", "o", "pé", "ku", "erre", "esse", "té", "u", "vé", "double vé", "ixe", "i grec", "zède"};
    static const char* de[26] = {"a", "be", "ce", "de", "e", "eff", "ge", "ha", "i", "jott", "ka", "ell", "emm", "enn",
                                 "o", "pe", "ku", "err", "ess", "te", "u", "vau", "we", "ix", "üpsilon", "zett"};
    static const char* en[26] = {"ay", "bee", "see", "dee", "ee", "ef", "gee", "aitch", "eye", "jay", "kay", "el", "em",
                                 "en", "oh", "pee", "cue", "ar", "ess", "tee", "you", "vee", "double you", "ex", "why", "zee"};
    static const char* ru[26] = {"эй", "би", "си", "ди", "и", "эф", "джи", "эйч", "ай", "джей", "кей", "эл", "эм", "эн",
                                 "оу", "пи", "кью", "ар", "эс", "ти", "ю", "ви", "дабл ю", "экс", "уай", "зед"};
    static const char* cyr[32] = {"а", "бэ", "вэ", "гэ", "дэ", "е", "жэ", "зэ", "и", "и краткое", "ка", "эль", "эм",
                                  "эн", "о", "пэ", "эр", "эс", "тэ", "у", "эф", "ха", "цэ", "че", "ша", "ща",
                                  "твёрдый знак", "ы", "мягкий знак", "э", "ю", "я"};
    const char** t = L == ES ? es : L == PT ? pt : L == IT ? it : L == FR ? fr : L == DE ? de : L == RU ? ru : en;
    if (c >= 'a' && c <= 'z') c -= 32;
    if (c >= 'A' && c <= 'Z') return t[c - 'A'];
    if (c == 0xD1 || c == 0xF1) return L == ES ? "eñe" : "ene";
    if (c == 0x401 || c == 0x451) return "ё";
    if (c >= 0x430 && c <= 0x44F) c -= 32;
    if (c >= 0x410 && c <= 0x42F) return cyr[c - 0x410];
    return NULL;
}
static int vowel(uint32_t c) {
    if (c >= 'a' && c <= 'z') c -= 32;
    if (c >= 0x430 && c <= 0x44F) c -= 32;
    return strchr("AEIOUY", (int)(c < 0x80 ? c : 1)) != NULL || (c >= 0xC0 && c <= 0xDC && c != 0xD1 && c != 0xC7) ||
           c == 0x410 || c == 0x415 || c == 0x401 || c == 0x418 || c == 0x41E || c == 0x423 || c == 0x42B ||
           c == 0x42D || c == 0x42E || c == 0x42F;
}
// can these capitals be read as a word? at least one vowel, and every consonant cluster one a word can have
static int pronounceable(const uint32_t* c, int n) {
    static const char* const onset[] = {"BL", "BR", "CL", "CR", "DR", "FL", "FR", "GL", "GR", "PL", "PR", "TR", "CH",
                                        "LL", "QU", "SH", "TH", "ST", "SP", "SC", "SK", "SM", "SN", "KR", "KL", "PH",
                                        "WH", 0};
    static const char* const coda[] = {"NS", "RS", "ND", "NT", "ST", "RT", "RD", "LD", "LT", "NK", "CK", "SK", "SS",
                                       "LL", "TH", "SH", "CH", "NG", "RK", "RM", "RN", "LM", "LS", "FT", "CT", "PT", "XT", "LF", "MP", 0};
    int nv = 0;
    for (int i = 0; i < n; i++) nv += vowel(c[i]);
    if (!nv || n < 3) return 0;
    if (n >= 5 && nv >= 2) return 1;   // SELECT, KEYDOWN, UNESCO: long enough to be words
    for (int i = 0; i < n;) {
        if (vowel(c[i])) {
            i++;
            continue;
        }
        int j = i;
        while (j < n && !vowel(c[j])) j++;
        int len = j - i;
        char cl[8] = {0};
        for (int k = 0; k < len && k < 7; k++) cl[k] = c[i + k] < 0x80 ? (char)c[i + k] : '#';
        if (i == 0 && len > 1 && !(len == 2 && in_list(cl, onset))) return 0;          // GPU, FBI
        if (j == n && len > 1 && !(len == 2 && in_list(cl, coda))) return 0;   // USB, OMS, AMD
        if (i > 0 && j < n && len > 3) return 0;
        i = j;
    }
    return 1;
}

// all-caps token c[0..n) (n >= 2): spelled letter by letter unless said as a word; returns 1 if it wrote something
static int acronym(Buf* b, int L, const uint32_t* c, int n) {
    char up[64];
    int k = 0;
    for (int i = 0; i < n && k < 60; i++) {
        if (c[i] >= 0x80) {
            k = 60;
            break;
        }
        up[k++] = (char)c[i];
    }
    up[k < 60 ? k : 0] = 0;
    // said as words (whatever the letters look like) and spelled (though they could be read)
    static const char* const as_word[] = {"NASA", "OTAN", "NATO", "UNESCO", "UNICEF", "FIFA", "UEFA", "OPEP", "OVNI",
                                          "SIDA", "AIDS", "IKEA", "COVID", "RAM", "ROM", "LED", "PIN", "SIM", "GIF",
                                          "WIFI", "LASER", "RADAR", "SEGA", "SONY", "LEGO", "ATARI", "CAPCOM", "ONU",
                                          "IVA", "INTERPOL", "MERCOSUR", "PYME", "ESA", "FAQ", "ASCII", "BIOS", "SWAT", 0};
    static const char* const spelled[] = {"USA", "API", "ONG", "ADN", "ARN", "EU", "UE", "UK", "AI", "IA", "OS", "ID",
                                          "UI", "UX", "EA", "AMD", "IBM", "HTML", "URL", "SQL", "JSON", "CSS", "PHP",
                                          "XML", "USB", "DVD", "CPU", "GPU", "PDF", "LLM", "OEM", 0};
    // short capitals that are words of the language (EN, LA, NO, EU = "I" in Portuguese, ...)
    static const char* const two[NL_][24] = {
        {"NO", "SI", "YA", "YO", "TU", "MI", "EL", "LA", "LE", "LO", "EN", "ES", "UN", "SE", "DE", "TE", "ME", "IR", "OK", "OH", "AH", "EH", "AL", 0},
        {"IS", "IT", "IN", "ON", "AT", "TO", "DO", "GO", "SO", "NO", "UP", "WE", "HE", "BE", "MY", "BY", "OR", "IF", "AN", "AS", "OF", "AM", "OK", 0},
        {"EU", "EM", "OS", "AS", "UM", "NA", "NO", "DO", "DA", "AO", "SE", "ME", "TE", "JA", "JÁ", "SÓ", "OK", "OI", "NÃO", "DE", "LA", "LÁ", "AÍ", 0},
        {"OU", "ET", "LE", "LA", "DU", "IL", "JE", "TU", "UN", "EN", "NE", "SE", "CE", "MA", "TA", "SA", "MON", "TON", "SON", "OK", "AH", "OH", "DE", 0},
        {"ER", "ES", "DA", "JA", "SO", "ZU", "IM", "AM", "UM", "OB", "DU", "WO", "OK", "AN", "IN", "AB", "NU", 0},
        {"DI", "IL", "IO", "MA", "TU", "LA", "LO", "NO", "SI", "SE", "CI", "MI", "TI", "VI", "UN", "IN", "HO", "HA", "OK", "DA", "FA", 0},
        {0}};
    if (k < 60 && in_list(up, as_word)) return 0;
    if (L >= 0 && n <= 3 && in_list(up, two[L])) return 0;   // (before the spelled list: EU in Portuguese is "I")
    int spell = k < 60 && in_list(up, spelled);
    if (!spell && pronounceable(c, n)) return 0;
    if (L < 0) return 0;
    sp(b);
    for (int i = 0; i < n; i++) {
        const char* nm = letter_name(L, c[i]);
        if (!nm) return 0;
        // Spanish: with hyphens (ele-ele-eme), which T3 says more clearly than with spaces
        if (i) put(b, L == ES ? "-" : " ");
        put(b, nm);
    }
    return 1;
}

// ---------------------------------------------------------------------------------------------------- the pass
typedef struct {
    const char* s;
    int n;
    int L;
    Buf* b;
    char prev[32];      // the previous word, lower case (for "в" before a year, "am" before "1. Mai", ...)
    int prev_cap;       // ... and whether it was capitalised (regnal numbers: Felipe II)
    int century;        // a century word was just read (siglo, século, secolo): Roman numerals follow
    int num_end;        // where the last number ended in s (units after it: 5 kg), and its value
    uint64_t num_v;
} St;

static const char* dec_word(int L, char sepc) {
    if (sepc == ',') {
        static const char* w[NL_] = {"coma", "point", "vírgula", "virgule", "Komma", "virgola", "запятая"};
        return w[L];
    }
    static const char* w[NL_] = {"punto", "point", "ponto", "point", "Punkt", "punto", "точка"};
    return w[L];
}
static const char* pct_word(int L) {
    static const char* w[NL_] = {"por ciento", "percent", "por cento", "pour cent", "Prozent", "per cento", ""};
    return w[L];
}

// a number written as digits, at s[i] (a digit); returns the index after what it consumed
static int number(St* st, int i) {
    const char* s = st->s;
    int n = st->n, L = st->L;
    Buf* b = st->b;
    int a = i;
    while (i < n && is_digit(s[i])) i++;
    int int_end = i;
    uint64_t v = 0;
    int nd = int_end - a, big = nd > 12;
    for (int k = a; k < int_end && !big; k++) v = v * 10 + (uint64_t)(s[k] - '0');
    // thousands groups: 1.250.000 / 1,250,000 / 25 000 (fr, ru, de, or a narrow no-break space)
    char tsep = 0;
    int grouped = 0;
    if (nd <= 3 && !big)
        for (;;) {
            int sl = 1;
            char c = i < n ? s[i] : 0;
            int nbsp = i + 1 < n && (uint8_t)s[i] == 0xC2 && (uint8_t)s[i + 1] == 0xA0;
            int thin = i + 2 < n && (uint8_t)s[i] == 0xE2 && (uint8_t)s[i + 1] == 0x80 && (uint8_t)s[i + 2] == 0xAF;
            if (nbsp) sl = 2, c = ' ';
            if (thin) sl = 3, c = ' ';
            if (!(c == '.' || c == ',' || c == ' ')) break;
            if (c == ' ' && !(nbsp || thin || L == FR || L == RU || L == DE)) break;
            if (tsep && c != tsep) break;
            int j = i + sl;
            if (!(j + 2 < n && is_digit(s[j]) && is_digit(s[j + 1]) && is_digit(s[j + 2])) || (j + 3 < n && is_digit(s[j + 3])))
                break;
            // one group after a single separator: thousands only where the language writes them so
            // (es / pt / it / de / fr / ru "3.500", en / es "3,500"; "3,141" in de, pt, it, fr, ru is a decimal)
            if (!tsep) {
                int more = j + 4 < n && s[j + 3] == c && is_digit(s[j + 4]);
                if (!more && c == '.' && L == EN) break;
                if (!more && c == ',' && L != EN && L != ES) break;
            }
            tsep = c, grouped = 1;
            v = v * 1000 + (uint64_t)((s[j] - '0') * 100 + (s[j + 1] - '0') * 10 + (s[j + 2] - '0'));
            i = j + 3;
        }
    // time: 10:30 (hours up to 24, two-digit minutes)
    if (!grouped && v <= 24 && nd <= 2 && i + 2 < n && s[i] == ':' && is_digit(s[i + 1]) && is_digit(s[i + 2]) &&
        !(i + 3 < n && is_digit(s[i + 3])) && (s[i + 1] - '0') * 10 + (s[i + 2] - '0') < 60) {
        int mm = (s[i + 1] - '0') * 10 + (s[i + 2] - '0');
        i += 3;
        switch (L) {
        case ES: case PT: case IT:
            card(b, L, v, G_F);
            if (mm) word(b, L == PT || L == IT ? "e" : "y"), card(b, L, (uint64_t)mm, G_F);
            break;
        case FR:
            card(b, L, v, G_F), word(b, v == 1 ? "heure" : "heures");
            if (mm) card(b, L, (uint64_t)mm, G_F);
            break;
        case DE:
            card(b, L, v, G_M), word(b, "Uhr");
            if (mm) card(b, L, (uint64_t)mm, G_ALONE);
            {   // "10:30 Uhr": Uhr is said already
                int k = i;
                while (k < n && s[k] == ' ') k++;
                if (k + 3 <= n && !strncmp(s + k, "Uhr", 3) && !alnum_at(s, k + 3, n)) i = k + 3;
            }
            break;
        case RU:
            card(b, L, v, G_M);
            if (mm) {
                if (mm < 10) word(b, "ноль");
                card(b, L, (uint64_t)mm, G_F);
            } else word(b, ru_plural(v, "час", "часа", "часов"));
            break;
        default:
            en_card(b, v);
            if (mm && mm < 10) word(b, "oh");
            if (mm) en_card(b, (uint64_t)mm);
            else word(b, "o'clock");
        }
        return i;
    }
    // French hours: 10h30, 10 h 30, 10h
    if (L == FR && !grouped && v <= 24 && nd <= 2) {
        int k = i;
        if (k < n && s[k] == ' ') k++;
        if (k < n && s[k] == 'h' && !(k + 1 < n && is_letter((uint8_t)s[k + 1]))) {
            int m = k + 1, mm = -1;
            if (m < n && s[m] == ' ') m++;
            if (m + 1 < n && is_digit(s[m]) && is_digit(s[m + 1]) && !(m + 2 < n && is_digit(s[m + 2])))
                mm = (s[m] - '0') * 10 + (s[m + 1] - '0'), k = m + 2;
            else k++;
            card(b, L, v, G_F), word(b, v == 1 ? "heure" : "heures");
            if (mm > 0) card(b, L, (uint64_t)mm, G_F);
            return k;
        }
    }
    // decimal part
    int frac_a = 0, frac_n = 0;
    char dsep = 0;
    if (i + 1 < n && (s[i] == ',' || s[i] == '.') && is_digit(s[i + 1]) && !(grouped && s[i] == tsep)) {
        dsep = s[i];
        frac_a = i + 1;
        int j = frac_a;
        while (j < n && is_digit(s[j])) j++;
        frac_n = j - frac_a;
        i = j;
    }
    // what follows: ordinal marks, %, °, currency, a range, the next word
    int L2 = L < 0 ? EN : L;
    int is_ord = 0, ord_fem = 0, ru_cs = -1, de_ord = 0;
    if (!dsep && !grouped) {
        const uint8_t* q = (const uint8_t*)s + i;
        int left = n - i;
        if (left >= 2 && q[0] == 0xC2 && (q[1] == 0xBA || q[1] == 0xAA)) is_ord = 1, ord_fem = q[1] == 0xAA, i += 2;
        else if (left >= 3 && q[0] == '.' && q[1] == 0xC2 && (q[2] == 0xBA || q[2] == 0xAA))
            is_ord = 1, ord_fem = q[2] == 0xAA, i += 3;
        else if ((L == IT || L == PT) && left >= 3 && q[0] == 0xC2 && q[1] == 0xB0 && q[2] == ' ') {
            char w[32];   // "1° posto": an ordinal when a lower-case word follows (not C / F, degrees)
            int ws = next_word(s, n, i + 2, w, sizeof(w));
            if (ws >= 0 && is_lower((uint8_t)s[ws]) && strcmp(w, "c") && strcmp(w, "f")) is_ord = 1, i += 2;
        } else if (L == EN && left >= 2) {
            char x = (char)(q[0] | 32), y = (char)(q[1] | 32);
            if (((x == 's' && y == 't') || (x == 'n' && y == 'd') || (x == 'r' && y == 'd') || (x == 't' && y == 'h')) &&
                !alnum_at(s, i + 2, n))
                is_ord = 1, i += 2;
        } else if (L == FR && left >= 1) {
            static const char* const suf[] = {"ère", "ème", "ers", "res", "er", "re", "nde", "nd", "e", 0};
            for (int k = 0; suf[k]; k++) {
                int l = (int)strlen(suf[k]);
                if (left >= l && !strncmp((const char*)q, suf[k], l) && !alnum_at(s, i + l, n)) {
                    is_ord = 1, ord_fem = suf[k][0] == 'r' || !strcmp(suf[k], "ère") || !strcmp(suf[k], "nde");
                    i += l;
                    break;
                }
            }
        } else if (L == RU && left >= 3 && q[0] == '-') {
            static const char* const suf[] = {"ого", "ому", "ый", "ий", "ой", "ом", "ым", "ая", "ое", "ых", "ые", "го",
                                              "му", "й", "м", "я", "е", "х", 0};
            static const int cs[] = {2, 4, 0, 0, 0, 1, 3, 5, 6, 7, 8, 2, 4, 0, 1, 5, 6, 7};
            for (int k = 0; suf[k]; k++) {
                int l = (int)strlen(suf[k]);
                if (left >= 1 + l && !strncmp((const char*)q + 1, suf[k], l) && !alnum_at(s, i + 1 + l, n)) {
                    ru_cs = cs[k], is_ord = 1, i += 1 + l;
                    break;
                }
            }
            if (ru_cs == 6) {   // "1990-е годы": plural
                char w[32];
                if (next_word(s, n, i, w, sizeof(w)) >= 0 && !strncmp(w, "год", 6) && strcmp(w, "год")) ru_cs = 8;
            }
        } else if (L == DE && left >= 2 && q[0] == '.' && q[1] == ' ') {
            char w[32];   // am 1. Mai, der 2. Weltkrieg (the period is not a sentence end: speech_text.c)
            int ws = next_word(s, n, i + 1, w, sizeof(w));
            static const char* const months[] = {"januar", "februar", "märz", "april", "mai", "juni", "juli", "august",
                                                 "september", "oktober", "november", "dezember", 0};
            static const char* const art[] = {"der", "die", "das", "den", "dem", "des", "am", "im", "zum", "vom", "beim", "zur", 0};
            if (ws >= 0 && (in_list(w, months) || in_list(st->prev, art))) {
                is_ord = 1, de_ord = 1, i += 1;
                ord_fem = !strcmp(st->prev, "der") || !strcmp(st->prev, "die") || !strcmp(st->prev, "das") ? 0 : 1;
            }
        }
    }
    if (big || (nd > 1 && s[a] == '0' && !dsep)) {   // 0800, long numbers: digit by digit
        digits(b, L2, s + a, i - a);
        return i;
    }
    char nw[48];
    int nws = next_word(s, n, i, nw, sizeof(nw));
    if (nws < 0) nw[0] = 0;
    if (is_ord) {
        if (L == RU) ru_ord(b, v, ru_cs < 0 ? 0 : ru_cs);
        else if (L == DE) {
            ord(b, L, v, 0, 0);
            put(b, de_ord && ord_fem ? "en" : "e");
        } else {
            int fem = ord_fem || (L != EN && L != FR && noun_gender(L, nw, v) == G_F && L != ES);
            static const char* const not_noun[] = {"de", "del", "y", "e", "o", "en", "a", 0};
            ord(b, L2, v, fem, nws >= 0 && !in_list(nw, not_noun));
        }
        return i;
    }
    // Russian years: в 1985 году -> в тысяча девятьсот восемьдесят пятом году
    if (L == RU && !dsep && !grouped) {
        int cs = -1, skip = 0;
        if (!strcmp(nw, "году")) cs = 1;
        else if (v >= 1000 && v <= 2100) {
            if (!strcmp(nw, "год")) cs = 0;
            else if (!strcmp(nw, "года")) cs = 2;
            else if (!strcmp(nw, "годом")) cs = 3;
            else if (!strcmp(nw, "г.")) {
                cs = !strcmp(st->prev, "в") || !strcmp(st->prev, "во") ? 1 : 2;
                skip = 1;
            }
        }
        if (cs >= 0) {
            ru_ord(b, v, cs);
            if (skip) {
                word(b, cs == 1 ? "году" : "года");
                i = nws + (int)strlen("г.");
            }
            return i;
        }
    }
    // percent
    {
        int k = i;
        if (k < n && s[k] == ' ') k++;
        if (k < n && s[k] == '%') {
            if (dsep || (L == RU && in_list(st->prev, (const char* const[]){"около", "более", "менее", "до", "от",
                                                                             "из", "свыше", "порядка", 0})))
                goto say_decimal;   // (said below, then the word)
            card(b, L2, v, L == RU ? G_M : G_ALONE);
            if (L == RU) word(b, ru_plural(v, "процент", "процента", "процентов"));
            else word(b, pct_word(L2));
            return k + 1;
        }
    }
say_decimal:;
    // currency after the number (25 €, 2,50 euros) or before it (handled by the caller: it passes cur)
    int k = i;
    if (k < n && s[k] == ' ') k++;
    const char* cur = NULL;
    int cur_end = k;
    {
        const uint8_t* q = (const uint8_t*)s + k;
        if (k + 2 < n && q[0] == 0xE2 && q[1] == 0x82 && q[2] == 0xAC) cur = "€", cur_end = k + 3;
        else if (k < n && q[0] == '$') cur = "$", cur_end = k + 1;
        else if (k + 1 < n && q[0] == 0xC2 && q[1] == 0xA3) cur = "£", cur_end = k + 2;
    }
    if (st->prev[0] == '$' || !strcmp(st->prev, "€") || !strcmp(st->prev, "£")) cur = st->prev, cur_end = i;
    // the integer part
    int g = noun_gender(L2, nw, v);
    if (cur) g = !strcmp(cur, "£") && (L == ES || L == PT || L == IT || L == FR) ? G_F : G_M;
    if (dsep) g = L == RU ? G_F : G_ALONE;
    // "2,50 euros": said as a price (dos euros con cincuenta)
    static const char* const money[] = {"euros", "euro", "dólares", "dólar", "pesos", "soles", "reais", "libras",
                                        "dollars", "dollar", "pounds", "dollari", 0};
    int money_w = !cur && dsep && frac_n == 2 && L != RU && nws >= 0 && nws <= i + 1 && in_list(nw, money);
    if (money_w) g = noun_gender(L2, nw, v);
    // Russian after около, более, до, от ...: the genitive (около пятнадцати процентов)
    static const char* const ru_gen[] = {"около", "более", "менее", "до", "от", "из", "свыше", "после", "для", "без",
                                         "кроме", "больше", "меньше", "порядка", 0};
    int gen = L == RU && !dsep && v < 1000000 && in_list(st->prev, ru_gen);
    int k2 = i;
    while (k2 < n && s[k2] == ' ') k2++;
    int pct = k2 < n && s[k2] == '%';
    if (L == EN && !dsep && !grouped && !cur && nd == 4 && ((v >= 1100 && v <= 1999) || (v >= 2010 && v <= 2099)))
        en_year(b, (int)v);
    else if (L == DE && !dsep && !grouped && !cur && nd == 4 && v >= 1100 && v <= 1999) {
        sp(b);
        de_lt100(b, (int)(v / 100), 0);
        put(b, "hundert");
        if (v % 100) de_lt100(b, (int)(v % 100), 1);
    } else if (gen) ru_card_gen(b, v, g);
    else card(b, L2, v, g);
    if (money_w) {
        char tmp[48];
        int we = lower_word(s, n, nws, tmp, sizeof(tmp));
        sp(b);
        putn(b, s + nws, we - nws);
        static const char* conj[NL_] = {"con", "", "e", "", "", "e", ""};
        if (*conj[L2]) word(b, conj[L2]);
        card(b, L2, (uint64_t)((s[frac_a] - '0') * 10 + (s[frac_a + 1] - '0')), G_ALONE);
        return we;
    }
    // the decimals
    int cents = cur && dsep && frac_n == 2;
    if (dsep && !cents) {
        if (L == RU && frac_n <= 3) {
            word(b, ru_plural(v, "целая", "целых", "целых"));
            uint64_t f = 0;
            for (int j = 0; j < frac_n; j++) f = f * 10 + (uint64_t)(s[frac_a + j] - '0');
            card(b, L, f, G_F);
            static const char* d1[3][2] = {{"десятая", "десятых"}, {"сотая", "сотых"}, {"тысячная", "тысячных"}};
            word(b, ru_plural(f, d1[frac_n - 1][0], d1[frac_n - 1][1], d1[frac_n - 1][1]));
        } else {
            word(b, dec_word(L2, L == EN ? '.' : dsep));
            if (frac_n <= 2 && s[frac_a] != '0' && L != EN && L != DE) {
                uint64_t f = 0;
                for (int j = 0; j < frac_n; j++) f = f * 10 + (uint64_t)(s[frac_a + j] - '0');
                card(b, L2, f, G_ALONE);
            } else digits(b, L2, s + frac_a, frac_n);
        }
    }
    if (pct) {
        if (gen) word(b, v % 10 == 1 && v % 100 != 11 ? "процента" : "процентов");
        else if (L == RU) word(b, dsep ? "процента" : ru_plural(v, "процент", "процента", "процентов"));
        else word(b, pct_word(L2));
        return k2 + 1;
    }
    if (cur) {
        static const char* eur[NL_][2] = {{"euro", "euros"}, {"euro", "euros"}, {"euro", "euros"}, {"euro", "euros"},
                                          {"Euro", "Euro"}, {"euro", "euro"}, {"евро", "евро"}};
        static const char* usd[NL_][2] = {{"dólar", "dólares"}, {"dollar", "dollars"}, {"dólar", "dólares"},
                                          {"dollar", "dollars"}, {"Dollar", "Dollar"}, {"dollaro", "dollari"}, {"доллар", "долларов"}};
        static const char* gbp[NL_][2] = {{"libra", "libras"}, {"pound", "pounds"}, {"libra", "libras"},
                                          {"livre", "livres"}, {"Pfund", "Pfund"}, {"sterlina", "sterline"}, {"фунт", "фунтов"}};
        const char* (*t)[2] = cur[0] == '$' ? usd : !strcmp(cur, "£") ? gbp : eur;
        if (L == RU && t != eur) word(b, ru_plural(v, t[L][0], cur[0] == '$' ? "доллара" : "фунта", t[L][1]));
        else word(b, v == 1 && !dsep ? t[L2][0] : t[L2][1]);
        if (cents) {
            static const char* conj[NL_] = {"con", "", "e", "", "", "e", ""};
            if (*conj[L2]) word(b, conj[L2]);
            uint64_t f = (uint64_t)((s[frac_a] - '0') * 10 + (s[frac_a + 1] - '0'));
            card(b, L2, f, G_ALONE);
        }
        return cur_end > i ? cur_end : i;
    }
    // degrees: 120°C, 20 °C, 5°
    {
        int d = i;
        if (d < n && s[d] == ' ') d++;
        if (d + 1 < n && (uint8_t)s[d] == 0xC2 && (uint8_t)s[d + 1] == 0xB0) {
            static const char* deg[NL_][2] = {{"grado", "grados"}, {"degree", "degrees"}, {"grau", "graus"},
                                              {"degré", "degrés"}, {"Grad", "Grad"}, {"grado", "gradi"}, {"градус", "градусов"}};
            if (L == RU) word(b, ru_plural(v, "градус", "градуса", "градусов"));
            else word(b, v == 1 && !dsep ? deg[L2][0] : deg[L2][1]);
            d += 2;
            if (d < n && (s[d] == 'C' || s[d] == 'F') && !alnum_at(s, d + 1, n)) {
                word(b, L == RU ? (s[d] == 'C' ? "по Цельсию" : "по Фаренгейту") : s[d] == 'C' ? "Celsius" : "Fahrenheit");
                d++;
            }
            return d;
        }
    }
    // a range: 5-7, 1914-1918 (not 2023-10-12)
    if (i + 1 < n && s[i] == '-' && is_digit(s[i + 1])) {
        int j = i + 1;
        while (j < n && is_digit(s[j])) j++;
        if (!(j < n && s[j] == '-' && j + 1 < n && is_digit(s[j + 1])) && !(j < n && is_letter((uint8_t)s[j]))) {
            static const char* to[NL_] = {"a", "to", "a", "à", "bis", "a", ""};
            if (*to[L2]) word(b, to[L2]);
            return i + 1;   // the second number is read next, as a number of its own
        }
    }
    return i;
}

// units after a number: 5 kg, 20 km, 30 min (singular after 1; Russian by its plural rule)
static int unit(St* st, int i, int j) {
    static const char* const u[] = {"kg", "g", "km", "m", "cm", "mm", "ml", "h", "min", 0};
    static const char* w[9][NL_][2] = {
        {{"kilo", "kilos"}, {"kilo", "kilos"}, {"quilo", "quilos"}, {"kilo", "kilos"}, {"Kilo", "Kilo"}, {"chilo", "chili"}, {0, 0}},
        {{"gramo", "gramos"}, {"gram", "grams"}, {"grama", "gramas"}, {"gramme", "grammes"}, {"Gramm", "Gramm"}, {"grammo", "grammi"}, {0, 0}},
        {{"kilómetro", "kilómetros"}, {"kilometer", "kilometers"}, {"quilômetro", "quilômetros"}, {"kilomètre", "kilomètres"},
         {"Kilometer", "Kilometer"}, {"chilometro", "chilometri"}, {0, 0}},
        {{"metro", "metros"}, {"meter", "meters"}, {"metro", "metros"}, {"mètre", "mètres"}, {"Meter", "Meter"}, {"metro", "metri"}, {0, 0}},
        {{"centímetro", "centímetros"}, {"centimeter", "centimeters"}, {"centímetro", "centímetros"}, {"centimètre", "centimètres"},
         {"Zentimeter", "Zentimeter"}, {"centimetro", "centimetri"}, {0, 0}},
        {{"milímetro", "milímetros"}, {"millimeter", "millimeters"}, {"milímetro", "milímetros"}, {"millimètre", "millimètres"},
         {"Millimeter", "Millimeter"}, {"millimetro", "millimetri"}, {0, 0}},
        {{"mililitro", "mililitros"}, {"milliliter", "milliliters"}, {"mililitro", "mililitros"}, {"millilitre", "millilitres"},
         {"Milliliter", "Milliliter"}, {"millilitro", "millilitri"}, {0, 0}},
        {{"hora", "horas"}, {"hour", "hours"}, {"hora", "horas"}, {"heure", "heures"}, {"Stunde", "Stunden"}, {"ora", "ore"}, {0, 0}},
        {{"minuto", "minutos"}, {"minute", "minutes"}, {"minuto", "minutos"}, {"minute", "minutes"}, {"Minute", "Minuten"},
         {"minuto", "minuti"}, {0, 0}}};
    static const char* ru[9][3] = {{"килограмм", "килограмма", "килограммов"}, {"грамм", "грамма", "граммов"},
                                   {"километр", "километра", "километров"}, {"метр", "метра", "метров"},
                                   {"сантиметр", "сантиметра", "сантиметров"}, {"миллиметр", "миллиметра", "миллиметров"},
                                   {"миллилитр", "миллилитра", "миллилитров"}, {"час", "часа", "часов"}, {"минута", "минуты", "минут"}};
    if (!(st->num_end == i || (st->num_end == i - 1 && st->s[i - 1] == ' '))) return 0;
    for (int k = 0; u[k]; k++) {
        int l = (int)strlen(u[k]);
        if (j - i != l || strncmp(st->s + i, u[k], l)) continue;
        // km/h
        if (k == 2 && j + 1 < st->n && st->s[j] == '/' && st->s[j + 1] == 'h' && !alnum_at(st->s, j + 2, st->n)) {
            static const char* kmh[NL_] = {"kilómetros por hora", "kilometers per hour", "quilômetros por hora",
                                           "kilomètres heure", "Kilometer pro Stunde", "chilometri orari", "километров в час"};
            word(st->b, kmh[st->L]);
            return j + 2;
        }
        if (st->L == RU) word(st->b, ru_plural(st->num_v, ru[k][0], ru[k][1], ru[k][2]));
        else word(st->b, w[k][st->L][st->num_v == 1 ? 0 : 1]);
        return j;
    }
    return 0;
}

// a word of letters at s[i] (possibly followed by digits: PS4, MP3, Qwen3): returns the index after the letters
static int letters(St* st, int i) {
    const char* s = st->s;
    int n = st->n, L = st->L;
    Buf* b = st->b;
    uint32_t c[64];
    int nc = 0, j = i, upper = 1;
    while (j < n && nc < 64) {
        int l;
        uint32_t x = cp(s + j, &l);
        if (!is_letter(x)) break;
        if (!is_upper(x)) upper = 0;
        c[nc++] = x;
        j += l;
    }
    char low[48];
    lower_word(s, n, i, low, sizeof(low));
    int u = unit(st, i, j);
    if (u) return u;
    // Roman numerals: siglo XIX, Felipe II, XIXe siècle, XIX век
    int core = j, fr_e = 0;   // French "XIXe", "Ier": the capitals before the ordinal mark
    if (!upper && L == FR && nc >= 2 && c[nc - 1] == 'e') core = j - 1, fr_e = 1;
    if (!upper && L == FR && nc >= 3 && c[nc - 2] == 'e' && c[nc - 1] == 'r') core = j - 2, fr_e = 2;
    int all_up = 1;
    for (int k = i; k < core; k++) all_up &= s[k] >= 'A' && s[k] <= 'Z';
    int r = all_up && core > i ? roman(s + i, core - i) : 0;
    if (fr_e && !r) core = j, fr_e = 0;
    if (r) {
        char nw[32];
        if (next_word(s, n, j, nw, sizeof(nw)) < 0) nw[0] = 0;
        int ru_cs = -1;
        if (L == RU) ru_cs = !strcmp(nw, "век") ? 0 : !strcmp(nw, "века") ? 2 : !strcmp(nw, "веке") ? 1 : !strcmp(nw, "веков") ? 7 : -1;
        int ctx = st->century || ru_cs >= 0 || fr_e || !strcmp(nw, "siècle") || !strcmp(nw, "siècles");
        // after a ruler's name: Felipe II -> Felipe segundo (a game or a war: Diablo III, World War II -> three, two)
        static const char* const rulers[] = {"felipe", "carlos", "luis", "juan", "pablo", "enrique", "isabel", "fernando",
                                             "alfonso", "pedro", "jaime", "benedicto", "pío", "león", "gregorio", "juana",
                                             "ricardo", "eduardo", "jorge", "guillermo", "federico", "nicolás", "iván",
                                             "catalina", "alejandro", "constantino", "ramsés", "ptolomeo", "napoleón",
                                             "henry", "charles", "louis", "elizabeth", "edward", "george", "william",
                                             "richard", "james", "mary", "philip", "john", "paul", "leo", "pius", "henri",
                                             "françois", "philippe", "jean", "léon", "felipe", "joão", "dom", "manuel",
                                             "afonso", "luís", "carlo", "vittorio", "umberto", "giovanni", "paolo",
                                             "enrico", "luigi", "federico", "ludwig", "wilhelm", "friedrich", "karl",
                                             "heinrich", "otto", "пётр", "петр", "иван", "николай", "александр",
                                             "екатерина", "павел", "елизавета", "анна", "мария", "людовик", "карл", 0};
        int regnal = core - i >= 2 && in_list(st->prev, rulers);
        if (ctx || core - i >= 2) {
            if (L == RU) ru_ord(b, (uint64_t)r, ru_cs >= 0 ? ru_cs : 0);
            else if (L == FR) {
                if (ctx || r == 1) ord(b, L, (uint64_t)r, 0, 0);
                else fr_card(b, (uint64_t)r, G_M);
            } else if ((L == ES || L == PT) && (ctx || regnal) && r <= 10) ord(b, L, (uint64_t)r, 0, 0);
            else if (L == IT && (ctx || regnal)) ord(b, L, (uint64_t)r, 0, 0);
            else if (L == EN && regnal) word(b, "the"), ord(b, L, (uint64_t)r, 0, 0);
            else if (L == DE && regnal) word(b, "der"), ord(b, L, (uint64_t)r, 0, 0), put(b, "e");
            else card(b, L, (uint64_t)r, G_M);
            if (st->century) st->century = 2;   // (siglos XV y XVI)
            return j;
        }
    }
    static const char* const cent[] = {"siglo", "siglos", "século", "séculos", "secolo", "secoli", 0};
    int is_cent = in_list(low, cent);
    // abbreviations with their periods
    static const char* const ab[][3] = {
        // lang, as written (from here, with the periods), said
        {"es", "a.C.", "antes de Cristo"}, {"es", "a. C.", "antes de Cristo"}, {"es", "d.C.", "después de Cristo"},
        {"es", "d. C.", "después de Cristo"}, {"pt", "a.C.", "antes de Cristo"}, {"pt", "d.C.", "depois de Cristo"},
        {"pt", "a. C.", "antes de Cristo"}, {"pt", "d. C.", "depois de Cristo"},
        {"it", "a.C.", "avanti Cristo"}, {"it", "d.C.", "dopo Cristo"}, {"it", "a. C.", "avanti Cristo"},
        {"it", "d. C.", "dopo Cristo"}, {"fr", "av. J.-C.", "avant Jésus-Christ"}, {"fr", "apr. J.-C.", "après Jésus-Christ"},
        {"fr", "av. J-C", "avant Jésus-Christ"}, {"fr", "apr. J-C", "après Jésus-Christ"},
        {"de", "v. Chr.", "vor Christus"}, {"de", "n. Chr.", "nach Christus"}, {"de", "v.Chr.", "vor Christus"},
        {"de", "n.Chr.", "nach Christus"}, {"ru", "до н. э.", "до нашей эры"}, {"ru", "до н.э.", "до нашей эры"},
        {"ru", "н. э.", "нашей эры"}, {"ru", "н.э.", "нашей эры"}, {"ru", "т. е.", "то есть"}, {"ru", "т.е.", "то есть"},
        {"ru", "т. д.", "так далее"}, {"ru", "т.д.", "так далее"}, {"ru", "т. п.", "тому подобное"},
        {"es", "EE. UU.", "Estados Unidos"}, {"es", "EE.UU.", "Estados Unidos"}, {"es", "p. ej.", "por ejemplo"},
        {"es", "etc.", "etcétera"}, {"pt", "p. ex.", "por exemplo"}, {"fr", "p. ex.", "par exemple"},
        {"en", "e.g.", "for example"}, {"en", "i.e.", "that is"}, {"en", "etc.", "et cetera"},
        {"de", "z. B.", "zum Beispiel"}, {"de", "z.B.", "zum Beispiel"}, {"de", "d. h.", "das heißt"},
        {"de", "usw.", "und so weiter"}, {"de", "bzw.", "beziehungsweise"}, {"it", "ecc.", "eccetera"},
        {"fr", "etc.", "et cetera"}, {"pt", "etc.", "et cetera"}, {"it", "etc.", "eccetera"}, {0, 0, 0}};
    static const char* names[NL_] = {"es", "en", "pt", "fr", "de", "it", "ru"};
    for (int k = 0; ab[k][0]; k++) {
        int l = (int)strlen(ab[k][1]);
        if (strcmp(ab[k][0], names[L])) continue;
        if (n - i >= l && !strncmp(s + i, ab[k][1], l) && !alnum_at(s, i + l, n)) {
            word(b, ab[k][2]);
            // its period also ended the sentence ("753 a.C. Luego ..."): keep one
            int q = i + l;
            while (q < n && s[q] == ' ') q++;
            int lc;
            if (ab[k][1][l - 1] == '.' && (q >= n || is_upper(cp(s + q, &lc)))) put(b, ".");
            return i + l;
        }
    }
    // dotted capitals: B.C., U.S.A.
    {
        int k = i, cnt = 0;
        while (k + 1 < n && s[k] >= 'A' && s[k] <= 'Z' && s[k + 1] == '.') k += 2, cnt++;
        if (cnt >= 2 && !alnum_at(s, k, n)) {
            sp(b);
            for (int m = 0; m < cnt; m++) {
                if (m) put(b, " ");
                put(b, letter_name(L, (uint8_t)s[i + 2 * m]));
            }
            return k;
        }
    }
    // all-caps acronyms (with digits after: PS4, MP3; a plural s after: GPUs)
    int plural_s = !upper && nc >= 3 && c[nc - 1] == 's';
    for (int k = 0; plural_s && k < nc - 1; k++) plural_s &= is_upper(c[k]);
    // (part of an identifier in code read as prose: pygame.KEYDOWN, K_LEFT, CONFIG.value: as written)
    int ident = (i >= 2 && s[i - 1] == '.' && alnum_at(s, i - 2, n)) || (j + 1 < n && s[j] == '.' && alnum_at(s, j + 1, n)) ||
                (j < n && s[j] == '_') || (i >= 1 && s[i - 1] == '_');
    if (((upper && nc >= 2) || plural_s) && !ident) {
        if (acronym(b, L, c, plural_s ? nc - 1 : nc)) {
            if (plural_s && L == EN) put(b, "s");
            return j;
        }
    }
    // a single letter glued to digits: 9B, 4K, x86, i7, 1080p
    if (nc == 1 && ((j < n && is_digit(s[j])) || (i > 0 && is_digit(s[i - 1])))) {
        word(b, letter_name(L, c[0]) ? letter_name(L, c[0]) : "");
        return j;
    }
    // a plain word: copied (Garlic's own name in Cyrillic for Russian: T3 misreads the Latin letters)
    if (L == RU && j - i == 6 && !strncmp(s + i, "Garlic", 6)) put(b, "Гарлик");
    else putn(b, s + i, j - i);
    st->century = is_cent;
    return j;
}

int st_speak(char* s, int n, int max, const char* lang) {
    static char out[4096];
    Buf b = {out, 0, (int)sizeof(out) < max ? (int)sizeof(out) : max};
    out[0] = 0;
    St st = {s, n, lang_of(lang), &b, {0}, 0, 0, -1, 0};
    if (st.L < 0) return n;   // (Greek and the rest: as written)
    for (int i = 0; i < n;) {
        int l;
        uint32_t c = cp(s + i, &l);
        int j;
        if (is_digit(s[i])) {
            // a minus sign before it: "-5 °C" (copied just before)
            if (i >= 1 && s[i - 1] == '-' && (i == 1 || s[i - 2] == ' ') && b.n && b.p[b.n - 1] == '-') {
                static const char* minus[NL_] = {"menos", "minus", "menos", "moins", "minus", "meno", "минус"};
                b.p[--b.n] = 0;
                word(&b, minus[st.L]);
            }
            if (b.n && b.p[b.n - 1] != ' ' && prev_alnum(s, i)) put(&b, " ");   // PS4 -> pe ese cuatro
            uint64_t v = 0;
            for (int k = i; k < n && is_digit(s[k]) && k < i + 18; k++) v = v * 10 + (uint64_t)(s[k] - '0');
            j = number(&st, i);
            if (j <= i) j = i + 1, putn(&b, s + i, 1);
            st.num_end = j, st.num_v = v;
            st.prev[0] = 0, st.prev_cap = 0;
        } else if (is_letter(c) && (!prev_alnum(s, i) || is_digit(s[i - 1]))) {
            if (i > 0 && is_digit(s[i - 1]) && b.n && b.p[b.n - 1] != ' ') put(&b, " ");   // 5kg, 9B
            char w[32];
            lower_word(s, n, i, w, sizeof(w));
            int cen = st.century;
            j = letters(&st, i);
            if (st.century == 2) st.century = cen;
            snprintf(st.prev, sizeof(st.prev), "%s", w);
            st.prev_cap = is_upper(c);
        } else if ((c == '$' || c == 0x20AC || c == 0xA3) && i + l < n &&
                   (is_digit(s[i + l]) || (s[i + l] == ' ' && i + l + 1 < n && is_digit(s[i + l + 1])))) {
            // currency before a number: $25, € 30
            snprintf(st.prev, sizeof(st.prev), "%s", c == '$' ? "$" : c == 0xA3 ? "£" : "€");
            j = i + l;
            if (s[j] == ' ') j++;
            sp(&b);
            int jj = number(&st, j);
            st.prev[0] = 0;
            j = jj > j ? jj : j + 1;
        } else if (c == 0xBA || c == 0xAA) {   // a lone º ª: Nº 5, n.º 5 -> número; else dropped
            j = i + l;
            int k = b.n;
            if (k >= 2 && b.p[k - 1] == '.' && (b.p[k - 2] | 32) == 'n' && (k == 2 || b.p[k - 3] == ' ')) k -= 2;
            else if (k >= 1 && (b.p[k - 1] | 32) == 'n' && (k == 1 || b.p[k - 2] == ' ')) k -= 1;
            if (k != b.n) {
                static const char* num[NL_] = {"número", "number", "número", "numéro", "Nummer", "numero", "номер"};
                b.n = k, b.p[k] = 0;
                word(&b, num[st.L]);
            }
        } else if (c == '$') {   // a dollar sign not before a number (u$s): dropped
            j = i + 1;
        } else {
            if (st.century && c != ' ' && c != ',' && c != '-') st.century = 0;
            if (!(c == ' ' && b.n && b.p[b.n - 1] == ' ')) putn(&b, s + i, l);
            j = i + l;
        }
        i = j;
    }
    // tidy: single spaces, none at the ends
    int o = 0;
    for (int k = 0; k < b.n; k++) {
        char ch = out[k];
        if (ch == ' ' && (o == 0 || out[o - 1] == ' ')) continue;
        out[o++] = ch;
    }
    while (o && out[o - 1] == ' ') o--;
    if (o >= max) o = max - 1;
    memcpy(s, out, o);
    s[o] = 0;
    return o;
}
