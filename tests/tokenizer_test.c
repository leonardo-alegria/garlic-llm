// Host test for chat/tokenizer.c with the app's own vocabulary (chat/assets/tokenizer.bin): pre-tokens longer than
// 512 bytes are merged whole (they were cut at 512 bytes), and those of 512 bytes or fewer as before. The file is
// included, so its static functions are tested too.
// Build and run (from the repository's root): zig cc -std=gnu99 -Wall -Wextra -g -O1 -I chat tests/tokenizer_test.c -o build/tokenizer_test.exe && build/tokenizer_test.exe [chat/assets/tokenizer.bin]
#include <stdio.h>
#include <stdlib.h>

static int no_memory;   // malloc fails (bpe_long's fallback: windows of 512 bytes)
static void* test_malloc(size_t n) { return no_memory ? NULL : malloc(n); }
#define malloc(n) test_malloc(n)
#include "tokenizer.c"
#undef malloc

static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

static Tokenizer tok;
static uint32_t rnd = 12345;
static uint32_t next_rand(void) { return rnd = rnd * 1664525u + 1013904223u; }

// some text: n bytes of characters drawn from one alphabet (whole UTF-8 characters, so up to 3 bytes short)
static const char* alphabets[] = {
    "abcdefghijklmnopqrstuvwxyz",
    "aaaaaaaab",
    "=-_*#~",
    "  \n",
    "0123456789abcdef",
    "\xe4\xb8\x80\xe6\x98\xaf\xe4\xb8\x8d\xe4\xba\x86\xe4\xba\xba\xe6\x88\x91\xe5\x9c\xa8\xe6\x9c\x89\xe4\xbb\x96",   // Chinese
    "\xe0\xb8\x81\xe0\xb8\xb2\xe0\xb8\xa3\xe0\xb8\x99\xe0\xb8\xb5\xe0\xb9\x89\xe0\xb8\xa1\xe0\xb8\xad",           // Thai
    "\xf0\x9f\x98\x82\xf0\x9f\x91\x8d\xe2\x9d\xa4",                                                              // emoji
    "\xd0\xb0\xd0\xb1\xd0\xb2\xd0\xb3\xd0\xb4\xd0\xb5\xd0\xb6",                                                  // Cyrillic
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",                                          // base64
};
#define N_ALPHABETS (int)(sizeof(alphabets) / sizeof(alphabets[0]))

static int utf8_len(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4; }

static size_t make_text(char* out, size_t n, int alphabet) {
    const char* a = alphabets[alphabet];
    int chars = 0, starts[64];
    for (int i = 0; a[i]; i += utf8_len((unsigned char)a[i])) starts[chars++] = i;
    size_t k = 0;
    for (;;) {
        int c = starts[next_rand() % chars], l = utf8_len((unsigned char)a[c]);
        if (k + l > n) break;
        memcpy(out + k, a + c, l);
        k += l;
    }
    out[k] = 0;
    return k;
}

// the text a sequence of tokens stands for
static size_t decode(const uint32_t* ids, int n, char* out) {
    size_t k = 0;
    for (int i = 0; i < n; i++) {
        const uint8_t* b;
        int l = tok_piece(&tok, ids[i], &b);
        memcpy(out + k, b, l);
        k += l;
    }
    out[k] = 0;
    return k;
}

// The heap's merges are the scan's: for pre-tokens of up to 512 bytes, every alphabet, every length.
static void test_heap_is_scan(void) {
    static char s[600];
    static uint32_t a[600], b[600];
    static uint8_t mem[512 * 40];
    int same = 0, cases = 0;
    for (int r = 0; r < 20000; r++) {
        size_t n = make_text(s, 1 + next_rand() % 512, r % N_ALPHABETS);
        if (!n) continue;
        int ka = bpe(&tok, (const uint8_t*)s, (int)n, a, 600);
        int kb = bpe_heap(&tok, mem, 512, (const uint8_t*)s, (int)n, b);
        cases++;
        same += ka == kb && !memcmp(a, b, (size_t)ka * sizeof(uint32_t));
    }
    CHECK(same == cases);
    printf("heap = scan on %d pre-tokens of up to 512 bytes: %d the same\n", cases, same);
}

// Windows stitched together give the tokens of the whole pre-token at once, for lengths past 512 bytes.
static void test_windows_are_whole(void) {
    static char s[70000];
    static uint32_t whole[70000], win[70000];
    uint8_t* mem = malloc((size_t)70000 * 40);
    int same = 0, cases = 0;
    for (int r = 0; r < 300; r++) {
        size_t lens[] = {513, 1000, 4096, 4097, 5000, 9000, 20000, 65000};
        size_t n = make_text(s, lens[r % 8] + next_rand() % 200, r % N_ALPHABETS);
        if (r % 23 == 0) memset(s, 'a', n);   // (one letter again and again: every pair merges)
        int kw = bpe_heap(&tok, mem, 70000, (const uint8_t*)s, (int)n, whole);
        no_memory = r % 5 == 4;   // (now and then the fallback, windows of 512 bytes)
        int kl = bpe_long(&tok, (const uint8_t*)s, (int)n, win, 70000);
        no_memory = 0;
        cases++;
        int ok = kw == kl && !memcmp(whole, win, (size_t)kw * sizeof(uint32_t));
        same += ok;
        if (!ok) printf("  differs: alphabet %d, %zu bytes, %d tokens whole, %d in windows\n", r % N_ALPHABETS, n, kw, kl);
    }
    free(mem);
    CHECK(same == cases);
    printf("windows = whole on %d pre-tokens of 513 to 65200 bytes: %d the same\n", cases, same);
}

// Long runs come back whole, byte for byte (before, a pre-token kept 512 bytes); max still limits the tokens.
static void test_long_text(void) {
    static char s[300000], back[300000];
    static uint32_t ids[300000];
    // base64 of 48 KiB, a URL of 2 KB, a minified line, and runs of one kind (each one pre-token)
    size_t n = make_text(s, 65536, 9);
    int k = tok_encode(&tok, s, ids, 300000);
    CHECK(decode(ids, k, back) == n && !strcmp(back, s));
    strcpy(s, "https://example.com/search?q=");
    n = strlen(s);
    n += make_text(s + n, 2048, 0);
    strcpy(s + n, "&page=2");
    k = tok_encode(&tok, s, ids, 300000);
    CHECK(decode(ids, k, back) == strlen(s) && !strcmp(back, s));
    s[0] = 0;
    for (int i = 0; i < 300; i++) strcat(s, "function(e){return e&&e.__esModule?e:{default:e}};");
    k = tok_encode(&tok, s, ids, 300000);
    CHECK(decode(ids, k, back) == strlen(s) && !strcmp(back, s));
    for (int a = 0; a < N_ALPHABETS; a++) {
        n = make_text(s, 100000, a);
        k = tok_encode(&tok, s, ids, 300000);
        CHECK(decode(ids, k, back) == n && !strcmp(back, s));
    }
    memset(s, 'a', 200000);
    s[200000] = 0;
    k = tok_encode(&tok, s, ids, 300000);
    CHECK(decode(ids, k, back) == 200000);
    CHECK(tok_encode(&tok, s, ids, 10) == 10);   // (max as before: the first tokens only)
    printf("long text decodes back to itself\n");
}

// What a few pre-tokens of 512 bytes or fewer give, as before (each a merge result: no byte is left alone).
static void test_short_text(void) {
    static uint32_t ids[64];
    static char back[256];
    const char* t = "Hola, ¿qué tal? Hello world 123 ===== 😂😂 你好世界";
    int k = tok_encode(&tok, t, ids, 64);
    CHECK(k > 5 && k < 40 && decode(ids, k, back) == strlen(t) && !strcmp(back, t));
    static char s[600], b2[600];
    memset(s, 'x', 512);   // exactly 512 bytes: the short path
    s[512] = 0;
    static uint32_t i512[600];
    k = tok_encode(&tok, s, i512, 600);
    CHECK(decode(i512, k, b2) == 512);
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "chat/assets/tokenizer.bin";
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("cannot read %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = malloc((size_t)size);
    if (!data || fread(data, 1, (size_t)size, f) != (size_t)size || tok_load(&tok, data, (size_t)size)) {
        printf("%s is not a tokenizer\n", path);
        return 1;
    }
    fclose(f);
    test_heap_is_scan();
    test_windows_are_whole();
    test_long_text();
    test_short_text();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
