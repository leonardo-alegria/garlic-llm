// Host tests for chat/models.c (the models folder: which folders hold a model, whether their files match).
// Build and run (any C compiler; zig cc on Windows):
//   zig cc -std=c99 -Wall -Wextra -I chat tests/models_test.c chat/models.c -o build/models_test.exe
//   build/models_test.exe build/models_test [<folder with the real qwen3.5-9b-q4s.bin and qwen3.5-9b-mtp.bin>]
// The first argument is a scratch folder (made if missing; each run writes a few small files in a new folder in it).
// With the second, the real weights files are also checked against chat/assets_9b's model.idx and mtp.idx (they are
// only opened, not copied). Run from the repository's root.
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#define make_dir(p) _mkdir(p)
#else
#define make_dir(p) mkdir(p, 0777)
#endif

#include "models.h"

static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("  log: ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

static char root[512];

static const char* at(const char* rel) {   // root/rel (one of a few rotating buffers)
    static char buf[8][600];
    static int k;
    k = (k + 1) % 8;
    snprintf(buf[k], sizeof(buf[k]), "%s/%s", root, rel);
    return buf[k];
}

static void put(const char* path, const void* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        printf("cannot write %s\n", path);
        exit(2);
    }
    fwrite(data, 1, n, f);
    fclose(f);
}

static void put_u32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
static void put_u64(uint8_t* p, uint64_t v) { memcpy(p, &v, 8); }

// A small model index like chat/assets_9b/model.idx: PMQ5, flags 7 (untied head, embedding rows on the CPU, a table
// of token ids), four tensors: final_norm, one matrix, the embedding rows (1000 rows), the ids (50).
enum { N_T = 4, IDX_SIZE = 224 + N_T * 36, WEIGHTS_SIZE = 12296, IDS_OFF = 12096, IDS_N = 50, EMB_ROWS = 1000 };
static uint8_t idx[IDX_SIZE];

static void make_index(const char* weights_path) {
    memset(idx, 0, sizeof(idx));
    memcpy(idx, "PMQ5", 4);
    put_u32(idx + 4, 1), put_u32(idx + 8, 64), put_u32(idx + 32, 8192);   // (a shape; not read by models.c)
    snprintf((char*)idx + 52, 128, "%s", weights_path);
    put_u64(idx + 180, 4096);
    put_u32(idx + 188, N_T);
    put_u32(idx + 220, 7);
    const uint64_t rec[N_T][6] = {{4096, 8000, EMB_ROWS, 64, 32, 4},
                                  {0, 1024, 0, 0, 0, 0},
                                  {1024, 2048, 128, 64, 1024, 6},
                                  {IDS_OFF, IDS_N * 4, IDS_N, 1, 0, 7}};
    for (int i = 0; i < N_T; i++) {
        uint8_t* r = idx + 224 + i * 36;
        put_u64(r, rec[i][0]), put_u64(r + 8, rec[i][1]), put_u32(r + 16, (uint32_t)rec[i][2]);
        put_u32(r + 20, (uint32_t)rec[i][3]), put_u64(r + 24, rec[i][4]), put_u32(r + 32, (uint32_t)rec[i][5]);
    }
}

static void make_weights(const char* path, size_t size, uint32_t bad_id) {   // bad_id != 0: one id out of range
    uint8_t* w = calloc(1, size);
    for (int i = 0; i < IDS_N && IDS_OFF + (i + 1) * 4 <= (int)size; i++) put_u32(w + IDS_OFF + i * 4, (uint32_t)(i * 7));
    if (bad_id && IDS_OFF + 40 <= (int)size) put_u32(w + IDS_OFF + 36, bad_id);
    put(path, w, size);
    free(w);
}

static void make_model(const char* name, size_t size, uint32_t bad_id) {
    make_dir(at(name));
    char p[600];
    snprintf(p, sizeof(p), "%s/test-q4s.bin", at(name));
    make_weights(p, size, bad_id);
}

// mtp.idx: PMTP, 128 draft rows, the path, 300 bytes of tensors + 200 of draft table, one record
enum { MTP_IDX_SIZE = 156 + 36, MTP_SIZE = 500 };
static uint8_t mtp_idx[MTP_IDX_SIZE];

static void make_mtp_index(const char* path) {
    memset(mtp_idx, 0, sizeof(mtp_idx));
    memcpy(mtp_idx, "PMTP", 4);
    put_u32(mtp_idx + 4, 128);
    snprintf((char*)mtp_idx + 8, 128, "%s", path);
    put_u64(mtp_idx + 136, 300), put_u64(mtp_idx + 144, 200);
    put_u32(mtp_idx + 152, 1);
    put_u64(mtp_idx + 156 + 8, 300);
}

static void test_names(void) {
    CHECK(model_name_ok("Qwen3.5-9B-abliterated"));
    CHECK(model_name_ok("my model (coder)"));
    CHECK(model_name_ok("日本語モデル"));
    CHECK(model_name_ok("abcdefghijklmnopqrstuvwxyz012345"));    // 32 bytes
    CHECK(!model_name_ok("abcdefghijklmnopqrstuvwxyz0123456"));  // 33
    CHECK(!model_name_ok(""));
    CHECK(!model_name_ok(".hidden"));
    CHECK(!model_name_ok(".."));
    CHECK(!model_name_ok("a/b"));
    CHECK(!model_name_ok("a\\b"));
    CHECK(!model_name_ok("tab\there"));
    CHECK(!model_name_ok("bad\xff"));
    CHECK(!model_name_ok("over\xc0\xaflong"));
    CHECK(!model_name_ok("cut\xe6\x97"));
}

static void test_scan_and_check(void) {
    char names[MODELS_MAX][MODEL_NAME_MAX], why[300], shown[160], file[64];
    CHECK(models_scan(at("missing"), "test-q4s.bin", names, MODELS_MAX) == 0);

    make_dir(at("models"));
    CHECK(models_scan(at("models"), "test-q4s.bin", names, MODELS_MAX) == 0);
    make_index("/data/test-q4s.bin");
    put(at("model.idx"), idx, sizeof(idx));
    CHECK(model_idx_file(at("model.idx"), file, sizeof(file)) == 0 && !strcmp(file, "test-q4s.bin"));

    make_model("models/b-model", WEIGHTS_SIZE, 0);
    make_model("models/a-model", WEIGHTS_SIZE, 0);
    make_model("models/.hidden", WEIGHTS_SIZE, 0);
    make_model("models/abcdefghijklmnopqrstuvwxyz0123456", WEIGHTS_SIZE, 0);   // (a name too long: left out)
    make_dir(at("models/empty"));
    put(at("models/loose-file.bin"), "x", 1);
    int n = models_scan(at("models"), "test-q4s.bin", names, MODELS_MAX);
    CHECK(n == 2);
    CHECK(n >= 2 && !strcmp(names[0], "a-model") && !strcmp(names[1], "b-model"));
    CHECK(models_scan(at("models"), "test-q4s.bin", names, 1) == 1);

    // a folder without its own model.idx: the built-in index decides
    CHECK(model_check(at("model.idx"), at("models/a-model"), why, sizeof(why), shown, sizeof(shown)) == 0);
    printf("  a-model: %s\n", why);
    // with its own model.idx, equal except for the weights path
    make_index("/data/garlic/models/b-model/test-q4s.bin");
    put(at("models/b-model/model.idx"), idx, sizeof(idx));
    CHECK(model_check(at("model.idx"), at("models/b-model"), why, sizeof(why), shown, sizeof(shown)) == 0);
    // one byte short
    make_model("models/c-short", WEIGHTS_SIZE - 1, 0);
    CHECK(model_check(at("model.idx"), at("models/c-short"), why, sizeof(why), shown, sizeof(shown)) == -1);
    printf("  c-short: %s | %s\n", why, shown);
    CHECK(strstr(shown, "test-q4s.bin") && strstr(shown, "12 295 / 12 296 B"));
    // longer than the index says
    make_model("models/c-long", WEIGHTS_SIZE + 4096, 0);
    CHECK(model_check(at("model.idx"), at("models/c-long"), why, sizeof(why), shown, sizeof(shown)) == -1);
    // an index with another tensor size
    make_model("models/d-other", WEIGHTS_SIZE, 0);
    make_index("/data/test-q4s.bin");
    put_u64(idx + 224 + 2 * 36 + 8, 2304);
    put(at("models/d-other/model.idx"), idx, sizeof(idx));
    CHECK(model_check(at("model.idx"), at("models/d-other"), why, sizeof(why), shown, sizeof(shown)) == -1);
    printf("  d-other: %s | %s\n", why, shown);
    CHECK(!strcmp(shown, "model.idx"));
    // an index of another length
    make_model("models/d-short-idx", WEIGHTS_SIZE, 0);
    make_index("/data/test-q4s.bin");
    put(at("models/d-short-idx/model.idx"), idx, sizeof(idx) - 36);
    CHECK(model_check(at("model.idx"), at("models/d-short-idx"), why, sizeof(why), shown, sizeof(shown)) == -1);
    // a token id past the vocabulary
    make_model("models/e-ids", WEIGHTS_SIZE, EMB_ROWS);
    CHECK(model_check(at("model.idx"), at("models/e-ids"), why, sizeof(why), shown, sizeof(shown)) == -1);
    printf("  e-ids: %s | %s\n", why, shown);
    CHECK(strstr(shown, "ids") != NULL);
    // the largest id that is still a token
    make_model("models/e-ids-ok", WEIGHTS_SIZE, EMB_ROWS - 1);
    CHECK(model_check(at("model.idx"), at("models/e-ids-ok"), why, sizeof(why), shown, sizeof(shown)) == 0);
    // no weights at all
    make_dir(at("models/f-none"));
    CHECK(model_check(at("model.idx"), at("models/f-none"), why, sizeof(why), shown, sizeof(shown)) == -1);
    // the built-in index missing or not an index
    CHECK(model_check(at("nothing.idx"), at("models/a-model"), why, sizeof(why), shown, sizeof(shown)) == -1);
    put(at("junk.idx"), "PMQ5", 4);
    CHECK(model_check(at("junk.idx"), at("models/a-model"), why, sizeof(why), shown, sizeof(shown)) == -1);
}

static void test_mtp(void) {
    char why[300], file[64];
    make_mtp_index("/data/test-mtp.bin");
    put(at("mtp.idx"), mtp_idx, sizeof(mtp_idx));
    CHECK(model_idx_file(at("mtp.idx"), file, sizeof(file)) == 0 && !strcmp(file, "test-mtp.bin"));
    CHECK(model_check_mtp(at("mtp.idx"), at("models/a-model"), why, sizeof(why)) == 0);   // none of its own
    CHECK(model_check_mtp(at("no-mtp.idx"), at("models/a-model"), why, sizeof(why)) == 0);   // the package has none
    static uint8_t blob[MTP_SIZE + 1];
    put(at("models/a-model/test-mtp.bin"), blob, MTP_SIZE);
    CHECK(model_check_mtp(at("mtp.idx"), at("models/a-model"), why, sizeof(why)) == 1);
    put(at("models/b-model/test-mtp.bin"), blob, MTP_SIZE - 1);
    CHECK(model_check_mtp(at("mtp.idx"), at("models/b-model"), why, sizeof(why)) == -1);
    printf("  b-model mtp: %s\n", why);
    put(at("models/b-model/test-mtp.bin"), blob, MTP_SIZE);
    make_mtp_index("/data/garlic/models/b-model/test-mtp.bin");   // its own mtp.idx, another path: fine
    put(at("models/b-model/mtp.idx"), mtp_idx, sizeof(mtp_idx));
    CHECK(model_check_mtp(at("mtp.idx"), at("models/b-model"), why, sizeof(why)) == 1);
    make_mtp_index("/data/test-mtp.bin");
    put_u64(mtp_idx + 136, 400);   // another split between tensors and table
    put(at("models/b-model/mtp.idx"), mtp_idx, sizeof(mtp_idx));
    CHECK(model_check_mtp(at("mtp.idx"), at("models/b-model"), why, sizeof(why)) == -1);
}

static void test_last(void) {
    char name[MODEL_NAME_MAX];
    model_last_get(at("no-such-file.txt"), name, sizeof(name));
    CHECK(name[0] == 0);
    model_last_set(at("last.txt"), "a-model");
    model_last_get(at("last.txt"), name, sizeof(name));
    CHECK(!strcmp(name, "a-model"));
    model_last_set(at("last.txt"), "");
    model_last_get(at("last.txt"), name, sizeof(name));
    CHECK(name[0] == 0);
    put(at("last.txt"), "b-model\r\n", 9);   // (written by hand)
    model_last_get(at("last.txt"), name, sizeof(name));
    CHECK(!strcmp(name, "b-model"));
    put(at("last.txt"), "0123456789012345678901234567890123456789", 40);   // too long for a name: none
    model_last_get(at("last.txt"), name, sizeof(name));
    CHECK(name[0] == 0);
}

static void test_real(const char* dir) {   // the shipped indexes against real weights files
    char why[300], shown[160], file[64];
    CHECK(model_idx_file("chat/assets_9b/model.idx", file, sizeof(file)) == 0 && !strcmp(file, "qwen3.5-9b-q4s.bin"));
    CHECK(model_idx_file("chat/assets_9b/mtp.idx", file, sizeof(file)) == 0 && !strcmp(file, "qwen3.5-9b-mtp.bin"));
    CHECK(model_check("chat/assets_9b/model.idx", dir, why, sizeof(why), shown, sizeof(shown)) == 0);
    printf("  real 9B: %s %s\n", why, shown);
    CHECK(model_check_mtp("chat/assets_9b/mtp.idx", dir, why, sizeof(why)) == 1);
    printf("  real MTP: %s\n", why);
    // the 9B's index against the MTP head's folder: the MTP file is not a 9B
    CHECK(model_check("chat/assets_9b/model.idx", root, why, sizeof(why), shown, sizeof(shown)) == -1);
}

int main(int argc, char** argv) {
    const char* base = argc > 1 ? argv[1] : "build/models_test";
    make_dir(base);
    snprintf(root, sizeof(root), "%s/run-%lld", base, (long long)time(NULL));   // (a fresh folder each run)
    make_dir(root);
    test_names();
    test_scan_and_check();
    test_mtp();
    test_last();
    if (argc > 2) test_real(argv[2]);
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
