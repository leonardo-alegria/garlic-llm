// The models folder (see models.h): which folders hold a model, and whether a folder's files match the built-in one.
#include "models.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _WIN32   // (the host tests)
#include <io.h>
#define seek_to(f, o) _lseeki64(f, (long long)(o), SEEK_SET)
#define seek_end(f) _lseeki64(f, 0, SEEK_END)
#else
#define seek_to(f, o) lseek(f, (off_t)(o), SEEK_SET)
#define seek_end(f) lseek(f, 0, SEEK_END)
#endif
#ifndef O_BINARY
#define O_BINARY 0
#endif

void say(const char* fmt, ...);   // the log (main.c)

// model.idx (pc/export_chat.py, pc/convert_9b_q4s.py): "PMQ5" (or "PMDL" for Qwen3), the shape, the weights path
// (128 bytes at 52), the size of the part that goes to GPU memory (8 at 180) and the number of tensors (4 at 188);
// "PMQ5" has 8 uint32 more, the last one flags (bit 2: the last record is the table of token ids, fmt 7); then 36
// bytes per tensor: offset and size (8 each), rows and columns (4 each), scales (8), format (4).
#define IDX_PATH 52
// mtp.idx (pc/convert_9b_q4s.py --mtp): "PMTP", the draft rows (4), the weights path (128 at 8), the size of the
// tensors (8 at 136) and of the draft table (8 at 144), the number of tensors (4 at 152), then their records.
#define MTP_PATH 8
#define PATH_LEN 128
#define FMT_EMBED_ROWS 4

static uint8_t* slurp(const char* path, size_t* size) {   // a whole index; NULL if missing, empty or unreadable
    int f = open(path, O_RDONLY | O_BINARY);
    if (f < 0) return NULL;
    long long n = seek_end(f);
    uint8_t* d = n > 0 && n < (16 << 20) && seek_to(f, 0) == 0 ? malloc((size_t)n) : NULL;
    size_t done = 0;
    while (d && done < (size_t)n) {
        int r = (int)read(f, d + done, (unsigned)((size_t)n - done));
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(f);
    if (!d || done != (size_t)n) {
        free(d);
        return NULL;
    }
    *size = done;
    return d;
}

static long long file_size(const char* path) {   // -1 if it cannot be opened
    int f = open(path, O_RDONLY | O_BINARY);
    if (f < 0) return -1;
    long long n = seek_end(f);
    close(f);
    return n;
}

static uint64_t u64_at(const uint8_t* p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

static uint32_t u32_at(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

// The file name at the end of an index's weights path (128 bytes at p). 0 = done.
static int path_file(const uint8_t* p, char* out, int n) {
    char path[PATH_LEN + 1];
    memcpy(path, p, PATH_LEN);
    path[PATH_LEN] = 0;
    const char* base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (!*base || (int)strlen(base) >= n) return -1;
    snprintf(out, (size_t)n, "%s", base);
    return 0;
}

// A model index: its records and, from them, how long its weights file is.
typedef struct {
    uint8_t* d;
    size_t size;
    uint32_t n, rec0, flags;
} Idx;

static int idx_parse(Idx* x) {
    if (!x->d || x->size < 192 || (memcmp(x->d, "PMQ5", 4) && memcmp(x->d, "PMDL", 4))) return -1;
    int q5 = !memcmp(x->d, "PMQ5", 4);
    x->n = u32_at(x->d + 188);
    x->rec0 = q5 ? 224 : 192;
    x->flags = q5 && x->size >= 224 ? u32_at(x->d + 220) : 0;
    if (!x->n || x->n > 1000000 || x->size < x->rec0 + (size_t)x->n * 36) return -1;
    return 0;
}

static const uint8_t* idx_rec(const Idx* x, uint32_t i) { return x->d + x->rec0 + (size_t)i * 36; }

static uint64_t idx_end(const Idx* x) {   // the end of the last tensor: the weights file's length
    uint64_t end = 0;
    for (uint32_t i = 0; i < x->n; i++) {
        const uint8_t* r = idx_rec(x, i);
        uint64_t e = u64_at(r) + u64_at(r + 8);
        if (e > end) end = e;
    }
    return end;
}

// The first byte where two indexes differ, outside the weights path [skip, skip + PATH_LEN); -1 if none.
static long first_difference(const uint8_t* a, size_t na, const uint8_t* b, size_t nb, size_t skip) {
    size_t n = na < nb ? na : nb;
    for (size_t i = 0; i < n; i++)
        if (a[i] != b[i] && (i < skip || i >= skip + PATH_LEN)) return (long)i;
    return na != nb ? (long)n : -1;
}

// "4641752088" -> "4 641 752 088" (sizes on the screen, readable in every language)
static void grouped(long long v, char* out, int n) {
    char d[32];
    int len = snprintf(d, sizeof(d), "%lld", v < 0 ? 0 : v), k = 0;
    for (int i = 0; i < len && k < n - 2; i++) {
        if (i && (len - i) % 3 == 0) out[k++] = ' ';
        out[k++] = d[i];
    }
    out[k] = 0;
}

int model_idx_file(const char* idx_path, char* out, int n) {
    size_t size;
    uint8_t* d = slurp(idx_path, &size);
    int r = -1;
    if (d && size >= 192 && (!memcmp(d, "PMQ5", 4) || !memcmp(d, "PMDL", 4))) r = path_file(d + IDX_PATH, out, n);
    else if (d && size >= 156 && !memcmp(d, "PMTP", 4)) r = path_file(d + MTP_PATH, out, n);
    free(d);
    return r;
}

int model_name_ok(const char* s) {
    size_t n = strlen(s);
    if (!n || n >= MODEL_NAME_MAX || s[0] == '.') return 0;
    for (size_t i = 0; i < n;) {
        unsigned char c = (unsigned char)s[i];
        int more = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
        if (more < 0 || c < 0x20 || c == 0x7F || c == '/' || c == '\\' || (more && c < 0xC2)) return 0;
        for (int k = 1; k <= more; k++)
            if (i + k >= n || ((unsigned char)s[i + k] & 0xC0) != 0x80) return 0;
        i += 1 + more;
    }
    return 1;
}

static int by_name(const void* a, const void* b) { return strcmp((const char*)a, (const char*)b); }

static int has_maple_idx(const char* dir) {
    char p[512];
    snprintf(p, sizeof(p), "%s/model.idx", dir);
    int f = open(p, O_RDONLY | O_BINARY);
    if (f < 0) return 0;
    char magic[4] = {0};
    int ok = read(f, magic, 4) == 4 && !memcmp(magic, "PMAP", 4);
    close(f);
    if (!ok) return 0;
    snprintf(p, sizeof(p), "%s/kernels.bin", dir);
    f = open(p, O_RDONLY | O_BINARY);
    if (f < 0) return 0;
    close(f);
    return 1;
}

int models_scan(const char* root, const char* weights, char (*names)[MODEL_NAME_MAX], int max) {
    DIR* d = opendir(root);
    if (!d) return 0;
    int n = 0;
    struct dirent* e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;   // (".", "..", hidden folders)
        char p[512];
        snprintf(p, sizeof(p), "%s/%s/%s", root, e->d_name, weights);
        int f = open(p, O_RDONLY | O_BINARY);
        if (f >= 0) {
            close(f);
        } else {
            snprintf(p, sizeof(p), "%s/%s", root, e->d_name);
            if (!has_maple_idx(p)) continue;   // (not a folder, or no weights / Maple PMAP index in it)
        }
        if (!model_name_ok(e->d_name)) {
            say("models: %.80s is left out: a model's folder name has 1 to %d bytes, without '/' or '\\'",
                e->d_name, MODEL_NAME_MAX - 1);
            continue;
        }
        if (n == max) {
            say("models: more than %d folders in %s, the rest are left out", max, root);
            break;
        }
        snprintf(names[n++], MODEL_NAME_MAX, "%s", e->d_name);
    }
    closedir(d);
    if (n > 1) qsort(names, (size_t)n, MODEL_NAME_MAX, by_name);
    return n;
}

int model_check(const char* idx_path, const char* dir, char* why, int why_n, char* shown, int shown_n) {
    why[0] = shown[0] = 0;
    Idx b = {0}, m = {0};
    int ret = -1;
    char file[64], p[512];
    b.d = slurp(idx_path, &b.size);
    if (idx_parse(&b) || path_file(b.d + IDX_PATH, file, sizeof(file))) {
        snprintf(why, (size_t)why_n, "cannot read the built-in model's index %s", idx_path);
        snprintf(shown, (size_t)shown_n, "%s", idx_path);
        goto out;
    }
    snprintf(p, sizeof(p), "%s/model.idx", dir);
    m.d = slurp(p, &m.size);   // (the folder's own index is optional; if it is there it must be the same, or PMAP)
    if (m.d && m.size >= 4 && !memcmp(m.d, "PMAP", 4)) {
        snprintf(p, sizeof(p), "%s/kernels.bin", dir);
        if (file_size(p) <= 0) {
            snprintf(why, (size_t)why_n, "Maple model folder %s is missing kernels.bin", dir);
            snprintf(shown, (size_t)shown_n, "kernels.bin");
            goto out;
        }
        snprintf(why, (size_t)why_n, "%s/model.idx (Maple PMAP)", dir);
        ret = 0;
        goto out;
    }
    if (m.d) {
        long at = first_difference(b.d, b.size, m.d, m.size, IDX_PATH);
        if (at >= 0) {
            snprintf(why, (size_t)why_n, "its model.idx differs from the built-in model's at byte %ld (%zu and %zu "
                     "bytes): another shape, format or conversion", at, m.size, b.size);
            snprintf(shown, (size_t)shown_n, "model.idx");
            goto out;
        }
    }
    uint64_t end = idx_end(&b);
    snprintf(p, sizeof(p), "%s/%s", dir, file);
    long long have = file_size(p);
    if (have != (long long)end) {
        char g1[32], g2[32];
        grouped(have, g1, sizeof(g1)), grouped((long long)end, g2, sizeof(g2));
        if (have < 0) snprintf(why, (size_t)why_n, "cannot open %s", p);
        else snprintf(why, (size_t)why_n, "%s has %lld bytes, the model has %llu: incomplete or another conversion",
                      p, have, (unsigned long long)end);
        snprintf(shown, (size_t)shown_n, "%s  ·  %s / %s B", file, g1, g2);
        goto out;
    }
    if (b.flags & 4) {   // the table of token ids (row of the trimmed lm_head -> token): every id a real token
        const uint8_t* ids = idx_rec(&b, b.n - 1);
        const uint8_t* emb = idx_rec(&b, 0);
        uint64_t off = u64_at(ids), count = u32_at(ids + 16);
        uint32_t bound = u32_at(emb + 32) == FMT_EMBED_ROWS ? u32_at(emb + 16) : 0;   // the input embedding's rows
        int f = bound ? open(p, O_RDONLY | O_BINARY) : -1;
        uint32_t* buf = f >= 0 ? malloc(1 << 16) : NULL;
        uint64_t done = 0, bad = 0;
        if (buf && seek_to(f, off) == (long long)off)
            while (done < count && !bad) {
                uint64_t k = count - done < (1 << 14) ? count - done : (1 << 14);
                if (read(f, buf, (unsigned)(k * 4)) != (int)(k * 4)) break;
                for (uint64_t i = 0; i < k; i++)
                    if (buf[i] >= bound) bad = done + i + 1;
                done += k;
            }
        free(buf);
        if (f >= 0) close(f);
        if (bound && (done != count || bad)) {
            if (bad) snprintf(why, (size_t)why_n, "%s: entry %llu of its vocabulary table is not a token", p,
                              (unsigned long long)(bad - 1));
            else snprintf(why, (size_t)why_n, "%s: cannot read its vocabulary table", p);
            snprintf(shown, (size_t)shown_n, "%s  ·  ids", file);
            goto out;
        }
    }
    snprintf(why, (size_t)why_n, "%s matches the built-in model (%llu bytes)", p, (unsigned long long)end);
    ret = 0;
out:
    free(b.d);
    free(m.d);
    return ret;
}

int model_check_mtp(const char* mtp_idx_path, const char* dir, char* why, int why_n) {
    why[0] = 0;
    size_t nb, nm;
    char file[64], p[512];
    uint8_t* b = slurp(mtp_idx_path, &nb);
    if (!b || nb < 156 || memcmp(b, "PMTP", 4) || path_file(b + MTP_PATH, file, sizeof(file))) {
        free(b);
        snprintf(why, (size_t)why_n, "the package has no MTP head (%s)", mtp_idx_path);
        return 0;
    }
    uint64_t total = u64_at(b + 136) + u64_at(b + 144);
    snprintf(p, sizeof(p), "%s/%s", dir, file);
    long long have = file_size(p);
    if (have < 0) {
        free(b);
        snprintf(why, (size_t)why_n, "no %s", p);
        return 0;
    }
    int ret = 1;
    snprintf(p, sizeof(p), "%s/mtp.idx", dir);
    uint8_t* m = slurp(p, &nm);
    long at = m ? first_difference(b, nb, m, nm, MTP_PATH) : -1;
    if (at >= 0) {
        snprintf(why, (size_t)why_n, "its mtp.idx differs from the built-in head's at byte %ld", at);
        ret = -1;
    } else if (have != (long long)total) {
        snprintf(why, (size_t)why_n, "%s/%s has %lld bytes, the head has %llu", dir, file, have,
                 (unsigned long long)total);
        ret = -1;
    } else {
        snprintf(why, (size_t)why_n, "%s/%s (%llu bytes)", dir, file, (unsigned long long)total);
    }
    free(b);
    free(m);
    return ret;
}

void model_last_get(const char* path, char* name, int n) {
    name[0] = 0;
    size_t size;
    uint8_t* d = slurp(path, &size);
    if (!d) return;
    while (size && (d[size - 1] == '\n' || d[size - 1] == '\r' || d[size - 1] == ' ')) size--;
    if (size < (size_t)n) {
        memcpy(name, d, size);
        name[size] = 0;
    }
    free(d);
}

void model_last_set(const char* path, const char* name) {
    int f = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0666);
    if (f < 0) return;
    if (write(f, name, (unsigned)strlen(name)) < 0) say("models: cannot write %s", path);
    close(f);
}
