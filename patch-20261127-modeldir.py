# patch-20261127-modeldir.py — model directory / hot-swap / /v1/models list for the Garlic chat app.
# Anchored replacements on the branch tree; aborts on any anchor miss. Idempotency: run once.
# New behaviour:
#   /data/garlic_model.txt   one name (or a full .idx path): which model boots; absent = today's default
#   /data/garlic.manifest.txt one line per model: "<name> [idx path]" (no path = /data/<name>.idx)
#   GET  /v1/models          {"object":"list","data":[{"id":..,"object":"model"}],"loaded":"<current>"}
#   POST /v1/model           {"name":"X"} -> swap X's weights into the arena that is already mapped
#                            (same shape/record table enforced; conversation starts fresh; 409 while a
#                            chat request is being answered)
import sys
ROOT = __import__("pathlib").Path(__file__).resolve().parent   # branch root

EDITS = []

def edit(path, old, new):
    EDITS.append((path, old, new))

# ---------------------------------------------------------------- engine.h: two new fields
edit("chat/engine.h",
"""    char error[160];
} Engine;""",
"""    uint32_t idx_max_seq;       // max_seq the .idx gives, before TurboQuant doubles e->max_seq (engine_swap's compare)
    char model_name[64];        // name of the loaded model ("" = the default) — evidence for /v1/models + the swap
    char error[160];
} Engine;""")

# ---------------------------------------------------------------- engine.c: boot selection + the swap
edit("chat/engine.c",
"""static int load_index(Engine* e, const char* dir) {
    char path[256];
    size_t size;
    uint8_t* d = read_file("/data/model.idx", &size);   // a model dropped on the HDD wins
    if (!d || (memcmp(d, "PMDL", 4) && memcmp(d, "PMQ5", 4))) {
        snprintf(path, sizeof(path), "%s/model.idx", dir);
        d = read_file(path, &size);
    }""",
"""// Which .idx a model name means: /data/garlic.manifest.txt line "<name> [idx path]"; a name without a line
// (or no manifest) = /data/<name>.idx. Names: 1..63 of [A-Za-z0-9._-]. 0 = path written, -1 = bad name.
static int model_idx_path(char* out, size_t cap, const char* name) {
    if (!name || !name[0]) return -1;
    for (int i = 0; name[i]; i++) {
        if (i >= 63) return -1;
        int c = (unsigned char)name[i];
        int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                 c == '.' || c == '_' || c == '-';
        if (!ok) return -1;
    }
    size_t fs;
    char* mf = (char*)read_file("/data/garlic.manifest.txt", &fs);
    char custom[256];
    custom[0] = 0;
    if (mf && fs) {
        char* p = mf;
        while ((size_t)(p - mf) < fs) {
            char* q = p;
            while ((size_t)(q - mf) < fs && *q != '\\n') q++;
            char* sp = p;
            while (sp < q && *sp != ' ') sp++;
            size_t nlen = (size_t)(sp - p);
            if (nlen == strlen(name) && !strncmp(p, name, nlen)) {
                char* path2 = sp < q ? sp + 1 : NULL;
                while (path2 && (*path2 == ' ' || *path2 == '\\r')) path2++;
                if (path2 && *path2 && *path2 == '/' && (size_t)(q - path2) < 250) {
                    size_t l = (size_t)(q - path2);
                    while (l && (path2[l - 1] == ' ' || path2[l - 1] == '\\r')) l--;
                    memcpy(custom, path2, l);
                    custom[l] = 0;
                }
                break;
            }
            p = q + 1;
        }
    }
    free(mf);
    if (custom[0]) snprintf(out, cap, "%s", custom);
    else snprintf(out, cap, "/data/%s.idx", name);
    return 0;
}

static int load_index(Engine* e, const char* dir) {
    char path[256];
    size_t size;
    uint8_t* d = NULL;
    {   // /data/garlic_model.txt = name (or a full .idx path): which model boots
        char* sw = (char*)read_file("/data/garlic_model.txt", &size);
        char nm[80];
        nm[0] = 0;
        if (sw) {
            int k = 0;
            while (k < 79 && sw[k] && sw[k] <= 32) k++;
            int j = 0;
            while (j < 79 && sw[k] > 32) nm[j++] = sw[k++];
            nm[j] = 0;
            free(sw);
        }
        if (nm[0] && model_idx_path(path, sizeof(path), nm) == 0) {
            d = read_file(path, &size);
            if (d && (memcmp(d, "PMDL", 4) && memcmp(d, "PMQ5", 4))) {
                free(d);
                d = NULL;
            }
            if (d) snprintf(e->model_name, sizeof(e->model_name), "%s", nm);
        } else if (!nm[0]) d = read_file("/data/model.idx", &size);   // a model dropped on the HDD wins
    }
    if (!d || (memcmp(d, "PMDL", 4) && memcmp(d, "PMQ5", 4))) {
        snprintf(path, sizeof(path), "%s/model.idx", dir);
        d = read_file(path, &size);
    }""")

edit("chat/engine.c",
"""    e->head_dim = v[5], e->vocab = v[6], e->max_seq = v[7], e->eos = v[8], e->eos2 = v[9];""",
"""    e->head_dim = v[5], e->vocab = v[6], e->max_seq = v[7], e->eos = v[8], e->eos2 = v[9];
    e->idx_max_seq = v[7];""")

edit("chat/engine.c",
"""#define TQ_TABLE \\""",
"""// Hot-swap: a candidate model's weights go over the model block that is already mapped (no unmap, no reboot).
// Its .idx must give the identical shape and record table (class-identical by construction); after the copy the
// conversation starts fresh: engine_reset + api_run.inc's api_break makes the next API request a new one.
int engine_swap(Engine* e, const char* name) {
    if (e->model_name[0] && !strcmp(e->model_name, name)) {   // asked for what is already loaded
        say("swap: %s already loaded", name);
        return 0;
    }
    char path[256];
    size_t size;
    if (model_idx_path(path, sizeof(path), name)) {
        fail(e, "swap: bad model name");
        return -1;
    }
    uint8_t* d = read_file(path, &size);
    if (!d || (memcmp(d, "PMDL", 4) && memcmp(d, "PMQ5", 4))) {
        fail(e, "swap: cannot read model index %s", path);
        return -1;
    }
    int q35 = !memcmp(d, "PMQ5", 4);
    if (q35 != (e->qwen35 != 0)) {
        free(d);
        fail(e, "swap: model family differs (PMDL/PMQ5)");
        return -1;
    }
    uint32_t v[10];
    memcpy(v, d + 4, 40);
    if (v[0] != e->n_layers || v[1] != e->hidden || v[2] != e->inter || v[3] != e->n_heads || v[4] != e->n_kv ||
        v[5] != e->head_dim || v[6] != e->vocab || v[7] != e->idx_max_seq || v[8] != e->eos || v[9] != e->eos2) {
        free(d);
        fail(e, "swap: shape differs");
        return -1;
    }
    float eps2, th2;
    memcpy(&eps2, d + 44, 4);
    memcpy(&th2, d + 48, 4);
    if (eps2 != e->eps || th2 != e->rope_theta) {
        free(d);
        fail(e, "swap: eps/theta differ");
        return -1;
    }
    uint64_t wsize;
    uint32_t nten;
    memcpy(&wsize, d + 180, 8);
    memcpy(&nten, d + 188, 4);
    if (wsize != e->weights_size || nten != e->n_tensors) {
        free(d);
        fail(e, "swap: sizes differ");
        return -1;
    }
    uint32_t rec0 = 192;
    if (q35) {
        uint32_t x[8];
        memcpy(x, d + 192, 32);
        if (x[0] != e->n_kh || x[1] != e->n_vh || x[2] != e->dk || x[3] != e->dv || x[4] != e->rope_dim ||
            x[5] != e->full_mask || x[6] != e->im_start || (x[7] & 1) != (uint32_t)e->has_head ||
            ((x[7] >> 1) & 1) != (uint32_t)e->cpu_embed || ((x[7] >> 2) & 1) != (uint32_t)(e->ids != NULL)) {
            free(d);
            fail(e, "swap: qwen3.5 block differs");
            return -1;
        }
        rec0 = 224;
    }
    for (uint32_t i = 0; i < nten; i++) {
        const uint8_t* r = d + rec0 + i * 36;
        const Tensor* t = &e->t[i];
        uint64_t v1, v2, v3;
        uint32_t v4, v5, v6;
        memcpy(&v1, r, 8);
        memcpy(&v2, r + 8, 8);
        memcpy(&v4, r + 16, 4);
        memcpy(&v5, r + 20, 4);
        memcpy(&v3, r + 24, 8);
        memcpy(&v6, r + 32, 4);
        if (v1 != t->off || v2 != t->size || v4 != t->rows || v5 != t->cols || v3 != t->scales || v6 != t->fmt) {
            free(d);
            fail(e, "swap: record %u differs", i);
            return -1;
        }
    }
    char wp[129];
    memcpy(wp, d + 52, 128);
    wp[128] = 0;
    free(d);
    weights_locate(wp);   // same resolution as the load (package first), and it logs "weights: %s"
    {   // the file must be whole before anything is overwritten
        int f = open(wp, O_RDONLY);
        if (f < 0) {
            fail(e, "swap: cannot open %s", wp);
            return -1;
        }
        off_t n = lseek(f, 0, SEEK_END);
        close(f);
        if (n < 0 || (uint64_t)n != e->weights_size) {
            fail(e, "swap: weights file size wrong (%lld)", (long long)n);
            return -1;
        }
    }
    // ---- point of no return: the model block takes the new weights ----
    if (e->fd >= 0) {
        close(e->fd);
        e->fd = -1;
    }
    int f = open(wp, O_RDONLY);
    uint64_t done = 0, chunk = 32ull << 20;
    while (done < e->weights_size) {
        uint64_t n = e->weights_size - done < chunk ? e->weights_size - done : chunk;
        ssize_t r = read(f, (uint8_t*)(uintptr_t)e->model + done, n);
        if (r <= 0) break;
        done += r;
    }
    close(f);
    if (done != e->weights_size) {
        fail(e, "swap incomplete (%llu of %llu bytes)", (unsigned long long)done, (unsigned long long)e->weights_size);
        engine_reset(e);
        return -1;
    }
    if (e->head_idx && e->head_idx != e->n_tensors - 1 && e->ids) {   // the row -> token id table
        const Tensor* t = &e->t[e->n_tensors - 1];
        int fi = open(wp, O_RDONLY);
        int ok = fi >= 0 && pread(fi, e->ids, (size_t)t->rows * 4, (off_t)t->off) == (ssize_t)t->rows * 4;
        if (fi >= 0) close(fi);
        if (!ok) {
            fail(e, "swap: cannot read the vocabulary table");
            engine_reset(e);
            return -1;
        }
    }
    if (e->cpu_embed) {   // the embedding rows: reopen the file, forget the cached rows
        e->fd = open(wp, O_RDONLY);
        if (e->fd < 0) {
            fail(e, "swap: cannot reopen for the embedding rows");
            engine_reset(e);
            return -1;
        }
        for (int i = 0; i < ROW_CACHE; i++) e->row_tag[i] = -1;
    }
    memcpy(e->weights_path, wp, 129);
    snprintf(e->model_name, sizeof(e->model_name), "%s", name);
    engine_reset(e);
    say("model swapped: %s (the conversation starts fresh)", name);
    return 0;
}

#define TQ_TABLE \\""")

# ---------------------------------------------------------------- main.c: accessors + evidence name
edit("chat/main.c",
"""static Engine eng;""",
"""static Engine eng;
static volatile int api_break;   // set by a model swap: the next API request starts the conversation fresh""")

edit("chat/main.c",
"""static int api_busy;        // the model is answering an API request (api_run.inc)""",
"""static int api_busy;        // the model is answering an API request (api_run.inc)
Engine* engine_ptr(void) { return &eng; }
int api_is_busy(void) { return api_busy; }
void api_break_set(void) { api_break = 1; }""")

edit("chat/main.c",
"""    api_set_model(MODEL_NAME);""",
"""    api_set_model(eng.model_name[0] ? eng.model_name : MODEL_NAME);   // the boot's pick, if garlic_model.txt chose one""")

# ---------------------------------------------------------------- api_run.inc: the forced fresh start
edit("chat/api_run.inc",
"""    int cont = api_pos > 0 && api_pos == pos &&""",
"""    int cont = !api_break && api_pos > 0 && api_pos == pos &&""")

edit("chat/api_run.inc",
"""    if (!cont) engine_reset(&eng);   // (continuing: the Gated DeltaNet states stay)""",
"""    if (!cont) engine_reset(&eng);   // (continuing: the Gated DeltaNet states stay)
    api_break = 0;   // a swap's forced fresh start is consumed by this processed request""")

# ---------------------------------------------------------------- api.c: /v1/models list + POST /v1/model
edit("chat/api.c",
"""#include "api.h"

#include <stdint.h>""",
"""#include "api.h"
#include "engine.h"   // the Engine type, for the model swap endpoint

#include <stdint.h>""")

edit("chat/api.c",
"""static void handle(int fd) {""",
"""// The models this engine can serve now: /data/garlic.manifest.txt lines "<name> [idx path]"; a name is
// listed only while its .idx is on the HDD. The loaded one is named in "loaded" (paired with the log's
// weights: line — the id alone is never evidence).
static void models_json(Sb* out) {
    const char* loaded = engine_ptr() ? engine_ptr()->model_name : "";
    if (!loaded[0]) loaded = api_model();
    char names[24][64];
    int nn = 0, li = -1;
    size_t fs;
    char* mf = (char*)read_all("/data/garlic.manifest.txt", &fs);
    if (mf && fs) {
        char* p = mf;
        while ((size_t)(p - mf) < fs && nn < 24) {
            char* q = p;
            while ((size_t)(q - mf) < fs && *q != '\\n') q++;
            char* sp = p;
            while (sp < q && *sp != ' ') sp++;
            size_t nlen = (size_t)(sp - p);
            if (nlen > 0 && nlen < 64) {
                char nm[64];
                memcpy(nm, p, nlen);
                nm[nlen] = 0;
                int bad = 0;
                for (size_t i = 0; i < nlen; i++)
                    if (nm[i] <= 32 || nm[i] == '/') bad = 1;
                if (!bad) {
                    char idxp[256];
                    char* path2 = sp < q ? sp + 1 : NULL;
                    while (path2 && (*path2 == ' ' || *path2 == '\\r')) path2++;
                    if (path2 && *path2 == '/' && (size_t)(q - path2) < 250) {
                        size_t l = (size_t)(q - path2);
                        while (l && (path2[l - 1] == ' ' || path2[l - 1] == '\\r')) l--;
                        memcpy(idxp, path2, l);
                        idxp[l] = 0;
                    } else snprintf(idxp, sizeof(idxp), "/data/%s.idx", nm);
                    int vf = open(idxp, O_RDONLY);
                    if (vf >= 0) {
                        close(vf);
                        snprintf(names[nn], 64, "%s", nm);
                        if (!strcmp(names[nn], loaded)) li = nn;
                        nn++;
                    }
                }
            }
            p = q + 1;
        }
        free(mf);
    }
    sb_str(out, "{\\"object\\":\\"list\\",\\"data\\":[");
    for (int i = 0; i < nn; i++) {
        if (i) sb_str(out, ",");
        sb_str(out, "{\\"id\\":\\"");
        sb_json_str(out, names[i], strlen(names[i]));
        sb_str(out, "\\",\\"object\\":\\"model\\"}");
    }
    if (li < 0) {   // the loaded model belongs in its own list
        if (nn) sb_str(out, ",");
        sb_str(out, "{\\"id\\":\\"");
        sb_json_str(out, loaded, strlen(loaded));
        sb_str(out, "\\",\\"object\\":\\"model\\"}");
    }
    sb_str(out, "],\\"loaded\\":");
    sb_json_str(out, loaded, strlen(loaded));
    sb_str(out, "}");
}

// {"name":"X"} from a request body into out (64 bytes); 1 = well formed, printable, 1..63 chars.
static int swap_name(HttpRequest* req, char* out) {
    if (!req->body || req->body_len < 8) return 0;
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    int ok = 0;
    if (pool) {
        int root = json_parse(req->body, req->body_len, pool, API_MAX_NODES);
        if (root >= 0) {
            int nm = json_get(pool, root, "name");
            if (nm >= 0) {
                Sb o;
                sb_init(&o);
                json_str_sb(pool, nm, &o);
                if (!o.oom && o.n > 0 && o.n < 64) {
                    int bad = 0;
                    for (size_t i = 0; i < o.n; i++) {
                        unsigned char c = (unsigned char)o.p[i];
                        int good = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                                   c == '.' || c == '_' || c == '-';
                        if (!good) bad = 1;
                    }
                    if (!bad) {
                        memcpy(out, o.p, o.n);
                        out[o.n] = 0;
                        ok = 1;
                    }
                }
                sb_free(&o);
            }
        }
        free(pool);
    }
    return ok;
}

static void handle(int fd) {""")

edit("chat/api.c",
"""        api_json_models(&s, model_name, 0);""",
"""        models_json(&s);""")

edit("chat/api.c",
"""        return;   // (serve_chat keeps the connection until the main loop has written the reply)
    } else {""",
"""        return;   // (serve_chat keeps the connection until the main loop has written the reply)
    } else if (!strcmp(req.method, "POST") && !strcmp(req.path, "/v1/model")) {
        char name[64];
        if (swap_name(&req, name)) {
            if (api_is_busy()) {
                reply_json_error(fd, 409, "Conflict", "The model is answering a request; swap when it is idle.",
                                 "invalid_request_error", "model_busy");
            } else {
                Engine* e = engine_ptr();
                if (!engine_swap(e, name)) {
                    api_break_set();
                    api_set_model(name);
                    char body[128];
                    int bn = snprintf(body, sizeof(body), "{\\"object\\":\\"model\\",\\"id\\":\\"%s\\",\\"swapped\\":true}",
                                      name);
                    reply(fd, 200, "OK", "application/json", body, (size_t)bn);
                } else {
                    reply_json_error(fd, 500, "Internal Server Error", e->error, "invalid_request_error",
                                     "swap_failed");
                }
            }
        } else {
            reply_json_error(fd, 400, "Bad Request", "Missing or malformed \\"name\\".", "invalid_request_error",
                             "invalid_name");
        }
    } else {""")

fail = 0
for path, old, new in EDITS:
    p = ROOT / path
    txt = p.read_text()
    n = txt.count(old)
    if n != 1:
        print(f"ANCHOR {path} count={n} — abort")
        fail = 1
if fail:
    sys.exit(1)
for path, old, new in EDITS:
    p = ROOT / path
    p.write_text(p.read_text().replace(old, new, 1))
    print(f"patched {path}")
print("model-dir build 1.21: manifest scan + boot pick + hot-swap + /v1/models list OK")
