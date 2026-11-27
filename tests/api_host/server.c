// PC test server: the console's OpenAI-compatible API on Windows, for real HTTP clients. chat/api.c is the server as
// it is (its own thread, on a real socket; the console's calls are stand-ins: orbis/*.h, host.c), chat/api_run.inc the
// job logic as main.c includes it, on a mock model instead of the engine. The mock reads the prompt and writes a
// reply (see brain): it greets, reasons when asked, calls the tools it is given, reads their results back, and writes
// long replies slowly enough for a client to leave in the middle or to ask /health meanwhile. The tokens are the
// app's own (chat/tokenizer.c with chat/assets/tokenizer.bin), so a prompt takes as much of the context as on the
// console. It listens on 127.0.0.1 only. test.py talks to it.
//
// Build (from the repository's root, with zig cc as for the other tests):
//   zig cc -std=gnu99 -Wall -Wextra -g -DAPI_PORT=18080 -I tests/api_host -I chat tests/api_host/server.c chat/api.c chat/api_proto.c chat/tokenizer.c tests/api_host/host.c -lws2_32 -o build/api_server.exe
// Run: build/api_server.exe [--key KEY] [--delay MS] [--no-spec] [--assets chat/assets]
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "api.h"
#include "api_proto.h"
#include "host.h"
#include "tokenizer.h"

// ---- what main.c gives api_run.inc ----------------------------------------------------------------------------------
static uint32_t im_start;         // <|im_start|> in the tokenizer's vocabulary; the others follow it, as in Qwen3.5's
#define IM_START im_start
#define IM_END (im_start + 1)
#define THINK (im_start + 23)
#define THINK_END (im_start + 24)
#define EOS IM_END                // (the mock ends every reply with <|im_end|>)
#define EOS2 IM_END

static struct {
    int qwen35;
    uint32_t ctx_limit, max_seq, eos, eos2;
    char error[160];
    struct {
        uint64_t ws_begin, ws_end;   // room for the kept state of a prompt's beginning
    } asr;
} eng = {1, 16384, 16384, 0, 0, "", {0, 1 << 20}};
static Tokenizer tok;
static int api_snap_n, pos, need_close, hist_n, hist_done, thinking, think_turn, sys_pre;
static unsigned seed = 1;
static uint32_t next_token;
static struct {
    int scroll;
} view;
static int think_on, think_cap = 1024;   // the console's reasoning setting (off) and its cap
#define THINK_T 0.7f
#define THINK_CLOSE "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
#define GARLIC_SPEC 1             // (as the 9B app; --no-spec: one token per step)
static int spec_on = 1;
static int delay_ms = 10;         // the time of one step of the reply
static const char* api_key;       // --key: what /data/garlic_api_key.txt holds
static const char* assets = "chat/assets";

void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("%10.3f ", host_now());
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    fflush(stdout);
}

uint8_t* read_all(const char* path, size_t* size) {
    if (strstr(path, "garlic_api_key.txt")) {
        if (!api_key) return NULL;
        *size = strlen(api_key);
        uint8_t* k = malloc(*size + 1);
        if (k) memcpy(k, api_key, *size + 1);
        return k;
    }
    char p[512];
    const char* name = strrchr(path, '/');
    snprintf(p, sizeof(p), "%s%s", assets, name ? name : "/");
    FILE* f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* d = malloc((size_t)n + 1);
    if (d && fread(d, 1, (size_t)n, f) != (size_t)n) {
        free(d);
        d = NULL;
    }
    fclose(f);
    if (d) d[n] = 0, *size = (size_t)n;
    return d;
}

static double now(void) { return host_now(); }

static int encode_into(uint32_t* ids, int n, int max, const char* text) {   // (as main.c)
    return n + tok_encode(&tok, text, ids + n, max - n);
}

// ---- the mock model -------------------------------------------------------------------------------------------------
static uint32_t* script;          // the reply: script[k], then IM_END
static int script_len, script_cap, script_k, prompt_phase, steps;

static uint64_t engine_state_size(const void* e) {
    (void)e;
    return 4096;
}
static int engine_state_copy(void* e, uint64_t buf, int save) {
    (void)e;
    (void)buf;
    (void)save;
    return 0;
}
static void engine_reset(void* e) { (void)e; }

static void script_add(uint32_t id) {
    if (script_len == script_cap) {
        script_cap = script_cap ? script_cap * 2 : 4096;
        script = realloc(script, sizeof(uint32_t) * (size_t)script_cap);
    }
    script[script_len++] = id;
}

// text into the reply, with <tool_call>, </tool_call> and </think> as the model writes them (special tokens)
static void script_text(const char* s) {
    const struct {
        const char* tag;
        uint32_t id;
    } tags[] = {{"<tool_call>", IM_START + 13}, {"</tool_call>", IM_START + 14}, {"</think>", THINK_END}};
    while (*s) {
        const char* best = NULL;
        uint32_t id = 0;
        size_t tl = 0;
        for (int k = 0; k < 3; k++) {
            const char* h = strstr(s, tags[k].tag);
            if (h && (!best || h < best)) best = h, id = tags[k].id, tl = strlen(tags[k].tag);
        }
        size_t plain = best ? (size_t)(best - s) : strlen(s);
        char* piece = malloc(plain + 1);
        uint32_t* ids = malloc(sizeof(uint32_t) * (plain + 1));
        memcpy(piece, s, plain);
        piece[plain] = 0;
        int k = tok_encode(&tok, piece, ids, (int)plain + 1);
        for (int i = 0; i < k; i++) script_add(ids[i]);
        free(ids);
        free(piece);
        if (!best) break;
        script_add(id);
        s = best + tl;
    }
}

static void brain(int n);   // (below: it reads api_ids, which api_run.inc defines)

static int engine_run(void* e, const uint32_t* tokens, int n, int pos0, float temperature, unsigned sd, uint32_t* next) {
    (void)e;
    (void)tokens;
    (void)temperature;
    (void)sd;
    if (pos0 + n > (int)eng.ctx_limit) {
        snprintf(eng.error, sizeof(eng.error), "mock engine: position %d past the context", pos0 + n);
        return -1;
    }
    if (prompt_phase) {   // (the last piece of the prompt draws the reply's first token)
        brain(pos0 + n);
        script_k = 0;
        *next = script_len ? script[0] : IM_END;
        return 0;
    }
    steps++;
    host_sleep_ms(delay_ms);
    *next = script_k < script_len ? script[script_k++] : IM_END;
    return 0;
}

static int spec_active(const void* e) {
    (void)e;
    return spec_on;
}

// as engine_spec.inc: tok goes in at pos0, up to 2 drafted tokens are kept (fewer than max_new), the next is drawn
static int engine_spec_step(void* e, uint32_t tk, int pos0, float temperature, unsigned sd, int max_new, uint32_t* out,
                            int* n_out, uint32_t* next) {
    (void)e;
    (void)tk;
    (void)temperature;
    (void)sd;
    int k = 0;
    while (k < 2 && k < max_new - 1 && script_k < script_len) out[k++] = script[script_k++];
    if (pos0 + 1 + k > (int)eng.ctx_limit) {
        snprintf(eng.error, sizeof(eng.error), "mock engine: position %d past the context", pos0 + 1 + k);
        return -1;
    }
    *n_out = k;
    steps++;
    host_sleep_ms(delay_ms);
    *next = script_k < script_len ? script[script_k++] : IM_END;
    return 0;
}

// what main.c shows of a request on the screen: here, the log
static double job_t0;
static void api_chat_begin(void) {
    prompt_phase = 1, steps = 0, job_t0 = now();
}
static void api_chat_reply(void) {
    prompt_phase = 0;
    if (script_k < script_len) script_k++;   // (the prompt drew script[0])
}
static int api_chat_step(int produced) {
    (void)produced;
    return 0;   // (no Circle on a PC)
}
static void api_chat_end(void) { say("job: %d steps in %.2f s", steps, now() - job_t0); }

#include "api_run.inc"

// ---- the mock model's replies ---------------------------------------------------------------------------------------
// The prompt as text, its special tokens written out.
static void render(int n, Sb* out) {
    static const char* names[] = {"<|im_start|>", "<|im_end|>", "<think>", "</think>"};
    for (int i = 0; i < n; i++) {
        uint32_t t = api_ids[i];
        const uint8_t* b;
        if (t == IM_START || t == IM_END) sb_str(out, names[t - IM_START]);
        else if (t == THINK || t == THINK_END) sb_str(out, names[2 + t - THINK]);
        else if (t == IM_START + 13) sb_str(out, "<tool_call>");
        else if (t == IM_START + 14) sb_str(out, "</tool_call>");
        else if (t == IM_START + 21) sb_str(out, "<tool_response>");
        else if (t == IM_START + 22) sb_str(out, "</tool_response>");
        else {
            int k = tok_piece(&tok, t, &b);
            if (k > 0) sb_add(out, (const char*)b, (size_t)k);
        }
    }
}

static const char* rfind(const char* s, const char* end, const char* needle) {
    size_t n = strlen(needle);
    for (const char* p = end - n; p >= s; p--)
        if (!memcmp(p, needle, n)) return p;
    return NULL;
}

// The call of function name with the tools of the prompt (the JSON lines between <tools> and </tools>): each required
// parameter (or the first) with a value of its type.
static void call_text(const char* tools, const char* name, int with_name, Sb* out) {
    if (with_name) {
        sb_str(out, name);
        sb_str(out, ">\n");
    }
    const char* line = tools;
    while (line && *line && strncmp(line, "</tools>", 8)) {
        const char* eol = strchr(line, '\n');
        size_t len = eol ? (size_t)(eol - line) : strlen(line);
        int root;
        JNode* pool = json_parse_grow(line, len, 256, &root);
        int fn = pool ? json_get(pool, root, "function") : -1;
        if (pool && fn < 0) fn = root;
        int nm = fn >= 0 ? json_get(pool, fn, "name") : -1;
        if (nm >= 0 && pool[nm].slen == strlen(name) && !memcmp(pool[nm].s, name, pool[nm].slen)) {
            int params = json_get(pool, fn, "parameters");
            int props = params >= 0 ? json_get(pool, params, "properties") : -1;
            int req = params >= 0 ? json_get(pool, params, "required") : -1;
            for (int p = props >= 0 ? pool[props].first : -1; p >= 0; p = pool[p].next) {
                int needed = p == pool[props].first && (req < 0 || pool[req].count == 0);
                for (int r = req >= 0 ? pool[req].first : -1; r >= 0; r = pool[r].next)
                    if (pool[r].slen == pool[p].klen && !memcmp(pool[r].s, pool[p].key, pool[p].klen)) needed = 1;
                if (!needed) continue;
                int tp = json_get(pool, p, "type"), en = json_get(pool, p, "enum");
                const char* type = tp >= 0 ? pool[tp].s : "string";
                sb_str(out, "<parameter=");
                sb_add(out, pool[p].key, pool[p].klen);
                sb_str(out, ">\n");
                if (en >= 0 && pool[en].first >= 0) json_str_sb(pool, pool[en].first, out);
                else if (!strncmp(type, "integer", 7) || !strncmp(type, "number", 6)) sb_str(out, "3");
                else if (!strncmp(type, "boolean", 7)) sb_str(out, "true");
                else if (!strncmp(type, "array", 5)) sb_str(out, "[\"a\"]");
                else if (!strncmp(type, "object", 6)) sb_str(out, "{}");
                else sb_str(out, "Lima");
                sb_str(out, "\n</parameter>\n");
            }
            free(pool);
            break;
        }
        free(pool);
        line = eol ? eol + 1 : NULL;
    }
    sb_str(out, "</function>\n</tool_call>");
}

// the name of the n-th tool (0 first) in the prompt's tool lines, into name
static int tool_name(const char* tools, int n, char* name, size_t cap) {
    for (const char* line = tools; line && *line && strncmp(line, "</tools>", 8); n--) {
        const char* f = strstr(line, "\"name\": \"");
        const char* eol = strchr(line, '\n');
        if (n == 0 && f && (!eol || f < eol)) {
            f += 9;
            const char* q = strchr(f, '"');
            snprintf(name, cap, "%.*s", (int)(q - f), f);
            return 1;
        }
        line = eol ? eol + 1 : NULL;
    }
    return 0;
}

// What the mock model answers the prompt (the n ids of api_ids).
static void brain(int n) {
    Sb p, out;
    sb_init(&p);
    sb_init(&out);
    render(n, &p);
    script_len = 0;
    const char* end = p.p + p.n;
    const char* turn = rfind(p.p, end, "<|im_start|>assistant\n");
    const char* tail = turn ? turn + 22 : end;
    const char* user = rfind(p.p, turn ? turn : end, "<|im_start|>user\n");
    const char* uend = user ? strstr(user, "<|im_end|>") : NULL;
    Sb said;
    sb_init(&said);
    if (user && uend) sb_add(&said, user + 17, (size_t)(uend - user - 17));
    const char* tools = strstr(p.p, "<tools>\n");
    if (tools) tools += 8;
    char first[128] = "", second[128] = "";
    if (tools) tool_name(tools, 0, first, sizeof(first));
    if (tools && !tool_name(tools, 1, second, sizeof(second))) strcpy(second, first);
    if (!strncmp(tail, "<think>\n", 8) && strncmp(tail, "<think>\n\n</think>", 17))
        sb_str(&out, "The user wants a short answer.</think>\n\n");
    const char* forced = strstr(tail, "<tool_call>\n<function=");
    const char* resp = said.p ? strstr(said.p, "<tool_response>\n") : NULL;
    const char* lng = said.p ? strstr(said.p, "long ") : NULL;
    if (forced) {   // tool_choice: the call has begun; a named one with its name
        const char* nm = forced + 22;
        if (nm < end && strchr(nm, '>')) {
            char name[128];
            snprintf(name, sizeof(name), "%.*s", (int)(strchr(nm, '>') - nm), nm);
            call_text(tools, name, 0, &out);
        } else {
            call_text(tools, first, 1, &out);
        }
    } else if (resp) {   // the results of the calls: the last one read back
        const char* last = resp;
        for (const char* r = resp; (r = strstr(r + 1, "<tool_response>\n")) != NULL;) last = r;
        const char* rend = strstr(last, "\n</tool_response>");
        sb_str(&out, "The tool said: ");
        sb_add(&out, last + 16, rend ? (size_t)(rend - last - 16) : 0);
        sb_str(&out, ".");
    } else if (tools && said.p && (strstr(said.p, "call") || strstr(said.p, "weather"))) {
        sb_str(&out, "Let me check.\n\n<tool_call>\n<function=");
        call_text(tools, first, 1, &out);
        if (strstr(said.p, "two calls")) {
            sb_str(&out, "\n<tool_call>\n<function=");
            call_text(tools, second, 1, &out);
        }
    } else if (lng) {   // "long N": N numbered words
        int words = atoi(lng + 5);
        char w[24];
        for (int i = 1; i <= words; i++) {
            snprintf(w, sizeof(w), i % 12 ? "word%d " : "word%d\n", i);
            sb_str(&out, w);
        }
        sb_str(&out, "The end.");
    } else {
        sb_str(&out, "Hello! You said: ");
        if (said.p) sb_str(&out, said.p);
    }
    if (out.p) script_text(out.p);
    sb_free(&said);
    sb_free(&out);
    sb_free(&p);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--key") && i + 1 < argc) api_key = argv[++i];
        else if (!strcmp(argv[i], "--delay") && i + 1 < argc) delay_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--assets") && i + 1 < argc) assets = argv[++i];
        else if (!strcmp(argv[i], "--no-spec")) spec_on = 0;
        else {
            fprintf(stderr, "usage: %s [--key KEY] [--delay MS] [--no-spec] [--assets DIR]\n", argv[0]);
            return 2;
        }
    }
    char path[512];
    size_t size = 0;
    snprintf(path, sizeof(path), "%s/tokenizer.bin", assets);
    uint8_t* data = read_all(path, &size);
    if (!data || tok_load(&tok, data, size)) {
        fprintf(stderr, "cannot read %s (run from the repository's root, or give --assets)\n", path);
        return 1;
    }
    for (uint32_t i = 0; i < tok.n_vocab && !im_start; i++)
        if (tok.special[i] && tok.raw_len[i] == 12 && !memcmp(tok.raw[i], "<|im_start|>", 12)) im_start = i;
    static const struct {
        uint32_t off;
        const char* text;
    } specials[] = {{1, "<|im_end|>"}, {13, "<tool_call>"}, {14, "</tool_call>"}, {21, "<tool_response>"},
                    {22, "</tool_response>"}, {23, "<think>"}, {24, "</think>"}};
    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        uint32_t id = im_start + specials[i].off;
        size_t n = strlen(specials[i].text);
        if (!im_start || id >= tok.n_vocab || tok.raw_len[id] != n || memcmp(tok.raw[id], specials[i].text, n)) {
            fprintf(stderr, "%s is not where api_run.inc expects it in %s\n", specials[i].text, path);
            return 1;
        }
    }
    eng.eos = EOS, eng.eos2 = EOS2;
    api_set_model("Qwen3.5-9B");
    api_set_context((int)eng.max_seq);
    if (api_enable(1)) return 1;
    say("test server on http://127.0.0.1:%d/v1 (a step every %d ms, %s)", API_PORT, delay_ms,
        spec_on ? "up to 3 tokens a step" : "one token a step");
    for (;;) {
        ApiJob* job = api_take();
        if (!job) {
            host_sleep_ms(2);
            continue;
        }
        api_job(job);
        api_done(job);
    }
}
