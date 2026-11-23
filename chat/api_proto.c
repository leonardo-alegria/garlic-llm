// OpenAI-compatible HTTP API, protocol part (see api_proto.h).
#include "api_proto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- growing string -----------------------------------------------------------------------------------------------

void sb_init(Sb* s) {
    s->p = NULL;
    s->n = s->cap = 0;
    s->oom = 0;
}

void sb_free(Sb* s) {
    free(s->p);
    sb_init(s);
}

void sb_add(Sb* s, const char* p, size_t n) {
    if (s->oom || n == 0) return;
    if (s->n + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap : 256;
        while (cap < s->n + n + 1) cap *= 2;
        char* q = realloc(s->p, cap);
        if (!q) {
            s->oom = 1;
            return;
        }
        s->p = q;
        s->cap = cap;
    }
    memcpy(s->p + s->n, p, n);
    s->n += n;
    s->p[s->n] = 0;
}

void sb_str(Sb* s, const char* z) { sb_add(s, z, strlen(z)); }

void sb_json_str(Sb* s, const char* p, size_t n) {
    sb_add(s, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)p[i];
        switch (c) {
        case '"': sb_add(s, "\\\"", 2); break;
        case '\\': sb_add(s, "\\\\", 2); break;
        case '\n': sb_add(s, "\\n", 2); break;
        case '\r': sb_add(s, "\\r", 2); break;
        case '\t': sb_add(s, "\\t", 2); break;
        default:
            if (c < 0x20) {
                char esc[8];
                snprintf(esc, sizeof(esc), "\\u%04x", c);
                sb_add(s, esc, 6);
            } else {
                sb_add(s, (const char*)&p[i], 1);   // UTF-8 bytes pass through
            }
        }
    }
    sb_add(s, "\"", 1);
}

void sb_int(Sb* s, long long v) {
    char b[32];
    snprintf(b, sizeof(b), "%lld", v);
    sb_str(s, b);
}

// ---- HTTP request -------------------------------------------------------------------------------------------------

static int ci_eq(const char* a, size_t n, const char* b) {
    size_t m = strlen(b);
    if (n != m) return 0;
    for (size_t i = 0; i < n; i++) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x += 32;
        if (x != y) return 0;
    }
    return 1;
}

int api_http_parse(const char* buf, size_t len, HttpRequest* req) {
    memset(req, 0, sizeof(*req));
    size_t head = 0, term = 4;
    for (size_t i = 0; i + 1 < len; i++) {
        if (i + 3 < len && !memcmp(buf + i, "\r\n\r\n", 4)) {
            head = i + 4;
            break;
        }
        if (!memcmp(buf + i, "\n\n", 2)) {
            head = i + 2;
            term = 2;
            break;
        }
    }
    if (!head) return len > 16384 ? -1 : 0;
    (void)term;
    // request line: METHOD SP target SP HTTP/1.x
    const char* p = buf;
    const char* end = buf + head;
    const char* eol = memchr(p, '\n', end - p);
    if (!eol) return -1;
    const char* sp1 = memchr(p, ' ', eol - p);
    if (!sp1 || sp1 - p >= (long)sizeof(req->method)) return -1;
    memcpy(req->method, p, sp1 - p);
    const char* sp2 = memchr(sp1 + 1, ' ', eol - sp1 - 1);
    if (!sp2) return -1;
    size_t plen = sp2 - (sp1 + 1);
    const char* q = memchr(sp1 + 1, '?', plen);
    if (q) plen = q - (sp1 + 1);
    if (plen == 0 || plen >= sizeof(req->path)) return -1;
    memcpy(req->path, sp1 + 1, plen);
    req->path[plen] = 0;
    if (memchr(sp2 + 1, 'H', eol - sp2 - 1) == NULL) return -1;
    // headers
    p = eol + 1;
    while (p < end) {
        eol = memchr(p, '\n', end - p);
        if (!eol) break;
        const char* line_end = eol;
        if (line_end > p && line_end[-1] == '\r') line_end--;
        if (line_end == p) break;
        const char* colon = memchr(p, ':', line_end - p);
        if (colon) {
            const char* v = colon + 1;
            while (v < line_end && (*v == ' ' || *v == '\t')) v++;
            size_t vlen = line_end - v;
            if (ci_eq(p, colon - p, "content-length")) {
                char num[24];
                if (vlen == 0 || vlen >= sizeof(num)) return -1;
                memcpy(num, v, vlen);
                num[vlen] = 0;
                char* ep;
                long cl = strtol(num, &ep, 10);
                if (*ep || cl < 0 || cl > 1 << 20) return -1;
                req->content_length = cl;
            } else if (ci_eq(p, colon - p, "authorization")) {
                size_t k = vlen < sizeof(req->auth) - 1 ? vlen : sizeof(req->auth) - 1;
                memcpy(req->auth, v, k);
                req->auth[k] = 0;
            }
        }
        p = eol + 1;
    }
    req->head_len = head;
    if (len < head + (size_t)req->content_length) return 0;
    req->body = buf + head;
    req->body_len = (size_t)req->content_length;
    return 1;
}

// ---- JSON reader --------------------------------------------------------------------------------------------------

typedef struct {
    const char* s;
    size_t n, i;
    JNode* pool;
    int max, used, depth;
} JP;

static void jp_ws(JP* P) {
    while (P->i < P->n && (P->s[P->i] == ' ' || P->s[P->i] == '\t' || P->s[P->i] == '\n' || P->s[P->i] == '\r')) P->i++;
}

static int jp_new(JP* P) {
    if (P->used >= P->max) return -1;
    JNode* d = &P->pool[P->used];
    memset(d, 0, sizeof(*d));
    d->first = d->next = -1;
    return P->used++;
}

// raw string at P->i (on the opening quote): sets node s/slen, moves past the closing quote
static int jp_string(JP* P, int node, const char** s, size_t* len) {
    if (P->i >= P->n || P->s[P->i] != '"') return -1;
    P->i++;
    size_t start = P->i;
    while (P->i < P->n) {
        char c = P->s[P->i];
        if (c == '\\') {
            P->i += 2;
            continue;
        }
        if (c == '"') {
            *s = P->s + start;
            *len = P->i - start;
            P->i++;
            (void)node;
            return 0;
        }
        P->i++;
    }
    return -1;
}

static int jp_value(JP* P, int* out);

static int jp_value(JP* P, int* out) {
    jp_ws(P);
    if (P->i >= P->n || ++P->depth > 64) return -1;
    int r = -1;
    char c = P->s[P->i];
    int idx = jp_new(P);
    if (idx < 0) goto done;
    JNode* d = &P->pool[idx];
    if (c == '{' || c == '[') {
        int obj = c == '{';
        d->type = obj ? J_OBJ : J_ARR;
        P->i++;
        jp_ws(P);
        int last = -1;
        if (P->i < P->n && P->s[P->i] == (obj ? '}' : ']')) {
            P->i++;
            r = 0;
            goto done_ok;
        }
        for (;;) {
            const char* key = NULL;
            size_t klen = 0;
            if (obj) {
                jp_ws(P);
                if (jp_string(P, idx, &key, &klen)) goto done;
                jp_ws(P);
                if (P->i >= P->n || P->s[P->i] != ':') goto done;
                P->i++;
            }
            int child;
            if (jp_value(P, &child)) goto done;
            P->pool[child].key = key;
            P->pool[child].klen = klen;
            if (last < 0) P->pool[idx].first = child;
            else P->pool[last].next = child;
            last = child;
            P->pool[idx].count++;
            jp_ws(P);
            if (P->i >= P->n) goto done;
            if (P->s[P->i] == ',') {
                P->i++;
                continue;
            }
            if (P->s[P->i] == (obj ? '}' : ']')) {
                P->i++;
                break;
            }
            goto done;
        }
    done_ok:
        r = 0;
    } else if (c == '"') {
        d->type = J_STR;
        if (jp_string(P, idx, &d->s, &d->slen)) goto done;
        r = 0;
    } else if (c == 't' || c == 'f' || c == 'n') {
        const char* lit = c == 't' ? "true" : c == 'f' ? "false" : "null";
        size_t m = strlen(lit);
        if (P->n - P->i < m || memcmp(P->s + P->i, lit, m)) goto done;
        P->i += m;
        d->type = c == 'n' ? J_NULL : J_BOOL;
        d->b = c == 't';
        r = 0;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        size_t start = P->i;
        while (P->i < P->n && strchr("+-0123456789.eE", P->s[P->i])) P->i++;
        size_t m = P->i - start;
        if (m == 0 || m > 40) goto done;
        char num[48];
        memcpy(num, P->s + start, m);
        num[m] = 0;
        char* ep;
        d->num = strtod(num, &ep);
        if (*ep) goto done;
        d->type = J_NUM;
        r = 0;
    }
done:
    if (r == 0) *out = idx;
    P->depth--;
    return r;
}

int json_parse(const char* s, size_t n, JNode* pool, int max) {
    JP P = {s, n, 0, pool, max, 0, 0};
    int root;
    if (jp_value(&P, &root)) return -1;
    jp_ws(&P);
    if (P.i != n) return -1;
    return root;
}

int json_get(const JNode* pool, int obj, const char* key) {
    if (obj < 0 || pool[obj].type != J_OBJ) return -1;
    size_t m = strlen(key);
    for (int c = pool[obj].first; c >= 0; c = pool[c].next)
        if (pool[c].klen == m && !memcmp(pool[c].key, key, m)) return c;
    return -1;
}

static int hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void utf8_put(Sb* out, uint32_t cp) {
    char b[4];
    if (cp < 0x80) {
        b[0] = (char)cp;
        sb_add(out, b, 1);
    } else if (cp < 0x800) {
        b[0] = (char)(0xC0 | (cp >> 6));
        b[1] = (char)(0x80 | (cp & 0x3F));
        sb_add(out, b, 2);
    } else if (cp < 0x10000) {
        b[0] = (char)(0xE0 | (cp >> 12));
        b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[2] = (char)(0x80 | (cp & 0x3F));
        sb_add(out, b, 3);
    } else {
        b[0] = (char)(0xF0 | (cp >> 18));
        b[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        b[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[3] = (char)(0x80 | (cp & 0x3F));
        sb_add(out, b, 4);
    }
}

// the 4 hex digits at p (p[0..3]); -1 if not hex
static long hex4(const char* p, const char* end) {
    if (end - p < 4) return -1;
    long v = 0;
    for (int i = 0; i < 4; i++) {
        int h = hexv(p[i]);
        if (h < 0) return -1;
        v = v * 16 + h;
    }
    return v;
}

void json_str_sb(const JNode* pool, int node, Sb* out) {
    if (node < 0 || pool[node].type != J_STR) return;
    const char* p = pool[node].s;
    const char* end = p + pool[node].slen;
    while (p < end) {
        if (*p != '\\') {
            sb_add(out, p++, 1);
            continue;
        }
        p++;
        if (p >= end) break;
        char c = *p++;
        switch (c) {
        case 'n': sb_add(out, "\n", 1); break;
        case 't': sb_add(out, "\t", 1); break;
        case 'r': sb_add(out, "\r", 1); break;
        case 'b': sb_add(out, "\b", 1); break;
        case 'f': sb_add(out, "\f", 1); break;
        case 'u': {
            long cp = hex4(p, end);
            if (cp < 0) return;
            p += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                long lo = hex4(p + 2, end);
                if (lo >= 0xDC00 && lo < 0xE000) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                }
            }
            utf8_put(out, (uint32_t)cp);
            break;
        }
        default: sb_add(out, &c, 1); break;   // \" \\ \/
        }
    }
}

int api_message(const JNode* pool, int msg, char* role, size_t rcap, Sb* out) {
    if (msg < 0 || pool[msg].type != J_OBJ) return -1;
    int r = json_get(pool, msg, "role");
    if (r < 0 || pool[r].type != J_STR) return -1;
    Sb rs;
    sb_init(&rs);
    json_str_sb(pool, r, &rs);
    size_t k = rs.n < rcap - 1 ? rs.n : rcap - 1;
    if (rs.n) memcpy(role, rs.p, k);
    role[k] = 0;
    sb_free(&rs);
    int c = json_get(pool, msg, "content");
    if (c < 0 || pool[c].type == J_NULL) return 0;
    if (pool[c].type == J_STR) {
        json_str_sb(pool, c, out);
        return 0;
    }
    if (pool[c].type != J_ARR) return -1;
    for (int part = pool[c].first; part >= 0; part = pool[part].next) {
        if (pool[part].type != J_OBJ) return -1;
        int t = json_get(pool, part, "text");
        if (t >= 0 && pool[t].type == J_STR) json_str_sb(pool, t, out);
    }
    return 0;
}

// ---- JSON writers -------------------------------------------------------------------------------------------------

void api_json_models(Sb* out, const char* model, long long created) {
    sb_str(out, "{\"object\":\"list\",\"data\":[{\"id\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"object\":\"model\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"owned_by\":\"garlic-llm\"}]}");
}

static void api_json_tool_calls(Sb* out, const ApiToolCall* calls, int ncalls, unsigned call_seq, int ollama) {
    sb_str(out, ",\"tool_calls\":[");
    for (int i = 0; i < ncalls; i++) {
        if (i) sb_str(out, ",");
        if (!ollama) {
            char cid[64];
            snprintf(cid, sizeof(cid), "call_%u_%d", call_seq, i);
            sb_str(out, "{\"id\":");
            sb_json_str(out, cid, strlen(cid));
            sb_str(out, ",\"type\":\"function\",\"function\":{\"name\":");
        } else {
            sb_str(out, "{\"function\":{\"name\":");
        }
        sb_json_str(out, calls[i].name, strlen(calls[i].name));
        sb_str(out, ",\"arguments\":");
        if (ollama && calls[i].arguments[0] == '{') sb_str(out, calls[i].arguments);
        else sb_json_str(out, calls[i].arguments, strlen(calls[i].arguments));
        sb_str(out, "}}");
    }
    sb_str(out, "]");
}

void api_json_completion(Sb* out, const char* id, const char* model, long long created, const char* text, size_t tlen,
                         const char* finish, int prompt_tokens, int completion_tokens, const ApiToolCall* calls,
                         int ncalls, unsigned call_seq) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    if (ncalls > 0 && tlen == 0) sb_str(out, "null");
    else sb_json_str(out, text ? text : "", text ? tlen : 0);
    if (ncalls > 0 && calls) api_json_tool_calls(out, calls, ncalls, call_seq, 0);
    sb_str(out, "},\"finish_reason\":");
    sb_json_str(out, finish, strlen(finish));
    sb_str(out, "}],\"usage\":{\"prompt_tokens\":");
    sb_int(out, prompt_tokens);
    sb_str(out, ",\"completion_tokens\":");
    sb_int(out, completion_tokens);
    sb_str(out, ",\"total_tokens\":");
    sb_int(out, prompt_tokens + completion_tokens);
    sb_str(out, "}}");
}

void api_json_chunk(Sb* out, const char* id, const char* model, long long created, const char* role, const char* text,
                    size_t tlen, const char* finish) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion.chunk\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"delta\":{");
    int any = 0;
    if (role) {
        sb_str(out, "\"role\":");
        sb_json_str(out, role, strlen(role));
        any = 1;
    }
    if (tlen) {
        if (any) sb_str(out, ",");
        sb_str(out, "\"content\":");
        sb_json_str(out, text, tlen);
    }
    sb_str(out, "},\"finish_reason\":");
    if (finish) sb_json_str(out, finish, strlen(finish));
    else sb_str(out, "null");
    sb_str(out, "}]}");
}

void api_json_tool_chunk(Sb* out, const char* id, const char* model, long long created, int index, const char* call_id,
                         const char* name, const char* arguments) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion.chunk\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    sb_int(out, index);
    sb_str(out, ",\"id\":");
    sb_json_str(out, call_id, strlen(call_id));
    sb_str(out, ",\"type\":\"function\",\"function\":{\"name\":");
    sb_json_str(out, name, strlen(name));
    sb_str(out, ",\"arguments\":");
    sb_json_str(out, arguments ? arguments : "", arguments ? strlen(arguments) : 0);
    sb_str(out, "}}]},\"finish_reason\":null}]}");
}

static void ollama_head(Sb* out, const char* model, const char* created) {
    sb_str(out, "{\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"created_at\":");
    sb_json_str(out, created, strlen(created));
}

static void ollama_done(Sb* out, int done, const char* reason, int prompt_n, int eval_n) {
    sb_str(out, ",\"done\":");
    sb_str(out, done ? "true" : "false");
    if (done) {
        sb_str(out, ",\"done_reason\":");
        sb_json_str(out, reason ? reason : "stop", strlen(reason ? reason : "stop"));
        sb_str(out, ",\"prompt_eval_count\":");
        sb_int(out, prompt_n);
        sb_str(out, ",\"eval_count\":");
        sb_int(out, eval_n);
    }
    sb_str(out, "}");
}

void api_json_ollama_chat(Sb* out, const char* model, const char* created, const char* text, size_t tlen, int done,
                          const char* reason, int prompt_n, int eval_n, const ApiToolCall* calls, int ncalls) {
    ollama_head(out, model, created);
    sb_str(out, ",\"message\":{\"role\":\"assistant\",\"content\":");
    sb_json_str(out, text ? text : "", text ? tlen : 0);
    if (ncalls > 0 && calls) api_json_tool_calls(out, calls, ncalls, 0, 1);
    sb_str(out, "}");
    ollama_done(out, done, reason, prompt_n, eval_n);
}

void api_json_ollama_gen(Sb* out, const char* model, const char* created, const char* text, size_t tlen, int done,
                         const char* reason, int prompt_n, int eval_n) {
    ollama_head(out, model, created);
    sb_str(out, ",\"response\":");
    sb_json_str(out, text ? text : "", text ? tlen : 0);
    ollama_done(out, done, reason, prompt_n, eval_n);
}

void api_json_ollama_tags(Sb* out, const char* model) {
    sb_str(out, "{\"models\":[{\"name\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"modified_at\":\"1970-01-01T00:00:00Z\",\"size\":0,\"digest\":\"\","
               "\"details\":{\"parent_model\":\"\",\"format\":\"darkgarlic\",\"family\":\"qwen3\","
               "\"families\":[\"qwen3\"],\"parameter_size\":\"\",\"quantization_level\":\"Q4\"}}]}");
}

void api_json_ollama_show(Sb* out, const char* model) {
    sb_str(out, "{\"modelfile\":\"# DarkGarlic\\n\",\"parameters\":\"\",\"template\":\"\",\"details\":{"
               "\"parent_model\":\"\",\"format\":\"darkgarlic\",\"family\":\"qwen3\",\"families\":[\"qwen3\"],"
               "\"parameter_size\":\"\",\"quantization_level\":\"Q4\"},\"model_info\":{\"general.architecture\":"
               "\"qwen3\"},\"model\":");
    // the name the client listed: Open WebUI shows whatever /api/tags returned
    sb_json_str(out, model, strlen(model));
    sb_str(out, "}");
}

void api_json_ollama_version(Sb* out) { sb_str(out, "{\"version\":\"1.20-darkgarlic\"}"); }

void api_json_ollama_error(Sb* out, const char* message) {
    sb_str(out, "{\"error\":");
    sb_json_str(out, message ? message : "", message ? strlen(message) : 0);
    sb_str(out, "}");
}

static int span_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

static void trim_span(const char** s, size_t* n) {
    const char* p = *s;
    size_t m = *n;
    while (m && span_ws(*p)) {
        p++;
        m--;
    }
    while (m && span_ws(p[m - 1])) m--;
    *s = p;
    *n = m;
}

static int num_i64(double v, long long* out) {
    if (v != v) return 0;
    if (v > 9223372036854774784.0 || v < -9223372036854774784.0) return 0;
    long long i = (long long)v;
    if ((double)i != v) return 0;
    *out = i;
    return 1;
}

// One JSON value, with the spacing Python's json.dumps uses (", " and ": "). An integral number is written
// without a decimal point. A string is copied raw, escapes included, because that is the text the parser stored.
static void json_emit(const JNode* pool, int node, Sb* out) {
    if (node < 0) {
        sb_str(out, "null");
        return;
    }
    const JNode* d = &pool[node];
    if (d->type == J_NULL) {
        sb_str(out, "null");
    } else if (d->type == J_BOOL) {
        sb_str(out, d->b ? "true" : "false");
    } else if (d->type == J_NUM) {
        long long iv;
        if (num_i64(d->num, &iv)) sb_int(out, iv);
        else if (d->num != d->num || d->num > 1e300 || d->num < -1e300) sb_str(out, "0");
        else {
            char b[64];
            snprintf(b, sizeof(b), "%.16g", d->num);
            sb_str(out, b);
        }
    } else if (d->type == J_STR) {
        sb_add(out, "\"", 1);
        if (d->s && d->slen) sb_add(out, d->s, d->slen);
        sb_add(out, "\"", 1);
    } else if (d->type == J_ARR) {
        sb_str(out, "[");
        int first = 1;
        for (int c = d->first; c >= 0; c = pool[c].next) {
            if (!first) sb_str(out, ", ");
            first = 0;
            json_emit(pool, c, out);
        }
        sb_str(out, "]");
    } else if (d->type == J_OBJ) {
        sb_str(out, "{");
        int first = 1;
        for (int c = d->first; c >= 0; c = pool[c].next) {
            if (!first) sb_str(out, ", ");
            first = 0;
            sb_add(out, "\"", 1);
            if (pool[c].key && pool[c].klen) sb_add(out, pool[c].key, pool[c].klen);
            sb_add(out, "\"", 1);
            sb_str(out, ": ");
            json_emit(pool, c, out);
        }
        sb_str(out, "}");
    } else {
        sb_str(out, "null");
    }
}

static void ollama_emit_msg(Sb* out, int* first, const char* role, const char* content, size_t n) {
    if (!*first) sb_str(out, ",");
    *first = 0;
    sb_str(out, "{\"role\":");
    sb_json_str(out, role, strlen(role));
    sb_str(out, ",\"content\":");
    sb_json_str(out, content ? content : "", content ? n : 0);
    sb_str(out, "}");
}

static int ollama_num(const JNode* pool, int obj, const char* key, double* v) {
    int n = json_get(pool, obj, key);
    if (n < 0 || pool[n].type == J_NULL) return 0;
    if (pool[n].type != J_NUM) return -1;
    *v = pool[n].num;
    return 1;
}

int api_ollama_as_openai(const char* body, size_t n, int generate, Sb* out, const char** err) {
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    if (!pool) {
        *err = "Out of memory.";
        return -1;
    }
    int root = json_parse(body, n, pool, API_MAX_NODES);
    if (root < 0 || pool[root].type != J_OBJ) {
        free(pool);
        *err = "The body must be a JSON object.";
        return -1;
    }
    int opt = json_get(pool, root, "options");
    if (opt >= 0 && pool[opt].type != J_OBJ && pool[opt].type != J_NULL) {
        free(pool);
        *err = "'options' must be an object.";
        return -1;
    }
    if (opt >= 0 && pool[opt].type != J_OBJ) opt = -1;
    int stream = 1;
    int st = json_get(pool, root, "stream");
    if (st >= 0 && pool[st].type != J_NULL) {
        if (pool[st].type != J_BOOL) {
            free(pool);
            *err = "'stream' must be true or false.";
            return -1;
        }
        stream = pool[st].b;
    }
    int has_temp = 0, has_pred = 0, has_seed = 0;
    double temp = 0.7, pred = 0, seed = 0;
    const char* keys[] = {"temperature", "num_predict", "seed"};
    double* vals[] = {&temp, &pred, &seed};
    int* has[] = {&has_temp, &has_pred, &has_seed};
    for (int pass = 0; pass < 2; pass++) {
        int obj = pass == 0 ? root : opt;
        if (obj < 0) continue;
        for (int k = 0; k < 3; k++) {
            double v = 0;
            int r = ollama_num(pool, obj, keys[k], &v);
            if (r < 0) {
                free(pool);
                *err = "'temperature', 'num_predict' and 'seed' must be numbers.";
                return -1;
            }
            if (r) *vals[k] = v, *has[k] = 1;
        }
    }
    if (has_temp && (temp < 0 || temp > 2)) {
        free(pool);
        *err = "'temperature' must be a number from 0 to 2.";
        return -1;
    }
    if (has_seed && seed < 0) {
        free(pool);
        *err = "'seed' must be a non-negative number.";
        return -1;
    }
    int max_tokens = 512;
    if (has_pred) {
        if (pred < 0) max_tokens = 4096;
        else if (pred < 1) max_tokens = 512;
        else if (pred > 16384) max_tokens = 16384;
        else max_tokens = (int)pred;
    }
    int first = 1, nmsg = 0;
    if (generate) {
        int sys = json_get(pool, root, "system");
        if (sys >= 0 && pool[sys].type != J_NULL && pool[sys].type != J_STR) {
            free(pool);
            *err = "'system' must be a string.";
            return -1;
        }
        int pr = json_get(pool, root, "prompt");
        if (pr < 0 || pool[pr].type != J_STR) {
            free(pool);
            *err = "'prompt' must be a string.";
            return -1;
        }
        sb_str(out, "{\"messages\":[");
        if (sys >= 0 && pool[sys].type == J_STR) {
            Sb s;
            sb_init(&s);
            json_str_sb(pool, sys, &s);
            ollama_emit_msg(out, &first, "system", s.p, s.n);
            sb_free(&s);
            nmsg++;
        }
        Sb p;
        sb_init(&p);
        json_str_sb(pool, pr, &p);
        ollama_emit_msg(out, &first, "user", p.p, p.n);
        sb_free(&p);
        nmsg++;
    } else {
        int m = json_get(pool, root, "messages");
        if (m < 0 || pool[m].type != J_ARR || pool[m].count < 1) {
            free(pool);
            *err = "'messages' must be a non-empty array.";
            return -1;
        }
        sb_str(out, "{\"messages\":[");
        for (int c = pool[m].first; c >= 0; c = pool[c].next) {
            Sb content;
            sb_init(&content);
            char role[16];
            if (api_message(pool, c, role, sizeof(role), &content)) {
                sb_free(&content);
                free(pool);
                *err = "Each message must be an object with 'role' and 'content'.";
                return -1;
            }
            ollama_emit_msg(out, &first, role, content.p, content.n);
            sb_free(&content);
            nmsg++;
        }
    }
    Sb tools_js;
    sb_init(&tools_js);
    if (!generate) {
        int tl = json_get(pool, root, "tools");
        if (tl >= 0 && pool[tl].type != J_NULL) {
            if (pool[tl].type != J_ARR) {
                sb_free(&tools_js);
                free(pool);
                *err = "'tools' must be an array.";
                return -1;
            }
            if (pool[tl].count > 0) json_emit(pool, tl, &tools_js);
        }
    }
    int tools_oom = tools_js.oom;
    free(pool);
    if (tools_oom) {
        sb_free(&tools_js);
        *err = "Out of memory.";
        return -1;
    }
    if (!nmsg) {
        sb_free(&tools_js);
        *err = "The request has no messages.";
        return -1;
    }
    sb_str(out, "]");
    if (tools_js.n) {
        sb_str(out, ",\"tools\":");
        sb_add(out, tools_js.p, tools_js.n);
    }
    sb_free(&tools_js);
    if (has_temp) {
        char b[64];
        snprintf(b, sizeof(b), ",\"temperature\":%.6g", temp);
        sb_str(out, b);
    }
    {
        char b[64];
        snprintf(b, sizeof(b), ",\"max_tokens\":%d", max_tokens);
        sb_str(out, b);
    }
    sb_str(out, stream ? ",\"stream\":true" : ",\"stream\":false");
    if (has_seed) {
        char b[64];
        unsigned sd = seed > 4294967295.0 ? 4294967295u : (unsigned)seed;
        snprintf(b, sizeof(b), ",\"seed\":%u", sd);
        sb_str(out, b);
    }
    sb_str(out, "}");
    if (out->oom) {
        *err = "Out of memory.";
        return -1;
    }
    return 0;
}

void api_json_error(Sb* out, const char* message, const char* type, const char* code) {
    sb_str(out, "{\"error\":{\"message\":");
    sb_json_str(out, message, strlen(message));
    sb_str(out, ",\"type\":");
    sb_json_str(out, type, strlen(type));
    sb_str(out, ",\"code\":");
    if (code) sb_json_str(out, code, strlen(code));
    else sb_str(out, "null");
    sb_str(out, "}}");
}

// ---- tools --------------------------------------------------------------------------------------------------------

static const char* find_lit(const char* s, size_t n, const char* lit, size_t m) {
    if (!s || m == 0 || n < m) return NULL;
    for (size_t i = 0; i + m <= n; i++)
        if (!memcmp(s + i, lit, m)) return s + i;
    return NULL;
}

static void decode_raw(const char* s, size_t n, char* dst, size_t cap) {
    if (cap == 0) return;
    dst[0] = 0;
    if (!s) n = 0;
    JNode node;
    memset(&node, 0, sizeof(node));
    node.type = J_STR;
    node.s = s ? s : "";
    node.slen = n;
    node.first = node.next = -1;
    Sb o;
    sb_init(&o);
    json_str_sb(&node, 0, &o);
    size_t k = o.n < cap - 1 ? o.n : cap - 1;
    if (k && o.p) memcpy(dst, o.p, k);
    dst[k] = 0;
    sb_free(&o);
}

static int param_type_name(const char* s) {
    if (!strcmp(s, "integer")) return API_P_INT;
    if (!strcmp(s, "number")) return API_P_NUM;
    if (!strcmp(s, "boolean")) return API_P_BOOL;
    if (!strcmp(s, "object")) return API_P_OBJ;
    if (!strcmp(s, "array")) return API_P_ARR;
    return API_P_STR;
}

static int copy_one_schema(const JNode* pool, int tool, ApiToolSchema* sch) {
    memset(sch, 0, sizeof(*sch));
    int src = tool;
    int fn = json_get(pool, tool, "function");
    if (fn >= 0 && pool[fn].type == J_OBJ) src = fn;
    int nm = json_get(pool, src, "name");
    if (nm < 0 || pool[nm].type != J_STR) return -1;
    decode_raw(pool[nm].s, pool[nm].slen, sch->name, sizeof(sch->name));
    if (!sch->name[0]) return -1;
    int params = json_get(pool, src, "parameters");
    if (params < 0 || pool[params].type != J_OBJ) return 0;
    int props = json_get(pool, params, "properties");
    if (props < 0 || pool[props].type != J_OBJ) return 0;
    for (int p = pool[props].first; p >= 0 && sch->n_params < API_MAX_PARAMS; p = pool[p].next) {
        ApiParam* ap = &sch->params[sch->n_params];
        decode_raw(pool[p].key, pool[p].klen, ap->name, sizeof(ap->name));
        if (!ap->name[0]) continue;
        ap->type = API_P_STR;
        int ty = json_get(pool, p, "type");
        if (ty >= 0 && pool[ty].type == J_STR) {
            char tb[16];
            decode_raw(pool[ty].s, pool[ty].slen, tb, sizeof(tb));
            ap->type = param_type_name(tb);
        }
        sch->n_params++;
    }
    return 0;
}

static const char TOOL_HEAD[] = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const char TOOL_TAIL[] =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
    "within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but "
    "NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do "
    "not tell the user about function calls\n"
    "</IMPORTANT>";

int api_tools_system(const JNode* pool, int tools, int first_msg, Sb* out, int* ate_system) {
    *ate_system = 0;
    if (tools < 0 || pool[tools].type != J_ARR) return -1;
    sb_str(out, TOOL_HEAD);
    for (int t = pool[tools].first; t >= 0; t = pool[t].next) {
        sb_str(out, "\n");
        json_emit(pool, t, out);
    }
    sb_str(out, TOOL_TAIL);
    if (first_msg >= 0) {
        char role[16];
        Sb c;
        sb_init(&c);
        if (api_message(pool, first_msg, role, sizeof(role), &c)) {
            sb_free(&c);
            return -1;
        }
        if (!strcmp(role, "system")) {
            *ate_system = 1;
            const char* p = c.p ? c.p : "";
            size_t n = c.n;
            trim_span(&p, &n);
            if (n) {
                sb_str(out, "\n\n");
                sb_add(out, p, n);
            }
        }
        sb_free(&c);
    }
    return out->oom ? -1 : 0;
}

static void xml_value(const JNode* pool, int node, Sb* out) {
    if (node < 0 || pool[node].type == J_NULL) {
        sb_str(out, "None");
        return;
    }
    if (pool[node].type == J_OBJ || pool[node].type == J_ARR) {
        json_emit(pool, node, out);
        return;
    }
    if (pool[node].type == J_STR) {
        json_str_sb(pool, node, out);
        return;
    }
    if (pool[node].type == J_BOOL) {
        sb_str(out, pool[node].b ? "True" : "False");
        return;
    }
    long long iv;
    if (num_i64(pool[node].num, &iv)) sb_int(out, iv);
    else {
        char b[64];
        snprintf(b, sizeof(b), "%.16g", pool[node].num);
        sb_str(out, b);
    }
}

int api_message_tool_xml(const JNode* pool, int msg, int content_nonempty, Sb* out) {
    int calls = json_get(pool, msg, "tool_calls");
    if (calls < 0 || pool[calls].type == J_NULL) return 0;
    if (pool[calls].type != J_ARR) return -1;
    int n = 0;
    for (int c = pool[calls].first; c >= 0; c = pool[c].next, n++) {
        if (pool[c].type != J_OBJ) return -1;
        int src = c;
        int fn = json_get(pool, c, "function");
        if (fn >= 0 && pool[fn].type == J_OBJ) src = fn;
        int nm = json_get(pool, src, "name");
        if (nm < 0 || pool[nm].type != J_STR) return -1;
        Sb name;
        sb_init(&name);
        json_str_sb(pool, nm, &name);
        if (!name.p || !name.n) {
            sb_free(&name);
            return -1;
        }
        JNode* scratch = NULL;
        char* kept = NULL;   // scratch nodes point into this copy of the arguments text
        const JNode* ap = pool;
        int root = -1;
        int arg = json_get(pool, src, "arguments");
        if (arg >= 0 && pool[arg].type != J_NULL) {
            if (pool[arg].type == J_OBJ) {
                root = arg;
            } else if (pool[arg].type == J_STR) {
                Sb s;
                sb_init(&s);
                json_str_sb(pool, arg, &s);
                const char* p = s.p ? s.p : "";
                size_t pn = s.n;
                trim_span(&p, &pn);
                if (pn) {
                    kept = malloc(pn + 1);
                    scratch = malloc(sizeof(JNode) * 256);
                    if (!kept || !scratch) {
                        free(kept);
                        free(scratch);
                        sb_free(&s);
                        sb_free(&name);
                        return -1;
                    }
                    memcpy(kept, p, pn);
                    kept[pn] = 0;
                    int r = json_parse(kept, pn, scratch, 256);
                    if (r < 0 || scratch[r].type != J_OBJ) {
                        free(scratch);
                        free(kept);
                        sb_free(&s);
                        sb_free(&name);
                        return -1;
                    }
                    ap = scratch;
                    root = r;
                }
                sb_free(&s);
            } else {
                sb_free(&name);
                return -1;
            }
        }
        if (n == 0) {
            if (content_nonempty) sb_str(out, "\n\n");
        } else {
            sb_str(out, "\n");
        }
        sb_str(out, "<tool_call>\n<function=");
        sb_add(out, name.p, name.n);
        sb_str(out, ">\n");
        sb_free(&name);
        if (root >= 0) {
            for (int f = ap[root].first; f >= 0; f = ap[f].next) {
                char kbuf[256];
                decode_raw(ap[f].key, ap[f].klen, kbuf, sizeof(kbuf));
                sb_str(out, "<parameter=");
                sb_str(out, kbuf);
                sb_str(out, ">\n");
                xml_value(ap, f, out);
                sb_str(out, "\n</parameter>\n");
            }
        }
        free(scratch);
        free(kept);
        sb_str(out, "</function>\n</tool_call>");
    }
    return out->oom ? -1 : 0;
}

static int parse_i64_text(const char* s, size_t n, long long* out) {
    if (n == 0 || n > 18) return 0;
    size_t i = 0;
    int neg = 0;
    if (s[0] == '-') {
        neg = 1;
        i++;
        if (i == n) return 0;
    }
    long long v = 0;
    for (; i < n; i++) {
        if (s[i] < '0' || s[i] > '9') return 0;
        v = v * 10 + (s[i] - '0');
    }
    *out = neg ? -v : v;
    return 1;
}

static int is_json_number(const char* s, size_t n) {
    if (n == 0 || n > 40) return 0;
    size_t i = 0;
    if (s[0] == '-') {
        i++;
        if (i == n) return 0;
    }
    if (s[i] == '0') i++;
    else if (s[i] >= '1' && s[i] <= '9') {
        while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    } else {
        return 0;
    }
    if (i < n && s[i] == '.') {
        i++;
        if (i == n || s[i] < '0' || s[i] > '9') return 0;
        while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    }
    if (i < n && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        if (i < n && (s[i] == '+' || s[i] == '-')) i++;
        if (i == n || s[i] < '0' || s[i] > '9') return 0;
        while (i < n && s[i] >= '0' && s[i] <= '9') i++;
    }
    return i == n;
}

static int schema_param_type(const ApiToolSchema* schemas, int nschemas, const char* fn, const char* key) {
    if (!schemas || !fn || !key) return API_P_STR;
    for (int i = 0; i < nschemas; i++) {
        if (strcmp(schemas[i].name, fn)) continue;
        for (int p = 0; p < schemas[i].n_params; p++)
            if (!strcmp(schemas[i].params[p].name, key)) return schemas[i].params[p].type;
        return API_P_STR;
    }
    return API_P_STR;
}

static void emit_typed(Sb* out, int typ, const char* v, size_t n) {
    const char* t = v;
    size_t tn = n;
    if (typ != API_P_STR) trim_span(&t, &tn);
    if (typ == API_P_INT) {
        long long x;
        if (parse_i64_text(t, tn, &x)) {
            sb_int(out, x);
            return;
        }
    } else if (typ == API_P_NUM) {
        if (is_json_number(t, tn)) {
            sb_add(out, t, tn);
            return;
        }
    } else if (typ == API_P_BOOL) {
        if (tn == 4 && !memcmp(t, "true", 4)) {
            sb_str(out, "true");
            return;
        }
        if (tn == 5 && !memcmp(t, "false", 5)) {
            sb_str(out, "false");
            return;
        }
    } else if ((typ == API_P_OBJ || typ == API_P_ARR) && tn > 0 && tn < 65536) {
        JNode* pool = malloc(sizeof(JNode) * 256);
        if (pool) {
            int root = json_parse(t, tn, pool, 256);
            int want = typ == API_P_OBJ ? J_OBJ : J_ARR;
            if (root >= 0 && pool[root].type == want) {
                json_emit(pool, root, out);
                free(pool);
                return;
            }
            free(pool);
        }
    }
    sb_json_str(out, typ == API_P_STR ? v : t, typ == API_P_STR ? n : tn);
}

// The bytes between <tool_call> and </tool_call>. Returns 1 when call holds a JSON object that fits.
static int store_call(const char* body, size_t n, const ApiToolSchema* schemas, int nschemas, ApiToolCall* call) {
    const char* p = body;
    const char* end = body + n;
    while (p < end && span_ws(*p)) p++;
    if ((size_t)(end - p) < 10 || memcmp(p, "<function=", 10)) return 0;
    p += 10;
    const char* gt = memchr(p, '>', (size_t)(end - p));
    if (!gt || gt == p) return 0;
    size_t nl = (size_t)(gt - p);
    if (nl >= API_TOOL_NAME) nl = API_TOOL_NAME - 1;
    memcpy(call->name, p, nl);
    call->name[nl] = 0;
    p = gt + 1;
    if (p < end && *p == '\n') p++;
    Sb args;
    sb_init(&args);
    sb_str(&args, "{");
    int first = 1, ok = 1;
    while (p < end) {
        while (p < end && span_ws(*p)) p++;
        if (p >= end) break;
        if ((size_t)(end - p) >= 11 && !memcmp(p, "</function>", 11)) break;
        if ((size_t)(end - p) < 11 || memcmp(p, "<parameter=", 11)) {
            ok = 0;
            break;
        }
        p += 11;
        gt = memchr(p, '>', (size_t)(end - p));
        if (!gt || gt == p) {
            ok = 0;
            break;
        }
        size_t kl = (size_t)(gt - p);
        const char* key_full = p;
        p = gt + 1;
        if (p < end && *p == '\n') p++;
        const char* vend = find_lit(p, (size_t)(end - p), "</parameter>", 12);
        if (!vend) {
            ok = 0;
            break;
        }
        size_t vn = (size_t)(vend - p);
        if (vn && p[vn - 1] == '\n') vn--;
        if (vn && p[vn - 1] == '\r') vn--;
        if (!first) sb_str(&args, ",");
        first = 0;
        sb_json_str(&args, key_full, kl);
        sb_str(&args, ":");
        int typ = API_P_STR;
        if (kl < API_PARAM_NAME) {
            char key[API_PARAM_NAME];
            memcpy(key, key_full, kl);
            key[kl] = 0;
            typ = schema_param_type(schemas, nschemas, call->name, key);
        }
        emit_typed(&args, typ, p, vn);
        p = vend + 12;
        if (p < end && *p == '\n') p++;
    }
    while (p < end && span_ws(*p)) p++;
    if ((size_t)(end - p) < 11 || memcmp(p, "</function>", 11)) ok = 0;
    sb_str(&args, "}");
    if (!ok || args.oom || !args.p || args.n >= API_TOOL_ARGS) {
        sb_free(&args);
        return 0;
    }
    memcpy(call->arguments, args.p, args.n + 1);
    sb_free(&args);
    return 1;
}

int api_parse_tool_calls(const char* text, size_t n, const ApiToolSchema* schemas, int nschemas, Sb* content,
                         ApiToolCall* calls, int max_calls, int* ncalls, int* cut) {
    if (ncalls) *ncalls = 0;
    if (cut) *cut = 0;
    if (!text) {
        text = "";
        n = 0;
    }
    const char* mark = find_lit(text, n, "<tool_call>", 11);
    size_t prefix_n = mark ? (size_t)(mark - text) : n;
    int stored = 0;
    if (mark) {
        const char* p = mark;
        const char* end = text + n;
        while (p + 11 <= end && !memcmp(p, "<tool_call>", 11)) {
            const char* inner = p + 11;
            const char* close = find_lit(inner, (size_t)(end - inner), "</tool_call>", 12);
            if (!close) {
                if (cut) *cut = 1;
                break;
            }
            if (calls && stored < max_calls &&
                store_call(inner, (size_t)(close - inner), schemas, nschemas, &calls[stored]))
                stored++;
            p = close + 12;
            if (p < end && *p == '\n') p++;
        }
    }
    if (stored == 0) sb_add(content, text, n);
    else {
        const char* tp = text;
        size_t tn = prefix_n;
        trim_span(&tp, &tn);
        sb_add(content, tp, tn);
    }
    if (ncalls) *ncalls = stored;
    return 0;
}

size_t api_tool_stream_end(const char* s, size_t n) {
    const char mark[] = "<tool_call>";
    const size_t mlen = 11;
    if (!s) return 0;
    for (size_t i = 0; i + mlen <= n; i++)
        if (!memcmp(s + i, mark, mlen)) return i;
    size_t max = n < mlen - 1 ? n : mlen - 1;
    for (size_t k = max; k > 0; k--)
        if (!memcmp(s + (n - k), mark, k)) return n - k;
    return n;
}

// ---- chat request -------------------------------------------------------------------------------------------------

int api_chat_parse(const char* body, size_t len, JNode* pool, int max, ChatRequest* req, const char** err) {
    memset(req, 0, sizeof(*req));
    req->temperature = 0.7;
    req->max_tokens = 512;
    req->messages = -1;
    req->tools = -1;
    req->root = json_parse(body, len, pool, max);
    if (req->root < 0 || pool[req->root].type != J_OBJ) {
        *err = "The body must be a JSON object.";
        return -1;
    }
    int m = json_get(pool, req->root, "messages");
    if (m < 0 || pool[m].type != J_ARR || pool[m].count < 1) {
        *err = "'messages' must be a non-empty array.";
        return -1;
    }
    for (int c = pool[m].first; c >= 0; c = pool[c].next) {
        if (pool[c].type != J_OBJ) {
            *err = "Each message must be an object with 'role' and 'content'.";
            return -1;
        }
        req->n_messages++;
    }
    req->messages = m;
    int t = json_get(pool, req->root, "temperature");
    if (t >= 0 && pool[t].type != J_NULL) {
        if (pool[t].type != J_NUM || pool[t].num < 0 || pool[t].num > 2) {
            *err = "'temperature' must be a number from 0 to 2.";
            return -1;
        }
        req->temperature = pool[t].num;
    }
    int mt = json_get(pool, req->root, "max_completion_tokens");
    if (mt < 0) mt = json_get(pool, req->root, "max_tokens");
    if (mt >= 0 && pool[mt].type != J_NULL) {
        if (pool[mt].type != J_NUM || pool[mt].num < 1 || pool[mt].num > 16384) {
            *err = "'max_tokens' must be a number from 1 to 16384.";
            return -1;
        }
        req->max_tokens = (int)pool[mt].num;
    }
    int s = json_get(pool, req->root, "stream");
    if (s >= 0 && pool[s].type != J_NULL) {
        if (pool[s].type != J_BOOL) {
            *err = "'stream' must be true or false.";
            return -1;
        }
        req->stream = pool[s].b;
    }
    int n = json_get(pool, req->root, "n");
    if (n >= 0 && pool[n].type == J_NUM && pool[n].num != 1) {
        *err = "Only n = 1 is supported.";
        return -1;
    }
    int sd = json_get(pool, req->root, "seed");
    if (sd >= 0 && pool[sd].type == J_NUM && pool[sd].num >= 0) {
        req->seed = (unsigned)pool[sd].num;
        req->has_seed = 1;
    }
    int tools = json_get(pool, req->root, "tools");
    if (tools >= 0 && pool[tools].type != J_NULL) {
        if (pool[tools].type != J_ARR) {
            *err = "'tools' must be an array.";
            return -1;
        }
        for (int c = pool[tools].first; c >= 0; c = pool[c].next) {
            if (pool[c].type != J_OBJ) {
                *err = "Each tool must be an object.";
                return -1;
            }
            ApiToolSchema one;
            if (copy_one_schema(pool, c, &one)) {
                *err = "Each tool needs a function name.";
                return -1;
            }
            if (req->n_schemas < API_MAX_TOOLS) req->schemas[req->n_schemas++] = one;
            req->n_tools++;
        }
        if (req->n_tools) req->tools = tools;
    }
    return 0;
}
