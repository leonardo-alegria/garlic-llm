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
    req->head_len = head;
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
                if (*ep || cl < 0) return -1;
                req->content_length = cl;
                if (cl > API_MAX_BODY) return -2;
            } else if (ci_eq(p, colon - p, "authorization")) {
                size_t k = vlen < sizeof(req->auth) - 1 ? vlen : sizeof(req->auth) - 1;
                memcpy(req->auth, v, k);
                req->auth[k] = 0;
            }
        }
        p = eol + 1;
    }
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
    int full;                   // the pool ran out of nodes
} JP;

static void jp_ws(JP* P) {
    while (P->i < P->n && (P->s[P->i] == ' ' || P->s[P->i] == '\t' || P->s[P->i] == '\n' || P->s[P->i] == '\r')) P->i++;
}

static int jp_new(JP* P) {
    if (P->used >= P->max) {
        P->full = 1;
        return -1;
    }
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
        d->s = P->s + start;   // (the number as written, for json_write)
        d->slen = m;
        r = 0;
    }
done:
    if (r == 0) *out = idx;
    P->depth--;
    return r;
}

int json_parse(const char* s, size_t n, JNode* pool, int max) {
    JP P = {s, n, 0, pool, max, 0, 0, 0};
    int root;
    if (jp_value(&P, &root)) return P.full ? -2 : -1;
    jp_ws(&P);
    if (P.i != n) return -1;
    return root;
}

JNode* json_parse_grow(const char* s, size_t n, int first, int* root) {
    for (int max = first;; max *= 8) {
        if (max > API_MAX_VALUES) max = API_MAX_VALUES;
        JNode* pool = malloc(sizeof(JNode) * (size_t)max);
        if (!pool) {
            *root = -3;
            return NULL;
        }
        *root = json_parse(s, n, pool, max);
        if (*root >= 0) return pool;
        free(pool);
        if (*root != -2 || max == API_MAX_VALUES) return NULL;
    }
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

static int is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

int api_think_switch(Sb* text) {
    static const struct {
        const char* word;
        int on;
    } words[] = {{"/no_think", 0}, {"/nothink", 0}, {"/think", 1}};
    size_t n = text->n;
    while (n && is_space(text->p[n - 1])) n--;
    for (size_t w = 0; w < sizeof(words) / sizeof(words[0]); w++) {
        size_t k = strlen(words[w].word);
        if (n < k || memcmp(text->p + n - k, words[w].word, k)) continue;
        if (n > k && !is_space(text->p[n - k - 1])) continue;   // (part of a longer word or path, not the switch)
        n -= k;
        while (n && is_space(text->p[n - 1])) n--;
        text->n = n;
        text->p[n] = 0;
        return words[w].on;
    }
    return -1;
}

void api_trim(Sb* text) {
    if (!text->p) return;
    size_t a = 0, b = text->n;
    while (a < b && is_space(text->p[a])) a++;
    while (b > a && is_space(text->p[b - 1])) b--;
    memmove(text->p, text->p + a, b - a);
    text->n = b - a;
    text->p[text->n] = 0;
}

void api_drop_think(Sb* text, Sb* reasoning) {
    if (!text->p) return;
    const char* first = strstr(text->p, "</think>");
    if (!first) return;
    if (reasoning) {   // (before the first </think>, after the last <think> there)
        const char* r = text->p;
        for (const char* f = text->p; (f = strstr(f, "<think>")) != NULL && f < first; f += 7) r = f + 7;
        sb_add(reasoning, r, (size_t)(first - r));
        api_trim(reasoning);
    }
    const char* end = first + 8;
    for (const char* f = end; (f = strstr(f, "</think>")) != NULL; f += 8) end = f + 8;
    while (*end == '\n' || *end == '\r') end++;
    size_t k = (size_t)(end - text->p);
    memmove(text->p, end, text->n - k + 1);
    text->n -= k;
}

static void json_out(const JNode* pool, int node, Sb* out, int spaced) {
    if (node < 0) return;
    const JNode* d = &pool[node];
    switch (d->type) {
    case J_NULL: sb_str(out, "null"); break;
    case J_BOOL: sb_str(out, d->b ? "true" : "false"); break;
    case J_NUM:
        if (d->s && d->slen) sb_add(out, d->s, d->slen);
        else sb_int(out, (long long)d->num);
        break;
    case J_STR:
        sb_add(out, "\"", 1);
        sb_add(out, d->s, d->slen);
        sb_add(out, "\"", 1);
        break;
    case J_ARR:
    case J_OBJ:
        sb_add(out, d->type == J_ARR ? "[" : "{", 1);
        for (int c = d->first; c >= 0; c = pool[c].next) {
            if (c != d->first) sb_str(out, spaced ? ", " : ",");
            if (d->type == J_OBJ) {
                sb_add(out, "\"", 1);
                sb_add(out, pool[c].key, pool[c].klen);
                sb_str(out, spaced ? "\": " : "\":");
            }
            json_out(pool, c, out, spaced);
        }
        sb_add(out, d->type == J_ARR ? "]" : "}", 1);
        break;
    }
}

void json_write(const JNode* pool, int node, Sb* out) { json_out(pool, node, out, 0); }

void json_write_spaced(const JNode* pool, int node, Sb* out) { json_out(pool, node, out, 1); }

void api_free_tool_calls(ApiToolCall* calls, int n) {
    for (int i = 0; i < n; i++) sb_free(&calls[i].arguments);
}

// The members of one OpenAI tool call: "id", "type" and "function" (name, and the arguments as a JSON string).
static void write_tool_call(Sb* out, const ApiToolCall* c) {
    sb_str(out, "\"id\":");
    sb_json_str(out, c->id, strlen(c->id));
    sb_str(out, ",\"type\":\"function\",\"function\":{\"name\":");
    sb_json_str(out, c->name, strlen(c->name));
    sb_str(out, ",\"arguments\":");
    sb_json_str(out, c->arguments.p ? c->arguments.p : "{}", c->arguments.p ? c->arguments.n : 2);
    sb_str(out, "}");
}

// ---- JSON writers -------------------------------------------------------------------------------------------------

void api_json_model(Sb* out, const char* model, long long created, int context_length) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"object\":\"model\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"owned_by\":\"garlic-llm\",\"context_length\":");
    sb_int(out, context_length);
    sb_str(out, "}");
}

void api_json_models(Sb* out, const char* model, long long created, int context_length) {
    sb_str(out, "{\"object\":\"list\",\"data\":[");
    api_json_model(out, model, created, context_length);
    sb_str(out, "]}");
}

// "usage":{...} (completion_tokens counts the reasoning too; its own count only when there was reasoning)
static void write_usage(Sb* out, int prompt_tokens, int completion_tokens, int reasoning_tokens) {
    sb_str(out, "\"usage\":{\"prompt_tokens\":");
    sb_int(out, prompt_tokens);
    sb_str(out, ",\"completion_tokens\":");
    sb_int(out, completion_tokens);
    sb_str(out, ",\"total_tokens\":");
    sb_int(out, prompt_tokens + completion_tokens);
    if (reasoning_tokens) {
        sb_str(out, ",\"completion_tokens_details\":{\"reasoning_tokens\":");
        sb_int(out, reasoning_tokens);
        sb_str(out, "}");
    }
    sb_str(out, "}");
}

void api_json_health(Sb* out, int busy, int waiting) {
    sb_str(out, "{\"status\":\"ok\",\"busy\":");
    sb_str(out, busy ? "true" : "false");
    sb_str(out, ",\"waiting\":");
    sb_int(out, waiting > 0 ? waiting : 0);
    sb_str(out, "}");
}

static void sb_base64(Sb* s, const uint8_t* data, size_t len) {
    static const char tbl[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= (uint32_t)data[i + 2];
        char b[4] = {tbl[(v >> 18) & 63], tbl[(v >> 12) & 63], i + 1 < len ? tbl[(v >> 6) & 63] : '=',
                     i + 2 < len ? tbl[v & 63] : '='};
        sb_add(s, b, 4);
    }
}

void api_json_completion_audio(Sb* out, const char* id, const char* model, long long created, const char* reasoning,
                               size_t rlen, const char* text, size_t tlen, const ApiToolCall* calls, int n_calls,
                               const char* audio_id, const uint8_t* audio_data, size_t audio_len, const char* finish,
                               int prompt_tokens, int completion_tokens, int reasoning_tokens) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    if (n_calls > 0 && !tlen) sb_str(out, "null");
    else sb_json_str(out, text ? text : "", tlen);
    if (rlen) {
        sb_str(out, ",\"reasoning_content\":");
        sb_json_str(out, reasoning, rlen);
    }
    if (n_calls > 0) {
        sb_str(out, ",\"tool_calls\":[");
        for (int i = 0; i < n_calls; i++) {
            sb_str(out, i ? ",{" : "{");
            write_tool_call(out, &calls[i]);
            sb_str(out, "}");
        }
        sb_add(out, "]", 1);
    }
    if (audio_data && audio_len > 0) {
        const char* aid = audio_id ? audio_id : "audio-1";
        sb_str(out, ",\"audio\":{\"id\":");
        sb_json_str(out, aid, strlen(aid));
        sb_str(out, ",\"data\":\"");
        sb_base64(out, audio_data, audio_len);
        sb_str(out, "\",\"transcript\":");
        sb_json_str(out, text ? text : "", tlen);
        sb_str(out, "}");
    }
    sb_str(out, "},\"finish_reason\":");
    sb_json_str(out, finish, strlen(finish));
    sb_str(out, "}],");
    write_usage(out, prompt_tokens, completion_tokens, reasoning_tokens);
    sb_str(out, "}");
}

void api_json_completion(Sb* out, const char* id, const char* model, long long created, const char* reasoning,
                         size_t rlen, const char* text, size_t tlen, const ApiToolCall* calls, int n_calls,
                         const char* finish, int prompt_tokens, int completion_tokens, int reasoning_tokens) {
    api_json_completion_audio(out, id, model, created, reasoning, rlen, text, tlen, calls, n_calls, NULL, NULL, 0,
                              finish, prompt_tokens, completion_tokens, reasoning_tokens);
}

void api_json_chunk_audio(Sb* out, const char* id, const char* model, long long created, const char* audio_id,
                          const uint8_t* audio_data, size_t audio_len, const char* transcript, size_t tlen) {
    const char* aid = audio_id ? audio_id : "audio-1";
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion.chunk\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"delta\":{\"audio\":{\"id\":");
    sb_json_str(out, aid, strlen(aid));
    sb_str(out, ",\"data\":\"");
    sb_base64(out, audio_data, audio_len);
    sb_str(out, "\",\"transcript\":");
    sb_json_str(out, transcript ? transcript : "", tlen);
    sb_str(out, "}},\"finish_reason\":null}]}");
}

void api_json_chunk(Sb* out, const char* id, const char* model, long long created, const char* role,
                    const char* reasoning, size_t rlen, const char* text, size_t tlen, const char* finish) {
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
    if (rlen) {
        if (any) sb_str(out, ",");
        sb_str(out, "\"reasoning_content\":");
        sb_json_str(out, reasoning, rlen);
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

void api_json_chunk_tool(Sb* out, const char* id, const char* model, long long created, int index,
                         const ApiToolCall* call) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion.chunk\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
    sb_int(out, index);
    sb_str(out, ",");
    write_tool_call(out, call);
    sb_str(out, "}]},\"finish_reason\":null}]}");
}

void api_json_chunk_usage(Sb* out, const char* id, const char* model, long long created, int prompt_tokens,
                          int completion_tokens, int reasoning_tokens) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion.chunk\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[],");
    write_usage(out, prompt_tokens, completion_tokens, reasoning_tokens);
    sb_str(out, "}");
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

// ---- chat request -------------------------------------------------------------------------------------------------

#define API_STR(x) #x
#define API_XSTR(x) API_STR(x)

// A tool's function object: {"type":"function","function":{...}} as OpenAI sends it, or the function itself.
static int tool_func_node(const JNode* pool, int tool) {
    if (tool < 0 || pool[tool].type != J_OBJ) return -1;
    int fn = json_get(pool, tool, "function");
    return fn >= 0 && pool[fn].type == J_OBJ ? fn : tool;
}

static int named_function(const JNode* pool, int tool) {
    int fn = tool_func_node(pool, tool);
    int nm = fn >= 0 ? json_get(pool, fn, "name") : -1;
    return nm >= 0 && pool[nm].type == J_STR && pool[nm].slen > 0 ? nm : -1;
}

// tools and tool_choice (see ChatRequest). Returns 0, or -1 with the reason in err.
static int parse_tools(const JNode* pool, ChatRequest* req, const char** err) {
    req->tools = -1;
    int tls = json_get(pool, req->root, "tools");
    if (tls >= 0 && pool[tls].type != J_NULL) {
        if (pool[tls].type != J_ARR) {
            *err = "'tools' must be an array.";
            return -1;
        }
        for (int c = pool[tls].first; c >= 0; c = pool[c].next) {
            if (named_function(pool, c) < 0) {
                *err = "Each tool must have a function 'name'.";
                return -1;
            }
            req->n_tools++;
        }
        if (req->n_tools > 0) {
            req->tools = tls;
            req->tool_choice = TOOL_AUTO;
        }
    }
    int tc = json_get(pool, req->root, "tool_choice");
    if (tc < 0 || pool[tc].type == J_NULL) return 0;
    if (pool[tc].type == J_STR) {
        const char* v = pool[tc].s;
        size_t n = pool[tc].slen;
        if (n == 4 && !memcmp(v, "none", 4)) req->tool_choice = TOOL_NONE;
        else if (n == 4 && !memcmp(v, "auto", 4)) req->tool_choice = req->n_tools > 0 ? TOOL_AUTO : TOOL_NONE;
        else if (n == 8 && !memcmp(v, "required", 8) && req->n_tools > 0) req->tool_choice = TOOL_REQUIRED;
        else {
            *err = n == 8 && !memcmp(v, "required", 8) ? "'tool_choice' needs 'tools'." : "Invalid 'tool_choice'.";
            return -1;
        }
        return 0;
    }
    int nm = pool[tc].type == J_OBJ ? named_function(pool, tc) : -1;
    if (nm < 0 || req->n_tools == 0) {
        *err = req->n_tools == 0 && nm >= 0 ? "'tool_choice' needs 'tools'." : "Invalid 'tool_choice'.";
        return -1;
    }
    Sb s;
    sb_init(&s);
    json_str_sb(pool, nm, &s);
    int found = 0;   // (the function must be one of the tools, as OpenAI requires)
    for (int t = pool[req->tools].first; t >= 0 && !found && s.p; t = pool[t].next) {
        Sb name;
        sb_init(&name);
        json_str_sb(pool, named_function(pool, t), &name);
        found = name.n == s.n && name.p && !memcmp(name.p, s.p, s.n);
        sb_free(&name);
    }
    if (!found || s.n >= sizeof(req->tool_name)) {
        sb_free(&s);
        *err = "'tool_choice' names a function that is not in 'tools'.";
        return -1;
    }
    memcpy(req->tool_name, s.p, s.n);
    req->tool_name[s.n] = 0;
    sb_free(&s);
    req->tool_choice = TOOL_NAMED;
    return 0;
}

int api_chat_parse(const char* body, size_t len, JNode* pool, int max, ChatRequest* req, const char** err) {
    memset(req, 0, sizeof(*req));
    req->temperature = 0.7;
    req->parallel_tool_calls = 1;
    req->messages = -1;
    req->tools = -1;
    req->root = json_parse(body, len, pool, max);
    if (req->root == -2) {
        *err = "The request has more than " API_XSTR(API_MAX_VALUES) " JSON values, the most the API reads.";
        return -2;
    }
    if (req->root < 0) {
        *err = len ? "The body is not valid JSON."
                   : "The request has no body. Send the JSON with a Content-Length header.";
        return -1;
    }
    if (pool[req->root].type != J_OBJ) {
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
    if (parse_tools(pool, req, err)) return -1;
    int t = json_get(pool, req->root, "temperature");
    if (t >= 0 && pool[t].type != J_NULL) {
        if (pool[t].type != J_NUM || pool[t].num < 0 || pool[t].num > 2) {
            *err = "'temperature' must be a number from 0 to 2.";
            return -1;
        }
        req->temperature = pool[t].num;
    }
    int mt = json_get(pool, req->root, "max_completion_tokens");
    if (mt < 0 || pool[mt].type == J_NULL) mt = json_get(pool, req->root, "max_tokens");
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
    int so = json_get(pool, req->root, "stream_options");
    int iu = so >= 0 && pool[so].type == J_OBJ ? json_get(pool, so, "include_usage") : -1;
    req->include_usage = iu >= 0 && pool[iu].type == J_BOOL && pool[iu].b;
    int ptc = json_get(pool, req->root, "parallel_tool_calls");
    if (ptc >= 0 && pool[ptc].type == J_BOOL) req->parallel_tool_calls = pool[ptc].b;
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
    // reasoning: the chat template's switch (as clients of other servers send it), then the plainer spellings
    req->think = -1;
    int kw = json_get(pool, req->root, "chat_template_kwargs");
    int et = kw >= 0 && pool[kw].type == J_OBJ ? json_get(pool, kw, "enable_thinking") : -1;
    if (et < 0 || pool[et].type != J_BOOL) et = json_get(pool, req->root, "enable_thinking");
    if (et < 0 || pool[et].type != J_BOOL) et = json_get(pool, req->root, "think");
    if (et >= 0 && pool[et].type == J_BOOL) {
        req->think = pool[et].b;
    } else {
        int re = json_get(pool, req->root, "reasoning_effort");
        if (re >= 0 && pool[re].type == J_STR) {
            int off = (pool[re].slen == 4 && !memcmp(pool[re].s, "none", 4)) ||
                      (pool[re].slen == 7 && !memcmp(pool[re].s, "minimal", 7));
            req->think = !off;
        }
    }
    int mods = json_get(pool, req->root, "modalities");
    if (mods >= 0 && pool[mods].type == J_ARR) {
        for (int c = pool[mods].first; c >= 0; c = pool[c].next)
            if (pool[c].type == J_STR && pool[c].slen == 5 && !memcmp(pool[c].s, "audio", 5)) req->want_audio = 1;
    }
    int aud = json_get(pool, req->root, "audio");
    if (aud >= 0 && pool[aud].type != J_NULL) {
        if (pool[aud].type == J_BOOL) {
            req->want_audio = pool[aud].b;
        } else if (pool[aud].type == J_OBJ) {
            req->want_audio = 1;
            int fmt = json_get(pool, aud, "format");
            if (fmt >= 0 && pool[fmt].type == J_STR) {
                Sb fs;
                sb_init(&fs);
                json_str_sb(pool, fmt, &fs);
                const char* fv = fs.p ? fs.p : "";
                if (!strcmp(fv, "wav")) req->audio_pcm = 0;
                else if (!strcmp(fv, "pcm16") || !strcmp(fv, "pcm")) req->audio_pcm = 1;
                else {
                    sb_free(&fs);
                    *err = "Audio 'format' must be 'wav' or 'pcm16'.";
                    return -1;
                }
                sb_free(&fs);
            }
            int vc = json_get(pool, aud, "voice");
            if (vc >= 0 && pool[vc].type == J_STR) {
                Sb vs;
                sb_init(&vs);
                json_str_sb(pool, vc, &vs);
                size_t k = vs.n < sizeof(req->audio_voice) - 1 ? vs.n : sizeof(req->audio_voice) - 1;
                if (vs.p) memcpy(req->audio_voice, vs.p, k);
                req->audio_voice[k] = 0;
                sb_free(&vs);
            }
        } else {
            *err = "'audio' must be an object or boolean.";
            return -1;
        }
    }
    return 0;
}

int api_speech_parse(const char* body, size_t len, JNode* pool, int max, SpeechRequest* req, const char** err) {
    memset(req, 0, sizeof(*req));
    sb_init(&req->input);
    int root = json_parse(body, len, pool, max);
    if (root < 0 || pool[root].type != J_OBJ) {
        *err = "The body must be a JSON object.";
        return -1;
    }
    int inp = json_get(pool, root, "input");
    if (inp < 0 || pool[inp].type != J_STR || pool[inp].slen == 0) {
        *err = "'input' must be a non-empty string.";
        return -1;
    }
    json_str_sb(pool, inp, &req->input);
    int vc = json_get(pool, root, "voice");
    if (vc >= 0 && pool[vc].type == J_STR) {
        Sb vs;
        sb_init(&vs);
        json_str_sb(pool, vc, &vs);
        size_t k = vs.n < sizeof(req->voice) - 1 ? vs.n : sizeof(req->voice) - 1;
        if (vs.p) memcpy(req->voice, vs.p, k);
        req->voice[k] = 0;
        sb_free(&vs);
    }
    int fmt = json_get(pool, root, "response_format");
    if (fmt >= 0 && pool[fmt].type == J_STR) {
        Sb fs;
        sb_init(&fs);
        json_str_sb(pool, fmt, &fs);
        const char* fv = fs.p ? fs.p : "";
        if (!strcmp(fv, "pcm") || !strcmp(fv, "pcm16")) req->pcm = 1;
        sb_free(&fs);
    }
    return 0;
}

JNode* api_chat_parse_pool(const char* body, size_t len, ChatRequest* req, const char** err) {
    for (int max = API_MAX_NODES;; max *= 8) {
        if (max > API_MAX_VALUES) max = API_MAX_VALUES;
        JNode* pool = malloc(sizeof(JNode) * (size_t)max);
        if (!pool) {
            *err = NULL;
            return NULL;
        }
        int r = api_chat_parse(body, len, pool, max, req, err);
        if (r == 0) return pool;
        free(pool);
        if (r != -2 || max == API_MAX_VALUES) return NULL;
    }
}

// ---- Qwen3.5's tool format ----------------------------------------------------------------------------------------
// The text of Qwen3.5's chat template for tools, so the model sees tools the way it was trained on them.

void api_system_tools(const JNode* pool, const ChatRequest* req, const char* user_sys, Sb* out) {
    sb_str(out, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
    if (req->tools >= 0) {
        for (int c = pool[req->tools].first; c >= 0; c = pool[c].next) {
            sb_add(out, "\n", 1);
            json_write_spaced(pool, c, out);
        }
    }
    sb_str(out,
           "\n</tools>\n\n"
           "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
           "<tool_call>\n"
           "<function=example_function_name>\n"
           "<parameter=example_parameter_1>\n"
           "value_1\n"
           "</parameter>\n"
           "<parameter=example_parameter_2>\n"
           "This is the value for the second parameter\n"
           "that can span\n"
           "multiple lines\n"
           "</parameter>\n"
           "</function>\n"
           "</tool_call>\n\n"
           "<IMPORTANT>\n"
           "Reminder:\n"
           "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested "
           "within <tool_call></tool_call> XML tags\n"
           "- Required parameters MUST be specified\n"
           "- You may provide optional reasoning for your function call in natural language BEFORE the function call, "
           "but NOT after\n"
           "- If there is no function call available, answer the question like normal with your current knowledge and "
           "do not tell the user about function calls\n"
           "</IMPORTANT>");
    if (user_sys && user_sys[0]) {
        sb_str(out, "\n\n");
        sb_str(out, user_sys);
    }
}

// <parameter=key>\nvalue\n</parameter>\n for each member of an arguments object (strings as text, the rest as JSON).
static void write_tool_params(const JNode* pool, int args_obj, Sb* out) {
    if (args_obj < 0 || pool[args_obj].type != J_OBJ) return;
    for (int p = pool[args_obj].first; p >= 0; p = pool[p].next) {
        sb_str(out, "<parameter=");
        sb_add(out, pool[p].key, pool[p].klen);
        sb_str(out, ">\n");
        if (pool[p].type == J_STR) json_str_sb(pool, p, out);
        else json_write_spaced(pool, p, out);
        sb_str(out, "\n</parameter>\n");
    }
}

int api_assistant_tool_calls(const JNode* pool, int msg, Sb* out) {
    if (msg < 0 || pool[msg].type != J_OBJ) return -1;
    int tcs = json_get(pool, msg, "tool_calls");
    if (tcs < 0 || pool[tcs].type == J_NULL) return 0;
    if (pool[tcs].type != J_ARR) return -1;
    int count = 0;
    for (int c = pool[tcs].first; c >= 0; c = pool[c].next) {
        int fn = tool_func_node(pool, c);
        int nm = named_function(pool, c);
        if (nm < 0) return -1;
        if (count > 0) sb_add(out, "\n", 1);
        sb_str(out, "<tool_call>\n<function=");
        json_str_sb(pool, nm, out);
        sb_str(out, ">\n");
        int args = json_get(pool, fn, "arguments");
        if (args >= 0 && pool[args].type == J_OBJ) {
            write_tool_params(pool, args, out);
        } else if (args >= 0 && pool[args].type == J_STR) {   // (as OpenAI sends them: a JSON object in a string)
            Sb dec;
            sb_init(&dec);
            json_str_sb(pool, args, &dec);
            int aroot;
            JNode* apool = dec.p && dec.n ? json_parse_grow(dec.p, dec.n, 256, &aroot) : NULL;
            if (apool && apool[aroot].type == J_OBJ) write_tool_params(apool, aroot, out);
            free(apool);
            sb_free(&dec);
        }
        sb_str(out, "</function>\n</tool_call>");
        count++;
    }
    return count;
}

// 1 if parameter pname of function fname has a schema type other than string in the request's tools.
static int param_is_non_string(const JNode* pool, const ChatRequest* req, const char* fname, const char* pname,
                               size_t plen) {
    if (!pool || !req || req->tools < 0) return 0;
    size_t flen = strlen(fname);
    for (int t = pool[req->tools].first; t >= 0; t = pool[t].next) {
        int fn = tool_func_node(pool, t);
        int nm = named_function(pool, t);
        if (nm < 0 || pool[nm].slen != flen || memcmp(pool[nm].s, fname, flen)) continue;
        int params = json_get(pool, fn, "parameters");
        int props = params >= 0 ? json_get(pool, params, "properties") : -1;
        if (props < 0 || pool[props].type != J_OBJ) return 0;
        for (int p = pool[props].first; p >= 0; p = pool[p].next) {
            if (pool[p].klen == plen && !memcmp(pool[p].key, pname, plen)) {
                int tp = json_get(pool, p, "type");
                if (tp >= 0 && pool[tp].type == J_STR)
                    return !(pool[tp].slen == 6 && !memcmp(pool[tp].s, "string", 6));
                return 0;
            }
        }
    }
    return 0;
}

static const char* find_sub(const char* s, const char* end, const char* needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0 || (size_t)(end - s) < nlen) return NULL;
    for (const char* p = s; p + nlen <= end; p++)
        if (!memcmp(p, needle, nlen)) return p;
    return NULL;
}

// The <function=...> blocks of one <tool_call> (Qwen3.5's form), from s to end.
static int parse_xml_functions(const JNode* pool, const ChatRequest* req, const char* s, const char* end,
                               ApiToolCall* calls, int count, int max_calls) {
    const char* p = s;
    while (count < max_calls) {
        const char* ftag = find_sub(p, end, "<function=");
        if (!ftag) break;
        const char* fn_start = ftag + 10;
        const char* fn_end = memchr(fn_start, '>', (size_t)(end - fn_start));
        if (!fn_end) break;
        const char* fend = find_sub(fn_end + 1, end, "</function>");
        if (!fend) fend = end;
        p = fend < end ? fend + 11 : end;

        while (fn_start < fn_end && (*fn_start == ' ' || *fn_start == '\t' || *fn_start == '"')) fn_start++;
        const char* fe = fn_end;
        while (fe > fn_start && (fe[-1] == ' ' || fe[-1] == '\t' || fe[-1] == '"')) fe--;
        size_t flen = (size_t)(fe - fn_start);
        if (flen == 0) continue;

        ApiToolCall* tc = &calls[count];
        memset(tc, 0, sizeof(*tc));
        snprintf(tc->id, sizeof(tc->id), "call_%d", count + 1);
        size_t nk = flen < sizeof(tc->name) - 1 ? flen : sizeof(tc->name) - 1;
        memcpy(tc->name, fn_start, nk);
        tc->name[nk] = 0;
        sb_init(&tc->arguments);
        sb_add(&tc->arguments, "{", 1);

        int n_params = 0;
        const char* q = fn_end + 1;
        while (q < fend) {
            const char* ptag = find_sub(q, fend, "<parameter=");
            if (!ptag) break;
            const char* pn_start = ptag + 11;
            const char* pn_end = memchr(pn_start, '>', (size_t)(fend - pn_start));
            if (!pn_end) break;
            const char* pclose = find_sub(pn_end + 1, fend, "</parameter>");
            if (!pclose) break;
            q = pclose + 12;

            const char* val = pn_end + 1;   // the value, without the line breaks around it
            const char* vend = pclose;
            if (val < vend && *val == '\r') val++;
            if (val < vend && *val == '\n') val++;
            if (vend > val && vend[-1] == '\n') vend--;
            if (vend > val && vend[-1] == '\r') vend--;

            if (n_params++) sb_add(&tc->arguments, ",", 1);
            sb_json_str(&tc->arguments, pn_start, (size_t)(pn_end - pn_start));
            sb_add(&tc->arguments, ":", 1);

            size_t vlen = (size_t)(vend - val);
            int non_str = param_is_non_string(pool, req, tc->name, pn_start, (size_t)(pn_end - pn_start));
            int wrote_raw = 0;
            if (vlen > 0 && (non_str || val[0] == '{' || val[0] == '[')) {
                int vroot;
                JNode* vpool = json_parse_grow(val, vlen, 256, &vroot);
                if (vpool && (non_str ? vpool[vroot].type != J_STR
                                      : (vpool[vroot].type == J_OBJ || vpool[vroot].type == J_ARR))) {
                    json_write(vpool, vroot, &tc->arguments);
                    wrote_raw = 1;
                }
                free(vpool);
            }
            if (!wrote_raw) sb_json_str(&tc->arguments, val, vlen);
        }
        sb_add(&tc->arguments, "}", 1);
        count++;
    }
    return count;
}

// A <tool_call> written as JSON ({"name": ..., "arguments": {...}}), the form of earlier Qwen models.
static int parse_json_function(const char* s, const char* end, ApiToolCall* calls, int count, int max_calls) {
    if (count >= max_calls) return count;
    while (s < end && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')) s++;
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' || end[-1] == '\r')) end--;
    if (s >= end || *s != '{') return count;
    int root;
    JNode* jpool = json_parse_grow(s, (size_t)(end - s), 256, &root);
    int fn = jpool && jpool[root].type == J_OBJ ? tool_func_node(jpool, root) : -1;
    int nm = fn >= 0 ? named_function(jpool, root) : -1;
    if (nm < 0) {
        free(jpool);
        return count;
    }

    ApiToolCall* tc = &calls[count];
    memset(tc, 0, sizeof(*tc));
    snprintf(tc->id, sizeof(tc->id), "call_%d", count + 1);
    Sb ns;
    sb_init(&ns);
    json_str_sb(jpool, nm, &ns);
    size_t nk = ns.n < sizeof(tc->name) - 1 ? ns.n : sizeof(tc->name) - 1;
    if (ns.p) memcpy(tc->name, ns.p, nk);
    tc->name[nk] = 0;
    sb_free(&ns);

    sb_init(&tc->arguments);
    int args = json_get(jpool, fn, "arguments");
    if (args >= 0 && jpool[args].type == J_STR) json_str_sb(jpool, args, &tc->arguments);
    else if (args >= 0 && jpool[args].type == J_OBJ) json_write(jpool, args, &tc->arguments);
    else sb_str(&tc->arguments, "{}");
    free(jpool);
    return count + 1;
}

int api_parse_tool_calls(const JNode* pool, const ChatRequest* req, const char* text, size_t len, Sb* content_out,
                         ApiToolCall* calls, int max_calls) {
    if (!text || len == 0) return 0;
    const char* end = text + len;
    const char* first = find_sub(text, end, "<tool_call>");
    if (!first) return 0;
    if (content_out) {
        const char* ce = first;
        while (ce > text && (ce[-1] == ' ' || ce[-1] == '\t' || ce[-1] == '\n' || ce[-1] == '\r')) ce--;
        if (ce > text) sb_add(content_out, text, (size_t)(ce - text));
    }
    int count = 0;
    const char* p = first;
    while (p < end && count < max_calls) {
        const char* open = find_sub(p, end, "<tool_call>");
        if (!open) break;
        const char* body = open + 11;
        const char* close = find_sub(body, end, "</tool_call>");
        const char* block_end = close ? close : end;
        p = close ? close + 12 : end;
        int before = count;
        count = parse_xml_functions(pool, req, body, block_end, calls, count, max_calls);
        if (count == before) count = parse_json_function(body, block_end, calls, count, max_calls);
    }
    return count;
}
