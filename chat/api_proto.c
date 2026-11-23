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

void api_json_completion(Sb* out, const char* id, const char* model, long long created, const char* text, size_t tlen,
                         const char* finish, int prompt_tokens, int completion_tokens) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    sb_json_str(out, text, tlen);
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

int api_chat_parse(const char* body, size_t len, JNode* pool, int max, ChatRequest* req, const char** err) {
    memset(req, 0, sizeof(*req));
    req->temperature = 0.7;
    req->max_tokens = 512;
    req->messages = -1;
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
    return 0;
}
