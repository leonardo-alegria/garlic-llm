// DarkGarlic cluster protocol (cluster_proto.h). Plain C99, no console calls.
#include "cluster_proto.h"

#include <stdio.h>
#include <string.h>

uint32_t cluster_fnv_init(void) { return 2166136261u; }

uint32_t cluster_fnv_mix(uint32_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 16777619u;
    }
    return h;
}

uint32_t cluster_fnv(const void* p, size_t n) {
    return n ? cluster_fnv_mix(cluster_fnv_init(), p, n) : 0;
}

typedef char cluster_hello_bytes_ok[(sizeof(ClusterHello) == CLUSTER_HELLO_BYTES) ? 1 : -1];

static void put32(uint8_t* o, uint32_t v) {
    o[0] = (uint8_t)v, o[1] = (uint8_t)(v >> 8), o[2] = (uint8_t)(v >> 16), o[3] = (uint8_t)(v >> 24);
}
static uint32_t get32(const uint8_t* i) {
    return (uint32_t)i[0] | (uint32_t)i[1] << 8 | (uint32_t)i[2] << 16 | (uint32_t)i[3] << 24;
}
static void put16(uint8_t* o, uint16_t v) { o[0] = (uint8_t)v, o[1] = (uint8_t)(v >> 8); }
static uint16_t get16(const uint8_t* i) { return (uint16_t)(i[0] | i[1] << 8); }

void cluster_put_hdr(uint8_t o[CLUSTER_HDR], const ClusterMsg* m) {
    uint32_t t;
    memcpy(&t, &m->temperature, 4);
    put32(o, CLUSTER_MAGIC);
    put16(o + 4, CLUSTER_VERSION);
    put16(o + 6, m->op);
    put32(o + 8, m->seq);
    put32(o + 12, m->pos0);
    put32(o + 16, m->n);
    put32(o + 20, m->flags);
    put32(o + 24, t);
    put32(o + 28, m->seed);
    put32(o + 32, m->payload);
    put32(o + 36, m->check);
}

int cluster_get_hdr(const uint8_t i[CLUSTER_HDR], ClusterMsg* m) {
    m->magic = get32(i);
    m->version = get16(i + 4);
    m->op = get16(i + 6);
    m->seq = get32(i + 8);
    m->pos0 = get32(i + 12);
    m->n = get32(i + 16);
    m->flags = get32(i + 20);
    uint32_t t = get32(i + 24);
    memcpy(&m->temperature, &t, 4);
    m->seed = get32(i + 28);
    m->payload = get32(i + 32);
    m->check = get32(i + 36);
    if (m->magic != CLUSTER_MAGIC || m->version != CLUSTER_VERSION) return -1;
    if (m->op < CL_HELLO || m->op > CL_BYE) return -1;
    if (m->payload > CLUSTER_MAX_PAYLOAD) return -1;
    if ((m->op == CL_RUN || m->op == CL_ROWS) && (m->n == 0 || m->n > CLUSTER_MAX_ROWS)) return -1;
    return 0;
}

void cluster_put_hello(uint8_t o[CLUSTER_HELLO_BYTES], const ClusterHello* h) {
    const uint32_t v[8] = {h->node, h->nodes, h->layer_first, h->layer_end, h->n_layers, h->hidden, h->vocab, h->flags};
    for (int k = 0; k < 8; k++) put32(o + 4 * k, v[k]);
    memcpy(o + 32, h->name, 32);
    o[63] = 0;
    put32(o + 64, h->fingerprint);
}

void cluster_get_hello(const uint8_t i[CLUSTER_HELLO_BYTES], ClusterHello* h) {
    uint32_t* v[8] = {&h->node, &h->nodes, &h->layer_first, &h->layer_end, &h->n_layers, &h->hidden, &h->vocab, &h->flags};
    for (int k = 0; k < 8; k++) *v[k] = get32(i + 4 * k);
    memcpy(h->name, i + 32, 32);
    h->name[31] = 0;
    h->fingerprint = get32(i + 64);
}

static int is_space(char c) { return c == ' ' || c == '\t' || c == '\r'; }

int cluster_parse_ip(const char* s, size_t n, uint8_t ip[4], uint16_t* port) {
    size_t i = 0;
    *port = CLUSTER_PORT;
    for (int k = 0; k < 4; k++) {
        unsigned v = 0, digits = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9' && digits < 4) v = v * 10 + (unsigned)(s[i++] - '0'), digits++;
        if (!digits || v > 255) return -1;
        ip[k] = (uint8_t)v;
        if (k < 3) {
            if (i >= n || s[i] != '.') return -1;
            i++;
        }
    }
    if (i < n && s[i] == ':') {
        i++;
        unsigned v = 0, digits = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9' && digits < 6) v = v * 10 + (unsigned)(s[i++] - '0'), digits++;
        if (!digits || v == 0 || v > 65535) return -1;
        *port = (uint16_t)v;
    }
    return i == n ? 0 : -1;
}

int cluster_parse_config(const char* text, size_t n, ClusterConfig* c, char* err, size_t err_n) {
    memset(c, 0, sizeof(*c));
    c->timeout_s = 60;
    int line = 0;
    for (size_t p = 0; p < n;) {
        size_t e = p;
        while (e < n && text[e] != '\n') e++;
        line++;
        size_t a = p, b = e;
        for (size_t k = a; k < b; k++)
            if (text[k] == '#') {
                b = k;
                break;
            }
        while (a < b && is_space(text[a])) a++;
        while (b > a && is_space(text[b - 1])) b--;
        p = e + 1;
        if (a == b) continue;
        size_t eq = a;
        while (eq < b && text[eq] != '=') eq++;
        if (eq == b) {
            snprintf(err, err_n, "cluster.txt line %d: expected key = value", line);
            return -1;
        }
        size_t ka = a, kb = eq, va = eq + 1, vb = b;
        while (kb > ka && is_space(text[kb - 1])) kb--;
        while (va < vb && is_space(text[va])) va++;
        size_t kl = kb - ka;
        if (kl == 6 && !memcmp(text + ka, "worker", 6)) {
            if (c->n_workers == CLUSTER_MAX_WORKERS) {
                snprintf(err, err_n, "cluster.txt line %d: at most %d workers", line, CLUSTER_MAX_WORKERS);
                return -1;
            }
            if (cluster_parse_ip(text + va, vb - va, c->ip[c->n_workers], &c->port[c->n_workers])) {
                snprintf(err, err_n, "cluster.txt line %d: bad address (want 192.168.1.21 or 192.168.1.21:8091)",
                         line);
                return -1;
            }
            if (c->port[c->n_workers] != CLUSTER_PORT) {
                snprintf(err, err_n, "cluster.txt line %d: workers listen on port %d only", line, CLUSTER_PORT);
                return -1;
            }
            c->n_workers++;
        } else if (kl == 4 && !memcmp(text + ka, "name", 4)) {
            size_t l = vb - va < sizeof(c->name) - 1 ? vb - va : sizeof(c->name) - 1;
            memcpy(c->name, text + va, l);
            c->name[l] = 0;
        } else if (kl == 7 && !memcmp(text + ka, "timeout", 7)) {
            int v = 0;
            size_t k = va;
            while (k < vb && text[k] >= '0' && text[k] <= '9' && v < 100000) v = v * 10 + (text[k++] - '0');
            if (k != vb || v < 1) {
                snprintf(err, err_n, "cluster.txt line %d: timeout must be seconds", line);
                return -1;
            }
            c->timeout_s = v;
        } else {
            snprintf(err, err_n, "cluster.txt line %d: unknown setting", line);
            return -1;
        }
    }
    if (!c->n_workers) {
        snprintf(err, err_n, "cluster.txt: no worker lines");
        return -1;
    }
    return 0;
}

int cluster_check_chain(const ClusterHello* head, const ClusterHello* w, int nw, char* err, size_t err_n) {
    uint32_t nodes = (uint32_t)nw + 1;
    if (head->node != 0 || head->nodes != nodes) {
        snprintf(err, err_n, "this console's shard is node %u of %u, cluster.txt lists %u workers", head->node,
                 head->nodes, (unsigned)nw);
        return -1;
    }
    if (head->layer_first != 0 || !(head->flags & 1)) {
        snprintf(err, err_n, "node 0 must start at layer 0 and hold the embedding rows");
        return -1;
    }
    uint32_t next = head->layer_end;
    for (int i = 0; i < nw; i++) {
        const ClusterHello* h = &w[i];
        if (h->node != (uint32_t)i + 1 || h->nodes != nodes) {
            snprintf(err, err_n, "worker %d says it is node %u of %u (want %d of %u): check the order in cluster.txt",
                     i + 1, h->node, h->nodes, i + 1, nodes);
            return -1;
        }
        if (h->n_layers != head->n_layers || h->hidden != head->hidden || h->vocab != head->vocab ||
            strncmp(h->name, head->name, 32)) {
            snprintf(err, err_n, "worker %d has a different model (%.32s)", i + 1, h->name);
            return -1;
        }
        if (head->fingerprint && h->fingerprint && head->fingerprint != h->fingerprint) {
            snprintf(err, err_n, "worker %d was converted from different weights", i + 1);
            return -1;
        }
        if (h->layer_first != next || h->layer_end <= h->layer_first) {
            snprintf(err, err_n, "worker %d has layers %u..%u, the chain needs %u next", i + 1, h->layer_first,
                     h->layer_end, next);
            return -1;
        }
        int last = i == nw - 1;
        if (!!(h->flags & 2) != last) {
            snprintf(err, err_n, "worker %d %s lm_head", i + 1, last ? "lacks" : "should not have");
            return -1;
        }
        next = h->layer_end;
    }
    if (next != head->n_layers) {
        snprintf(err, err_n, "the layers end at %u of %u", next, head->n_layers);
        return -1;
    }
    if (head->flags & 2) {
        snprintf(err, err_n, "node 0 has lm_head but there are workers");
        return -1;
    }
    return 0;
}
