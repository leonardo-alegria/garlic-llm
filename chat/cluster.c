// DarkGarlic cluster: sockets (the console's sceNet calls, as api.c uses them) and the engine hooks. See cluster.h and
// docs/CLUSTER.md. Nothing here runs on a single console: main.c calls in only when the model is a cluster shard.
#include "cluster.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <orbis/Net.h>
#include <orbis/Sysmodule.h>
#include <orbis/libkernel.h>

#include "cluster_proto.h"

void say(const char* fmt, ...);   // main.c
uint8_t* read_all(const char* path, size_t* size);   // main.c

// BSD socket constants the SDK headers do not name (as in api.c)
#define CL_SOL_SOCKET 0xffff
#define CL_SO_REUSEADDR 0x0004
#define CL_SO_RCVTIMEO 0x1006
#define CL_SO_SNDTIMEO 0x1005
#define CL_IPPROTO_TCP 6
#define CL_TCP_NODELAY 1

typedef struct {
    uint8_t len, family;
    uint16_t port;
    uint32_t addr;
    char zero[8];
} ClSockaddrIn;
_Static_assert(sizeof(ClSockaddrIn) == sizeof(OrbisNetSockaddr), "sockaddr size");

static int net_ready;
static ClusterConfig cfg;
static char summary[200];

int cluster_net_init(void) {
    if (net_ready) return 0;
    int r = sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_NET);
    if (r < 0) {
        say("net: network module 0x%08X", r);
        return -1;
    }
    r = sceNetInit();
    if (r < 0) {
        say("net: sceNetInit 0x%08X", r);
        return -1;
    }
    net_ready = 1;
    return 0;
}

const char* cluster_name(void) { return cfg.name; }
const char* cluster_summary(void) { return summary; }

static void set_timeouts(OrbisNetId fd, long seconds) {   // 0 = wait for ever
    struct {
        long sec, usec;
    } tmo = {seconds, 0};
    sceNetSetsockopt(fd, CL_SOL_SOCKET, CL_SO_RCVTIMEO, &tmo, sizeof(tmo));
    sceNetSetsockopt(fd, CL_SOL_SOCKET, CL_SO_SNDTIMEO, &tmo, sizeof(tmo));
    int one = 1;   // small messages (one row per token) must not wait for Nagle
    sceNetSetsockopt(fd, CL_IPPROTO_TCP, CL_TCP_NODELAY, &one, sizeof(one));
}

static int send_all(OrbisNetId fd, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*)p;
    while (n) {
        int w = sceNetSend(fd, b, n, 0);
        if (w <= 0) return -1;
        b += w;
        n -= (size_t)w;
    }
    return 0;
}

static int recv_all(OrbisNetId fd, void* p, size_t n) {
    uint8_t* b = (uint8_t*)p;
    while (n) {
        int r = sceNetRecv(fd, b, n, 0);
        if (r <= 0) return -1;
        b += r;
        n -= (size_t)r;
    }
    return 0;
}

static int send_msg(OrbisNetId fd, ClusterMsg* m, const void* payload) {
    uint8_t h[CLUSTER_HDR];
    m->check = cluster_fnv(payload, m->payload);
    cluster_put_hdr(h, m);
    if (send_all(fd, h, sizeof(h))) return -1;
    return m->payload ? send_all(fd, payload, m->payload) : 0;
}

// Reads one message; the payload into buf (cap bytes). 0, or -1 (connection, framing or checksum).
static int recv_msg(OrbisNetId fd, ClusterMsg* m, void* buf, size_t cap) {
    uint8_t h[CLUSTER_HDR];
    if (recv_all(fd, h, sizeof(h)) || cluster_get_hdr(h, m) || m->payload > cap) return -1;
    if (m->payload && recv_all(fd, buf, m->payload)) return -1;
    return cluster_fnv(buf, m->payload) == m->check ? 0 : -1;
}

static void hello_of(const Engine* e, ClusterHello* h) {
    memset(h, 0, sizeof(*h));
    h->node = e->node, h->nodes = e->nodes, h->layer_first = e->l0, h->layer_end = e->l1;
    h->n_layers = e->n_layers, h->hidden = e->hidden, h->vocab = e->vocab;
    h->flags = (e->has_embed ? 1u : 0u) | (e->has_lm ? 2u : 0u);
    memcpy(h->name, e->shard_name, 31);
    h->fingerprint = e->idx_hash;
}

// ---- head ----------------------------------------------------------------------------------------------------------

typedef struct {
    OrbisNetId fd;
    ClusterHello hello;
    char where[24];
} Worker;

static Worker workers[CLUSTER_MAX_WORKERS];
static int n_workers;
static uint32_t seq;
static float *rows_a, *rows_b;   // cached copies of the rows on their way (the GPU's are in write-combined memory)
static EngineLink head_link;
static uint64_t sent_rows, runs;

static void cl_fail(Engine* e, const char* fmt, const char* a, const char* b) {
    snprintf(e->error, sizeof(e->error), fmt, a, b);
    say("cluster: %s", e->error);
}

static int head_rows(EngineLink* l, Engine* e, const float* rows, int n, int pos0, int final, float temperature,
                     uint32_t seed, uint32_t* row) {
    (void)l;
    size_t rowb = (size_t)e->hidden * 4;
    // pieces of at most CLUSTER_MAX_ROWS rows; only the last piece of a final chunk asks for the token
    for (int p0 = 0; p0 < n; p0 += CLUSTER_MAX_ROWS) {
        int k = n - p0 < CLUSTER_MAX_ROWS ? n - p0 : CLUSTER_MAX_ROWS;
        int want = final && p0 + k == n;
        memcpy(rows_a, rows + (size_t)p0 * e->hidden, (size_t)k * rowb);
        float* cur = rows_a;
        for (int w = 0; w < n_workers; w++) {
            Worker* wk = &workers[w];
            int last = w == n_workers - 1;
            ClusterMsg m = {0};
            m.op = CL_RUN, m.seq = ++seq, m.pos0 = (uint32_t)(pos0 + p0), m.n = (uint32_t)k;
            m.flags = want ? CL_F_SAMPLE : 0, m.temperature = temperature, m.seed = seed;
            m.payload = (uint32_t)(k * rowb);
            if (send_msg(wk->fd, &m, cur)) {
                cl_fail(e, "worker %s went away (send)%s", wk->where, "");
                return -1;
            }
            ClusterMsg r;
            float* dst = cur == rows_a ? rows_b : rows_a;
            if (recv_msg(wk->fd, &r, dst, (size_t)CLUSTER_MAX_ROWS * rowb) || r.seq != m.seq) {
                cl_fail(e, "worker %s went away or answered out of order%s", wk->where, "");
                return -1;
            }
            if (r.op == CL_ERROR) {
                char msg[120];
                size_t ml = r.payload < sizeof(msg) - 1 ? r.payload : sizeof(msg) - 1;
                memcpy(msg, dst, ml);
                msg[ml] = 0;
                cl_fail(e, "worker %s: %s", wk->where, msg);
                return -1;
            }
            if (!last) {
                if (r.op != CL_ROWS || r.n != (uint32_t)k || r.payload != m.payload) {
                    cl_fail(e, "worker %s sent the wrong rows%s", wk->where, "");
                    return -1;
                }
                cur = dst;
            } else {
                if (r.op != CL_TOKEN || (want && r.payload != 4)) {
                    cl_fail(e, "worker %s did not send a token%s", wk->where, "");
                    return -1;
                }
                if (want) memcpy(row, dst, 4);
            }
        }
        sent_rows += (uint64_t)k;
    }
    runs++;
    return 0;
}

static int broadcast(Engine* e, uint16_t op, uint32_t flags) {
    for (int w = 0; w < n_workers; w++) {
        ClusterMsg m = {0}, r;
        m.op = op, m.seq = ++seq, m.flags = flags;
        uint8_t buf[128];
        if (send_msg(workers[w].fd, &m, NULL) || recv_msg(workers[w].fd, &r, buf, sizeof(buf)) || r.seq != m.seq ||
            r.op != CL_OK) {
            cl_fail(e, "worker %s did not confirm (op %s)", workers[w].where, op == CL_RESET ? "reset" : "state");
            return -1;
        }
    }
    return 0;
}

static int head_reset(EngineLink* l, Engine* e) {
    (void)l;
    return broadcast(e, CL_RESET, 0);
}

static int head_state(EngineLink* l, Engine* e, int save) {
    (void)l;
    return broadcast(e, CL_STATE, save ? CL_F_SAVE : 0);
}

static void head_abort(Engine* e, int n) {
    e->link = NULL;
    for (int i = 0; i < n && i < CLUSTER_MAX_WORKERS; i++) {
        if (workers[i].fd >= 0) {
            sceNetSocketClose(workers[i].fd);
            workers[i].fd = -1;
        }
    }
    free(rows_a);
    free(rows_b);
    rows_a = rows_b = NULL;
    n_workers = 0;
}

static OrbisNetId connect_to(const uint8_t ip[4], uint16_t port) {
    OrbisNetId s = sceNetSocket("dg_cluster", ORBIS_NET_AF_INET, ORBIS_NET_SOCK_STREAM, CL_IPPROTO_TCP);
    if (s < 0) return -1;
    ClSockaddrIn a;
    memset(&a, 0, sizeof(a));
    a.len = sizeof(a);
    a.family = ORBIS_NET_AF_INET;
    a.port = sceNetHtons(port);
    memcpy(&a.addr, ip, 4);
    if (sceNetConnect(s, (const OrbisNetSockaddr*)&a, sizeof(a))) {
        sceNetSocketClose(s);
        return -1;
    }
    set_timeouts(s, 30);
    return s;
}

int cluster_head_start(Engine* e, ClusterStatus status, char* err, size_t err_n) {
    if (!e->hidden || e->hidden > CLUSTER_MAX_HIDDEN) {
        snprintf(err, err_n, "hidden %u does not fit the cluster link (max %u)", e->hidden, CLUSTER_MAX_HIDDEN);
        return -1;
    }
    size_t n = 0;
    uint8_t* text = read_all(CLUSTER_CONFIG, &n);
    if (!text) {
        snprintf(err, err_n, "this model is node 0 of a %u-console cluster but %s is missing", e->nodes,
                 CLUSTER_CONFIG);
        return -1;
    }
    int r = cluster_parse_config((const char*)text, n, &cfg, err, err_n);
    free(text);
    if (r) return -1;
    if (cluster_net_init()) {
        snprintf(err, err_n, "the network library did not start");
        return -1;
    }
    size_t cap = (size_t)CLUSTER_MAX_ROWS * e->hidden * 4;
    rows_a = malloc(cap);
    rows_b = malloc(cap);
    if (!rows_a || !rows_b) {
        free(rows_a);
        free(rows_b);
        rows_a = rows_b = NULL;
        snprintf(err, err_n, "no memory for the cluster's row buffers");
        return -1;
    }
    n_workers = cfg.n_workers;
    for (int w = 0; w < n_workers; w++) workers[w].fd = -1;
    ClusterHello hellos[CLUSTER_MAX_WORKERS], me;
    for (int w = 0; w < n_workers; w++) {
        Worker* wk = &workers[w];
        snprintf(wk->where, sizeof(wk->where), "%u.%u.%u.%u", cfg.ip[w][0], cfg.ip[w][1], cfg.ip[w][2], cfg.ip[w][3]);
        uint64_t t0 = sceKernelGetProcessTime();
        for (;;) {
            char line[120];
            snprintf(line, sizeof(line), "Waiting for worker %d of %d at %s:%u", w + 1, n_workers, wk->where,
                     cfg.port[w]);
            if (status) status(line);
            wk->fd = connect_to(cfg.ip[w], cfg.port[w]);
            if (wk->fd >= 0) break;
            if (sceKernelGetProcessTime() - t0 > (uint64_t)cfg.timeout_s * 1000000) {
                snprintf(err, err_n, "worker %d (%s:%u) did not answer in %d s: is DarkGarlic open on it?", w + 1,
                         wk->where, cfg.port[w], cfg.timeout_s);
                head_abort(e, w + 1);
                return -1;
            }
            sceKernelUsleep(1000000);
        }
        ClusterMsg m = {0}, rr;
        m.op = CL_HELLO, m.seq = ++seq;
        uint8_t hb[CLUSTER_HELLO_BYTES];
        memset(hb, 0, sizeof(hb));
        // 64 bytes: a worker from before the fingerprint. The last word stays 0.
        if (send_msg(wk->fd, &m, NULL) || recv_msg(wk->fd, &rr, hb, sizeof(hb)) || rr.op != CL_HELLO ||
            (rr.payload != CLUSTER_HELLO_BYTES && rr.payload != CLUSTER_HELLO_BYTES - 4)) {
            snprintf(err, err_n, "worker %d (%s) did not say hello", w + 1, wk->where);
            head_abort(e, w + 1);
            return -1;
        }
        cluster_get_hello(hb, &hellos[w]);
        wk->hello = hellos[w];
        say("cluster: worker %d at %s: node %u of %u, layers %u..%u", w + 1, wk->where, hellos[w].node,
            hellos[w].nodes, hellos[w].layer_first, hellos[w].layer_end - 1);
    }
    hello_of(e, &me);
    if (cluster_check_chain(&me, hellos, n_workers, err, err_n)) {
        head_abort(e, n_workers);
        return -1;
    }
    head_link.rows = head_rows;
    head_link.reset = head_reset;
    head_link.state = head_state;
    e->link = &head_link;
    if (broadcast(e, CL_RESET, 0)) {   // every console starts from an empty conversation
        snprintf(err, err_n, "%s", e->error);
        head_abort(e, n_workers);
        return -1;
    }
    snprintf(summary, sizeof(summary), "cluster of %u: layers 0..%u here", e->nodes, e->l1 - 1);
    for (int w = 0, o = (int)strlen(summary); w < n_workers && o < (int)sizeof(summary); w++)
        o += snprintf(summary + o, sizeof(summary) - o, ", %u..%u on %s", hellos[w].layer_first,
                      hellos[w].layer_end - 1, workers[w].where);
    say("cluster: %s", summary);
    return 0;
}

// ---- worker --------------------------------------------------------------------------------------------------------

static float* out_rows;   // the rows this worker produced for the current RUN (worker_rows appends)
static int out_n;
static EngineLink worker_link;

static int worker_rows(EngineLink* l, Engine* e, const float* rows, int n, int pos0, int final, float temperature,
                       uint32_t seed, uint32_t* row) {
    (void)l, (void)pos0, (void)final, (void)temperature, (void)seed, (void)row;
    if (out_n + n > CLUSTER_MAX_ROWS) {
        snprintf(e->error, sizeof(e->error), "more rows out than in");
        return -1;
    }
    memcpy(out_rows + (size_t)out_n * e->hidden, rows, (size_t)n * e->hidden * 4);
    out_n += n;
    return 0;
}

static OrbisNetId worker_listen(void) {
    OrbisNetId s = sceNetSocket("dg_worker", ORBIS_NET_AF_INET, ORBIS_NET_SOCK_STREAM, CL_IPPROTO_TCP);
    if (s < 0) return -1;
    int one = 1;
    sceNetSetsockopt(s, CL_SOL_SOCKET, CL_SO_REUSEADDR, &one, sizeof(one));
    ClSockaddrIn a;
    memset(&a, 0, sizeof(a));
    a.len = sizeof(a);
    a.family = ORBIS_NET_AF_INET;
    a.port = sceNetHtons(CLUSTER_PORT);
    if (sceNetBind(s, (const OrbisNetSockaddr*)&a, sizeof(a)) || sceNetListen(s, 1)) {
        sceNetSocketClose(s);
        return -1;
    }
    return s;
}

static void reply_error(OrbisNetId fd, uint32_t s, const char* msg) {
    ClusterMsg m = {0};
    m.op = CL_ERROR, m.seq = s, m.payload = (uint32_t)strlen(msg);
    send_msg(fd, &m, msg);
}

static void serve(Engine* e, OrbisNetId fd, float* in, ClusterStatus status) {
    size_t cap = (size_t)CLUSTER_MAX_ROWS * e->hidden * 4;
    uint64_t tokens = 0, t_last = sceKernelGetProcessTime();
    for (;;) {
        ClusterMsg m, r = {0};
        if (recv_msg(fd, &m, in, cap)) return;   // the head went away, or a broken message: wait for it again
        r.seq = m.seq;
        switch (m.op) {
        case CL_HELLO: {
            ClusterHello h;
            uint8_t hb[CLUSTER_HELLO_BYTES];
            hello_of(e, &h);
            cluster_put_hello(hb, &h);
            r.op = CL_HELLO, r.payload = CLUSTER_HELLO_BYTES;
            if (send_msg(fd, &r, hb)) return;
            break;
        }
        case CL_RUN: {
            if (m.payload != m.n * e->hidden * 4) {
                reply_error(fd, m.seq, "RUN payload does not match its rows");
                return;
            }
            out_n = 0;
            uint32_t row = 0;
            int sample = (m.flags & CL_F_SAMPLE) && e->has_lm;
            if (engine_run_rows(e, in, (int)m.n, (int)m.pos0, m.temperature, m.seed, sample, &row)) {
                reply_error(fd, m.seq, e->error);
                break;
            }
            if (e->has_lm) {
                r.op = CL_TOKEN, r.n = sample ? 1 : 0, r.payload = sample ? 4 : 0;
                if (send_msg(fd, &r, &row)) return;
            } else {
                if (out_n != (int)m.n) {
                    reply_error(fd, m.seq, "this node made a different number of rows");
                    break;
                }
                r.op = CL_ROWS, r.pos0 = m.pos0, r.n = m.n, r.payload = m.payload;
                if (send_msg(fd, &r, out_rows)) return;
            }
            tokens += m.n;
            uint64_t t = sceKernelGetProcessTime();
            if (status && t - t_last > 2000000) {   // a status line every 2 s while it works
                char line[160];
                snprintf(line, sizeof(line), "DarkGarlic worker: node %u of %u, layers %u..%u - %llu positions served",
                         e->node, e->nodes, e->l0, e->l1 - 1, (unsigned long long)tokens);
                status(line);
                t_last = t;
            }
            break;
        }
        case CL_RESET:
            engine_reset(e);
            r.op = CL_OK;
            if (send_msg(fd, &r, NULL)) return;
            break;
        case CL_STATE:
            if (!e->state_slot || engine_state_copy(e, e->state_slot, (m.flags & CL_F_SAVE) != 0)) {
                reply_error(fd, m.seq, e->state_slot ? e->error : "no memory for the state slot");
                break;
            }
            r.op = CL_OK;
            if (send_msg(fd, &r, NULL)) return;
            break;
        case CL_PING:
            r.op = CL_OK;
            if (send_msg(fd, &r, NULL)) return;
            break;
        case CL_BYE:
            return;
        default:
            reply_error(fd, m.seq, "unexpected message");
            return;
        }
    }
}

void cluster_worker_main(Engine* e, ClusterStatus status) {
    char line[200];
    if (!e->hidden || e->hidden > CLUSTER_MAX_HIDDEN) {
        if (status) status("DarkGarlic worker: hidden size does not fit the cluster link");
        for (;;) sceKernelUsleep(1000000);
    }
    if (cluster_net_init()) {
        if (status) status("DarkGarlic worker: the network library did not start");
        for (;;) sceKernelUsleep(1000000);
    }
    size_t cap = (size_t)CLUSTER_MAX_ROWS * e->hidden * 4;
    float* in = malloc(cap);
    out_rows = malloc(cap);
    uint64_t ss = engine_state_size(e);
    e->state_slot = ss ? engine_alloc(e, ss) : 0;
    worker_link.rows = worker_rows;   // (reset and state come over the network, not from this console)
    e->link = &worker_link;
    OrbisNetId l = worker_listen();
    if (!in || !out_rows || l < 0) {
        if (status) status("DarkGarlic worker: cannot listen on the cluster port");
        for (;;) sceKernelUsleep(1000000);
    }
    for (;;) {
        snprintf(line, sizeof(line), "DarkGarlic worker: node %u of %u, layers %u..%u (%s) - waiting for the head on "
                 "port %d", e->node, e->nodes, e->l0, e->l1 - 1, e->shard_name, CLUSTER_PORT);
        if (status) status(line);
        say("cluster: %s", line);
        ClSockaddrIn peer;
        OrbisNetSocklen_t plen = sizeof(peer);
        OrbisNetId fd = sceNetAccept(l, (OrbisNetSockaddr*)&peer, &plen);
        if (fd < 0) {
            sceKernelUsleep(200000);
            continue;
        }
        set_timeouts(fd, 0);   // no timeout: the head may sit idle for hours between questions
        const uint8_t* b = (const uint8_t*)&peer.addr;
        snprintf(line, sizeof(line), "DarkGarlic worker: node %u of %u, layers %u..%u - head %u.%u.%u.%u connected",
                 e->node, e->nodes, e->l0, e->l1 - 1, b[0], b[1], b[2], b[3]);
        if (status) status(line);
        say("cluster: %s", line);
        serve(e, fd, in, status);
        sceNetSocketClose(fd);
        engine_reset(e);   // the next head starts a new conversation
    }
}
