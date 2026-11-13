// OpenAI-compatible HTTP API, server part on the console (see api.h). One thread accepts the connections and
// reads their requests; the chat requests go to the main loop one at a time (api_run.inc), which owns the model.
// The sockets are the console's sceNet calls, declared here as the OpenOrbis headers do not have them all.
#include "api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <orbis/Net.h>
#include <orbis/Sysmodule.h>
#include <orbis/libkernel.h>

#include "api_proto.h"
#include "ui.h"

#define API_ASSETS "/app0/assets"
#define API_KEY_FILE "/data/garlic_api_key.txt"   // optional: if it holds a key, requests need "Authorization: Bearer <key>"
#define API_MAX_BODY (1 << 20)

// Socket constants that the SDK's headers do not name (the BSD values the console's libSceNet uses)
#define API_SOL_SOCKET 0xffff
#define API_SO_REUSEADDR 0x0004
#define API_SO_RCVTIMEO 0x1006
#define API_SO_SNDTIMEO 0x1005
#define API_IPPROTO_TCP 6

typedef struct {                // the BSD sockaddr_in: 16 bytes, the same size as OrbisNetSockaddr
    uint8_t len, family;
    uint16_t port;              // network byte order
    uint32_t addr;
    char zero[8];
} ApiSockaddrIn;
_Static_assert(sizeof(ApiSockaddrIn) == sizeof(OrbisNetSockaddr), "sockaddr size");

static OrbisPthreadMutex api_mu;
static ApiJob* queued;          // waiting for the main loop
static int busy;                // a chat job is queued or running (api_done clears it)
static char model_name[64] = "Garlic";
static char* key;               // the API key, or NULL
static OrbisPthread api_th;

void api_set_model(const char* name) {
    snprintf(model_name, sizeof(model_name), "%s", name);
}

const char* api_model(void) { return model_name; }

int api_send(int fd, const char* p, size_t n) {
    while (n) {
        int w = sceNetSend(fd, p, n, 0);
        if (w <= 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

static void reply(int fd, int status, const char* reason, const char* ctype, const char* body, size_t n) {
    char head[512];
    int h = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                     "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                     status, reason, ctype, n);
    if (api_send(fd, head, h) == 0) api_send(fd, body, n);
}

static void reply_json_error(int fd, int status, const char* reason, const char* message, const char* type,
                             const char* code) {
    Sb s;
    sb_init(&s);
    api_json_error(&s, message, type, code);
    reply(fd, status, reason, "application/json", s.p, s.n);
    sb_free(&s);
}

ApiJob* api_take(void) {
    scePthreadMutexLock(&api_mu);
    ApiJob* j = queued;
    queued = NULL;
    scePthreadMutexUnlock(&api_mu);
    return j;
}

void api_done(ApiJob* job) {
    scePthreadMutexLock(&api_mu);
    busy = 0;
    scePthreadMutexUnlock(&api_mu);
    job->done = 1;
}

// the header value of Authorization must be "Bearer <key>"
static int authorized(const HttpRequest* r) {
    if (!key) return 1;
    size_t k = strlen(key);
    return strncmp(r->auth, "Bearer ", 7) == 0 && strlen(r->auth + 7) == k && !memcmp(r->auth + 7, key, k);
}

static int read_request(int fd, char** buf_out, size_t* len_out, HttpRequest* req) {
    size_t cap = 16384, len = 0;
    char* buf = malloc(cap);
    if (!buf) return -1;
    int r = 0;
    for (;;) {
        if (len == cap) {
            if (cap >= API_MAX_BODY + 16384) break;
            char* nb = realloc(buf, cap * 2);
            if (!nb) break;
            buf = nb;
            cap *= 2;
        }
        int n = sceNetRecv(fd, buf + len, cap - len, 0);
        if (n <= 0) break;
        len += (size_t)n;
        r = api_http_parse(buf, len, req);
        if (r != 0) break;
    }
    if (r == 1) {   // the body points into buf: keep the buffer until the reply is written
        *buf_out = buf;
        *len_out = len;
        return 1;
    }
    free(buf);
    return r == 0 ? 0 : -1;
}

static void serve_chat(int fd, HttpRequest* req) {
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    ChatRequest cr;
    const char* err = NULL;
    if (!pool || api_chat_parse(req->body, req->body_len, pool, API_MAX_NODES, &cr, &err)) {
        free(pool);
        reply_json_error(fd, 400, "Bad Request", err ? err : "Out of memory.", "invalid_request_error", "invalid_body");
        return;
    }
    free(pool);
    ApiJob* job = calloc(1, sizeof(ApiJob));
    char* body = malloc(req->body_len + 1);
    if (!job || !body) {
        free(job);
        free(body);
        reply_json_error(fd, 500, "Internal Server Error", "Out of memory.", "server_error", NULL);
        return;
    }
    memcpy(body, req->body, req->body_len);
    body[req->body_len] = 0;
    job->fd = fd;
    job->body = body;
    job->body_len = req->body_len;
    scePthreadMutexLock(&api_mu);
    int full = busy;
    if (!full) {
        busy = 1;
        queued = job;
    }
    scePthreadMutexUnlock(&api_mu);
    if (full) {
        free(body);
        free(job);
        reply_json_error(fd, 503, "Service Unavailable", "The model is busy with another request.", "server_error",
                         "busy");
        return;
    }
    say("api: chat request, %zu bytes, %s", req->body_len, cr.stream ? "streamed" : "whole reply");
    // the main loop writes the reply to fd; the connection is closed here once it is written
    while (!job->done) sceKernelUsleep(5000);
    free(job->body);
    free(job);
}

static void handle(int fd) {
    struct {
        long sec, usec;
    } tmo = {20, 0};
    sceNetSetsockopt(fd, API_SOL_SOCKET, API_SO_RCVTIMEO, &tmo, sizeof(tmo));
    sceNetSetsockopt(fd, API_SOL_SOCKET, API_SO_SNDTIMEO, &tmo, sizeof(tmo));
    char* buf = NULL;
    size_t len = 0;
    HttpRequest req;
    int r = read_request(fd, &buf, &len, &req);
    if (r != 1) {   // malformed, too big, or cut off before the end
        reply_json_error(fd, 400, "Bad Request", "Malformed or incomplete request.", "invalid_request_error", NULL);
        return;
    }
    if (!strcmp(req.method, "OPTIONS")) {
        const char* h = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
                        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        api_send(fd, h, strlen(h));
    } else if (!authorized(&req)) {
        reply_json_error(fd, 401, "Unauthorized", "Missing or wrong API key.", "invalid_request_error",
                         "invalid_api_key");
    } else if (!strcmp(req.method, "GET") && (!strcmp(req.path, "/v1/models") || !strcmp(req.path, "/models"))) {
        Sb s;
        sb_init(&s);
        api_json_models(&s, model_name, 0);
        reply(fd, 200, "OK", "application/json", s.p, s.n);
        sb_free(&s);
    } else if (!strcmp(req.method, "GET") && !strcmp(req.path, "/health")) {
        reply(fd, 200, "OK", "application/json", "{\"status\":\"ok\"}", 15);
    } else if (!strcmp(req.method, "GET") && (!strcmp(req.path, "/openapi.json") || !strcmp(req.path, "/"))) {
        size_t n = 0;
        uint8_t* spec = read_all(API_ASSETS "/openapi.json", &n);
        if (spec) reply(fd, 200, "OK", "application/json", (const char*)spec, n);
        else reply_json_error(fd, 404, "Not Found", "The API description is missing.", "invalid_request_error", NULL);
        free(spec);
    } else if (!strcmp(req.method, "POST") && !strcmp(req.path, "/v1/chat/completions")) {
        serve_chat(fd, &req);
        free(buf);
        return;   // (serve_chat keeps the connection until the main loop has written the reply)
    } else {
        reply_json_error(fd, 404, "Not Found", "No such endpoint.", "invalid_request_error", NULL);
    }
    free(buf);
}

static void* api_main(void* arg) {
    (void)arg;
    for (;;) {
        ApiSockaddrIn peer;
        OrbisNetSocklen_t plen = sizeof(peer);
        int fd = sceNetAccept(*(OrbisNetId*)arg, (OrbisNetSockaddr*)&peer, &plen);
        if (fd < 0) {
            sceKernelUsleep(100000);
            continue;
        }
        handle(fd);
        sceNetSocketClose(fd);
    }
    return NULL;
}

static OrbisNetId listen_fd = -1;

int api_start(void) {
    scePthreadMutexInit(&api_mu, NULL, "garlic_api");
    // the network library (as the SDK's net_http sample starts it), then the socket
    int r = sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_NET);
    if (r < 0) {
        say("api: network module 0x%08X; the API is off", r);
        return -1;
    }
    r = sceNetInit();
    if (r < 0) {
        say("api: sceNetInit 0x%08X; the API is off", r);
        return -1;
    }
    size_t n = 0;
    uint8_t* kd = read_all(API_KEY_FILE, &n);
    if (kd) {
        while (n && (kd[n - 1] == '\n' || kd[n - 1] == '\r' || kd[n - 1] == ' ')) n--;
        if (n) {
            key = malloc(n + 1);
            if (key) {
                memcpy(key, kd, n);
                key[n] = 0;
            }
        }
        free(kd);
    }
    listen_fd = sceNetSocket("garlic_api", ORBIS_NET_AF_INET, ORBIS_NET_SOCK_STREAM, API_IPPROTO_TCP);
    if (listen_fd < 0) {
        say("api: no socket (%d); the API is off", listen_fd);
        return -1;
    }
    int one = 1;
    sceNetSetsockopt(listen_fd, API_SOL_SOCKET, API_SO_REUSEADDR, &one, sizeof(one));
    ApiSockaddrIn a;
    memset(&a, 0, sizeof(a));
    a.len = sizeof(a);
    a.family = ORBIS_NET_AF_INET;
    a.port = sceNetHtons(API_PORT);
    a.addr = 0;   // any address
    if (sceNetBind(listen_fd, (const OrbisNetSockaddr*)&a, sizeof(a)) || sceNetListen(listen_fd, 4)) {
        say("api: cannot listen on port %d; the API is off", API_PORT);
        sceNetSocketClose(listen_fd);
        return -1;
    }
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 256 << 10);
    if (scePthreadCreate(&api_th, &attr, api_main, &listen_fd, "garlic_api")) {
        say("api: no thread; the API is off");
        return -1;
    }
    say("api: OpenAI-compatible API on port %d%s", API_PORT, key ? " (API key required)" : "");
    return 0;
}
