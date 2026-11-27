// The console's calls that chat/api.c makes (orbis/*.h), on Windows: sockets with Winsock, threads and the mutex with
// Win32. They behave as on the console where api.c relies on it:
// - addresses are BSD's (a length byte, then the family), turned into Windows' and back;
// - SO_RCVTIMEO and SO_SNDTIMEO take a struct of two longs (seconds, microseconds), as on the console;
// - recv takes MSG_PEEK (0x02) and MSG_DONTWAIT (0x80), which Winsock has no flag for;
// - the server listens on 127.0.0.1 only (a test server is not for the network), and the console's address is
//   not looked up (connect fails, so api_address() is ":<port>").
#include <winsock2.h>
#include <windows.h>

#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "orbis/Net.h"
#include "orbis/Sysmodule.h"
#include "orbis/libkernel.h"

#define BSD_MSG_PEEK 0x02
#define BSD_MSG_DONTWAIT 0x80
#define BSD_SO_REUSEADDR 0x0004
#define BSD_SO_NOSIGPIPE 0x0800
#define BSD_SO_SNDTIMEO 0x1005
#define BSD_SO_RCVTIMEO 0x1006

int sceNetInit(void) {
    WSADATA w;
    return WSAStartup(MAKEWORD(2, 2), &w) ? -1 : 0;
}

int sceSysmoduleLoadModuleInternal(unsigned int id) {
    (void)id;
    return 0;
}

int _sigaction(int sig, const void* act, void* old) {   // (no SIGPIPE on Windows: a send to a closed socket fails)
    (void)sig;
    (void)act;
    (void)old;
    return 0;
}

int sceNetSocket(const char* name, int family, int type, int protocol) {
    (void)name;
    SOCKET s = socket(family, type == ORBIS_NET_SOCK_STREAM ? SOCK_STREAM : SOCK_DGRAM, protocol);
    return s == INVALID_SOCKET ? -1 : (int)s;
}

int sceNetSocketClose(OrbisNetId s) { return closesocket((SOCKET)s) ? -1 : 0; }

int sceNetSocketAbort(OrbisNetId s, int flags) {   // (the test server never turns the API off)
    (void)s;
    (void)flags;
    return 0;
}

int sceNetSetsockopt(OrbisNetId s, int level, int optname, const void* optval, OrbisNetSocklen_t optlen) {
    if (optname == BSD_SO_REUSEADDR || optname == BSD_SO_NOSIGPIPE) return 0;   // (Windows' REUSEADDR shares a port)
    if (optname == BSD_SO_RCVTIMEO || optname == BSD_SO_SNDTIMEO) {
        const long* tv = (const long*)optval;
        DWORD ms = optlen >= 2 * sizeof(long) ? (DWORD)(tv[0] * 1000 + tv[1] / 1000) : 0;
        int name = optname == BSD_SO_RCVTIMEO ? SO_RCVTIMEO : SO_SNDTIMEO;
        return setsockopt((SOCKET)s, SOL_SOCKET, name, (const char*)&ms, sizeof(ms)) ? -1 : 0;
    }
    return setsockopt((SOCKET)s, level, optname, (const char*)optval, (int)optlen) ? -1 : 0;
}

static void to_win(const OrbisNetSockaddr* in, struct sockaddr_in* out) {
    memcpy(out, in, sizeof(*out));
    out->sin_family = in->sa_family;
}

static void to_bsd(const struct sockaddr_in* in, OrbisNetSockaddr* out) {
    memcpy(out, in, sizeof(*out));
    out->sa_len = sizeof(*out);
    out->sa_family = (uint8_t)in->sin_family;
}

int sceNetBind(OrbisNetId s, const OrbisNetSockaddr* addr, OrbisNetSocklen_t addrlen) {
    (void)addrlen;
    struct sockaddr_in a;
    to_win(addr, &a);
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    return bind((SOCKET)s, (struct sockaddr*)&a, sizeof(a)) ? -1 : 0;
}

int sceNetListen(OrbisNetId s, int backlog) { return listen((SOCKET)s, backlog) ? -1 : 0; }

int sceNetAccept(OrbisNetId s, OrbisNetSockaddr* addr, OrbisNetSocklen_t* addrlen) {
    struct sockaddr_in a;
    int len = sizeof(a);
    SOCKET c = accept((SOCKET)s, (struct sockaddr*)&a, &len);
    if (c == INVALID_SOCKET) return -1;
    if (addr) to_bsd(&a, addr);
    if (addrlen) *addrlen = sizeof(*addr);
    return (int)c;
}

int sceNetConnect(OrbisNetId s, const OrbisNetSockaddr* addr, OrbisNetSocklen_t addrlen) {
    (void)s;
    (void)addr;
    (void)addrlen;
    return -1;
}

int sceNetGetsockname(OrbisNetId s, OrbisNetSockaddr* addr, OrbisNetSocklen_t* addrlen) {
    struct sockaddr_in a;
    int len = sizeof(a);
    if (getsockname((SOCKET)s, (struct sockaddr*)&a, &len)) return -1;
    to_bsd(&a, addr);
    *addrlen = sizeof(*addr);
    return 0;
}

int sceNetSend(OrbisNetId s, const void* msg, size_t len, int flags) {
    (void)flags;
    int n = send((SOCKET)s, (const char*)msg, len > 1 << 30 ? 1 << 30 : (int)len, 0);
    return n == SOCKET_ERROR ? -1 : n;
}

int sceNetRecv(OrbisNetId s, void* buf, size_t len, int flags) {
    if (flags & BSD_MSG_DONTWAIT) {   // nothing to read now: "would block"
        fd_set r;
        FD_ZERO(&r);
        FD_SET((SOCKET)s, &r);
        struct timeval zero = {0, 0};
        if (select(0, &r, NULL, NULL, &zero) != 1) return -1;
    }
    int n = recv((SOCKET)s, (char*)buf, len > 1 << 30 ? 1 << 30 : (int)len, (flags & BSD_MSG_PEEK) ? MSG_PEEK : 0);
    return n == SOCKET_ERROR ? -1 : n;
}

uint16_t sceNetHtons(uint16_t host16) { return htons(host16); }

// ---- threads and the mutex ----------------------------------------------------------------------------------------

struct ApiHostMutex {
    CRITICAL_SECTION cs;
};

int scePthreadMutexInit(OrbisPthreadMutex* m, const OrbisPthreadMutexattr* attr, const char* name) {
    (void)attr;
    (void)name;
    *m = malloc(sizeof(**m));
    if (!*m) return -1;
    InitializeCriticalSection(&(*m)->cs);
    return 0;
}

int scePthreadMutexLock(OrbisPthreadMutex* m) {
    EnterCriticalSection(&(*m)->cs);
    return 0;
}

int scePthreadMutexUnlock(OrbisPthreadMutex* m) {
    LeaveCriticalSection(&(*m)->cs);
    return 0;
}

int scePthreadAttrInit(OrbisPthreadAttr* a) {
    a->stack = 0;
    return 0;
}

int scePthreadAttrSetstacksize(OrbisPthreadAttr* a, size_t size) {
    a->stack = size;
    return 0;
}

typedef struct {
    void* (*fn)(void*);
    void* arg;
} Start;

static DWORD WINAPI start(LPVOID p) {
    Start st = *(Start*)p;
    free(p);
    st.fn(st.arg);
    return 0;
}

int scePthreadCreate(OrbisPthread* t, const OrbisPthreadAttr* a, void* (*fn)(void*), void* arg, const char* name) {
    (void)name;
    Start* st = malloc(sizeof(*st));
    if (!st) return -1;
    st->fn = fn;
    st->arg = arg;
    HANDLE h = CreateThread(NULL, a ? a->stack : 0, start, st, 0, NULL);   // (the console's stack size, 256 KiB)
    if (!h) {
        free(st);
        return -1;
    }
    *t = h;
    return 0;
}

int sceKernelUsleep(unsigned int usec) {
    Sleep(usec < 1000 ? 1 : usec / 1000);
    return 0;
}

// ---- for server.c ---------------------------------------------------------------------------------------------------

void host_sleep_ms(int ms) { Sleep(ms > 0 ? (DWORD)ms : 0); }

double host_now(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
