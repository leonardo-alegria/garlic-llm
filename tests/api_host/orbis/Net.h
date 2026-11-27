// PC stand-in for the console's network calls that chat/api.c uses, so the API's server runs on Windows for the PC
// test server (tests/api_host/server.c). Same names and BSD constants as on the console; host.c implements them with
// Winsock. Only the calls api.c makes are here.
#ifndef API_HOST_ORBIS_NET_H
#define API_HOST_ORBIS_NET_H

#include <stddef.h>
#include <stdint.h>

typedef int OrbisNetId;
typedef uint32_t OrbisNetSocklen_t;
typedef struct OrbisNetSockaddr {   // BSD layout: length, family, then the address (16 bytes in all)
    uint8_t sa_len, sa_family;
    char sa_data[14];
} OrbisNetSockaddr;

#define ORBIS_NET_AF_INET 2
#define ORBIS_NET_SOCK_STREAM 1
#define ORBIS_NET_SOCK_DGRAM 2

int sceNetInit(void);
int sceNetSocket(const char* name, int family, int type, int protocol);
int sceNetSocketClose(OrbisNetId s);
int sceNetSocketAbort(OrbisNetId s, int flags);
int sceNetSetsockopt(OrbisNetId s, int level, int optname, const void* optval, OrbisNetSocklen_t optlen);
int sceNetBind(OrbisNetId s, const OrbisNetSockaddr* addr, OrbisNetSocklen_t addrlen);
int sceNetListen(OrbisNetId s, int backlog);
int sceNetAccept(OrbisNetId s, OrbisNetSockaddr* addr, OrbisNetSocklen_t* addrlen);
int sceNetConnect(OrbisNetId s, const OrbisNetSockaddr* addr, OrbisNetSocklen_t addrlen);
int sceNetGetsockname(OrbisNetId s, OrbisNetSockaddr* addr, OrbisNetSocklen_t* addrlen);
int sceNetSend(OrbisNetId s, const void* msg, size_t len, int flags);
int sceNetRecv(OrbisNetId s, void* buf, size_t len, int flags);
uint16_t sceNetHtons(uint16_t host16);

#endif
