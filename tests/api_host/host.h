// What the PC test server (server.c) needs of Windows besides the console's calls (orbis/*.h); host.c.
#ifndef API_HOST_H
#define API_HOST_H

void host_sleep_ms(int ms);
double host_now(void);   // seconds, from an arbitrary start

#endif
