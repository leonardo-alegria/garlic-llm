// PC stand-in for the console's threads, mutex and sleep that chat/api.c uses (see Net.h); host.c implements them
// with Win32.
#ifndef API_HOST_ORBIS_LIBKERNEL_H
#define API_HOST_ORBIS_LIBKERNEL_H

#include <stddef.h>

typedef struct ApiHostMutex* OrbisPthreadMutex;
typedef void* OrbisPthreadMutexattr;
typedef void* OrbisPthread;
typedef struct {
    size_t stack;
} OrbisPthreadAttr;

int scePthreadMutexInit(OrbisPthreadMutex* m, const OrbisPthreadMutexattr* attr, const char* name);
int scePthreadMutexLock(OrbisPthreadMutex* m);
int scePthreadMutexUnlock(OrbisPthreadMutex* m);
int scePthreadAttrInit(OrbisPthreadAttr* a);
int scePthreadAttrSetstacksize(OrbisPthreadAttr* a, size_t size);
int scePthreadCreate(OrbisPthread* t, const OrbisPthreadAttr* a, void* (*fn)(void*), void* arg, const char* name);
int sceKernelUsleep(unsigned int usec);

#endif
