// PC stand-in for the console's module loader (see Net.h): the network library is always there.
#ifndef API_HOST_ORBIS_SYSMODULE_H
#define API_HOST_ORBIS_SYSMODULE_H

#define ORBIS_SYSMODULE_INTERNAL_NET 0x80000009

int sceSysmoduleLoadModuleInternal(unsigned int id);

#endif
