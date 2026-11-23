// DarkGarlic cluster on the console: the network side of docs/CLUSTER.md (protocol: cluster_proto.h).
//
// Node 0 (the head) runs the app as usual; after engine_init, cluster_head_start connects to the workers listed in
// /data/darkgarlic/cluster.txt and hooks the engine, so every engine_run on the head goes through all the consoles
// (main.c and api_run.inc do not change). A console whose shard says node >= 1 calls cluster_worker_main instead of
// the chat loop: it serves the head and never returns.
#ifndef CHAT_CLUSTER_H
#define CHAT_CLUSTER_H

#include <stddef.h>

#include "engine.h"

#define CLUSTER_CONFIG "/data/darkgarlic/cluster.txt"

typedef void (*ClusterStatus)(const char* line);

// The network library, once (api.c uses it too). 0, or -1 (logged).
int cluster_net_init(void);
// Head. Returns 0 when every worker answered and the chain checks out (the engine is hooked), -1 with err set.
int cluster_head_start(Engine* e, ClusterStatus status, char* err, size_t err_n);
// The model name from cluster.txt ("" if none) and the summary line for the log / screen.
const char* cluster_name(void);
const char* cluster_summary(void);
// Worker: never returns.
void cluster_worker_main(Engine* e, ClusterStatus status);

#endif
