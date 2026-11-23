// DarkGarlic cluster: the wire protocol and the head's configuration file. No console calls in here, so the host
// tests (tests/cluster_proto_test.c) build it with gcc. The transport and the engine hooks are in cluster.c; the
// design is in docs/CLUSTER.md.
//
// Topology: a star. The head (node 0: screen, tokenizer, API, the first layers) opens one TCP connection to every
// worker (port CLUSTER_PORT) and drives them in order: it sends a chunk of hidden rows to node 1, reads node 1's
// rows back, sends them to node 2, ... and the last node answers with the sampled lm_head row. Workers never talk to
// each other and need no configuration: their shard's model.idx says which node they are.
//
// Every message is a 40-byte header (little endian, ClusterMsg) followed by `payload` bytes. `check` is the FNV-1a
// hash of the payload (0 for none), so a framing bug shows up as an error and not as a wrong answer.
#ifndef CHAT_CLUSTER_PROTO_H
#define CHAT_CLUSTER_PROTO_H

#include <stddef.h>
#include <stdint.h>

#define CLUSTER_PORT 8091
#define CLUSTER_MAGIC 0x4C434744u   // "DGCL"
#define CLUSTER_VERSION 1
#define CLUSTER_HDR 40
#define CLUSTER_MAX_WORKERS 7
#define CLUSTER_MAX_ROWS 64          // hidden rows per RUN message (engine PF_ROWS on a PS4 Pro)
#define CLUSTER_MAX_HIDDEN 8192
#define CLUSTER_MAX_PAYLOAD ((uint32_t)CLUSTER_MAX_ROWS * CLUSTER_MAX_HIDDEN * 4)

enum {
    CL_HELLO = 1,       // head -> worker: no payload. Reply CL_HELLO with a ClusterHello payload.
    CL_RUN = 2,         // head -> worker: n rows of `hidden` floats for positions pos0..pos0+n-1.
                        //   flags & CL_F_SAMPLE: draw the token after the last row (only the last node can).
                        //   Reply CL_ROWS (n rows out) from a middle node, CL_TOKEN from the last node.
    CL_ROWS = 3,
    CL_TOKEN = 4,       // payload: u32 lm_head row (n = 1), or none when the chunk did not ask for a token
    CL_RESET = 5,       // a new conversation: clear the DeltaNet states. Reply CL_OK.
    CL_STATE = 6,       // flags & CL_F_SAVE: save the DeltaNet states to the worker's slot, else put them back. CL_OK.
    CL_PING = 7,        // reply CL_OK
    CL_OK = 8,
    CL_ERROR = 9,       // payload: a message (UTF-8, no terminator)
    CL_BYE = 10,        // head -> worker: the head is closing; the worker waits for the next connection
};

#define CL_F_SAMPLE 1u
#define CL_F_SAVE 2u

typedef struct {
    uint32_t magic;
    uint16_t version, op;
    uint32_t seq;            // the head numbers its requests; replies carry the same number
    uint32_t pos0, n;        // CL_RUN / CL_ROWS: first position and row count
    uint32_t flags;
    float temperature;
    uint32_t seed;
    uint32_t payload;        // bytes after the header
    uint32_t check;          // FNV-1a of the payload
} ClusterMsg;

// What a worker says about itself (CL_HELLO reply): 68 bytes. A peer from before the fingerprint sends 64;
// the missing word is read as 0 and the chain check skips the hash.
typedef struct {
    uint32_t node, nodes, layer_first, layer_end;   // its shard (layer_end exclusive)
    uint32_t n_layers, hidden, vocab, flags;        // the whole model's shape; flags 1 = embed rows, 2 = lm_head
    char name[32];                                  // the model's name from the shard
    uint32_t fingerprint;                           // hash of the shape; 0 = not set
} ClusterHello;
#define CLUSTER_HELLO_BYTES 68

uint32_t cluster_fnv(const void* p, size_t n);
uint32_t cluster_fnv_init(void);
uint32_t cluster_fnv_mix(uint32_t h, const void* p, size_t n);
void cluster_put_hdr(uint8_t out[CLUSTER_HDR], const ClusterMsg* m);
// 0 if the header is ours (magic, version, sane payload size), -1 otherwise.
int cluster_get_hdr(const uint8_t in[CLUSTER_HDR], ClusterMsg* m);
void cluster_put_hello(uint8_t out[CLUSTER_HELLO_BYTES], const ClusterHello* h);
void cluster_get_hello(const uint8_t in[CLUSTER_HELLO_BYTES], ClusterHello* h);

// The head's file (/data/darkgarlic/cluster.txt), one setting per line, '#' starts a comment:
//   worker = 192.168.1.21          (node 1; then node 2 ... in order; optional ":8091" only)
//   name = Abliterated Qwen3.5-27B (optional: the screen / API model name)
//   timeout = 60                   (optional: seconds to wait for the workers at start-up)
typedef struct {
    int n_workers;
    uint8_t ip[CLUSTER_MAX_WORKERS][4];
    uint16_t port[CLUSTER_MAX_WORKERS];
    char name[64];
    int timeout_s;
} ClusterConfig;

// Parses the text (not NUL-terminated: n bytes). Returns 0, or -1 with a message in err (line number included).
int cluster_parse_config(const char* text, size_t n, ClusterConfig* c, char* err, size_t err_n);
int cluster_parse_ip(const char* s, size_t n, uint8_t ip[4], uint16_t* port);

// The head's own shard plus every worker's HELLO must chain: node i of n, layers contiguous from 0 to n_layers, the
// same shape and name everywhere, lm_head only on the last. When both fingerprints are non-zero they must match
// (0 means an older peer that did not send one). 0, or -1 with a message.
int cluster_check_chain(const ClusterHello* head, const ClusterHello* workers, int n_workers, char* err, size_t err_n);

#endif
