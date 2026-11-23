// OpenAI-compatible HTTP API, protocol part: HTTP/1.1 request parsing, a small JSON reader and the JSON the API
// answers with. No console calls here, so the same code is tested on Windows and Linux (tests/api_proto_test.c).
// The server itself is api.c (PS4 sockets); the work on the model is api_run.inc (main.c).
#ifndef CHAT_API_PROTO_H
#define CHAT_API_PROTO_H

#include <stddef.h>
#include <stdint.h>

// ---- growing string -----------------------------------------------------------------------------------------------
typedef struct {
    char* p;
    size_t n, cap;
    int oom;
} Sb;

void sb_init(Sb* s);
void sb_free(Sb* s);
void sb_add(Sb* s, const char* p, size_t n);
void sb_str(Sb* s, const char* z);
void sb_json_str(Sb* s, const char* p, size_t n);   // a JSON string literal, quotes included
void sb_int(Sb* s, long long v);

// ---- HTTP request ---------------------------------------------------------------------------------------------------
typedef struct {
    char method[8];
    char path[256];
    char auth[256];             // the Authorization header's value ("" if none)
    long content_length;        // 0 if none
    const char* body;           // points into the buffer given to api_http_parse
    size_t body_len;
    size_t head_len;            // bytes of head (up to and including the blank line)
} HttpRequest;

// Returns 1 when buf holds a whole request, 0 when more bytes are needed, -1 on a malformed or too big request.
int api_http_parse(const char* buf, size_t len, HttpRequest* req);

// ---- JSON reader --------------------------------------------------------------------------------------------------
// The whole document is parsed into a pool of nodes; children are linked (first child, then next sibling). An
// object's children carry their key in key/klen.
enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ };
typedef struct JNode {
    uint8_t type;
    int b;                      // J_BOOL value
    double num;                 // J_NUM value
    const char* s;              // J_STR: the raw text between the quotes (escapes not decoded)
    size_t slen;
    const char* key;            // member name in an object (raw, as above)
    size_t klen;
    int first, next, count;     // node indices; -1 = none
} JNode;

// Parses one JSON value from s (n bytes) into pool (max nodes). Returns the root index, or -1 on error.
int json_parse(const char* s, size_t n, JNode* pool, int max);
// The member named key of object node obj (-1 if absent or obj is not an object).
int json_get(const JNode* pool, int obj, const char* key);
// Appends the decoded text of a string node to out (escapes, \uXXXX and surrogate pairs turned into UTF-8).
void json_str_sb(const JNode* pool, int node, Sb* out);
// The text of one chat message: its role (copied into role, NUL-terminated) and its content into out. Content may be
// a string or an array of parts (their "text" parts are joined). Returns 0, or -1 if the message is malformed.
int api_message(const JNode* pool, int msg, char* role, size_t rcap, Sb* out);

// ---- JSON writers used by the API ------------------------------------------------------------------------------------
// The JSON of GET /v1/models: one model, the one loaded.
void api_json_models(Sb* out, const char* model, long long created);
// A complete chat.completion (non-streaming).
void api_json_completion(Sb* out, const char* id, const char* model, long long created, const char* text, size_t tlen,
                         const char* finish, int prompt_tokens, int completion_tokens);
// One chat.completion.chunk (streaming). role != NULL: the first chunk, with the role. text may be empty.
void api_json_chunk(Sb* out, const char* id, const char* model, long long created, const char* role, const char* text,
                    size_t tlen, const char* finish);
// The error object: {"error": {"message", "type", "code"}}.
void api_json_error(Sb* out, const char* message, const char* type, const char* code);

// ---- chat request -------------------------------------------------------------------------------------------------
#define API_MAX_NODES 1024
typedef struct {
    int root;                   // root node (in the pool the caller gives)
    int messages;               // the messages array node (-1 if missing)
    int n_messages;
    double temperature;         // default 0.7
    int max_tokens;             // default 512
    int stream;                 // 0 or 1
    unsigned seed;              // default 0 = the engine's own seed
    int has_seed;
} ChatRequest;

// Reads a chat.completions body. Returns 0 if valid (fills req; messages are pool nodes, each an object with role
// and content), or -1 with the reason in err (a fixed string, for the 400 answer).
int api_chat_parse(const char* body, size_t len, JNode* pool, int max, ChatRequest* req, const char** err);

#endif
