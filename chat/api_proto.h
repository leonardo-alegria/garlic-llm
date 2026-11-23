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

// ---- tools (Qwen3.5 chat template) -----------------------------------------------------------------------------------
// Schemas are copied out of the JSON pool: the pool is freed before the reply is parsed. Names that do not fit are
// cut; a tool past API_MAX_TOOLS is still rendered into the prompt, and its parameters are typed as strings.
#define API_MAX_TOOLS 8
#define API_MAX_PARAMS 8
#define API_MAX_CALLS 8
#define API_TOOL_NAME 64
#define API_PARAM_NAME 48
#define API_TOOL_ARGS 2048
enum { API_P_STR, API_P_INT, API_P_NUM, API_P_BOOL, API_P_OBJ, API_P_ARR };
typedef struct {
    char name[API_PARAM_NAME];
    int type;                   // API_P_*
} ApiParam;
typedef struct {
    char name[API_TOOL_NAME];
    int n_params;
    ApiParam params[API_MAX_PARAMS];
} ApiToolSchema;
typedef struct {
    char name[API_TOOL_NAME];
    char arguments[API_TOOL_ARGS];   // a JSON object, as text
} ApiToolCall;

// ---- JSON writers used by the API ------------------------------------------------------------------------------------
// The JSON of GET /v1/models: one model, the one loaded.
void api_json_models(Sb* out, const char* model, long long created);
// A complete chat.completion (non-streaming). ncalls == 0 keeps the previous JSON. With calls, an empty content is
// null and finish is the caller's string (the job passes "tool_calls"). arguments is a JSON string. call_seq builds
// ids call_<seq>_<index>.
void api_json_completion(Sb* out, const char* id, const char* model, long long created, const char* text, size_t tlen,
                         const char* finish, int prompt_tokens, int completion_tokens, const ApiToolCall* calls,
                         int ncalls, unsigned call_seq);
// One chat.completion.chunk (streaming). role != NULL: the first chunk, with the role. text may be empty.
void api_json_chunk(Sb* out, const char* id, const char* model, long long created, const char* role, const char* text,
                    size_t tlen, const char* finish);
// One streaming tool call: the whole arguments object in a single chunk.
void api_json_tool_chunk(Sb* out, const char* id, const char* model, long long created, int index, const char* call_id,
                         const char* name, const char* arguments);
// The error object: {"error": {"message", "type", "code"}}.
void api_json_error(Sb* out, const char* message, const char* type, const char* code);

// Ollama (same port as the OpenAI API). Chat and generate replies are one JSON object, or NDJSON when streamed.
// done_reason is "stop" or "length". Counts are written only on the final object.
// ncalls == 0 keeps the previous JSON. With calls, arguments that start with '{' are written as a JSON object.
void api_json_ollama_chat(Sb* out, const char* model, const char* created_at, const char* text, size_t tlen, int done,
                          const char* reason, int prompt_n, int eval_n, const ApiToolCall* calls, int ncalls);
void api_json_ollama_gen(Sb* out, const char* model, const char* created_at, const char* text, size_t tlen, int done,
                         const char* reason, int prompt_n, int eval_n);
void api_json_ollama_tags(Sb* out, const char* model);
void api_json_ollama_show(Sb* out, const char* model);
void api_json_ollama_version(Sb* out);
void api_json_ollama_error(Sb* out, const char* message);
// Turns an Ollama /api/chat (generate = 0) or /api/generate (generate = 1) body into an OpenAI chat.completions
// body, so the job runner stays one path. stream defaults to true. num_predict -1 becomes 4096. Returns 0, or -1
// with a fixed string in *err.
int api_ollama_as_openai(const char* body, size_t n, int generate, Sb* out, const char** err);

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
    int tools;                  // the tools array node, or -1 when there are none
    int n_tools;                // 0 when tools are absent or empty: the prompt is unchanged
    int n_schemas;              // how many of schemas[] were copied (at most API_MAX_TOOLS)
    ApiToolSchema schemas[API_MAX_TOOLS];
} ChatRequest;

// The system turn Qwen3.5 prepends when tools is a non-empty list: the <tools> block, the call format, and, when
// the first message is system, its trimmed text. *ate_system is 1 when that first message was merged (the caller
// skips it). Returns 0, or -1 if the first message is malformed.
int api_tools_system(const JNode* pool, int tools, int first_msg, Sb* out, int* ate_system);
// Appends the <tool_call> XML of an assistant message. arguments may be a JSON object or a JSON string of one.
// content_nonempty puts a blank line before the first call, as the template does. Returns 0, or -1 if a call is
// malformed.
int api_message_tool_xml(const JNode* pool, int msg, int content_nonempty, Sb* out);
// Reads model text. Complete calls are stored (an unfinished last call is dropped, *cut = 1). When at least one
// call is stored, content is the trimmed text before the first <tool_call>; otherwise content is the raw text.
// A call whose arguments do not fit API_TOOL_ARGS is dropped. Returns 0.
int api_parse_tool_calls(const char* text, size_t n, const ApiToolSchema* schemas, int nschemas, Sb* content,
                         ApiToolCall* calls, int max_calls, int* ncalls, int* cut);
// How many leading bytes of s are safe to stream: everything before a <tool_call> marker. A suffix that is only a
// prefix of the marker is held, so a marker split across tokens is not shown.
size_t api_tool_stream_end(const char* s, size_t n);

// Reads a chat.completions body. Returns 0 if valid (fills req; messages are pool nodes, each an object with role
// and content), or -1 with the reason in err (a fixed string, for the 400 answer).
int api_chat_parse(const char* body, size_t len, JNode* pool, int max, ChatRequest* req, const char** err);

#endif
