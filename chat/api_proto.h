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
    size_t head_len;            // bytes of head (up to and including the blank line), once the head is whole
} HttpRequest;

#define API_MAX_BODY (1 << 20)    // the largest body the API reads (1 MiB)
// Returns 1 when buf holds a whole request, 0 when more bytes are needed, -1 on a malformed request, or -2 if its
// Content-Length is over API_MAX_BODY (req->content_length says how much the client sends).
int api_http_parse(const char* buf, size_t len, HttpRequest* req);

// ---- JSON reader --------------------------------------------------------------------------------------------------
// The whole document is parsed into a pool of nodes; children are linked (first child, then next sibling). An
// object's children carry their key in key/klen.
enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ };
typedef struct JNode {
    uint8_t type;
    int b;                      // J_BOOL value
    double num;                 // J_NUM value
    const char* s;              // J_STR: the raw text between the quotes (escapes not decoded); J_NUM: as written
    size_t slen;
    const char* key;            // member name in an object (raw, as above)
    size_t klen;
    int first, next, count;     // node indices; -1 = none
} JNode;

// Parses one JSON value from s (n bytes) into pool (max nodes). Returns the root index, -1 if s is not valid JSON,
// or -2 if it has more values than max.
int json_parse(const char* s, size_t n, JNode* pool, int max);
// The same into a pool that grows as s needs (first nodes, then 8 times more each time, up to API_MAX_VALUES).
// Returns the pool, which the caller frees, with the root in *root; or NULL (*root -1: not valid JSON, -2: too many
// values, -3: out of memory).
JNode* json_parse_grow(const char* s, size_t n, int first, int* root);
// The member named key of object node obj (-1 if absent or obj is not an object).
int json_get(const JNode* pool, int obj, const char* key);
// Appends the decoded text of a string node to out (escapes, \uXXXX and surrogate pairs turned into UTF-8).
void json_str_sb(const JNode* pool, int node, Sb* out);
// The text of one chat message: its role (copied into role, NUL-terminated) and its content into out. Content may be
// a string or an array of parts (their "text" parts are joined). Returns 0, or -1 if the message is malformed.
int api_message(const JNode* pool, int msg, char* role, size_t rcap, Sb* out);
// The reasoning switch a user message may end with: "/think" asks for reasoning in this reply, "/no_think" (or
// "/nothink") for none. It is taken out of text, with the spaces before it. Returns 1, 0, or -1 if there is none.
int api_think_switch(Sb* text);
// An earlier assistant message as the model reads it: reasoning a client sent back in the text (anything up to the
// last "</think>") is left out, as Qwen's chat template leaves it out, with the line breaks after it. If reasoning is
// not NULL it gets that reasoning (what Qwen's template keeps of it), trimmed.
void api_drop_think(Sb* text, Sb* reasoning);
// Removes the spaces and line breaks at both ends of text (the chat template trims every message).
void api_trim(Sb* text);
// Any parsed JSON value written back as compact JSON (strings and numbers exactly as they were written).
void json_write(const JNode* pool, int node, Sb* out);
// The same with a space after each ',' and ':', as the chat template writes the tools.
void json_write_spaced(const JNode* pool, int node, Sb* out);

// ---- tool calls ---------------------------------------------------------------------------------------------------
// Qwen3.5 calls a tool by writing <tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>
// (its chat template's format); the API turns that into OpenAI's tool_calls, and OpenAI's tools into that format.
#define API_MAX_TOOL_CALLS 16
enum { TOOL_NONE = 0, TOOL_AUTO = 1, TOOL_REQUIRED = 2, TOOL_NAMED = 3 };

typedef struct {
    char id[40];                // "call_1", ... (api_run.inc makes them unique: "call_<reply>_1")
    char name[128];             // the function's name (OpenAI allows 64 characters)
    Sb arguments;               // its arguments as a JSON object, e.g. {"query":"ps4"}
} ApiToolCall;

void api_free_tool_calls(ApiToolCall* calls, int n);

// ---- JSON writers used by the API ------------------------------------------------------------------------------------
// The JSON of GET /v1/models: one model, the one loaded, with its context window in tokens (context_length).
void api_json_models(Sb* out, const char* model, long long created, int context_length);
// That model alone (GET /v1/models/<id>).
void api_json_model(Sb* out, const char* model, long long created, int context_length);
// The JSON of GET /health: {"status":"ok","busy":..,"waiting":..}. busy: a request sent now would wait (the model is
// answering one, or the console is using it); waiting: requests in line besides the one being answered.
void api_json_health(Sb* out, int busy, int waiting);
// A complete chat.completion (non-streaming). The reasoning (rlen > 0) goes in message.reasoning_content, and its
// tokens, which completion_tokens includes, in usage.completion_tokens_details.reasoning_tokens. With tool calls
// (n_calls > 0) the message carries tool_calls, and its content is null when there is no text besides them.
void api_json_completion(Sb* out, const char* id, const char* model, long long created, const char* reasoning,
                         size_t rlen, const char* text, size_t tlen, const ApiToolCall* calls, int n_calls,
                         const char* finish, int prompt_tokens, int completion_tokens, int reasoning_tokens);
// A chat.completion with optional synthesized audio (base64 WAV/PCM16 + transcript).
void api_json_completion_audio(Sb* out, const char* id, const char* model, long long created, const char* reasoning,
                               size_t rlen, const char* text, size_t tlen, const ApiToolCall* calls, int n_calls,
                               const char* audio_id, const uint8_t* audio_data, size_t audio_len, const char* finish,
                               int prompt_tokens, int completion_tokens, int reasoning_tokens);
// One streaming chat.completion.chunk carrying synthesized audio (base64 data + transcript).
void api_json_chunk_audio(Sb* out, const char* id, const char* model, long long created, const char* audio_id,
                          const uint8_t* audio_data, size_t audio_len, const char* transcript, size_t tlen);
// One chat.completion.chunk (streaming). role != NULL: the first chunk, with the role. A piece of the reasoning goes
// in delta.reasoning_content, a piece of the answer in delta.content; either may be empty.
void api_json_chunk(Sb* out, const char* id, const char* model, long long created, const char* role,
                    const char* reasoning, size_t rlen, const char* text, size_t tlen, const char* finish);
// One chat.completion.chunk with one whole tool call (delta.tool_calls[0], at index).
void api_json_chunk_tool(Sb* out, const char* id, const char* model, long long created, int index,
                         const ApiToolCall* call);
// The last chunk of a stream that asked for stream_options.include_usage: no choices, the usage of the whole reply
// (the same numbers as a whole reply's usage).
void api_json_chunk_usage(Sb* out, const char* id, const char* model, long long created, int prompt_tokens,
                          int completion_tokens, int reasoning_tokens);
// The error object: {"error": {"message", "type", "code"}}.
void api_json_error(Sb* out, const char* message, const char* type, const char* code);

// ---- chat request -------------------------------------------------------------------------------------------------
// A request is read into a pool of API_MAX_NODES nodes (64 bytes each), and one 8 times bigger while that is too
// small, up to API_MAX_VALUES (4 MiB): an agent's request (a long conversation, many tools with their JSON schemas)
// has thousands of values. The pool is freed once the request has been read (the server's memory is tight).
#define API_MAX_NODES 4096
#define API_MAX_VALUES 65536
typedef struct {
    int root;                   // root node (in the pool the caller gives)
    int messages;               // the messages array node (-1 if missing)
    int n_messages;
    int tools;                  // the tools array node (-1 if missing or empty)
    int n_tools;
    int tool_choice;            // TOOL_NONE, TOOL_AUTO (the default with tools), TOOL_REQUIRED, TOOL_NAMED
    char tool_name[128];        // the function tool_choice names (TOOL_NAMED)
    int parallel_tool_calls;    // default 1; 0: at most one call per reply
    double temperature;         // default 0.7
    int max_tokens;             // what the request gives (max_completion_tokens or max_tokens), 0 if neither:
                                // then the reply may use the rest of the context window
    int stream;                 // 0 or 1
    int include_usage;          // stream_options.include_usage: a stream ends with a chunk of usage
    unsigned seed;              // default 0 = the engine's own seed
    int has_seed;
    // Reasoning asked for by the request: 1 or 0 from chat_template_kwargs.enable_thinking, enable_thinking, think
    // or reasoning_effort ("none" and "minimal" mean 0); -1 if none of them (the console's own setting decides).
    int think;
    int want_audio;             // 1 if audio reply was requested (via modalities or audio)
    int audio_pcm;              // 0 = wav (default), 1 = pcm16 (raw 24 kHz s16le)
    char audio_voice[32];       // "" / "auto", language code, or custom voice name
} ChatRequest;

typedef struct {
    Sb input;
    char voice[32];             // "auto", language code, or custom voice name
    int pcm;                    // 0 = wav, 1 = pcm16
} SpeechRequest;

// Reads a /v1/audio/speech body ({"input": "...", "voice": "...", "response_format": "wav"|"pcm"}).
int api_speech_parse(const char* body, size_t len, JNode* pool, int max, SpeechRequest* req, const char** err);

// Reads a chat.completions body. Returns 0 if valid (fills req; messages are pool nodes, each an object with role
// and content), or -1 with the reason in err (a fixed string, for the 400 answer); -2 if the body has more JSON values
// than the pool holds.
int api_chat_parse(const char* body, size_t len, JNode* pool, int max, ChatRequest* req, const char** err);
// api_chat_parse into a pool that grows as the body needs (see API_MAX_NODES). Returns the pool, which the caller
// frees, or NULL with the reason in err (err NULL: out of memory).
JNode* api_chat_parse_pool(const char* body, size_t len, ChatRequest* req, const char** err);

// The system message of a request with tools: Qwen3.5's tool block (each tool's JSON and how to call one), then the
// request's own system text (user_sys, may be empty). The tools take part of the context window like the messages.
void api_system_tools(const JNode* pool, const ChatRequest* req, const char* user_sys, Sb* out);
// An assistant message's tool_calls in Qwen3.5's format, into out. Returns how many, or -1 if malformed.
int api_assistant_tool_calls(const JNode* pool, int msg, Sb* out);
// The tool calls in the model's reply (text, len): up to max_calls into calls (OpenAI's form). The text before the
// first <tool_call> goes to content_out. Returns how many were read. Arguments whose schema is not a string (numbers,
// booleans, objects) are given as JSON values.
int api_parse_tool_calls(const JNode* pool, const ChatRequest* req, const char* text, size_t len, Sb* content_out,
                         ApiToolCall* calls, int max_calls);

#endif
