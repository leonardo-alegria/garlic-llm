// OpenAI-compatible HTTP API on the console (port API_PORT, on the LAN). The server thread (api.c) reads the
// requests and queues one chat job at a time; the main loop (api_run.inc) runs it on the model between turns
// and writes the reply to the socket. Protocol and JSON: api_proto.h.
// The API is off until api_enable turns it on (L3 on the chat screen, main.c): nothing listens before that.
#ifndef CHAT_API_H
#define CHAT_API_H

#include <stddef.h>

#ifndef API_PORT                  // (the PC test server, tests/api_host/server.c, picks its own)
#define API_PORT 8080
#endif

typedef struct ApiJob {
    int fd;                     // the connection: the main thread writes the whole reply to it, api_done closes it
    int kind;                   // 0 = /v1/chat/completions, 1 = /v1/audio/speech, 2 = /v1/audio/voices, 3 = /v1/models/select
    char* body;                 // the request's JSON (validated by the server thread)
    size_t body_len;
    struct ApiJob* next;
} ApiJob;

// Turns the API on (the server listens on API_PORT; the network library is loaded the first time) or off (it stops
// listening; a request already read is still answered). Returns 0, or -1 if it could not be turned on (logged).
int api_enable(int on);
int api_enabled(void);
// Where the API listens while it is on: "192.168.1.5:8080", or ":8080" if the console's address is not known.
const char* api_address(void);
// Main thread: the oldest job waiting for the model, or NULL. Its reply must be written to job->fd, then
// api_done(job), which closes the connection and frees the job. Chat requests wait in a line (up to 8, the one
// being answered included; more get 503). The server thread does not wait for them: it goes on answering other
// connections (/v1/models, /health) while the model writes a reply.
ApiJob* api_take(void);
void api_done(ApiJob* job);
// Writes all n bytes to fd. Returns 0, or -1 if the client went away.
int api_send(int fd, const char* p, size_t n);
// 1 if the client has closed fd (a request that waited and is no longer wanted), without waiting.
int api_client_gone(int fd);
// Main thread: whether the console itself is using the model (a chat reply, the keyboard, voice mode, dictation,
// a Python program), so a request would wait. GET /health reports it, with the requests in line.
void api_console_busy(int busy);
// The model's name as the API reports it.
void api_set_model(const char* name);
const char* api_model(void);
// The context window GET /v1/models reports (context_length), in tokens: the engine's max_seq. Clients size their
// prompts by it.
void api_set_context(int tokens);
// Durable task queue (task_queue.h): initialized once at startup, served at /v1/tasks, claimed by the main loop.
#include "task_queue.h"
int api_tasks_init(const char* journal_path);
int api_task_claim(GarlicTask* out);
void api_task_finish(uint32_t id, int ok, const char* text);

#endif
