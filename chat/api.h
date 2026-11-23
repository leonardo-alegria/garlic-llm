// OpenAI-compatible HTTP API on the console (port API_PORT, on the LAN). The server thread (api.c) reads the
// requests and queues one chat job at a time; the main loop (api_run.inc) runs it on the model between turns
// and writes the reply to the socket. Protocol and JSON: api_proto.h.
// The API is off until api_enable turns it on (L3 on the chat screen, main.c): nothing listens before that.
#ifndef CHAT_API_H
#define CHAT_API_H

#include <stddef.h>

#define API_PORT 8080

typedef struct ApiJob {
    int fd;                     // the connection: the main thread writes the whole reply to it
    char* body;                 // OpenAI chat.completions JSON (an Ollama body is converted before it is queued)
    size_t body_len;
    int wire;                   // 0 OpenAI, 1 Ollama /api/chat, 2 Ollama /api/generate
    volatile int done;          // 1 once the main thread has written the reply
    struct ApiJob* next;
} ApiJob;

// Turns the API on (the server listens on API_PORT; the network library is loaded the first time) or off (it stops
// listening; a request already read is still answered). Returns 0, or -1 if it could not be turned on (logged).
int api_enable(int on);
int api_enabled(void);
// Where the API listens while it is on: "192.168.1.5:8080", or ":8080" if the console's address is not known.
const char* api_address(void);
// Main thread: the job waiting for the model, or NULL. Its reply must be written to job->fd, then api_done(job).
ApiJob* api_take(void);
void api_done(ApiJob* job);
// Writes all n bytes to fd. Returns 0, or -1 if the client went away.
int api_send(int fd, const char* p, size_t n);
// The model's name as the API reports it.
void api_set_model(const char* name);
const char* api_model(void);

#endif
