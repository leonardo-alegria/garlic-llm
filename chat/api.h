// OpenAI-compatible HTTP API on the console (port API_PORT, on the LAN). The server thread (api.c) reads the
// requests and queues one chat job at a time; the main loop (api_run.inc) runs it on the model between turns
// and writes the reply to the socket. Protocol and JSON: api_proto.h.
#ifndef CHAT_API_H
#define CHAT_API_H

#include <stddef.h>

#define API_PORT 8080

typedef struct ApiJob {
    int fd;                     // the connection: the main thread writes the whole reply to it
    char* body;                 // the request's JSON (validated by the server thread)
    size_t body_len;
    volatile int done;          // 1 once the main thread has written the reply
    struct ApiJob* next;
} ApiJob;

// Starts the server (after the network module is loaded). Returns 0 if listening, -1 otherwise (logged).
int api_start(void);
// Main thread: the job waiting for the model, or NULL. Its reply must be written to job->fd, then api_done(job).
ApiJob* api_take(void);
void api_done(ApiJob* job);
// Writes all n bytes to fd. Returns 0, or -1 if the client went away.
int api_send(int fd, const char* p, size_t n);
// The model's name as the API reports it.
void api_set_model(const char* name);
const char* api_model(void);

#endif
