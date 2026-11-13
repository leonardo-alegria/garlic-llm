// Host test for chat/api_run.inc (the API's model side), with a mock engine and tokenizer, so the job logic runs
// without the PS4. The real file is included as main.c includes it.
// Build and run: gcc -std=gnu99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/api_run_test.c chat/api_proto.c -o build/api_run_test && build/api_run_test
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "api.h"
#include "api_proto.h"

// ---- mock model and tokenizer ----------------------------------------------------------------------------------------
#define IM_START 1000u
#define IM_END 1001u
#define THINK 1002u
#define THINK_END 1003u
#define EOS2 1004u

typedef struct {
    int qwen35, ctx_limit;
    uint32_t eos, eos2;
    char error[160];
} MockEngine;
static MockEngine eng = {1, 4096, 1005, EOS2, ""};
typedef struct {
    int dummy;
} Tokenizer;
static Tokenizer tok;
static int pos, need_close, hist_n, hist_done, thinking, think_turn, sys_pre, msgs_cleared;
static unsigned seed = 1;
static uint32_t next_token;
typedef struct {
    int scroll;
} ChatView;
static ChatView view;
static int fail_engine_at = -1;   // engine_run fails on this call (-1: never)

// Scripted reply: the model emits script[k], then IM_END. Tokens 500/501 are the two halves of "é".
static const uint32_t* script;
static int script_len, script_k, engine_calls, prompt_len_seen;
static char wire[1 << 16];
static size_t wire_n;
static int wire_fail;   // the client has gone: sends fail
static int sends_left = -1;   // sends that still succeed (-1: all)

int api_send(int fd, const char* p, size_t n) {
    (void)fd;
    if (wire_fail) return -1;
    if (sends_left == 0) return -1;
    if (sends_left > 0) sends_left--;
    if (wire_n + n >= sizeof(wire)) return -1;
    memcpy(wire + wire_n, p, n);
    wire_n += n;
    wire[wire_n] = 0;
    return 0;
}
const char* api_model(void) { return "Qwen3.5-9B"; }
static void say(const char* fmt, ...) {
    (void)fmt;
}
static void msgs_clear(void) { msgs_cleared++; }
static int tok_encode(const Tokenizer* t, const char* s, uint32_t* out, int max);
// as main.c
static int encode_into(uint32_t* ids, int n, int max, const char* text) { return n + tok_encode(&tok, text, ids + n, max - n); }
static void engine_reset(MockEngine* e) { (void)e; }

// one byte per token (id = byte + 1), so prompts are easy to count
static int tok_encode(const Tokenizer* t, const char* s, uint32_t* out, int max) {
    (void)t;
    int n = 0;
    for (const char* p = s; *p && n < max; p++) out[n++] = (uint8_t)*p + 1;
    return n;
}
static int tok_piece(const Tokenizer* t, uint32_t id, const uint8_t** bytes) {
    (void)t;
    static const uint8_t e_lead[] = {0xC3}, e_tail[] = {0xA9};
    if (id == 500) {
        *bytes = e_lead;
        return 1;
    }
    if (id == 501) {
        *bytes = e_tail;
        return 1;
    }
    if (id >= 1000) {
        *bytes = NULL;
        return 0;
    }
    static uint8_t one[1];
    one[0] = (uint8_t)(id - 1);
    *bytes = one;
    return 1;
}

static int engine_run(MockEngine* e, const uint32_t* tokens, int n, int pos0, float temperature, unsigned sd,
                      uint32_t* next) {
    (void)temperature;
    (void)sd;
    (void)pos0;
    engine_calls++;
    if (engine_calls == fail_engine_at) {
        snprintf(e->error, sizeof(e->error), "mock GPU fault");
        return -1;
    }
    if (n > 1) prompt_len_seen = n;   // the prompt (fed in one go)
    if (script_k < script_len) *next = script[script_k++];
    else *next = IM_END;
    (void)tokens;
    return 0;
}

#include "api_run.inc"

// ---- test helpers --------------------------------------------------------------------------------------------------
static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

static void run_job(const char* body, const uint32_t* sc, int sc_len) {
    memset(wire, 0, sizeof(wire));
    wire_n = 0;
    script = sc;
    script_len = sc_len;
    script_k = 0;
    engine_calls = 0;
    ApiJob job;
    memset(&job, 0, sizeof(job));
    job.body = (char*)body;
    job.body_len = strlen(body);
    job.fd = 7;
    api_job(&job);
}

static const char* body_of(const char* w) {
    const char* b = strstr(w, "\r\n\r\n");
    return b ? b + 4 : w;
}

static void test_completion(void) {
    static const uint32_t sc[] = {'H' + 1, 'o' + 1, 'l' + 1, 'a' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":10}", sc, 4);
    CHECK(strncmp(wire, "HTTP/1.1 200 OK\r\nContent-Type: application/json", 46) == 0);
    CHECK(strstr(wire, "\"content\":\"Hola\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"stop\"") != NULL);
    // prompt (one id per byte in the mock): <im_start> "user" "\n" "hi" <im_end> "\n" = 10;
    // <im_start> "assistant\n" = 11; <think> "\n\n" </think> "\n\n" = 6 (Qwen3.5 without thinking). Total 27.
    CHECK(prompt_len_seen == 27);
    CHECK(strstr(wire, "\"prompt_tokens\":27") != NULL);
    CHECK(strstr(wire, "\"completion_tokens\":4") != NULL);
    CHECK(pos == 0 && msgs_cleared >= 1);
}

static void test_length(void) {
    static const uint32_t sc[] = {'H' + 1, 'e' + 1, 'l' + 1, 'l' + 1, 'o' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":3}", sc, 5);
    CHECK(strstr(wire, "\"content\":\"Hel\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"length\"") != NULL);
    CHECK(strstr(wire, "\"completion_tokens\":3") != NULL);
}

static void test_stream(void) {
    static const uint32_t sc[] = {'A' + 1, 500, 501, 'B' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true}", sc, 4);
    CHECK(strstr(wire, "Content-Type: text/event-stream") != NULL);
    const char* role = strstr(wire, "\"delta\":{\"role\":\"assistant\"}");
    CHECK(role != NULL);
    CHECK(strstr(wire, "\"content\":\"A\"") != NULL);
    // the split é arrives as one event, whole
    CHECK(strstr(wire, "\"content\":\"\xc3\xa9\"") != NULL);
    CHECK(strstr(wire, "\"content\":\"B\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"stop\"") != NULL);
    CHECK(strstr(wire, "data: [DONE]\n\n") != NULL);
    CHECK(strstr(wire, "data: [DONE]\n\n") > strstr(wire, "\"finish_reason\":\"stop\""));
    CHECK(strstr(body_of(wire), "\"content\":\"\xc3\xa9\"") != NULL);
}

static void test_errors(void) {
    run_job("{\"messages\":[]}", NULL, 0);
    CHECK(strncmp(wire, "HTTP/1.1 400 Bad Request", 23) == 0);
    CHECK(strstr(wire, "\"type\":\"invalid_request_error\"") != NULL);

    eng.ctx_limit = 100;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":200}", NULL, 0);
    CHECK(strncmp(wire, "HTTP/1.1 400 Bad Request", 23) == 0);
    CHECK(strstr(wire, "context_length_exceeded") != NULL);
    eng.ctx_limit = 4096;

    static const uint32_t sc[] = {'A' + 1};
    fail_engine_at = 1;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", sc, 1);
    fail_engine_at = -1;
    CHECK(strncmp(wire, "HTTP/1.1 500", 12) == 0);
    CHECK(strstr(wire, "mock GPU fault") != NULL);
    CHECK(pos == 0);
}

static void test_stream_error(void) {
    static const uint32_t sc[] = {'A' + 1, 'B' + 1};
    fail_engine_at = 2;   // the second step (after the first token) fails
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true}", sc, 2);
    fail_engine_at = -1;
    CHECK(strstr(wire, "\"message\":\"mock GPU fault\"") != NULL);
    CHECK(strstr(wire, "data: [DONE]") == NULL);   // a failed stream does not pretend to finish
}

static void test_client_gone(void) {
    // the header and the role event go out; the client leaves before the first text event
    static const uint32_t sc[] = {'A' + 1, 'B' + 1, 'C' + 1};
    memset(wire, 0, sizeof(wire));
    wire_n = 0;
    sends_left = 2;
    script = sc;
    script_len = 3;
    script_k = 0;
    engine_calls = 0;
    ApiJob job;
    memset(&job, 0, sizeof(job));
    const char* body = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true}";
    job.body = (char*)body;
    job.body_len = strlen(body);
    job.fd = 7;
    api_job(&job);
    sends_left = -1;
    CHECK(engine_calls == 1);                       // the prefill only: no more tokens for a client that left
    CHECK(strstr(wire, "data: [DONE]") == NULL);
    CHECK(pos == 0);
}

static void test_utf8(void) {
    CHECK(api_utf8_whole((const uint8_t*)"\xc3", 1) == 0);
    CHECK(api_utf8_whole((const uint8_t*)"a\xc3", 2) == 1);
    CHECK(api_utf8_whole((const uint8_t*)"\xc3\xa9", 2) == 2);
    CHECK(api_utf8_whole((const uint8_t*)"\xf0\x9f\x98", 3) == 0);
    CHECK(api_utf8_whole((const uint8_t*)"ab", 2) == 2);
}

int main(void) {
    test_utf8();
    test_completion();
    test_length();
    test_stream();
    test_errors();
    test_stream_error();
    test_client_gone();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
