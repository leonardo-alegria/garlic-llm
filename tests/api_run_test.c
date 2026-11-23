// Host test for chat/api_run.inc (the API's model side), with a mock engine and tokenizer, so the job logic runs
// without the PS4. The real file is included as main.c includes it. Every test runs twice: one token per step
// (engine_run), and several per step as with the MTP head (engine_spec_step).
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
static int fail_engine_at = -1;   // the engine fails on this call (-1: never)
static int job_wire;               // ApiJob.wire for the next run_job (0 OpenAI, 1 Ollama chat, 2 generate)
#define GARLIC_SPEC 1             // (as the 9B app: api_run.inc asks spec_active before every step)
static int spec_on;               // the mock MTP head is on: a step keeps up to spec_keep tokens of its draft
static int spec_keep = 2;
static int next_pos, pos_errors;  // where the next token must go; steps given another position
static int fed;                   // tokens of the reply fed to the model
// what main.c shows of a job: begun and ended (they must pair), and Circle held from stop_at tokens on (-1: never)
static int chat_begun, chat_ended, chat_replies, stop_at = -1;

// Scripted reply: the model emits script[k], then IM_END. Tokens 500/501 are the two halves of "é".
static const uint32_t* script;
static int script_len, script_k, engine_calls, prompt_len_seen;
static char prompt_text[16384];
static int prompt_text_n;
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
static double now(void) { return 0; }
static void api_chat_begin(void) { chat_begun++, msgs_cleared++; }   // (the real one clears the conversation)
static void api_chat_reply(void) { chat_replies++; }
static int api_chat_step(int produced) { return stop_at >= 0 && produced >= stop_at; }
static void api_chat_end(void) { chat_ended++; }
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
    engine_calls++;
    if (engine_calls == fail_engine_at) {
        snprintf(e->error, sizeof(e->error), "mock GPU fault");
        return -1;
    }
    if (n > 1) {   // the prompt (fed in one go, from position 0)
        prompt_len_seen = n;
        next_pos = 0;
        prompt_text_n = 0;
        for (int i = 0; i < n && prompt_text_n + 1 < (int)sizeof(prompt_text); i++) {
            if (tokens[i] == 0 || tokens[i] >= 1000) continue;
            prompt_text[prompt_text_n++] = (char)(tokens[i] - 1);
        }
        prompt_text[prompt_text_n] = 0;
    }
    else fed++;
    if (pos0 != next_pos) pos_errors++;
    next_pos += n;
    if (script_k < script_len) *next = script[script_k++];
    else *next = IM_END;
    (void)tokens;
    return 0;
}

// as engine_spec.inc: tok goes in at pos0, up to max_new - 1 tokens of the draft are kept, and the next one is drawn
static int spec_active(const MockEngine* e) {
    (void)e;
    return spec_on;
}
static int engine_spec_step(MockEngine* e, uint32_t tk, int pos0, float temperature, unsigned sd, int max_new,
                            uint32_t* out, int* n_out, uint32_t* next) {
    (void)tk;
    (void)temperature;
    (void)sd;
    engine_calls++;
    if (engine_calls == fail_engine_at) {
        snprintf(e->error, sizeof(e->error), "mock GPU fault");
        return -1;
    }
    int k = 0;
    while (k < spec_keep && k < max_new - 1 && script_k < script_len) out[k++] = script[script_k++];
    *n_out = k;
    if (pos0 != next_pos) pos_errors++;
    next_pos += 1 + k;
    fed += 1 + k;
    if (script_k < script_len) *next = script[script_k++];
    else *next = IM_END;
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
    fed = 0;
    prompt_text_n = 0;
    prompt_text[0] = 0;
    ApiJob job;
    memset(&job, 0, sizeof(job));
    job.body = (char*)body;
    job.body_len = strlen(body);
    job.fd = 7;
    job.wire = job_wire;
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
    int cleared = msgs_cleared, begun = chat_begun;
    run_job("{\"messages\":[]}", NULL, 0);
    CHECK(strncmp(wire, "HTTP/1.1 400 Bad Request", 23) == 0);
    CHECK(strstr(wire, "\"type\":\"invalid_request_error\"") != NULL);

    eng.ctx_limit = 100;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":200}", NULL, 0);
    CHECK(strncmp(wire, "HTTP/1.1 400 Bad Request", 23) == 0);
    CHECK(strstr(wire, "context_length_exceeded") != NULL);
    eng.ctx_limit = 4096;
    CHECK(msgs_cleared == cleared && chat_begun == begun);   // a refused request leaves the conversation on screen

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

// Several tokens per step: the reply is the same, in fewer steps, and never past max_tokens.
static void test_spec(void) {
    if (!spec_on) return;
    static const uint32_t sc[] = {'H' + 1, 'o' + 1, 'l' + 1, 'a' + 1, ' ' + 1, 'P' + 1, 'S' + 1, '4' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":64}", sc, 8);
    CHECK(strstr(wire, "\"content\":\"Hola PS4\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"stop\"") != NULL);
    CHECK(strstr(wire, "\"completion_tokens\":8") != NULL);
    CHECK(engine_calls == 1 + 3);   // the prompt, then H + ol, a + " P", S + 4 (8 steps one token at a time)
    CHECK(fed == 8);

    spec_keep = 8;   // the draft would keep more than max_tokens allows
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}", sc, 8);
    spec_keep = 2;
    CHECK(strstr(wire, "\"content\":\"Hola\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"length\"") != NULL);
    CHECK(strstr(wire, "\"completion_tokens\":4") != NULL);
    CHECK(fed == 4);   // the model read what was written, nothing more

    // a draft never holds the end of the turn; if one did, the reply would end there all the same
    static const uint32_t end_in_draft[] = {'A' + 1, 'B' + 1, IM_END, 'C' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", end_in_draft, 4);
    CHECK(strstr(wire, "\"content\":\"AB\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"stop\"") != NULL);
    CHECK(strstr(wire, "\"completion_tokens\":2") != NULL);
}

// Circle on the controller: the reply ends there, cut short as by max_tokens, and the client gets what was written.
static void test_stop(void) {
    static const uint32_t sc[] = {'A' + 1, 'B' + 1, 'C' + 1, 'D' + 1, 'E' + 1, 'F' + 1};
    stop_at = 2;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", sc, 6);
    // one token per step stops at "AB"; with the MTP head the first step had already given "BCD" (B, C kept; D drawn)
    CHECK(strstr(wire, spec_on ? "\"content\":\"ABCD\"" : "\"content\":\"AB\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"length\"") != NULL);
    CHECK(strstr(wire, "HTTP/1.1 200 OK") != NULL);

    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true}", sc, 6);
    stop_at = -1;
    CHECK(strstr(wire, "\"content\":\"A\"") != NULL);
    CHECK(strstr(wire, "\"content\":\"F\"") == NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"length\"") != NULL);
    CHECK(strstr(wire, "data: [DONE]\n\n") != NULL);   // the stream closes as usual
    CHECK(pos == 0);
}

// The prompt's limit is counted in tokens, and nothing is ever written past api_ids.
static char big[40000];
static const char* long_body(int text_bytes, int empty_messages, int max_tokens) {
    size_t n = (size_t)sprintf(big, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
    memset(big + n, 'x', (size_t)text_bytes);
    n += (size_t)text_bytes;
    n += (size_t)sprintf(big + n, "\"}");
    for (int i = 0; i < empty_messages; i++) n += (size_t)sprintf(big + n, ",{\"role\":\"user\"}");
    sprintf(big + n, "],\"max_tokens\":%d}", max_tokens);
    return big;
}

static void test_long_prompt(void) {
    int cleared = msgs_cleared, begun = chat_begun;
    // many messages without text after a long one: once more ids than api_ids holds (it used to write past it)
    run_job(long_body(15000, 509, 1), NULL, 0);
    CHECK(strncmp(wire, "HTTP/1.1 400 Bad Request", 23) == 0);
    CHECK(strstr(wire, "context_length_exceeded") != NULL);
    CHECK(engine_calls == 0);
    CHECK(msgs_cleared == cleared && chat_begun == begun);

    // more than the old 15000 bytes of text is fine while the tokens fit the context
    static const uint32_t sc[] = {'O' + 1, 'K' + 1};
    eng.ctx_limit = API_IDS;
    run_job(long_body(15500, 0, 2), sc, 2);
    CHECK(strncmp(wire, "HTTP/1.1 200 OK", 15) == 0);
    // <im_start> "user" "\n" 15500 bytes <im_end> "\n" = 15508, and the assistant's turn 17 (see test_completion)
    CHECK(prompt_len_seen == 15525);
    CHECK(strstr(wire, "\"content\":\"OK\"") != NULL);

    // text that does not fit the ids at all (the encoder stops at the limit: never a cut prompt)
    run_job(long_body(16400, 0, 1), NULL, 0);
    CHECK(strstr(wire, "context_length_exceeded") != NULL);
    eng.ctx_limit = 4096;

    // a malformed message is not called too long
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":5}]}", NULL, 0);
    CHECK(strncmp(wire, "HTTP/1.1 400 Bad Request", 23) == 0);
    CHECK(strstr(wire, "invalid_body") != NULL);
    CHECK(strstr(wire, "context_length_exceeded") == NULL);
}

// A whole reply cut by max_tokens between the two halves of "é" ends before it: the JSON stays valid UTF-8.
static void test_cut_char(void) {
    static const uint32_t sc[] = {'a' + 1, 500, 501};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":2}", sc, 3);
    CHECK(strstr(wire, "\"content\":\"a\"") != NULL);
    CHECK(strchr(body_of(wire), '\xc3') == NULL);
    CHECK(strstr(wire, "\"completion_tokens\":2") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"length\"") != NULL);

    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":3}", sc, 3);   // whole: kept
    CHECK(strstr(wire, "\"content\":\"a\xc3\xa9\"") != NULL);
}

static void test_ollama_job(void) {
    static const uint32_t sc[] = {'H' + 1, 'o' + 1, 'l' + 1, 'a' + 1};
    job_wire = 1;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":10}", sc, 4);
    CHECK(strstr(wire, "Content-Type: application/x-ndjson") != NULL);
    CHECK(strstr(wire, "\"content\":\"H\"") != NULL);
    CHECK(strstr(wire, "\"content\":\"a\"") != NULL);
    CHECK(strstr(wire, "\"done\":true") != NULL);
    CHECK(strstr(wire, "\"done_reason\":\"stop\"") != NULL);
    CHECK(strstr(wire, "\"prompt_eval_count\":") != NULL);
    CHECK(strstr(wire, "data:") == NULL);
    CHECK(strstr(wire, "text/event-stream") == NULL);

    job_wire = 2;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":false,\"max_tokens\":10}", sc, 4);
    CHECK(strncmp(wire, "HTTP/1.1 200 OK\r\nContent-Type: application/json", 46) == 0);
    CHECK(strstr(wire, "\"response\":\"Hola\"") != NULL);
    CHECK(strstr(wire, "\"done\":true") != NULL);
    CHECK(strstr(wire, "\"done_reason\":\"stop\"") != NULL);
    CHECK(strstr(wire, "ndjson") == NULL);

    job_wire = 1;
    run_job("{\"messages\":[]}", NULL, 0);
    CHECK(strncmp(wire, "HTTP/1.1 400", 12) == 0);
    CHECK(strstr(wire, "{\"error\":") != NULL);
    CHECK(strstr(wire, "invalid_request_error") == NULL);

    fail_engine_at = 2;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,\"max_tokens\":10}", sc, 4);
    fail_engine_at = -1;
    CHECK(strstr(wire, "{\"error\":\"mock GPU fault\"}") != NULL);
    CHECK(strstr(wire, "data:") == NULL);
    job_wire = 0;
}

static int ids_of(const char* s, uint32_t* out, int max) {
    int n = 0;
    for (; *s && n < max; s++) out[n++] = (unsigned char)*s + 1;
    return n;
}

static void test_tools(void) {
    static const uint32_t hi[] = {'H' + 1, 'i' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[],\"max_tokens\":10}", hi, 2);
    CHECK(prompt_len_seen == 27);   // an empty tools list leaves the prompt as it was

    const char* hist =
        "{\"messages\":["
        "{\"role\":\"system\",\"content\":\"Be brief\"},"
        "{\"role\":\"user\",\"content\":\"read notes\"},"
        "{\"role\":\"assistant\",\"content\":\"Sure\",\"tool_calls\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"notes.txt\\\",\\\"limit\\\":5}\"}}]},"
        "{\"role\":\"tool\",\"content\":\"hello\"},"
        "{\"role\":\"user\",\"content\":\"thanks\"}"
        "],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"path\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}}],\"max_tokens\":8}";
    run_job(hist, hi, 2);
    CHECK(prompt_text_n > 0);
    CHECK(strstr(prompt_text, "# Tools") && strstr(prompt_text, "<tools>") && strstr(prompt_text, "</IMPORTANT>"));
    const char* brief = strstr(prompt_text, "Be brief");
    CHECK(brief && strstr(brief + 1, "Be brief") == NULL);
    CHECK(strstr(prompt_text, "<function=read_file>") && strstr(prompt_text, "<tool_response>"));
    CHECK(strstr(prompt_text, "hello") && strstr(prompt_text, "thanks"));

    const char* xml =
        "Ok\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n"
        "<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>";
    uint32_t sc[512];
    int nsc = ids_of(xml, sc, 512);
    const char* ask =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"read_file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
        "\"limit\":{\"type\":\"integer\"}}}}}],\"max_tokens\":400}";
    run_job(ask, sc, nsc);
    CHECK(strstr(wire, "\"finish_reason\":\"tool_calls\"") != NULL);
    CHECK(strstr(wire, "\"content\":\"Ok\"") != NULL);
    CHECK(strstr(wire, "\"name\":\"read_file\"") != NULL);
    CHECK(strstr(wire, "\\\"path\\\":\\\"notes.txt\\\"") != NULL);
    CHECK(strstr(wire, "\\\"limit\\\":5") != NULL);
    CHECK(strstr(wire, "<tool_call>") == NULL);
    CHECK(strstr(wire, "<function=") == NULL);

    char stream_body[1024];
    snprintf(stream_body, sizeof(stream_body),
             "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"tools\":[{\"type\":\"function\","
             "\"function\":{\"name\":\"read_file\",\"parameters\":{\"type\":\"object\",\"properties\":{"
             "\"path\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"}}}}}],\"max_tokens\":400}");
    run_job(stream_body, sc, nsc);
    CHECK(strstr(wire, "Content-Type: text/event-stream") != NULL);
    CHECK(strstr(wire, "\"content\":\"O\"") != NULL);
    CHECK(strstr(wire, "\"content\":\"k\"") != NULL);
    CHECK(strstr(wire, "<tool_call>") == NULL);
    CHECK(strstr(wire, "\"tool_calls\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"tool_calls\"") != NULL);
    CHECK(strstr(wire, "data: [DONE]\n\n") != NULL);

    const char* partial = "<tool_call>\n<function=read_file>";
    nsc = ids_of(partial, sc, 512);
    char cut_body[1024];
    snprintf(cut_body, sizeof(cut_body),
             "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[{\"type\":\"function\",\"function\":{"
             "\"name\":\"read_file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}"
             "}}}}],\"max_tokens\":%d}",
             nsc);
    run_job(cut_body, sc, nsc);
    CHECK(strstr(wire, "\"finish_reason\":\"length\"") != NULL);
    CHECK(strstr(wire, "<tool_call>") != NULL);
    CHECK(strstr(wire, "\"tool_calls\"") == NULL);

    job_wire = 1;
    run_job(ask, sc, ids_of(xml, sc, 512));
    CHECK(strstr(wire, "\"arguments\":{\"path\":\"notes.txt\",\"limit\":5}") != NULL);
    CHECK(strstr(wire, "\"done_reason\":\"stop\"") != NULL);
    CHECK(strstr(wire, "<tool_call>") == NULL);
    job_wire = 0;
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
    for (spec_on = 0; spec_on <= 1; spec_on++) {
        test_completion();
        test_length();
        test_stream();
        test_errors();
        test_stream_error();
        test_client_gone();
        test_spec();
        test_stop();
        test_long_prompt();
        test_cut_char();
        test_ollama_job();
        test_tools();
        CHECK(pos_errors == 0);           // every token went in at the position after the one before
        CHECK(chat_begun == chat_ended);   // every request that took the screen gave it back
        CHECK(chat_replies <= chat_begun);
    }
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
