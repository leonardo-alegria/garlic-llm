// Host test for chat/api_run.inc (the API's model side), with a mock engine and tokenizer, so the job logic runs
// without the PS4. The real file is included as main.c includes it. Every test runs twice: one token per step
// (engine_run), and several per step as with the MTP head (engine_spec_step).
// Build and run: gcc -std=gnu99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/api_run_test.c chat/api_proto.c -o build/api_run_test && build/api_run_test
// (on Windows: zig cc instead of gcc, -fsanitize=undefined only)
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
    struct {
        uint64_t ws_begin, ws_end;   // the dictation's work memory (0, 0: none, so the API keeps no state)
    } asr;
} MockEngine;
static MockEngine eng = {1, 4096, 1005, EOS2, "", {0, 0}};
static int api_snap_n;   // (main.c's)
static int saves, restores, restored_at;   // the state copies of the mock engine, and where reading went on after one
static uint64_t engine_state_size(const MockEngine* e) {
    (void)e;
    return 4096;
}
static int engine_state_copy(MockEngine* e, uint64_t buf, int save) {
    (void)e;
    (void)buf;
    if (save) saves++;
    else restores++, restored_at = -2;   // (the next prompt call says where it goes on)
    return 0;
}
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
#define GARLIC_SPEC 1             // (as the 9B app: api_run.inc asks spec_active before every step)
static int spec_on;               // the mock MTP head is on: a step keeps up to spec_keep tokens of its draft
static int spec_keep = 2;
static int next_pos, pos_errors;  // where the next token must go; steps given another position
static int ctx_errors;            // tokens given a position past the context window (eng.ctx_limit)
static int fed;                   // tokens of the reply fed to the model
// what main.c shows of a job: begun and ended (they must pair), and Circle held from stop_at tokens on (-1: never)
static int chat_begun, chat_ended, chat_replies, stop_at = -1;
// the console's reasoning setting (R3) and its cap, as main.c has them
static int think_on, think_cap = 1024;
#define THINK_T 0.7f
#define THINK_CLOSE "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
static float first_temp, last_temp;   // the temperature of the prompt, and of the last step
static int close_fed;                 // ids fed in one go after the prompt (the closing sentence of a cut reasoning)
static uint32_t prompt_last;          // the prompt's last id

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
static int client_left;   // the client of a waiting request has closed its connection
static int gone_after = -1;   // the client closes its connection once this many tokens of the reply are fed
int api_client_gone(int fd) {
    (void)fd;
    return client_left || (gone_after >= 0 && fed >= gone_after);
}
static void say(const char* fmt, ...) {
    (void)fmt;
}
static double now(void) { return 0; }
static int prompt_phase;   // between api_chat_begin and api_chat_reply: the engine reads the prompt
static void api_chat_begin(void) {   // (the real one clears the conversation)
    chat_begun++, msgs_cleared++;
    prompt_phase = 1, prompt_len_seen = 0;
}
static void api_chat_reply(void) {   // (the reply begins with the token the prompt's last piece drew)
    chat_replies++, prompt_phase = 0;
    if (script_k < script_len) script_k++;
}
static int api_chat_step(int produced) { return stop_at >= 0 && produced >= stop_at; }
static void api_chat_end(void) { chat_ended++, prompt_phase = 0; }
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
    (void)sd;
    engine_calls++;
    if (engine_calls == fail_engine_at) {
        snprintf(e->error, sizeof(e->error), "mock GPU fault");
        return -1;
    }
    if (prompt_phase) {   // the prompt, in pieces: from 0, or from where a restored state left it
        if (pos0 == 0) next_pos = 0;
        if (restored_at == -2) restored_at = pos0, next_pos = pos0;
        prompt_len_seen += n, first_temp = temperature, prompt_last = tokens[n - 1];
    } else {
        if (n > 1) close_fed++;
        fed += n;
    }
    last_temp = temperature;
    if (pos0 != next_pos) pos_errors++;
    if (pos0 + n > e->ctx_limit) ctx_errors++;
    next_pos += n;
    if (prompt_phase) *next = script_k < script_len ? script[script_k] : IM_END;   // (taken when the reply begins)
    else if (script_k < script_len) *next = script[script_k++];
    else *next = IM_END;
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
    (void)sd;
    last_temp = temperature;
    engine_calls++;
    if (engine_calls == fail_engine_at) {
        snprintf(e->error, sizeof(e->error), "mock GPU fault");
        return -1;
    }
    int k = 0;
    while (k < spec_keep && k < max_new - 1 && script_k < script_len) out[k++] = script[script_k++];
    *n_out = k;
    if (pos0 != next_pos) pos_errors++;
    if (pos0 + 1 + k > e->ctx_limit) ctx_errors++;
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
    close_fed = 0;
    saves = restores = 0, restored_at = -1;
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

// The id of the last reply's i-th tool call: "call_", the reply's id after "chatcmpl-", "_i".
static const char* call_id(int i) {
    static char out[64];
    const char* p = strstr(wire, "\"id\":\"chatcmpl-");
    const char* q = p ? strchr(p + 15, '"') : NULL;
    if (!q) return "?";
    snprintf(out, sizeof(out), "call_%.*s_%d", (int)(q - p - 15), p + 15, i);
    return out;
}

// The error message of the last answer (empty if none).
static const char* error_message(void) {
    static char out[1024];
    const char* m = strstr(wire, "\"error\":{\"message\":\"");
    const char* e = m ? strstr(m, "\",\"type\"") : NULL;
    out[0] = 0;
    long k = (long)strlen("\"error\":{\"message\":\"");
    if (m && e && e - m - k < (long)sizeof(out)) snprintf(out, sizeof(out), "%.*s", (int)(e - m - k), m + k);
    return out;
}

// 1 if text begins with head
static int starts(const char* text, const char* head) { return strncmp(text, head, strlen(head)) == 0; }

// 1 if text has word in it, in any case
static int has_word(const char* text, const char* word) {
    size_t n = strlen(word);
    for (const char* p = text; *p; p++) {
        size_t i = 0;
        while (i < n && p[i] && (p[i] | 32) == word[i]) i++;
        if (i == n) return 1;
    }
    return 0;
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
    // as OpenAI words it, so clients read the window and the prompt's size from it
    CHECK(!strcmp(error_message(), "This model's maximum context length is 100 tokens. However, you requested 226 tokens "
                                   "(26 in the messages, 200 in the completion). Please reduce the length of the "
                                   "messages or completion."));
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

    // a request that waited in line, whose client left before its turn: the model does not read it at all
    int begun = chat_begun;
    client_left = 1;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", sc, 3);
    client_left = 0;
    CHECK(engine_calls == 0 && wire_n == 0 && chat_begun == begun);

    // a whole reply (nothing is sent until it ends) whose client closes the connection meanwhile: seen within
    // API_LOOK tokens, and nothing more is written; the same in a tool call that is streamed
    static uint32_t many[400];
    for (int i = 0; i < 400; i++) many[i] = 'a' + 1;
    gone_after = 40;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}", many, 400);
    CHECK(fed >= 40 && fed <= 40 + API_LOOK + 3 && wire_n == 0);
    static uint32_t call[400];
    int k = 0;
    call[k++] = TOOL_CALL;
    while (k < 400) call[k++] = 'a' + 1;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"stream\":true,"
            "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}", call, 400);
    CHECK(fed >= 40 && fed <= 40 + API_LOOK + 3 && strstr(wire, "data: [DONE]") == NULL);
    gone_after = -1;
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
    CHECK(strstr(error_message(), "However, your messages take more than 16256 tokens.") != NULL);
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

// ---- reasoning -----------------------------------------------------------------------------------------------------
// The model reasons "ab", closes </think> itself, writes a blank line and answers "OK".
static const uint32_t think_sc[] = {'a' + 1, 'b' + 1, THINK_END, '\n' + 1, '\n' + 1, 'O' + 1, 'K' + 1};
#define THINK_SC_LEN 7

static void test_think_request(void) {
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":0.2,"
            "\"chat_template_kwargs\":{\"enable_thinking\":true}}", think_sc, THINK_SC_LEN);
    CHECK(strstr(wire, "\"content\":\"OK\",\"reasoning_content\":\"ab\"") != NULL);
    // the messages (10, see test_completion), then <im_start> "assistant\n" <think> "\n" = 13
    CHECK(prompt_len_seen == 23 && prompt_last == '\n' + 1);
    CHECK(strstr(wire, "\"completion_tokens\":4,") != NULL);   // the answer's 2 and the reasoning's 2
    CHECK(strstr(wire, "\"reasoning_tokens\":2") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"stop\"") != NULL);
    CHECK(first_temp == THINK_T && last_temp > 0.19f && last_temp < 0.21f);   // reasons at THINK_T, answers at 0.2
    CHECK(close_fed == 0);   // closed by the model: no closing sentence

    // max_tokens counts the answer only
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"think\":true}", think_sc,
            THINK_SC_LEN);
    CHECK(strstr(wire, "\"content\":\"O\",\"reasoning_content\":\"ab\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"length\"") != NULL);
}

// Who decides: the message's /think or /no_think, then the request, then the console (R3).
static void test_think_who(void) {
    think_on = 0;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi /think\"}]}", think_sc, THINK_SC_LEN);
    CHECK(prompt_len_seen == 23);   // " /think" is not in the prompt
    CHECK(strstr(wire, "\"reasoning_content\":\"ab\"") != NULL);

    think_on = 1;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", think_sc, THINK_SC_LEN);
    CHECK(prompt_len_seen == 23 && strstr(wire, "\"reasoning_content\":\"ab\"") != NULL);

    static const uint32_t plain[] = {'O' + 1, 'K' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi /no_think\"}]}", plain, 2);
    CHECK(prompt_len_seen == 27 && strstr(wire, "\"content\":\"OK\"}") != NULL);
    CHECK(strstr(wire, "reasoning") == NULL);
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"chat_template_kwargs\":{\"enable_thinking\":false}}",
            plain, 2);
    CHECK(prompt_len_seen == 27 && strstr(wire, "reasoning") == NULL);
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi /think\"}],\"reasoning_effort\":\"none\"}", think_sc,
            THINK_SC_LEN);
    CHECK(prompt_len_seen == 23);   // (the message's switch wins)
    think_on = 0;

    // an earlier message's switch is not this reply's
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"a /think\"},{\"role\":\"assistant\",\"content\":\"b\"},"
            "{\"role\":\"user\",\"content\":\"c\"}]}", plain, 2);
    CHECK(strstr(wire, "reasoning") == NULL);
}

static void test_think_stream(void) {
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"think\":true}", think_sc,
            THINK_SC_LEN);
    const char* ra = strstr(wire, "\"delta\":{\"reasoning_content\":\"a\"}");
    const char* rb = strstr(wire, "\"delta\":{\"reasoning_content\":\"b\"}");
    const char* co = strstr(wire, "\"delta\":{\"content\":\"O\"}");
    CHECK(ra && rb && co && ra < rb && rb < co);
    CHECK(strstr(wire, "\"content\":\"\\n\"") == NULL);   // the blank line after the reasoning is not sent
    CHECK(strstr(wire, "\"finish_reason\":\"stop\"") != NULL);
    CHECK(strstr(wire, "data: [DONE]\n\n") != NULL);
}

// The reasoning reaches its cap: Qwen's closing sentence and </think> are fed, and the answer follows.
static void test_think_cut(void) {
    static uint32_t sc[66];
    for (int i = 0; i < 65; i++) sc[i] = 'x' + 1;
    sc[65] = 'Y' + 1;
    think_cap = 64;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"think\":true}", sc, 66);
    think_cap = 1024;
    static char want[128];
    memcpy(want, "\"content\":\"Y\",\"reasoning_content\":\"", 36);
    memset(want + 36, 'x', 64);
    strcpy(want + 100, "\"");
    CHECK(strstr(wire, want) != NULL);   // the 64 tokens of the cap; the 65th is not written
    CHECK(close_fed == 1);
    CHECK(strstr(wire, "\"reasoning_tokens\":64") != NULL);
}

// Little room left in the context: the reply comes without reasoning rather than not at all.
static void test_think_room(void) {
    static const uint32_t plain[] = {'O' + 1, 'K' + 1};
    eng.ctx_limit = 150;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":10,\"think\":true}", plain, 2);
    eng.ctx_limit = 4096;
    CHECK(prompt_len_seen == 27 && strstr(wire, "\"content\":\"OK\"}") != NULL);
}

// Reasoning a client sends back inside an earlier assistant message is not read again.
static void test_think_history(void) {
    static const uint32_t plain[] = {'O' + 1, 'K' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"q\"},"
            "{\"role\":\"assistant\",\"content\":\"<think>\\nlong reasoning\\n</think>\\n\\nA\"},"
            "{\"role\":\"user\",\"content\":\"r\"}]}", plain, 2);
    // user "q" 9, assistant "A" 14, user "r" 9, the assistant's turn 17
    CHECK(prompt_len_seen == 49);
}

// ---- tools ---------------------------------------------------------------------------------------------------------
// The last prompt as text: one byte per id, the special ones as <S:offset from <|im_start|>> (0 = <|im_start|>).
static const char* prompt_text(int n) {
    static char out[1 << 15];
    size_t k = 0;
    for (int i = 0; i < n && k + 16 < sizeof(out); i++) {
        uint32_t t = api_ids[i];
        if (t >= IM_START) k += (size_t)snprintf(out + k, sizeof(out) - k, "<S:%u>", t - IM_START);
        else out[k++] = (char)(t - 1);
    }
    out[k] = 0;
    return out;
}

// A scripted reply: text (one id per byte), with "@" for <tool_call> and "#" for </tool_call>.
static uint32_t reply_sc[512];
static int reply_of(const char* s) {
    int n = 0;
    for (; *s && n < 512; s++) reply_sc[n++] = *s == '@' ? TOOL_CALL : *s == '#' ? TOOL_CALL_END : (uint8_t)*s + 1u;
    return n;
}

#define WEATHER_TOOL                                                                                            \
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Weather now\","   \
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}}," \
    "\"required\":[\"city\"]}}}]"
#define WEATHER_CALL "@\n<function=get_weather>\n<parameter=city>\nLima\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n#"

static void test_tools(void) {
    int k = reply_of(WEATHER_CALL);
    run_job("{\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Weather?\"}],"
            WEATHER_TOOL "}", reply_sc, k);
    // the tools in the system message, as the chat template writes them, with the request's own system text after
    const char* p = prompt_text(prompt_len_seen);
    CHECK(strncmp(p, "<S:0>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
                     "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"", 107) == 0);
    CHECK(strstr(p, "</IMPORTANT>\n\nBe brief.<S:1>\n<S:0>user\nWeather?<S:1>\n<S:0>assistant\n") != NULL);
    // the call, in OpenAI's form: integer arguments as numbers, content null
    char want[512];
    snprintf(want, sizeof(want), "\"content\":null,\"tool_calls\":[{\"id\":\"%s\",\"type\":\"function\",\"function\":"
             "{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Lima\\\",\\\"days\\\":3}\"}}]", call_id(1));
    CHECK(strstr(wire, want) != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"tool_calls\"") != NULL);
    char first_id[64];
    snprintf(first_id, sizeof(first_id), "%s", call_id(1));

    // streamed: the words before the call go out, the call's text does not, the call comes whole at the end
    k = reply_of("Checking.\n\n" WEATHER_CALL);
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}]," WEATHER_TOOL ",\"stream\":true}", reply_sc, k);
    CHECK(strstr(wire, "\"delta\":{\"content\":\"C\"}") != NULL);
    CHECK(strstr(wire, "function=") == NULL);
    CHECK(strstr(wire, "\"content\":\"\\n") == NULL);   // (the blank line before the call is not sent: as a whole reply)
    snprintf(want, sizeof(want), "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"%s\",\"type\":\"function\","
             "\"function\":{\"name\":\"get_weather\"", call_id(1));
    CHECK(strstr(wire, want) != NULL);
    CHECK(strcmp(call_id(1), first_id) != 0 && strncmp(first_id, "call_", 5) == 0);   // (another reply, another id)
    CHECK(strstr(wire, "\"finish_reason\":\"tool_calls\"") != NULL && strstr(wire, "data: [DONE]") != NULL);

    // a reply without a call is a plain answer, even with tools
    static const uint32_t plain[] = {'O' + 1, 'K' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]," WEATHER_TOOL "}", plain, 2);
    CHECK(strstr(wire, "\"content\":\"OK\"}") != NULL && strstr(wire, "\"finish_reason\":\"stop\"") != NULL);

    // tool_choice "none": no tools in the prompt
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]," WEATHER_TOOL ",\"tool_choice\":\"none\"}", plain, 2);
    CHECK(prompt_len_seen == 27);
}

// The results of a call go back to the model as Qwen's template writes them, after the call itself.
static void test_tool_results(void) {
    static const uint32_t plain[] = {'1' + 1, '8' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"},"
            "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
            "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Lima\\\"}\"}}]},"
            "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":\"18 C\"},"
            "{\"role\":\"tool\",\"tool_call_id\":\"call_2\",\"content\":\"sunny\"}]," WEATHER_TOOL "}", plain, 2);
    const char* p = prompt_text(prompt_len_seen);
    // (the mock's <think> and </think> are <|im_start|> + 2 and + 3)
    CHECK(strstr(p, "<S:0>user\nWeather?<S:1>\n<S:0>assistant\n<S:2>\n\n<S:3>\n\n<S:13>\n<function=get_weather>\n"
                    "<parameter=city>\nLima\n</parameter>\n</function>\n<S:14><S:1>\n"
                    "<S:0>user\n<S:21>\n18 C\n<S:22>\n<S:21>\nsunny\n<S:22><S:1>\n<S:0>assistant\n") != NULL);
    CHECK(strstr(wire, "\"content\":\"18\"") != NULL);
}

// tool_choice naming a function: the reply begins with its call, and the rest of the call is read with it.
static void test_tool_choice(void) {
    int k = reply_of("<parameter=city>\nCusco\n</parameter>\n</function>\n#");
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}]," WEATHER_TOOL ","
            "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}}", reply_sc, k);
    const char* p = prompt_text(prompt_len_seen);
    CHECK(strstr(p, "<S:0>assistant\n<S:2>\n\n<S:3>\n\n<S:13>\n<function=get_weather>\n") != NULL);
    CHECK(strstr(wire, "\"arguments\":\"{\\\"city\\\":\\\"Cusco\\\"}\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"tool_calls\"") != NULL);
}

// ---- what agent clients send and expect ------------------------------------------------------------------------------
// stream_options.include_usage: the stream ends with a chunk of usage (no choices) before [DONE].
static void test_usage(void) {
    static const uint32_t sc[] = {'O' + 1, 'K' + 1};
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,"
            "\"stream_options\":{\"include_usage\":true}}", sc, 2);
    const char* fin = strstr(wire, "\"finish_reason\":\"stop\"");
    const char* use = strstr(wire, "data: {\"id\":\"chatcmpl-");
    while (use && strstr(use + 1, "data: {\"id\":\"chatcmpl-")) use = strstr(use + 1, "data: {\"id\":\"chatcmpl-");
    const char* done = strstr(wire, "data: [DONE]\n\n");
    CHECK(fin && use && done && fin < use && use < done);   // (the last chunk)
    CHECK(use && strstr(use, "\"choices\":[],\"usage\":{\"prompt_tokens\":27,\"completion_tokens\":2,"
                             "\"total_tokens\":29}}\n\ndata: [DONE]") == strstr(use, "\"choices\""));
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true}", sc, 2);
    CHECK(strstr(wire, "usage") == NULL);   // (not asked for)
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"think\":true,"
            "\"stream_options\":{\"include_usage\":true}}", think_sc, THINK_SC_LEN);
    CHECK(strstr(wire, "\"choices\":[],\"usage\":{\"prompt_tokens\":23,\"completion_tokens\":4,\"total_tokens\":27,"
                       "\"completion_tokens_details\":{\"reasoning_tokens\":2}}}") != NULL);
}

// Without max_tokens the answer may use the rest of the context window: a long reply is not cut at 512 tokens.
static void test_default_max(void) {
    static uint32_t sc[700];
    for (int i = 0; i < 700; i++) sc[i] = 'a' + 1 + i % 26;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", sc, 700);
    CHECK(strstr(wire, "\"finish_reason\":\"stop\"") && strstr(wire, "\"completion_tokens\":700,"));
    eng.ctx_limit = 100;   // up to the end of the context: 27 tokens of prompt, 71 of answer (and 2 to spare)
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", sc, 700);
    CHECK(strstr(wire, "\"finish_reason\":\"length\"") && strstr(wire, "\"completion_tokens\":71,"));
    // a prompt that leaves no room says so, with its size
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"0123456789012345678901234567890123456789012345678901234567"
            "89012345678901234567890123456789\"}]}", sc, 700);
    CHECK(strstr(wire, "context_length_exceeded") != NULL);
    CHECK(!strcmp(error_message(), "This model's maximum context length is 100 tokens. However, your messages take "
                                   "115 tokens. Please reduce the length of the messages."));
    eng.ctx_limit = 4096;

    // reasoning starts only with API_ANSWER_MIN left for the answer
    think_on = 1;
    eng.ctx_limit = 700;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", think_sc, THINK_SC_LEN);
    CHECK(prompt_len_seen == 27 && strstr(wire, "reasoning") == NULL);
    // with room for it: and when it ends early, the answer has what it did not use (625 tokens without that)
    static uint32_t long_think[705];
    memcpy(long_think, think_sc, 5 * sizeof(uint32_t));   // "ab", </think>, the blank line
    for (int i = 5; i < 705; i++) long_think[i] = 'z' + 1;
    eng.ctx_limit = 900;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", long_think, 705);
    CHECK(prompt_len_seen == 23 && strstr(wire, "\"reasoning_content\":\"ab\"") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"stop\"") && strstr(wire, "\"completion_tokens\":702,"));
    eng.ctx_limit = 4096;
    think_on = 0;
}

// Tools and messages that do not fit: the numbers, and the tools' share, without the words "tool" or "function".
static void test_tools_too_long(void) {
    int cleared = msgs_cleared;
    eng.ctx_limit = 400;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}]," WEATHER_TOOL ",\"max_tokens\":10}", NULL, 0);
    CHECK(strncmp(wire, "HTTP/1.1 400 Bad Request", 23) == 0 && strstr(wire, "\"code\":\"context_length_exceeded\""));
    const char* m = error_message();
    CHECK(starts(m, "This model's maximum context length is 400 tokens. However, you requested "));
    CHECK(strstr(m, " in the messages, 10 in the completion). The schema sent with the request takes ") != NULL);
    CHECK(!has_word(m, "tool") && !has_word(m, "function"));
    eng.ctx_limit = 4096;
    CHECK(msgs_cleared == cleared && engine_calls == 0);
}

// What an agent client sends: a developer message, an assistant turn of calls without content, the results, tools
// and tool_choice "auto", streamed with usage.
static void test_agent_history(void) {
    int k = reply_of("Done: 18 C.");
    run_job("{\"model\":\"Qwen3.5-9B\",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":["
            "{\"role\":\"developer\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Weather?\"},"
            "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_9\",\"type\":\"function\",\"function\":"
            "{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Lima\\\",\\\"days\\\":2}\"}}]},"
            "{\"role\":\"tool\",\"tool_call_id\":\"call_9\",\"content\":\"18 C\"}]," WEATHER_TOOL ",\"tool_choice\":\"auto\"}",
            reply_sc, k);
    const char* p = prompt_text(prompt_len_seen);
    CHECK(starts(p, "<S:0>system\n# Tools\n\n"));   // (the developer message goes with the tools)
    CHECK(strstr(p, "</IMPORTANT>\n\nBe brief.<S:1>\n<S:0>user\nWeather?<S:1>\n<S:0>assistant\n<S:2>\n\n<S:3>\n\n"
                    "<S:13>\n<function=get_weather>\n<parameter=city>\nLima\n</parameter>\n<parameter=days>\n2\n"
                    "</parameter>\n</function>\n<S:14><S:1>\n<S:0>user\n<S:21>\n18 C\n<S:22><S:1>\n") != NULL);
    CHECK(strstr(wire, "\"delta\":{\"content\":\"D\"}") != NULL && strstr(wire, "\"finish_reason\":\"stop\""));
    CHECK(strstr(wire, "\"choices\":[],\"usage\":{\"prompt_tokens\":") != NULL && strstr(wire, "data: [DONE]"));

    // developer without tools: a system turn of its own
    static const uint32_t plain[] = {'O' + 1, 'K' + 1};
    run_job("{\"messages\":[{\"role\":\"developer\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"Hi\"}]}",
            plain, 2);
    CHECK(starts(prompt_text(prompt_len_seen), "<S:0>system\nBe brief.<S:1>\n<S:0>user\nHi<S:1>\n"));

    // calls an assistant message cannot have: said apart from a malformed message
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"},{\"role\":\"assistant\",\"tool_calls\":"
            "[{\"type\":\"function\",\"function\":{}}]},{\"role\":\"tool\",\"content\":\"x\"}]," WEATHER_TOOL "}", plain, 2);
    CHECK(strncmp(wire, "HTTP/1.1 400 Bad Request", 23) == 0 && strstr(wire, "invalid_body"));
    CHECK(strstr(error_message(), "An assistant message's 'tool_calls' are malformed") != NULL);
}

// tool_choice "required": the reply begins with a call, and the model writes which function; parallel_tool_calls
// false keeps only the first of several calls.
static void test_tool_choice_more(void) {
    int k = reply_of("get_weather>\n<parameter=city>\nLima\n</parameter>\n</function>\n#");
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]," WEATHER_TOOL ",\"tool_choice\":\"required\","
            "\"stream\":true}", reply_sc, k);
    CHECK(strstr(prompt_text(prompt_len_seen), "<S:0>assistant\n<S:2>\n\n<S:3>\n\n<S:13>\n<function=") != NULL);
    CHECK(strstr(wire, "\"delta\":{\"content\"") == NULL);   // (nothing of the call as text)
    CHECK(strstr(wire, "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Lima\\\"}\"}") != NULL);
    CHECK(strstr(wire, "\"finish_reason\":\"tool_calls\"") != NULL);

    k = reply_of(WEATHER_CALL "\n" WEATHER_CALL);
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]," WEATHER_TOOL "}", reply_sc, k);
    CHECK(strstr(wire, call_id(2)) != NULL);
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]," WEATHER_TOOL ",\"parallel_tool_calls\":false}",
            reply_sc, k);
    CHECK(strstr(wire, call_id(1)) != NULL && strstr(wire, call_id(2)) == NULL);
}

// A big agent request: 100 tools, a long conversation and more JSON values than the first pool holds. The reply's
// call is read with the schemas (an integer argument as a number), with the request read again at the end.
static char agent_big[80000];
static void test_big_agent_request(void) {
    size_t n = (size_t)sprintf(agent_big, "{\"meta\":[0");
    for (int i = 0; i < 5000; i++) n += (size_t)sprintf(agent_big + n, ",0");
    n += (size_t)sprintf(agent_big + n, "],\"messages\":[");
    for (int i = 0; i < 200; i++)
        n += (size_t)sprintf(agent_big + n, "%s{\"role\":\"%s\",\"content\":\"m\"}", i ? "," : "", i % 2 ? "assistant" : "user");
    n += (size_t)sprintf(agent_big + n, ",{\"role\":\"user\",\"content\":\"go\"}],\"tools\":[");
    for (int i = 0; i < 100; i++)
        n += (size_t)sprintf(agent_big + n, "%s{\"type\":\"function\",\"function\":{\"name\":\"t%d\"%s}}", i ? "," : "", i,
                             i == 57 ? ",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}" : "");
    sprintf(agent_big + n, "]}");
    int k = reply_of("@\n<function=t57>\n<parameter=n>\n42\n</parameter>\n</function>\n#");
    eng.ctx_limit = API_IDS;
    run_job(agent_big, reply_sc, k);
    eng.ctx_limit = 4096;
    CHECK(strncmp(wire, "HTTP/1.1 200 OK", 15) == 0);
    CHECK(strstr(wire, "\"name\":\"t57\",\"arguments\":\"{\\\"n\\\":42}\"") != NULL);
    CHECK(prompt_len_seen > 5000 && strstr(wire, "\"finish_reason\":\"tool_calls\""));
}

// ---- the kept state of the prompt's beginning -------------------------------------------------------------------------
static void test_kept_state(void) {
    static const uint32_t plain[] = {'O' + 1, 'K' + 1};
    eng.asr.ws_end = 1 << 20;   // (room for the copy)
    api_snap_n = 0;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", plain, 2);
    CHECK(prompt_len_seen == 27 && saves == 1 && restores == 0);
    CHECK(api_snap_n == 21);   // the messages (10) and <im_start> "assistant\n" (11)

    // the same conversation, one turn longer: read from the copy on
    const char* longer = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"OK\"},"
                         "{\"role\":\"user\",\"content\":\"more\"}]}";
    run_job(longer, plain, 2);
    CHECK(restores == 1 && restored_at == 21);
    CHECK(prompt_len_seen == 54 - 21);   // (user "hi" 10, assistant "OK" 15, user "more" 12, the assistant's turn 17)
    CHECK(api_snap_n == 48 && saves == 1);
    CHECK(strstr(wire, "\"content\":\"OK\"") != NULL);

    // another conversation: read from the start, and its own state kept
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"bye\"}]}", plain, 2);
    CHECK(restores == 0 && prompt_len_seen == 28 && api_snap_n == 22);

    // the chat used the model meanwhile (main.c sets api_snap_n to 0): read from the start
    api_snap_n = 0;
    run_job(longer, plain, 2);
    CHECK(restores == 0 && prompt_len_seen == 54);

    // tool results after the kept question: the question's state stays the one kept
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}]," WEATHER_TOOL "}", plain, 2);
    int kept = api_snap_n;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"},"
            "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\","
            "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}]},"
            "{\"role\":\"tool\",\"content\":\"18 C\"}]," WEATHER_TOOL "}", plain, 2);
    CHECK(kept > 0 && restores == 1 && restored_at == kept && saves == 0 && api_snap_n == kept);

    eng.asr.ws_end = 0;   // no room: nothing is kept
    api_snap_n = 0;
    run_job("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", plain, 2);
    CHECK(saves == 0 && api_snap_n == 0 && prompt_len_seen == 27);
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
        test_think_request();
        test_think_who();
        test_think_stream();
        test_think_cut();
        test_think_room();
        test_think_history();
        test_tools();
        test_tool_results();
        test_tool_choice();
        test_usage();
        test_default_max();
        test_tools_too_long();
        test_agent_history();
        test_tool_choice_more();
        test_big_agent_request();
        test_kept_state();
        CHECK(pos_errors == 0);           // every token went in at the position after the one before
        CHECK(ctx_errors == 0);           // and none past the context window
        CHECK(chat_begun == chat_ended);   // every request that took the screen gave it back
        CHECK(chat_replies <= chat_begun);
    }
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
