// Host tests for chat/api_proto.c (the OpenAI-compatible API's protocol part).
// Build and run: gcc -std=c99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/api_proto_test.c chat/api_proto.c -o build/api_proto_test && build/api_proto_test
// (on Windows: zig cc instead of gcc, -fsanitize=undefined only)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "api_proto.h"

static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

static void test_http(void) {
    HttpRequest r;
    const char* post = "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: ps4\r\nContent-Type: application/json\r\n"
                       "content-length: 13\r\nAuthorization: Bearer abc\r\n\r\n{\"a\":\"b\"}    ";
    CHECK(api_http_parse(post, strlen(post), &r) == 1);
    CHECK(!strcmp(r.method, "POST"));
    CHECK(!strcmp(r.path, "/v1/chat/completions"));
    CHECK(r.content_length == 13);
    CHECK(r.body_len == 13 && !memcmp(r.body, "{\"a\":\"b\"}    ", 13));
    CHECK(!strcmp(r.auth, "Bearer abc"));

    const char* partial = "POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc";
    CHECK(api_http_parse(partial, strlen(partial), &r) == 0);
    const char* head_only = "GET /v1/models HTTP/1.1\r\nHost: a\r\n";
    CHECK(api_http_parse(head_only, strlen(head_only), &r) == 0);
    const char* get = "GET /v1/models HTTP/1.1\r\nHost: a\r\n\r\n";
    CHECK(api_http_parse(get, strlen(get), &r) == 1);
    CHECK(!strcmp(r.path, "/v1/models") && r.content_length == 0);
    const char* bad = "GET\r\n\r\n";
    CHECK(api_http_parse(bad, strlen(bad), &r) == -1);
    const char* badcl = "POST / HTTP/1.1\r\nContent-Length: nope\r\n\r\n";
    CHECK(api_http_parse(badcl, strlen(badcl), &r) == -1);
    const char* bigcl = "POST / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n";
    CHECK(api_http_parse(bigcl, strlen(bigcl), &r) == -2);   // too big: refused before the body is read
    CHECK(r.content_length == 99999999 && r.head_len == strlen(bigcl));
    const char* limit = "POST / HTTP/1.1\r\nContent-Length: 1048576\r\n\r\n";
    CHECK(api_http_parse(limit, strlen(limit), &r) == 0);    // 1 MiB exactly: read
    const char* negcl = "POST / HTTP/1.1\r\nContent-Length: -5\r\n\r\n";
    CHECK(api_http_parse(negcl, strlen(negcl), &r) == -1);
}

static void test_json(void) {
    JNode pool[64];
    const char* doc = "{ \"a\" : [1, 2.5, -3e2, true, false, null], \"s\": \"x\\\"y\\\\\\n\\u00e9\\ud83d\\ude00\" }";
    int root = json_parse(doc, strlen(doc), pool, 64);
    CHECK(root >= 0);
    int a = json_get(pool, root, "a");
    CHECK(a >= 0 && pool[a].type == J_ARR && pool[a].count == 6);
    int c = pool[a].first;
    CHECK(pool[c].type == J_NUM && pool[c].num == 1);
    c = pool[c].next;
    CHECK(pool[c].num == 2.5);
    c = pool[c].next;
    CHECK(pool[c].num == -300);
    c = pool[c].next;
    CHECK(pool[c].type == J_BOOL && pool[c].b == 1);
    int s = json_get(pool, root, "s");
    Sb out;
    sb_init(&out);
    json_str_sb(pool, s, &out);
    CHECK(out.n == strlen("x\"y\\\n\xc3\xa9\xf0\x9f\x98\x80"));
    CHECK(out.p && !memcmp(out.p, "x\"y\\\n\xc3\xa9\xf0\x9f\x98\x80", out.n));
    sb_free(&out);

    CHECK(json_parse("{\"a\":}", 6, pool, 64) == -1);
    CHECK(json_parse("[1,2", 4, pool, 64) == -1);
    CHECK(json_parse("{} x", 4, pool, 64) == -1);
    CHECK(json_parse("[1,2]", 5, pool, 2) == -2);   // pool too small
    CHECK(json_parse("[1,2", 4, pool, 2) == -2);    // (it ran out before it saw the error)
    CHECK(json_get(pool, root, "missing") == -1);

    // a pool that grows: 2 nodes first, a document of 1001 values
    Sb big;
    sb_init(&big);
    sb_str(&big, "[0");
    for (int i = 1; i < 1000; i++) sb_str(&big, ",1");
    sb_str(&big, "]");
    int r2;
    JNode* grown = json_parse_grow(big.p, big.n, 2, &r2);
    CHECK(grown && r2 == 0 && grown[r2].count == 1000);
    free(grown);
    CHECK(json_parse_grow("[1,", 3, 2, &r2) == NULL && r2 == -1);
    sb_free(&big);
}

static void test_chat(void) {
    static JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
    const char* body =
        "{\"model\":\"x\",\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hi \"},{\"type\":\"text\",\"text\":\"there\"}]}],"
        "\"temperature\":0.2,\"max_tokens\":64,\"stream\":true,\"seed\":7}";
    CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_messages == 2 && req.stream == 1 && req.max_tokens == 64 && req.has_seed && req.seed == 7);
    CHECK(req.temperature > 0.19 && req.temperature < 0.21);
    char role[16];
    Sb content;
    sb_init(&content);
    int m = pool[req.messages].first;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "system"));
    CHECK(content.n == 9 && !memcmp(content.p, "Be brief.", 9));
    sb_free(&content);
    sb_init(&content);
    m = pool[m].next;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "user"));
    CHECK(content.n == 8 && !memcmp(content.p, "Hi there", 8));
    sb_free(&content);

    const char* defaults = "{\"messages\":[{\"role\":\"user\",\"content\":\"yo\"}]}";
    CHECK(api_chat_parse(defaults, strlen(defaults), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.max_tokens == 0 && req.stream == 0 && !req.has_seed);   // (0: the rest of the context window)
    CHECK(!req.include_usage && req.parallel_tool_calls == 1);

    // what an agent client sends besides the messages
    const char* agent = "{\"messages\":[{\"role\":\"user\",\"content\":\"yo\"}],\"stream\":true,"
                        "\"stream_options\":{\"include_usage\":true},\"parallel_tool_calls\":false,"
                        "\"max_completion_tokens\":null,\"max_tokens\":300}";
    CHECK(api_chat_parse(agent, strlen(agent), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.stream && req.include_usage && !req.parallel_tool_calls && req.max_tokens == 300);

    const char* bad[] = {
        "[]",
        "{\"messages\":[]}",
        "{\"messages\":[1]}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"n\":2}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"temperature\":3}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_tokens\":0}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"stream\":\"yes\"}",
        "{\"messages\":[{\"role\":\"user\"",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        CHECK(api_chat_parse(bad[i], strlen(bad[i]), pool, API_MAX_NODES, &req, &err) == -1);
        CHECK(err != NULL);
    }

    // each failure says what is wrong (issue #18: a valid body with many values was "not a JSON object")
    CHECK(api_chat_parse("[]", 2, pool, API_MAX_NODES, &req, &err) == -1 && !strcmp(err, "The body must be a JSON object."));
    CHECK(api_chat_parse("{\"messages\":", 12, pool, API_MAX_NODES, &req, &err) == -1 &&
          !strcmp(err, "The body is not valid JSON."));
    CHECK(api_chat_parse("", 0, pool, API_MAX_NODES, &req, &err) == -1 && strstr(err, "no body") != NULL);
    CHECK(api_chat_parse(defaults, strlen(defaults), pool, 4, &req, &err) == -2 && strstr(err, "65536 JSON values"));
}

// A request with far more JSON values than the first pool holds: an agent's tool list and a long conversation.
static void test_big_request(void) {
    Sb b;
    sb_init(&b);
    sb_str(&b, "{\"stream\":true,\"messages\":[{\"role\":\"system\",\"content\":\"Build it.\"}");
    for (int i = 0; i < 300; i++) sb_str(&b, ",{\"role\":\"user\",\"content\":\"a\"},{\"role\":\"assistant\",\"content\":\"b\"}");
    sb_str(&b, "],\"tools\":[");
    for (int t = 0; t < 200; t++) {
        if (t) sb_str(&b, ",");
        sb_str(&b, "{\"type\":\"function\",\"function\":{\"name\":\"look\",\"description\":\"d\",\"parameters\":{\"type\":"
                   "\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"mode\":{\"enum\":[\"a\",\"b\"]}}}}}");
    }
    sb_str(&b, "],\"tool_choice\":\"auto\"}");
    CHECK(!b.oom);
    ChatRequest req;
    const char* err = NULL;
    static JNode small[API_MAX_NODES];
    CHECK(api_chat_parse(b.p, b.n, small, API_MAX_NODES, &req, &err) == -2);   // more than the first pool holds
    JNode* pool = api_chat_parse_pool(b.p, b.n, &req, &err);
    CHECK(pool != NULL);
    if (pool) {
        CHECK(req.n_messages == 601 && req.stream == 1 && req.n_tools == 200 && req.tool_choice == TOOL_AUTO);
        int last = pool[req.messages].first;
        while (pool[last].next >= 0) last = pool[last].next;
        char role[16];
        Sb content;
        sb_init(&content);
        CHECK(api_message(pool, last, role, sizeof(role), &content) == 0 && !strcmp(role, "assistant"));
        sb_free(&content);
        free(pool);
    }
    // past API_MAX_VALUES the request is refused, with that reason
    Sb huge;
    sb_init(&huge);
    sb_str(&huge, "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"x\":[0");
    for (int i = 0; i < API_MAX_VALUES; i++) sb_str(&huge, ",0");
    sb_str(&huge, "]}");
    CHECK(api_chat_parse_pool(huge.p, huge.n, &req, &err) == NULL && err && strstr(err, "65536") != NULL);
    // (one that is also cut short runs out of values before the end: too big all the same)
    huge.p[huge.n - 1] = ',';
    CHECK(api_chat_parse_pool(huge.p, huge.n, &req, &err) == NULL && err && strstr(err, "65536") != NULL);
    // a big body that is not JSON is not called too big
    CHECK(api_chat_parse_pool(b.p, b.n - 1, &req, &err) == NULL && err && !strcmp(err, "The body is not valid JSON."));
    sb_free(&huge);
    sb_free(&b);
}

// The request's reasoning fields: the chat template's switch first, then the plainer spellings; none = -1.
static void test_think_fields(void) {
    static JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
    static const struct {
        const char* extra;
        int think;
    } cases[] = {
        {"", -1},
        {",\"chat_template_kwargs\":{\"enable_thinking\":true}", 1},
        {",\"chat_template_kwargs\":{\"enable_thinking\":false}", 0},
        {",\"enable_thinking\":true", 1},
        {",\"think\":false", 0},
        {",\"reasoning_effort\":\"high\"", 1},
        {",\"reasoning_effort\":\"low\"", 1},
        {",\"reasoning_effort\":\"none\"", 0},
        {",\"reasoning_effort\":\"minimal\"", 0},
        {",\"chat_template_kwargs\":{\"enable_thinking\":false},\"reasoning_effort\":\"high\"", 0},
        {",\"chat_template_kwargs\":{},\"think\":true", 1},
        {",\"enable_thinking\":\"yes\"", -1},   // (not a boolean: ignored, as unknown fields are)
    };
    char body[256];
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        snprintf(body, sizeof(body), "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]%s}", cases[i].extra);
        CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
        CHECK(req.think == cases[i].think);
    }
}

static int switch_of(const char* in, const char* left, int want) {
    Sb s;
    sb_init(&s);
    sb_str(&s, in);
    int r = api_think_switch(&s);
    int ok = r == want && s.n == strlen(left) && (!s.n || !memcmp(s.p, left, s.n));
    sb_free(&s);
    return ok;
}

static int drop_of(const char* in, const char* left) {
    Sb s;
    sb_init(&s);
    sb_str(&s, in);
    api_drop_think(&s, NULL);
    int ok = s.n == strlen(left) && (!s.n || !strcmp(s.p, left));
    sb_free(&s);
    return ok;
}

static void test_tool_fields(void) {
    static JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
#define MSG "\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]"
#define TOOL "{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}}"
    const char* ok = "{" MSG ",\"tools\":[" TOOL "]}";
    CHECK(api_chat_parse(ok, strlen(ok), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_tools == 1 && req.tools >= 0 && req.tool_choice == TOOL_AUTO);
    const char* none = "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":\"none\"}";
    CHECK(api_chat_parse(none, strlen(none), pool, API_MAX_NODES, &req, &err) == 0 && req.tool_choice == TOOL_NONE);
    const char* named = "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"f\"}}}";
    CHECK(api_chat_parse(named, strlen(named), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.tool_choice == TOOL_NAMED && !strcmp(req.tool_name, "f"));
    const char* plain = "{" MSG "}";
    CHECK(api_chat_parse(plain, strlen(plain), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_tools == 0 && req.tools == -1 && req.tool_choice == TOOL_NONE);
    const char* required = "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":\"required\"}";
    CHECK(api_chat_parse(required, strlen(required), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.tool_choice == TOOL_REQUIRED);
    // the newer form of a named function, without "function": {...}
    const char* flat = "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":{\"type\":\"function\",\"name\":\"f\"}}";
    CHECK(api_chat_parse(flat, strlen(flat), pool, API_MAX_NODES, &req, &err) == 0 && req.tool_choice == TOOL_NAMED);
    const char* none_alone = "{" MSG ",\"tool_choice\":\"none\"}";
    CHECK(api_chat_parse(none_alone, strlen(none_alone), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.tool_choice == TOOL_NONE);
    const char* unknown = "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"g\"}}}";
    CHECK(api_chat_parse(unknown, strlen(unknown), pool, API_MAX_NODES, &req, &err) == -1);
    CHECK(err && !strcmp(err, "'tool_choice' names a function that is not in 'tools'."));
    const char* named_alone = "{" MSG ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"f\"}}}";
    CHECK(api_chat_parse(named_alone, strlen(named_alone), pool, API_MAX_NODES, &req, &err) == -1);
    CHECK(err && !strcmp(err, "'tool_choice' needs 'tools'."));
    const char* bad[] = {
        "{" MSG ",\"tools\":{}}",
        "{" MSG ",\"tools\":[{\"type\":\"function\",\"function\":{}}]}",
        "{" MSG ",\"tool_choice\":\"required\"}",
        "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":\"sometimes\"}",
        "{" MSG ",\"tools\":[" TOOL "],\"tool_choice\":5}",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        CHECK(api_chat_parse(bad[i], strlen(bad[i]), pool, API_MAX_NODES, &req, &err) == -1);
        CHECK(err != NULL);
    }
#undef MSG
#undef TOOL
}

// An earlier call's arguments with more values than a small pool holds are all written back (they were left out).
static void test_tool_history_args(void) {
    Sb b;
    sb_init(&b);
    sb_str(&b, "{\"messages\":[{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"c\",\"type\":\"function\","
               "\"function\":{\"name\":\"todo\",\"arguments\":\"{\\\"items\\\":[0");
    for (int i = 1; i < 400; i++) sb_str(&b, ",1");
    sb_str(&b, "]}\"}}]}]}");
    ChatRequest req;
    const char* err = NULL;
    JNode* pool = api_chat_parse_pool(b.p, b.n, &req, &err);
    CHECK(pool != NULL);
    if (pool) {
        Sb out;
        sb_init(&out);
        CHECK(api_assistant_tool_calls(pool, pool[req.messages].first, &out) == 1);
        CHECK(out.p && strstr(out.p, "<parameter=items>\n[0, 1, 1") != NULL && strstr(out.p, "1]\n</parameter>") != NULL);
        sb_free(&out);
        free(pool);
    }
    sb_free(&b);
}

// A call written as JSON (the form of earlier Qwen models) is read too.
static void test_tool_json_call(void) {
    ApiToolCall calls[API_MAX_TOOL_CALLS];
    Sb before;
    sb_init(&before);
    const char* text = "Sure.\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>";
    int n = api_parse_tool_calls(NULL, NULL, text, strlen(text), &before, calls, API_MAX_TOOL_CALLS);
    CHECK(n == 1 && !strcmp(calls[0].name, "f") && calls[0].arguments.p && !strcmp(calls[0].arguments.p, "{\"a\":1}"));
    CHECK(before.n == 5 && !memcmp(before.p, "Sure.", 5));
    api_free_tool_calls(calls, n);
    sb_free(&before);
}

static void test_think_text(void) {
    CHECK(switch_of("Why is the sky blue? /think", "Why is the sky blue?", 1));
    CHECK(switch_of("Hi /no_think\n", "Hi", 0));
    CHECK(switch_of("Hi\n/nothink", "Hi", 0));
    CHECK(switch_of("/think", "", 1));
    CHECK(switch_of("Hello", "Hello", -1));
    CHECK(switch_of("look in /usr/think", "look in /usr/think", -1));   // (a path, not the switch)
    CHECK(switch_of("/think about it", "/think about it", -1));         // (only at the end)
    CHECK(drop_of("<think>\nfirst this\n</think>\n\nThe answer.", "The answer."));
    CHECK(drop_of("reasoning without its start</think>\nAnswer", "Answer"));
    CHECK(drop_of("a</think>b</think>\n\nc", "c"));
    CHECK(drop_of("No reasoning here.", "No reasoning here."));
}

static void test_writers(void) {
    Sb s;
    sb_init(&s);
    api_json_completion(&s, "chatcmpl-1", "Qwen3.5-9B", 1700000000, "", 0, "Hola \"mundo\"\n", 13, NULL, 0, "stop", 5, 4, 0);
    CHECK(s.p && strstr(s.p, "\"content\":\"Hola \\\"mundo\\\"\\n\"") != NULL);
    CHECK(strstr(s.p, "\"total_tokens\":9") != NULL);
    CHECK(strstr(s.p, "\"finish_reason\":\"stop\"") != NULL);
    CHECK(strstr(s.p, "reasoning") == NULL);   // (no reasoning: the same JSON as before reasoning existed)
    sb_free(&s);

    sb_init(&s);
    api_json_completion(&s, "c", "m", 1, "Let me see.", 11, "Yes.", 4, NULL, 0, "stop", 5, 30, 26);
    CHECK(strstr(s.p, "\"message\":{\"role\":\"assistant\",\"content\":\"Yes.\",\"reasoning_content\":\"Let me see.\"}"));
    CHECK(strstr(s.p, "\"completion_tokens\":30,\"total_tokens\":35,"
                      "\"completion_tokens_details\":{\"reasoning_tokens\":26}}") != NULL);
    sb_free(&s);

    sb_init(&s);
    api_json_chunk(&s, "c", "m", 1, "assistant", "", 0, "", 0, NULL);
    CHECK(!strcmp(s.p, "{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\","
                       "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}"));
    sb_free(&s);

    sb_init(&s);
    api_json_chunk(&s, "c", "m", 1, NULL, "Hmm", 3, "", 0, NULL);
    CHECK(strstr(s.p, "\"delta\":{\"reasoning_content\":\"Hmm\"},\"finish_reason\":null") != NULL);
    sb_free(&s);

    sb_init(&s);
    api_json_error(&s, "bad \x01 input", "invalid_request_error", "bad_body");
    CHECK(!strcmp(s.p, "{\"error\":{\"message\":\"bad \\u0001 input\",\"type\":\"invalid_request_error\","
                       "\"code\":\"bad_body\"}}"));
    sb_free(&s);

    sb_init(&s);
    api_json_health(&s, 1, 2);
    CHECK(!strcmp(s.p, "{\"status\":\"ok\",\"busy\":true,\"waiting\":2}"));
    sb_free(&s);
    sb_init(&s);
    api_json_health(&s, 0, 0);
    CHECK(!strcmp(s.p, "{\"status\":\"ok\",\"busy\":false,\"waiting\":0}"));
    sb_free(&s);

    sb_init(&s);
    api_json_models(&s, "Qwen3.5-9B", 0, 16384);
    CHECK(!strcmp(s.p, "{\"object\":\"list\",\"data\":[{\"id\":\"Qwen3.5-9B\",\"object\":\"model\",\"created\":0,"
                       "\"owned_by\":\"garlic-llm\",\"context_length\":16384}]}"));
    sb_free(&s);
    sb_init(&s);
    api_json_model(&s, "Qwen3.5-9B", 0, 8192);
    CHECK(!strcmp(s.p, "{\"id\":\"Qwen3.5-9B\",\"object\":\"model\",\"created\":0,\"owned_by\":\"garlic-llm\","
                       "\"context_length\":8192}"));
    sb_free(&s);

    // the usage chunk of stream_options.include_usage: no choices
    sb_init(&s);
    api_json_chunk_usage(&s, "c", "m", 1, 27, 4, 0);
    CHECK(!strcmp(s.p, "{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[],"
                       "\"usage\":{\"prompt_tokens\":27,\"completion_tokens\":4,\"total_tokens\":31}}"));
    sb_free(&s);
    sb_init(&s);
    api_json_chunk_usage(&s, "c", "m", 1, 5, 30, 26);
    CHECK(strstr(s.p, "\"completion_tokens_details\":{\"reasoning_tokens\":26}}}") != NULL);
    sb_free(&s);
}

int main(void) {
    test_http();
    test_json();
    test_chat();
    test_big_request();
    test_think_fields();
    test_tool_fields();
    test_tool_history_args();
    test_tool_json_call();
    test_think_text();
    test_writers();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
