// Host tests for chat/api_proto.c (the OpenAI-compatible API's protocol part).
// Build and run: gcc -std=c99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/api_proto_test.c chat/api_proto.c -o build/api_proto_test && build/api_proto_test
#include <stdio.h>
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
    CHECK(api_http_parse(bigcl, strlen(bigcl), &r) == -1);
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
    CHECK(json_parse("[1,2]", 5, pool, 2) == -1);   // pool too small
    CHECK(json_get(pool, root, "missing") == -1);
}

static void test_chat(void) {
    JNode pool[API_MAX_NODES];
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
    CHECK(req.max_tokens == 512 && req.stream == 0 && !req.has_seed);

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
}

static void test_writers(void) {
    Sb s;
    sb_init(&s);
    api_json_completion(&s, "chatcmpl-1", "Qwen3.5-9B", 1700000000, "Hola \"mundo\"\n", 13, "stop", 5, 4, NULL, 0, 0);
    CHECK(s.p && strstr(s.p, "\"content\":\"Hola \\\"mundo\\\"\\n\"") != NULL);
    CHECK(strstr(s.p, "\"total_tokens\":9") != NULL);
    CHECK(strstr(s.p, "\"finish_reason\":\"stop\"") != NULL);
    sb_free(&s);

    sb_init(&s);
    api_json_chunk(&s, "c", "m", 1, "assistant", "", 0, NULL);
    CHECK(!strcmp(s.p, "{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\","
                       "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}"));
    sb_free(&s);

    sb_init(&s);
    api_json_error(&s, "bad \x01 input", "invalid_request_error", "bad_body");
    CHECK(!strcmp(s.p, "{\"error\":{\"message\":\"bad \\u0001 input\",\"type\":\"invalid_request_error\","
                       "\"code\":\"bad_body\"}}"));
    sb_free(&s);

    sb_init(&s);
    api_json_models(&s, "Qwen3.5-9B", 0);
    CHECK(strstr(s.p, "\"data\":[{\"id\":\"Qwen3.5-9B\",\"object\":\"model\"") != NULL);
    sb_free(&s);
}

static void test_ollama(void) {
    Sb s;
    sb_init(&s);
    const char* err = NULL;
    const char* chat = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],"
                       "\"options\":{\"temperature\":0.2,\"num_predict\":32,\"seed\":3}}";
    CHECK(api_ollama_as_openai(chat, strlen(chat), 0, &s, &err) == 0);
    JNode pool[API_MAX_NODES];
    ChatRequest req;
    CHECK(api_chat_parse(s.p, s.n, pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.stream == 1 && req.max_tokens == 32 && req.has_seed && req.seed == 3 && req.n_messages == 1);
    CHECK(req.temperature > 0.19 && req.temperature < 0.21);
    sb_free(&s);

    sb_init(&s);
    const char* gen = "{\"prompt\":\"Hi\",\"system\":\"Be brief\",\"stream\":false,"
                      "\"options\":{\"num_predict\":-1,\"seed\":9}}";
    CHECK(api_ollama_as_openai(gen, strlen(gen), 1, &s, &err) == 0);
    CHECK(api_chat_parse(s.p, s.n, pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.stream == 0 && req.max_tokens == 4096 && req.n_messages == 2 && req.has_seed && req.seed == 9);
    char role[16];
    Sb content;
    sb_init(&content);
    int m = pool[req.messages].first;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "system"));
    CHECK(content.n == 8 && !memcmp(content.p, "Be brief", 8));
    sb_free(&content);
    sb_init(&content);
    CHECK(api_message(pool, pool[m].next, role, sizeof(role), &content) == 0 && !strcmp(role, "user"));
    CHECK(content.n == 2 && !memcmp(content.p, "Hi", 2));
    sb_free(&content);
    sb_free(&s);

    sb_init(&s);
    CHECK(api_ollama_as_openai("{\"prompt\":1}", 12, 1, &s, &err) == -1);
    CHECK(err && strstr(err, "prompt"));
    sb_free(&s);
    sb_init(&s);
    CHECK(api_ollama_as_openai("{\"messages\":[]}", 15, 0, &s, &err) == -1);
    sb_free(&s);

    sb_init(&s);
    api_json_ollama_version(&s);
    CHECK(s.p && !strcmp(s.p, "{\"version\":\"1.20-darkgarlic\"}"));
    sb_free(&s);
    sb_init(&s);
    api_json_ollama_tags(&s, "Qwen3.5-9B");
    CHECK(s.p && strstr(s.p, "\"name\":\"Qwen3.5-9B\"") && strstr(s.p, "\"family\":\"qwen3\""));
    sb_free(&s);
    sb_init(&s);
    api_json_ollama_show(&s, "Qwen3.5-9B");
    CHECK(s.p && strstr(s.p, "\"model\":\"Qwen3.5-9B\"") && strstr(s.p, "general.architecture"));
    sb_free(&s);
    sb_init(&s);
    api_json_ollama_error(&s, "nope");
    CHECK(s.p && !strcmp(s.p, "{\"error\":\"nope\"}"));
    sb_free(&s);
    sb_init(&s);
    api_json_ollama_chat(&s, "m", "2026-11-22T00:00:00Z", "Hi", 2, 1, "stop", 3, 1, NULL, 0);
    CHECK(strstr(s.p, "\"content\":\"Hi\"") && strstr(s.p, "\"done\":true") && strstr(s.p, "\"done_reason\":\"stop\""));
    CHECK(strstr(s.p, "\"prompt_eval_count\":3") && strstr(s.p, "\"eval_count\":1"));
    sb_free(&s);
    sb_init(&s);
    api_json_ollama_gen(&s, "m", "2026-11-22T00:00:00Z", "A", 1, 0, NULL, 0, 0);
    CHECK(strstr(s.p, "\"response\":\"A\"") && strstr(s.p, "\"done\":false") && !strstr(s.p, "done_reason"));
    sb_free(&s);
}

static void test_tools(void) {
    JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
    const char* body =
        "{\"messages\":[{\"role\":\"system\",\"content\":\"  Be brief  \"},{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"parameters\":{\"type\":\"object\","
        "\"properties\":{\"path\":{\"type\":\"string\"},\"limit\":{\"type\":\"integer\"},\"flag\":{\"type\":\"boolean\"},"
        "\"meta\":{\"type\":\"object\"}}}}}]}";
    CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_tools == 1 && req.tools >= 0 && req.n_schemas == 1);
    CHECK(!strcmp(req.schemas[0].name, "read_file") && req.schemas[0].n_params == 4);
    CHECK(!strcmp(req.schemas[0].params[0].name, "path") && req.schemas[0].params[0].type == API_P_STR);
    CHECK(!strcmp(req.schemas[0].params[1].name, "limit") && req.schemas[0].params[1].type == API_P_INT);
    CHECK(req.schemas[0].params[2].type == API_P_BOOL && req.schemas[0].params[3].type == API_P_OBJ);

    const char* empty = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[]}";
    CHECK(api_chat_parse(empty, strlen(empty), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_tools == 0 && req.tools == -1);
    const char* bad_tools = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":{}}";
    CHECK(api_chat_parse(bad_tools, strlen(bad_tools), pool, API_MAX_NODES, &req, &err) == -1);

    CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
    Sb sys;
    sb_init(&sys);
    int ate = 0;
    CHECK(api_tools_system(pool, req.tools, pool[req.messages].first, &sys, &ate) == 0);
    CHECK(ate == 1);
    CHECK(sys.p && strstr(sys.p, "<tools>") && strstr(sys.p, "read_file") && strstr(sys.p, "</IMPORTANT>"));
    CHECK(strstr(sys.p, "Be brief") != NULL);
    const char* once = strstr(sys.p, "Be brief");
    CHECK(once && strstr(once + 1, "Be brief") == NULL);
    sb_free(&sys);

    const char* as_str =
        "{\"role\":\"assistant\",\"content\":\"Sure\",\"tool_calls\":[{\"id\":\"c\",\"type\":\"function\","
        "\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"notes.txt\\\",\\\"limit\\\":5}\"}}]}";
    int root = json_parse(as_str, strlen(as_str), pool, API_MAX_NODES);
    Sb xml;
    sb_init(&xml);
    CHECK(root >= 0 && api_message_tool_xml(pool, root, 1, &xml) == 0);
    CHECK(xml.p && !strcmp(xml.p,
                           "\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n"
                           "<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>"));
    sb_free(&xml);

    const char* as_obj =
        "{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"function\":{\"name\":\"read_file\","
        "\"arguments\":{\"path\":\"notes.txt\",\"flag\":true}}}]}";
    root = json_parse(as_obj, strlen(as_obj), pool, API_MAX_NODES);
    sb_init(&xml);
    CHECK(root >= 0 && api_message_tool_xml(pool, root, 0, &xml) == 0);
    CHECK(xml.p && !strcmp(xml.p, "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n"
                                  "<parameter=flag>\nTrue\n</parameter>\n</function>\n</tool_call>"));
    sb_free(&xml);

    const char* model =
        "Ok\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n"
        "<parameter=limit>\n5\n</parameter>\n<parameter=code>\n007\n</parameter>\n</function>\n</tool_call>";
    ApiToolCall calls[4];
    int ncalls = 0, cut = 0;
    Sb content;
    sb_init(&content);
    CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(api_parse_tool_calls(model, strlen(model), req.schemas, req.n_schemas, &content, calls, 4, &ncalls, &cut) ==
          0);
    CHECK(ncalls == 1 && cut == 0 && content.n == 2 && !memcmp(content.p, "Ok", 2));
    CHECK(!strcmp(calls[0].name, "read_file"));
    CHECK(!strcmp(calls[0].arguments, "{\"path\":\"notes.txt\",\"limit\":5,\"code\":\"007\"}"));
    sb_free(&content);

    const char* partial = "Hi <tool_call>\n<function=read_file>";
    sb_init(&content);
    CHECK(api_parse_tool_calls(partial, strlen(partial), req.schemas, req.n_schemas, &content, calls, 4, &ncalls,
                               &cut) == 0);
    CHECK(ncalls == 0 && cut == 1 && content.n == strlen(partial) && !memcmp(content.p, partial, content.n));
    sb_free(&content);

    CHECK(api_tool_stream_end("hello", 5) == 5);
    CHECK(api_tool_stream_end("a<", 2) == 1);
    CHECK(api_tool_stream_end("hello <tool_ca", 14) == 6);
    const char* held = "hello <tool_call>\n";
    CHECK(api_tool_stream_end(held, strlen(held)) == 6);

    Sb s;
    sb_init(&s);
    ApiToolCall one;
    memset(&one, 0, sizeof(one));
    memcpy(one.name, "read_file", 10);
    memcpy(one.arguments, "{\"path\":\"notes.txt\"}", 20);
    api_json_completion(&s, "chatcmpl-1", "Qwen3.5-9B", 1700000000, "", 0, "tool_calls", 5, 4, &one, 1, 7);
    CHECK(s.p && strstr(s.p, "\"content\":null") && strstr(s.p, "\"finish_reason\":\"tool_calls\""));
    CHECK(strstr(s.p, "\"id\":\"call_7_0\"") && strstr(s.p, "\"name\":\"read_file\""));
    CHECK(strstr(s.p, "\"arguments\":\"{\\\"path\\\":\\\"notes.txt\\\"}\""));
    sb_free(&s);

    sb_init(&s);
    api_json_ollama_chat(&s, "m", "2026-11-22T00:00:00Z", "Ok", 2, 1, "stop", 3, 8, &one, 1);
    CHECK(strstr(s.p, "\"content\":\"Ok\"") && strstr(s.p, "\"arguments\":{\"path\":\"notes.txt\"}"));
    CHECK(strstr(s.p, "\"done_reason\":\"stop\""));
    sb_free(&s);

    sb_init(&s);
    const char* ochat = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"tools\":[{\"type\":\"function\","
                        "\"function\":{\"name\":\"ping\"}}],\"stream\":false}";
    CHECK(api_ollama_as_openai(ochat, strlen(ochat), 0, &s, &err) == 0);
    CHECK(api_chat_parse(s.p, s.n, pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_tools == 1 && !strcmp(req.schemas[0].name, "ping") && req.stream == 0);
    sb_free(&s);
}

int main(void) {
    test_http();
    test_json();
    test_chat();
    test_writers();
    test_ollama();
    test_tools();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
