"""Keep the conversation on the OpenAI endpoint (2026-11-22).
The engine's state already holds the previous request's conversation up to pos; a follow-up
request then feeds only the messages that came in since, closing the previous answer like the
chat loop does. Only when the state no longer matches (the screen chat typed in between, or the
client changed history) does the whole request get fed from zero, as before. Run on a clean api_run.inc."""
import pathlib
p = pathlib.Path("chat/api_run.inc")
s = p.read_text()

# 1) api_prompt gains a skip count: the first `skip` messages are already in the engine's state
old = ("static int api_prompt(const JNode* pool, const ChatRequest* cr) {\n"
       "    const int cap = API_IDS - API_TAIL;   // where the messages must end\n"
       "    int n = 0;\n"
       "    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next) {")
new = ("static int api_prompt(const JNode* pool, const ChatRequest* cr, int skip) {\n"
       "    const int cap = API_IDS - API_TAIL;   // where the messages must end\n"
       "    int n = 0;\n"
       "    for (int m = pool[cr->messages].first, i = 0; m >= 0; m = pool[m].next, i++) {\n"
       "        if (i < skip) continue;   // (already in the engine's state: only the new messages are fed)")
assert old in s, "api_prompt anchor missing"
s = s.replace(old, new, 1)

# 2) api_job: remember where the API conversation ends in the engine's state
old = ("static void api_job(ApiJob* job) {\n"
       "    static unsigned api_count;")
new = ("static void api_job(ApiJob* job) {\n"
       "    static unsigned api_count;\n"
       "    static int api_pos, api_msgs_fed;   // the conversation on the engine: where it ends, messages fed")
assert old in s, "api_job anchor missing"
s = s.replace(old, new, 1)

# 3) choose the path, then run
old = ("    int n = api_prompt(pool, &cr);\n"
       "    if (n == -1) {")
new = ("    int mcount = 0;\n"
       "    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next) mcount++;\n"
       "    // keep the conversation: the engine's state holds the earlier requests when nothing typed on the\n"
       "    // screen took the room since. the client's whole history is then fed from where it stopped.\n"
       "    int cont = api_pos > 0 && api_pos == pos && api_msgs_fed > 0 && api_msgs_fed < mcount &&\n"
       "               pos + api_prompt_max(pool, &cr, api_msgs_fed) + cr.max_tokens + 2 <= (int)eng.ctx_limit;\n"
       "    int n = api_prompt(pool, &cr, cont ? api_msgs_fed : 0);\n"
       "    if (n == -1) {")
assert old in s, "api_prompt call anchor missing"
s = s.replace(old, new, 1)

old = ("    if (n + cr.max_tokens + 2 > (int)eng.ctx_limit) {\n"
       "        api_error(job->fd, \"HTTP/1.1 400 Bad Request\", \"The messages and max_tokens do not fit the context window.\",\n"
       "                  \"invalid_request_error\", \"context_length_exceeded\");\n"
       "        return;\n"
       "    }")
new = ("    if (!cont && n + cr.max_tokens + 2 > (int)eng.ctx_limit) {\n"
       "        api_error(job->fd, \"HTTP/1.1 400 Bad Request\", \"The messages and max_tokens do not fit the context window.\",\n"
       "                  \"invalid_request_error\", \"context_length_exceeded\");\n"
       "        return;\n"
       "    }\n"
       "    if (cont && pos + n + cr.max_tokens + 2 > (int)eng.ctx_limit) {\n"
       "        api_error(job->fd, \"HTTP/1.1 400 Bad Request\", \"The conversation and max_tokens do not fit the context window anymore.\",\n"
       "                  \"invalid_request_error\", \"context_length_exceeded\");\n"
       "        return;\n"
       "    }")
assert old in s, "fit check anchor missing"
s = s.replace(old, new, 1)

# 4) the run itself: continue from pos or start over
old = ("    // the model answers this request: the conversation on screen starts again from nothing (a request refused\n"
       "    // above leaves it as it was)\n"
       "    pos = 0;\n"
       "    need_close = 0;\n"
       "    hist_n = hist_done = 0;\n"
       "    thinking = think_turn = 0;\n"
       "    sys_pre = 0;\n"
       "    view.scroll = 0;\n"
       "    api_chat_begin();   // (clears the conversation: a note says that a request is being answered)\n"
       "    if (cr.has_seed) seed = cr.seed;\n"
       "    char id[32];")
new = ("    // the model answers this request: the conversation on screen starts again from nothing (a request refused\n"
       "    // above leaves it as it was)\n"
       "    if (!cont) {\n"
       "        pos = 0;\n"
       "        need_close = 0;\n"
       "    }\n"
       "    hist_n = hist_done = 0;\n"
       "    thinking = think_turn = 0;\n"
       "    sys_pre = 0;\n"
       "    view.scroll = 0;\n"
       "    api_chat_begin();   // (clears the conversation: a note says that a request is being answered)\n"
       "    if (cont && need_close) {   // the last answer's end token was sampled but never fed (as the chat does)\n"
       "        static uint32_t close_ids[2];\n"
       "        if (!close_ids[0]) {\n"
       "            close_ids[0] = IM_END;\n"
       "            uint32_t one[8];\n"
       "            tok_n[0] = 0;\n"
       "            encode_into(one, 0, 8, \"\\n\");\n"
       "            close_ids[1] = one[0];\n"
       "        }\n"
       "        if (engine_run(&eng, close_ids, 2, pos, 0.0f, 1, &next_token)) {\n"
       "            api_error(job->fd, \"HTTP/1.1 500 Internal Server Error\", eng.error, \"server_error\", NULL);\n"
       "            free(pool);\n"
       "            return;\n"
       "        }\n"
       "        pos += 2;\n"
       "    }\n"
       "    if (cr.has_seed) seed = cr.seed;\n"
       "    char id[32];")
assert old in s, "reset block anchor missing"
s = s.replace(old, new, 1)

old = ("    engine_reset(&eng);\n"
       "    if (engine_run(&eng, api_ids, n, 0, (float)cr.temperature, seed++, &next_token)) {\n"
       "        api_error(job->fd, \"HTTP/1.1 500 Internal Server Error\", eng.error, \"server_error\", NULL);\n"
       "        pos = 0;\n"
       "        api_chat_end();\n"
       "        return;\n"
       "    }\n"
       "    pos = n;")
new = ("    if (!cont) engine_reset(&eng);   // (continuing: the Gated DeltaNet states stay)\n"
       "    if (engine_run(&eng, api_ids, n, pos, (float)cr.temperature, seed++, &next_token)) {\n"
       "        api_error(job->fd, \"HTTP/1.1 500 Internal Server Error\", eng.error, \"server_error\", NULL);\n"
       "        pos = cont ? pos : 0;\n"
       "        api_chat_end();\n"
       "        return;\n"
       "    }\n"
       "    pos += n;\n"
       "    api_pos = pos, api_msgs_fed = mcount;   // this conversation is now on the engine up to pos\n"
       "    need_close = 1;   // the answer's end token will be sampled but not fed: the next request closes it")
assert old in s, "run anchor missing"
s = s.replace(old, new, 1)

# 5) room probe for the continuation decision: upper bound of the new messages' tokens by bytes
helper = ("// Upper bound on the tokens the messages from `skip` on will take (a token is at least one byte;\n"
          "    // plus the wrapper per message and the assistant header).\nstatic int api_prompt_max(const JNode* pool, const ChatRequest* cr, int skip) {\n"
          "    int n = 0, i = 0;\n"
          "    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next, i++) {\n"
          "        if (i < skip) continue;\n"
          "        Sb c;\n"
          "        sb_init(&c);\n"
          "        char role[16] = \"user\";\n"
          "        if (api_message(pool, m, role, sizeof(role), &c) || c.oom) return API_IDS;   // (unknown: no continuation)\n"
          "        n += (int)c.n + (int)strlen(role) + 8;\n"
          "        sb_free(&c);\n"
          "    }\n"
          "    return n + 64;\n"
          "}\n\n")
anchor = "static void api_error(int fd, const char* status_line,"
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)

p.write_text(s)
print("api_run.inc patched: conversation kept across requests")
