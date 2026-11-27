// OpenAI-compatible API jobs (api.c queues them; included in main.c, so it sees the engine and the conversation).
// A job runs on the model between the chat's turns: the request's messages are the whole conversation, so the
// conversation on screen is cleared, and the reply is written to the connection (whole, or as server-sent events).
// The reply is generated as the chat generates its own: several tokens per step with the MTP head where the app has
// it (GARLIC_SPEC), and Circle on the controller stops it. What the screen shows meanwhile is main.c's part
// (api_chat_begin / _reply / _step / _end).
//
// The prompt is Qwen3.5's chat template (its chat_template.jinja): the messages trimmed, tools in the system message,
// tool results in user turns, the reasoning of the turns after the question kept.
//
// Reasoning works as in the chat (main.c, think_one): the turn opens <think> and the model reasons at THINK_T for up
// to think_cap tokens, then answers; if it has not closed </think> by then, Qwen's closing sentence and </think> are
// fed. The reasoning goes to the client apart from the answer (reasoning_content), and max_tokens counts the answer
// only. Whether a reply reasons: the last user message's /think or /no_think, else the request's own field
// (api_chat_parse), else the console's setting (R3). Without max_tokens the answer may use the rest of the context
// window (an agent's reply may be a long tool call), and reasoning starts only with API_ANSWER_MIN left for it.
//
// Tools: the model writes <tool_call> blocks (api_proto.c reads them into OpenAI's tool_calls). Streamed text stops
// at the first <tool_call>; the calls go out whole at the end. The tools' JSON is part of the prompt, so it counts
// toward the context window and prompt_tokens like the messages.
//
// The request's JSON pool (up to 4 MiB for a big agent request) is freed once the prompt is written; the reply's
// tool calls are read with the request read again, at the end.
//
// The prompt's beginning is kept between requests (a client sends the whole conversation every time): after the
// prompt is read up to its last "<|im_start|>assistant\n", the Gated DeltaNet state there is copied into the
// dictation's work memory (as the chat's reasoning does, think_save), with the ids it stands for (api_snap_ids). A
// next request that begins with those same ids starts from the copy and reads only the rest. Attention layers keep
// their cache by position, so they need no copy. Anything else that runs the model or uses that memory (the chat,
// dictation, voice, Python) sets api_snap_n to 0.

#define API_IDS 16384             // tokens a job's prompt may have: the longest context of the app
#define API_TAIL 128              // kept free for the assistant's turn after the messages (and a named tool's start)
#define API_THINK_MIN 64          // the least reasoning worth starting; with less room the reply comes without
#define API_ANSWER_MIN 512        // without max_tokens: the room reasoning leaves for the answer at least
#define API_CHUNK 256             // tokens read per engine call: a client that leaves or Circle stops a long prompt
#define API_LOOK 32               // tokens between two looks at a connection that nothing is sent to meanwhile
static uint32_t api_ids[API_IDS];        // the prompt of a job
static uint32_t api_snap_ids[API_IDS];   // slot 1: full conversation prefix (api_snap_n in main.c)
static uint32_t api_sys_ids[API_IDS];    // slot 0: system prompt + tools prefix (for new agent sessions)
static int api_sys_n;
static int api_last_sys_end;             // token offset right after the system/tools message of the last api_prompt()

// Qwen's tool tokens, by their place after <|im_start|> in its vocabulary
#define TOOL_CALL (IM_START + 13)        // <tool_call>
#define TOOL_CALL_END (IM_START + 14)    // </tool_call>
#define TOOL_RESP (IM_START + 21)        // <tool_response>
#define TOOL_RESP_END (IM_START + 22)    // </tool_response>

// How many bytes at the start of p (n) are whole UTF-8 characters (the rest waits for the next token).
static size_t api_utf8_whole(const uint8_t* p, size_t n) {
    size_t i = n;
    for (int back = 1; back <= 4 && i > 0; back++) {
        i--;
        uint8_t c = p[i];
        if ((c & 0xC0) == 0x80) continue;   // a continuation byte: keep looking for its lead byte
        size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        return back < (int)need ? i : n;
    }
    return n;
}

// Where a "<tool_call>" that may still be coming begins at the end of p (n bytes), or n: text before it can be sent.
static size_t api_tool_prefix(const char* p, size_t n) {
    static const char tag[] = "<tool_call>";
    size_t start = n > sizeof(tag) - 1 ? n - (sizeof(tag) - 1) : 0;
    for (size_t i = start; i < n; i++)
        if (p[i] == '<' && !memcmp(p + i, tag, n - i)) return i;
    return n;
}

// The first n bytes of p without the spaces and line breaks at their end.
static size_t api_trim_end(const char* p, size_t n) {
    while (n && (p[n - 1] == ' ' || p[n - 1] == '\n' || p[n - 1] == '\r' || p[n - 1] == '\t')) n--;
    return n;
}

// Text into ids, with the tool tags as Qwen's special tokens (the tokenizer reads them as plain text).
static int api_encode_text(uint32_t* ids, int n, int max, const char* text) {
    static const struct {
        const char* tag;
        size_t len;
        uint32_t off;
    } tags[] = {{"<tool_call>", 11, 13}, {"</tool_call>", 12, 14}, {"<tool_response>", 15, 21}, {"</tool_response>", 16, 22}};
    const char* p = text;
    Sb span;
    sb_init(&span);
    while (*p && n < max) {
        const char* best = NULL;
        int bk = -1;
        for (int k = 0; k < 4; k++) {
            const char* hit = strstr(p, tags[k].tag);
            if (hit && (!best || hit < best)) best = hit, bk = k;
        }
        if (!best) {
            n = encode_into(ids, n, max, p);
            break;
        }
        if (best > p) {
            span.n = 0;
            sb_add(&span, p, (size_t)(best - p));
            if (span.oom) break;
            n = encode_into(ids, n, max, span.p);
        }
        if (n < max) ids[n++] = IM_START + tags[bk].off;
        p = best + tags[bk].len;
    }
    sb_free(&span);
    return n;
}

// one server-sent event; -1 if the client went away
static int api_send_event(int fd, const char* json) {
    Sb s;
    sb_init(&s);
    sb_str(&s, "data: ");
    sb_str(&s, json);
    sb_str(&s, "\n\n");
    int r = s.p && !s.oom ? api_send(fd, s.p, s.n) : -1;
    sb_free(&s);
    return r;
}

// The first `upto` bytes of pend that are whole characters as one event (reasoning or answer), taken out of pend;
// -1 if the client went away.
static int api_stream_piece(int fd, const char* id, const char* model, long long created, Sb* pend, size_t upto,
                            int reasoning) {
    if (!upto) return 0;
    size_t whole = api_utf8_whole((const uint8_t*)pend->p, upto);
    if (!whole) return 0;
    Sb ev;
    sb_init(&ev);
    if (reasoning) api_json_chunk(&ev, id, model, created, NULL, pend->p, whole, "", 0, NULL);
    else api_json_chunk(&ev, id, model, created, NULL, "", 0, pend->p, whole, NULL);
    int r = ev.p && !ev.oom ? api_send_event(fd, ev.p) : -1;
    sb_free(&ev);
    memmove(pend->p, pend->p + whole, pend->n - whole);
    pend->n -= whole;
    pend->p[pend->n] = 0;
    return r;
}

static int api_role_is(const char* role, const char* r) { return !strcmp(role, r); }

// The role a message is read with: system (developer too), assistant, tool; anything else is the user's.
static void api_role(char* role) {
    if (api_role_is(role, "developer")) strcpy(role, "system");
    if (!api_role_is(role, "system") && !api_role_is(role, "assistant") && !api_role_is(role, "tool"))
        strcpy(role, "user");
}

// The messages of the prompt, as Qwen3.5's chat template writes them (see the top of this file). The last user
// message's reasoning switch is taken out of its text (into *think_switch: 1, 0 or -1, see api_think_switch).
// *tool_turn is 1 if the conversation ends with tool results. Returns the number of ids, -1 if a message is
// malformed, -3 if an assistant message's tool_calls are, or -2 if the messages do not fit API_IDS. The limit counts
// tokens, not bytes: a long conversation may use the whole context. (Every token stands for one byte or more, so the
// turn after the messages fits API_TAIL.)
static int api_prompt(const JNode* pool, const ChatRequest* cr, int* think_switch, int* tool_turn) {
    const int cap = API_IDS - API_TAIL;   // where the messages must end
    int has_tools = cr->n_tools > 0 && cr->tool_choice != TOOL_NONE;
    int n = 0, i = 0, query = -1;
    *think_switch = -1;
    *tool_turn = 0;
    api_last_sys_end = 0;
    // the question this reply answers: the last user message that is not only tool results
    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next, i++) {
        Sb c;
        sb_init(&c);
        char role[16] = "user";
        if (api_message(pool, m, role, sizeof(role), &c) == 0) {
            api_role(role);
            api_trim(&c);
            int results = c.n >= 31 && !strncmp(c.p, "<tool_response>", 15) && !strcmp(c.p + c.n - 16, "</tool_response>");
            if (api_role_is(role, "user") && !results) query = i;
        }
        sb_free(&c);
    }
    int m = pool[cr->messages].first;
    i = 0;
    if (has_tools) {   // the tools' system message, with the request's own system text after it
        Sb sys, own;
        sb_init(&sys);
        sb_init(&own);
        char role[16] = "user";
        int own_ok = m >= 0 && api_message(pool, m, role, sizeof(role), &own) == 0;
        if (own_ok) api_role(role);
        if (own_ok && api_role_is(role, "system")) {   // (it goes in this message: the loop below skips it)
            m = pool[m].next;
            i++;
            api_trim(&own);
        } else {
            own.n = 0;
        }
        api_system_tools(pool, cr, own.n ? own.p : "", &sys);
        sb_free(&own);
        if (sys.oom) {
            sb_free(&sys);
            return -1;
        }
        api_ids[n++] = IM_START;
        n = encode_into(api_ids, n, cap, "system\n");
        n = api_encode_text(api_ids, n, cap, sys.p);
        sb_free(&sys);
        if (n + 2 > cap) return -2;
        api_ids[n++] = IM_END;
        n = encode_into(api_ids, n, cap, "\n");
        api_last_sys_end = n;
    }
    int in_results = 0;   // inside the user turn that holds tool results
    for (; m >= 0; m = pool[m].next, i++) {
        Sb content, calls, reasoning;
        sb_init(&content);
        sb_init(&calls);
        sb_init(&reasoning);
        char role[16] = "user";
        int ok = api_message(pool, m, role, sizeof(role), &content) == 0;
        api_role(role);
        int assistant = api_role_is(role, "assistant");
        int n_calls = ok && assistant ? api_assistant_tool_calls(pool, m, &calls) : 0;
        if (!ok || n_calls < 0 || content.oom || calls.oom) {
            sb_free(&content);
            sb_free(&calls);
            sb_free(&reasoning);
            return ok && n_calls < 0 ? -3 : -1;
        }
        api_trim(&content);
        if (assistant) {   // its reasoning: the message's reasoning_content, else what its text holds before </think>
            int rc = json_get(pool, m, "reasoning_content");
            if (rc >= 0 && pool[rc].type == J_STR) {
                json_str_sb(pool, rc, &reasoning);
                api_trim(&reasoning);
                api_drop_think(&content, NULL);
            } else {
                api_drop_think(&content, &reasoning);
            }
        }
        if (i == query) *think_switch = api_think_switch(&content);
        if (in_results && !api_role_is(role, "tool")) {
            api_ids[n++] = IM_END;
            n = encode_into(api_ids, n, cap, "\n");
            in_results = 0;
        }
        if (api_role_is(role, "tool")) {
            if (!in_results) {
                api_ids[n++] = IM_START;
                n = encode_into(api_ids, n, cap, "user");
                in_results = 1;
            }
            n = encode_into(api_ids, n, cap, "\n");
            if (n < cap) api_ids[n++] = TOOL_RESP;
            n = encode_into(api_ids, n, cap, "\n");
            if (content.p) n = api_encode_text(api_ids, n, cap, content.p);
            n = encode_into(api_ids, n, cap, "\n");
            if (n < cap) api_ids[n++] = TOOL_RESP_END;
        } else {
            api_ids[n++] = IM_START;
            n = encode_into(api_ids, n, cap, role);
            n = encode_into(api_ids, n, cap, "\n");
            if (assistant && i > query && eng.qwen35) {   // (a turn after the question keeps its reasoning)
                if (n < cap) api_ids[n++] = THINK;
                n = encode_into(api_ids, n, cap, "\n");
                if (reasoning.p) n = encode_into(api_ids, n, cap, reasoning.p);
                n = encode_into(api_ids, n, cap, "\n");
                if (n < cap) api_ids[n++] = THINK_END;
                n = encode_into(api_ids, n, cap, "\n\n");
            }
            if (content.p) n = api_encode_text(api_ids, n, cap, content.p);
            if (n_calls) {
                if (content.n) n = encode_into(api_ids, n, cap, "\n\n");
                n = api_encode_text(api_ids, n, cap, calls.p);
            }
            if (n + 2 <= cap) {
                api_ids[n++] = IM_END;
                n = encode_into(api_ids, n, cap, "\n");
            }
            if (api_role_is(role, "system") && !api_last_sys_end) api_last_sys_end = n;
        }
        sb_free(&content);
        sb_free(&calls);
        sb_free(&reasoning);
        if (n + 2 > cap) return -2;   // the text filled what was left (it may have been cut): too long
    }
    if (in_results) {
        api_ids[n++] = IM_END;
        n = encode_into(api_ids, n, cap, "\n");
        *tool_turn = 1;
    }
    return n;
}

// The assistant's turn after the n ids of the messages: <think>\n to reason, or the empty thought block (as the chat
// does, enable_thinking=false), then the start of the call that tool_choice asks for (also in *forced). *head is
// where the turn's "assistant\n" ends (the state worth keeping, see the top of this file). Returns the number of ids.
static int api_turn(int n, int think, const ChatRequest* cr, Sb* forced, int* head) {
    api_ids[n++] = IM_START;
    n = encode_into(api_ids, n, API_IDS, "assistant\n");
    *head = n;
    if (eng.qwen35) {
        api_ids[n++] = THINK;
        if (think) return encode_into(api_ids, n, API_IDS, "\n");
        n = encode_into(api_ids, n, API_IDS, "\n\n");
        api_ids[n++] = THINK_END;
        n = encode_into(api_ids, n, API_IDS, "\n\n");
    }
    if (cr->n_tools > 0 && (cr->tool_choice == TOOL_REQUIRED || cr->tool_choice == TOOL_NAMED)) {
        sb_str(forced, "<tool_call>\n<function=");
        if (cr->tool_choice == TOOL_NAMED) {
            sb_str(forced, cr->tool_name);
            sb_str(forced, ">\n");
        }
        n = api_encode_text(api_ids, n, API_IDS, forced->p);
    }
    return n;
}

static int api_can_snap(void) {
    uint64_t need = engine_state_size(&eng);
#ifdef GARLIC_MAPLE_ENGINE
    int ok = eng.qwen35 || eng.maple;
#else
    int ok = eng.qwen35;
#endif
    return ok && eng.asr.ws_end && eng.asr.ws_end - eng.asr.ws_begin >= need;
}

static int api_can_snap2(void) {
    uint64_t need = engine_state_size(&eng);
#ifdef GARLIC_MAPLE_ENGINE
    int ok = eng.qwen35 || eng.maple;
#else
    int ok = eng.qwen35;
#endif
    return ok && eng.asr.ws_end && eng.asr.ws_end - eng.asr.ws_begin >= 2 * need;
}

// Reads ids[from, to) of the prompt in pieces of API_CHUNK, the last one drawing next_token at temperature temp.
// Returns 0, 1 if the client left or Circle stopped it (between pieces), or -1 on an engine error.
static int api_read(int fd, int from, int to, float temp) {
    for (int at = from; at < to; at += API_CHUNK) {
        if (at > from && (api_client_gone(fd) || api_chat_step(0))) return 1;
        int k = to - at < API_CHUNK ? to - at : API_CHUNK;
        if (engine_run(&eng, api_ids + at, k, at, temp, seed++, &next_token)) return -1;
    }
    return 0;
}

// Slot 1 (conversation prefix): saved at ws_begin + need when 2 slots fit, else at ws_begin.
static void api_snap_save(int head) {
    uint64_t off = api_can_snap2() ? engine_state_size(&eng) : 0;
    if (engine_state_copy(&eng, eng.asr.ws_begin + off, 1)) {
        api_snap_n = 0;
        return;
    }
    memcpy(api_snap_ids, api_ids, (size_t)head * sizeof(uint32_t));
    api_snap_n = head;
}

// Slot 0 (system prompt + tools prefix): saved at ws_begin when 2 slots fit, so new sessions reuse the system prompt.
static void api_sys_save(int sys_end) {
    if (!api_can_snap2() || sys_end <= 0) return;
    if (engine_state_copy(&eng, eng.asr.ws_begin, 1)) {
        api_sys_n = 0;
        return;
    }
    memcpy(api_sys_ids, api_ids, (size_t)sys_end * sizeof(uint32_t));
    api_sys_n = sys_end;
}

// Reads the prompt (n ids), starting from the best matching cached prefix (full conversation slot 1, or system+tools
// slot 0). Returns as api_read.
static int api_read_prompt(int fd, int n, int head, int tool_turn, float temp, int* cached) {
    *cached = 0;
    if (api_snap_n == 0) api_sys_n = 0;   // (chat/ASR/TTS/think cleared ws_begin)
    int two = api_can_snap2();
    uint64_t conv_off = two ? engine_state_size(&eng) : 0;
    int same = 0;
    while (same < api_snap_n && same < head && api_ids[same] == api_snap_ids[same]) same++;
    int sys_same = 0;
    if (two && api_sys_n > 0) {
        while (sys_same < api_sys_n && sys_same < head && api_ids[sys_same] == api_sys_ids[sys_same]) sys_same++;
    }
    if (api_snap_n > 0 && same == api_snap_n && api_can_snap() &&
        engine_state_copy(&eng, eng.asr.ws_begin + conv_off, 0) == 0) {
        *cached = api_snap_n;
    } else if (two && api_sys_n > 0 && sys_same == api_sys_n &&
               engine_state_copy(&eng, eng.asr.ws_begin, 0) == 0) {
        *cached = api_sys_n;
    } else {
        api_snap_n = 0;
        api_sys_n = 0;
        engine_reset(&eng);
    }
    int cur = *cached;
    // Save slot 0 (system prompt + tools) on first sight when 2 slots fit and it wasn't already cached.
    int sys_end = api_last_sys_end;
    if (two && sys_end > cur && sys_end < head && api_sys_n != sys_end) {
        int r = api_read(fd, cur, sys_end, temp);
        if (r) return r;
        api_sys_save(sys_end);
        cur = sys_end;
    }
    int keep = head > cur && head < n && api_can_snap();   // (a new state to keep: up to this turn's assistant)
    int r = api_read(fd, cur, keep ? head : n, temp);
    if (r || !keep) return r;
    // (in a run of tool calls the state of the question is kept: the next request begins the same way)
    if (!tool_turn || !*cached) api_snap_save(head);
    return api_read(fd, head, n, temp);
}

static void api_error(int fd, const char* status_line, const char* message, const char* type, const char* code) {
    Sb s;
    sb_init(&s);
    api_json_error(&s, message, type, code);
    Sb h;
    sb_init(&h);
    sb_str(&h, status_line);
    sb_str(&h, "\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, (long long)s.n);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && s.p && api_send(fd, h.p, h.n) == 0) api_send(fd, s.p, s.n);
    sb_free(&h);
    sb_free(&s);
}

// The tokens of the tool block alone (api_system_tools without the request's own system text), written into
// api_ids: only for the answer to a request that is refused (its prompt is not needed any more). -1 if it fills them.
static int api_tools_tokens(const JNode* pool, const ChatRequest* cr) {
    Sb sys;
    sb_init(&sys);
    api_system_tools(pool, cr, "", &sys);
    int n = sys.p && !sys.oom ? api_encode_text(api_ids, 0, API_IDS, sys.p) : -1;
    sb_free(&sys);
    return n >= API_IDS ? -1 : n;
}

// The answer to a request that does not fit the context window, worded as OpenAI's (clients read the numbers in it):
// prompt is the messages' tokens (-1: more than API_IDS - API_TAIL), completion the max_tokens asked for (0: none).
// It never says "tool" or "function": some clients take a 400 that does to mean that the server cannot call tools,
// and stop sending them. The tools are "the schemas sent with the request".
static void api_too_long(int fd, const JNode* pool, const ChatRequest* cr, int prompt, int completion) {
    char msg[512];
    int window = (int)eng.ctx_limit, k;
    if (prompt < 0)
        k = snprintf(msg, sizeof(msg), "This model's maximum context length is %d tokens. However, your messages take "
                     "more than %d tokens.", window, API_IDS - API_TAIL);
    else if (!completion)
        k = snprintf(msg, sizeof(msg), "This model's maximum context length is %d tokens. However, your messages take "
                     "%d tokens.", window, prompt);
    else
        k = snprintf(msg, sizeof(msg), "This model's maximum context length is %d tokens. However, you requested %d "
                     "tokens (%d in the messages, %d in the completion).", window, prompt + completion, prompt,
                     completion);
    if (cr->n_tools > 0 && cr->tool_choice != TOOL_NONE && k > 0 && k < (int)sizeof(msg)) {
        int t = api_tools_tokens(pool, cr);
        char what[48];
        if (cr->n_tools == 1) snprintf(what, sizeof(what), "The schema sent with the request takes");
        else snprintf(what, sizeof(what), "The %d schemas sent with the request take", cr->n_tools);
        if (t < 0) k += snprintf(msg + k, sizeof(msg) - k, " %s more than %d of them.", what, API_IDS - 1);
        else k += snprintf(msg + k, sizeof(msg) - k, " %s %d of them.", what, t);
    }
    if (k > 0 && k < (int)sizeof(msg))
        snprintf(msg + k, sizeof(msg) - k, completion ? " Please reduce the length of the messages or completion."
                                                      : " Please reduce the length of the messages.");
    say("api: request refused: %s", msg);
    api_error(fd, "HTTP/1.1 400 Bad Request", msg, "invalid_request_error", "context_length_exceeded");
}

static int api_is_blank(const uint8_t* p, int n) {
    for (int i = 0; i < n; i++)
        if (p[i] != '\n' && p[i] != '\r' && p[i] != ' ') return 0;
    return 1;
}

#ifdef GARLIC_API_AUDIO
static void api_wav_from_pcm16(Sb* out, const int16_t* samples, size_t n_samples, uint32_t sample_rate) {
    uint32_t data_bytes = (uint32_t)(n_samples * 2);
    uint32_t riff_size = 36 + data_bytes;
    uint32_t byte_rate = sample_rate * 2;
    uint8_t hdr[44] = {'R', 'I', 'F', 'F', (uint8_t)riff_size, (uint8_t)(riff_size >> 8), (uint8_t)(riff_size >> 16),
                       (uint8_t)(riff_size >> 24), 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 16, 0, 0, 0, 1, 0, 1, 0,
                       (uint8_t)sample_rate, (uint8_t)(sample_rate >> 8), (uint8_t)(sample_rate >> 16),
                       (uint8_t)(sample_rate >> 24), (uint8_t)byte_rate, (uint8_t)(byte_rate >> 8),
                       (uint8_t)(byte_rate >> 16), (uint8_t)(byte_rate >> 24), 2, 0, 16, 0, 'd', 'a', 't', 'a',
                       (uint8_t)data_bytes, (uint8_t)(data_bytes >> 8), (uint8_t)(data_bytes >> 16),
                       (uint8_t)(data_bytes >> 24)};
    sb_add(out, (const char*)hdr, sizeof(hdr));
    sb_add(out, (const char*)samples, data_bytes);
}

// Synthesizes 24 kHz 16-bit mono WAV (or raw PCM16 when cr->audio_pcm) for reply_text using Chatterbox (T3 + S3Gen).
static int api_synthesize_audio(const ChatRequest* cr, const char* reply_text, size_t reply_len, int produced,
                                Sb* audio_out) {
    if (!tts_ready || !reply_text || reply_len == 0) return -1;
    api_snap_n = 0;
    double t0 = now();
#ifdef GARLIC_SPEC
    if (spec_lent) {
        spec_unload(&eng);
        spec_lent = 0;
        if (t3_reload(&eng)) {
            say("api audio: T3 reload failed (%s)", eng.error);
            return -1;
        }
    }
#endif
    uint64_t wsize = 0, w = engine_kv_lend(&eng, &wsize);
    if (!w) {
        say("api audio: cannot lend KV memory for S3Gen");
        return -1;
    }
    if (!eng.s3.loaded && s3g_load(&eng, w, wsize, eng.asr.ws_begin, eng.asr.ws_end - eng.asr.ws_begin)) {
        say("api audio: cannot load S3Gen (%s)", eng.error);
        engine_kv_return(&eng);
        return -1;
    }
    const char* forced_lang = NULL;
    const char* custom_voice = NULL;
    char voice_buf[32];
    snprintf(voice_buf, sizeof(voice_buf), "%s", cr->audio_voice);
    char* colon = strchr(voice_buf, ':');
    const char* lang_part = voice_buf;
    if (colon) {
        *colon = 0;
        custom_voice = colon + 1;
    }
    static const char* known_langs[] = {"es", "en", "pt", "ru", "it", "fr", "de"};
    for (size_t i = 0; i < sizeof(known_langs) / sizeof(known_langs[0]); i++) {
        if (!strcmp(lang_part, known_langs[i])) {
            forced_lang = known_langs[i];
            break;
        }
    }
    if (!custom_voice && !forced_lang && voice_buf[0] && strcmp(voice_buf, "auto")) custom_voice = voice_buf;
    char base_lang[4];
    snprintf(base_lang, sizeof(base_lang), "%s", forced_lang ? forced_lang : st_lang(reply_text, lang_code()));

    Sb pcm;
    sb_init(&pcm);
    int cur = 0, len = (int)reply_len;
    while (cur < len && !pcm.oom) {
        if (api_chat_step(produced)) break;
        int end = st_next_piece(reply_text, len, cur, 1);
        if (end <= cur) break;
        while (end - cur < 80) {
            int e2 = st_next_piece(reply_text, len, end, 1);
            if (e2 <= end || st_spoken_len(reply_text + cur, e2 - cur) > 200) break;
            if (!forced_lang && strcmp(st_text_lang(reply_text + cur, end - cur, base_lang),
                                       st_text_lang(reply_text + end, e2 - end, base_lang)))
                break;
            end = e2;
        }
        int start = cur;
        cur = end;
        static char words[1024];
        int wl = st_clean(reply_text + start, end - start, words, sizeof(words) - 2);
        if (!wl) continue;
        const char* lang = forced_lang ? forced_lang : st_lang(words, base_lang);
        wl = st_speak(words, wl, (int)sizeof(words) - 2, lang);
        if (!wl) continue;
        for (char* c = words; *c; c++)
            if (*c == ':' && !(c > words && c[-1] >= '0' && c[-1] <= '9' && c[1] >= '0' && c[1] <= '9')) *c = ',';

        static uint16_t toks[512];
        int nt = ttstok_encode(&tts_tok, words, lang, toks, 400);
        int max_out = (int)strlen(words) * 3 + 40;
        if (max_out > 450) max_out = 450;
        t3_voice(&eng, lang);
        if (!custom_voice || !s3g_voice_named(&eng, custom_voice)) s3g_voice(&eng, lang);
        if (t3_start(&eng, toks, nt, max_out, ++tts.seed, 0)) break;
        int n_speech = 0;
        for (;;) {
            int t = t3_next(&eng);
            if (t >= 0 && n_speech < 512) {
                tts.speech[n_speech++] = (uint16_t)t;
                continue;
            }
            break;
        }
        static const uint16_t sil[6] = {6324, 6405, 4137, 4218, 6081, 6082};
        int run = 0;
        for (; run < n_speech; run++) {
            uint16_t s = tts.speech[n_speech - 1 - run];
            int is = 0;
            for (int q = 0; q < 6; q++) is |= s == sil[q];
            if (!is) break;
        }
        if (run > 7) n_speech -= run - 7;
        int s3r = n_speech >= 2 ? s3g_begin(&eng, tts.speech, n_speech, tts.seed) : 0;
        if (s3r > 0) {
            for (;;) {
                int r = s3g_step(&eng, tts_pcm, (int)(sizeof(tts_pcm) / sizeof(tts_pcm[0])));
                if (r < 0) break;
                if (r == 0) continue;
                float gain = eng.s3.gain;
                for (int i = 0; i < r; i++) {
                    float v = tts_pcm[i] * gain;
                    if (v > 1.0f) v = 1.0f;
                    if (v < -1.0f) v = -1.0f;
                    int16_t s16 = (int16_t)(v * 32767.0f);
                    uint8_t b[2] = {(uint8_t)(s16 & 0xFF), (uint8_t)((uint16_t)s16 >> 8)};
                    sb_add(&pcm, (const char*)b, 2);
                }
            }
        }
    }
    if (!tts_keep()) eng.s3.loaded = 0;
    engine_kv_return(&eng);
    if (pcm.oom || pcm.n == 0) {
        sb_free(&pcm);
        return -1;
    }
    size_t n_samples = pcm.n / 2;
    if (cr->audio_pcm) {
        *audio_out = pcm;
    } else {
        api_wav_from_pcm16(audio_out, (const int16_t*)pcm.p, n_samples, 24000);
        sb_free(&pcm);
    }
    say("api audio: %.2f s of 24 kHz speech synthesized in %.2f s (%s)", n_samples / 24000.0, now() - t0,
        cr->audio_pcm ? "pcm16" : "wav");
    return 0;
}

static void api_speech_job(ApiJob* job) {
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    SpeechRequest sr;
    const char* err = NULL;
    if (!pool || api_speech_parse(job->body, job->body_len, pool, API_MAX_NODES, &sr, &err)) {
        if (pool) sb_free(&sr.input);
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", err ? err : "Out of memory.", "invalid_request_error",
                  "invalid_body");
        return;
    }
    free(pool);
    if (!tts_ready) {
        sb_free(&sr.input);
        api_error(job->fd, "HTTP/1.1 503 Service Unavailable", "TTS models are not loaded.", "server_error", NULL);
        return;
    }
    api_chat_begin();
    ChatRequest cr;
    memset(&cr, 0, sizeof(cr));
    cr.want_audio = 1;
    cr.audio_pcm = sr.pcm;
    snprintf(cr.audio_voice, sizeof(cr.audio_voice), "%s", sr.voice);
    Sb audio_bin;
    sb_init(&audio_bin);
    if (api_synthesize_audio(&cr, sr.input.p, sr.input.n, 0, &audio_bin) == 0 && audio_bin.p && audio_bin.n > 0) {
        Sb head;
        sb_init(&head);
        sb_str(&head, "HTTP/1.1 200 OK\r\nContent-Type: ");
        sb_str(&head, sr.pcm ? "audio/pcm" : "audio/wav");
        sb_str(&head, "\r\nContent-Length: ");
        sb_int(&head, (long long)audio_bin.n);
        sb_str(&head, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if (head.p && api_send(job->fd, head.p, head.n) == 0) api_send(job->fd, audio_bin.p, audio_bin.n);
        sb_free(&head);
    } else {
        api_error(job->fd, "HTTP/1.1 500 Internal Server Error", eng.error[0] ? eng.error : "Speech synthesis failed.",
                  "server_error", NULL);
    }
    sb_free(&audio_bin);
    sb_free(&sr.input);
    api_chat_end();
#ifdef GARLIC_SPEC
    spec_lend();
#endif
}

static void api_voices_job(ApiJob* job) {
    Sb s, h;
    sb_init(&s);
    sb_init(&h);
    sb_str(&s, "{\"voices\":[");
    for (int i = 0; i < eng.s3.nvoice; i++) {
        if (i) sb_add(&s, ",", 1);
        sb_str(&s, "{\"id\":");
        sb_json_str(&s, eng.s3.voice[i].name, strlen(eng.s3.voice[i].name));
        int bi = eng.s3.voice[i].lang[0] || !strcmp(eng.s3.voice[i].name, "default");
        sb_str(&s, ",\"type\":");
        sb_json_str(&s, bi ? "builtin" : "custom", bi ? 7 : 6);
        sb_add(&s, "}", 1);
    }
    sb_str(&s, "],\"languages\":[\"auto\",\"en\",\"es\",\"pt\",\"ru\",\"it\",\"fr\",\"de\"]}");
    sb_str(&h, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, (long long)s.n);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && s.p && api_send(job->fd, h.p, h.n) == 0) api_send(job->fd, s.p, s.n);
    sb_free(&h);
    sb_free(&s);
}
static int switch_to_model(const char* name, char* err_out, size_t err_cap);   // main.c

static void api_model_select_job(ApiJob* job) {
    JNode* pool = malloc(sizeof(JNode) * 256);
    int root = pool ? json_parse(job->body, job->body_len, pool, 256) : -1;
    int mn = root >= 0 && pool[root].type == J_OBJ ? json_get(pool, root, "model") : -1;
    if (mn < 0) mn = root >= 0 && pool[root].type == J_OBJ ? json_get(pool, root, "id") : -1;
    if (mn < 0 || pool[mn].type != J_STR || pool[mn].slen == 0) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "Provide {\"model\": \"<folder-name>\"}.",
                  "invalid_request_error", "invalid_body");
        return;
    }
    Sb name;
    sb_init(&name);
    json_str_sb(pool, mn, &name);
    free(pool);
    char err[256] = {0};
    int r = name.p ? switch_to_model(name.p, err, sizeof(err)) : -1;
    sb_free(&name);
    if (r != 0) {
        api_error(job->fd, r == -2 ? "HTTP/1.1 404 Not Found" : "HTTP/1.1 500 Internal Server Error",
                  err[0] ? err : "Failed to switch model.", r == -2 ? "invalid_request_error" : "server_error", NULL);
        return;
    }
    Sb s, h;
    sb_init(&s);
    sb_init(&h);
    api_json_model(&s, api_model(), 0, (int)eng.max_seq);
    sb_str(&h, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, (long long)s.n);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && s.p && api_send(job->fd, h.p, h.n) == 0) api_send(job->fd, s.p, s.n);
    sb_free(&h);
    sb_free(&s);
}
#endif

static void api_job(ApiJob* job) {
#ifdef GARLIC_API_AUDIO
    if (job->kind == 3) {
        api_model_select_job(job);
        return;
    }
    if (job->kind == 2) {
        api_voices_job(job);
        return;
    }
    if (job->kind == 1) {
        api_speech_job(job);
        return;
    }
#endif
    static unsigned api_count;
    if (api_client_gone(job->fd)) {   // (it waited in line, and its client has given up meanwhile)
        say("api: the client of a waiting request left; nothing to answer");
        return;
    }
    ChatRequest cr;
    const char* err = NULL;
    JNode* pool = api_chat_parse_pool(job->body, job->body_len, &cr, &err);
    if (!pool) {
        say("api: request refused: %s", err ? err : "out of memory");
        if (err) api_error(job->fd, "HTTP/1.1 400 Bad Request", err, "invalid_request_error", "invalid_body");
        else api_error(job->fd, "HTTP/1.1 500 Internal Server Error", "Out of memory.", "server_error", NULL);
        return;
    }
    int think_switch, tool_turn;
    int n = api_prompt(pool, &cr, &think_switch, &tool_turn);
    if (n == -1 || n == -3) {
        free(pool);
        const char* why = n == -1 ? "A message is malformed: it needs a 'role', and 'content' must be a string or an "
                                    "array of parts."
                                  : "An assistant message's 'tool_calls' are malformed: each needs a function with a "
                                    "'name'.";
        say("api: request refused: %s", why);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", why, "invalid_request_error", "invalid_body");
        return;
    }
    if (n < 0) {
        api_too_long(job->fd, pool, &cr, -1, cr.max_tokens);
        free(pool);
        return;
    }
    int has_tools = cr.n_tools > 0 && cr.tool_choice != TOOL_NONE;
    int forced_call = has_tools && (cr.tool_choice == TOOL_REQUIRED || cr.tool_choice == TOOL_NAMED);
    int auto_max = !cr.max_tokens;   // no max_tokens: the answer may use the rest of the context window
    // Reasoning, when asked for and when the context has room for it besides the answer and the closing sentence
    // (not when tool_choice makes the reply begin with a call).
    int want = think_switch >= 0 ? think_switch : cr.think >= 0 ? cr.think : think_on;
    int think_budget = 0;
    if (want && eng.qwen35 && !forced_call) {
        think_budget = (int)eng.ctx_limit - (n + API_TAIL) - (auto_max ? API_ANSWER_MIN : cr.max_tokens) - 64;
        if (think_budget > think_cap) think_budget = think_cap;
        if (think_budget < API_THINK_MIN) think_budget = 0;
    }
    int think = think_budget > 0, head;
    Sb text;   // the reply (all of it), beginning with the call tool_choice asked for
    sb_init(&text);
    n = api_turn(n, think, &cr, &text, &head);
    if (auto_max) cr.max_tokens = (int)eng.ctx_limit - n - 2 - (think ? think_budget + 64 : 0);
    if (cr.max_tokens < 1 || n + cr.max_tokens + 2 > (int)eng.ctx_limit) {
        sb_free(&text);
        api_too_long(job->fd, pool, &cr, n, auto_max ? 0 : cr.max_tokens);
        free(pool);
        return;
    }
    free(pool);   // (the prompt is written: the request's pool is not needed while the model answers)
    pool = NULL;
    // the model answers this request: the conversation on screen starts again from nothing (a request refused
    // above leaves it as it was)
    pos = 0;
    need_close = 0;
    hist_n = hist_done = 0;
    thinking = think_turn = 0;
    sys_pre = 0;
    view.scroll = 0;
    api_chat_begin();   // (clears the conversation: a note says that a request is being answered)
    if (cr.has_seed) seed = cr.seed;
    char id[48];   // (the time in it: ids of the tool calls stay unique when the app starts again)
    snprintf(id, sizeof(id), "chatcmpl-%llx%u", (unsigned long long)time(NULL), ++api_count);
    const char* model = api_model();
    long long created = (long long)time(NULL);
    int fd = job->fd, cached = 0;
    double tp = now();
    int read = api_read_prompt(fd, n, head, tool_turn, think ? THINK_T : (float)cr.temperature, &cached);
    say("api: prompt of %d tokens (%d kept from the last request) in %.2f s, up to %d new, T %.2f%s%s%s", n, cached,
        now() - tp, cr.max_tokens, cr.temperature, think ? ", reasoning" : "", has_tools ? ", tools" : "",
        cr.stream ? ", streamed" : "");
    if (read < 0) {
        api_snap_n = 0;
        sb_free(&text);
        api_error(fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
        pos = 0;
        api_chat_end();
        return;
    }
    if (read > 0 && api_client_gone(fd)) {   // (no one to answer)
        sb_free(&text);
        pos = 0;
        api_chat_end();
        return;
    }
    pos = n;

    int streamed = 0, client_gone = 0, error = 0;
    if (cr.stream) {
        const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        if (api_send(fd, h, strlen(h))) client_gone = 1;
        streamed = 1;
        Sb first;
        sb_init(&first);
        api_json_chunk(&first, id, model, created, "assistant", "", 0, "", 0, NULL);
        if (!client_gone && first.p && api_send_event(fd, first.p)) client_gone = 1;
        sb_free(&first);
    }

    // What is still to be sent of the answer, and the reasoning before it (all of it, and what is still to be sent):
    // bytes of the last tokens that are not yet a whole character, or (pend) that may begin a <tool_call>.
    Sb pend, rtext, rpend;
    sb_init(&pend);
    sb_init(&rtext);
    sb_init(&rpend);
    int in_call = text.n > 0;   // the reply is inside a tool call: its text is not streamed
    int produced = 0, think_n = 0, stopped = read > 0, cut = 0, in_think = think, next_look = API_LOOK;
    const char* finish = "length";
    // The tokens to write: the one a step drew, after those it kept from the MTP head's draft (the model has read
    // these already). Without the head a step is one token.
    uint32_t q[9];
    int nq = 1, iq = 0;
    q[0] = next_token;
    double t0 = now();
    api_chat_reply();
    while (!stopped && produced < cr.max_tokens) {
        uint32_t t = q[iq++];
        const uint8_t* bytes = NULL;
        if (in_think) {
            int natural = t == THINK_END;
            if (natural || t == IM_END || t == eng.eos2 || t == eng.eos || think_n >= think_budget) {
                in_think = 0;   // the answer begins (the bytes of a cut character are left out of the reasoning)
                rpend.n = 0;
                // without max_tokens the answer has the rest of the context: what the reasoning did not use too
                // (it begins after </think> or the closing sentence; 8 to spare for the blank line after them)
                if (auto_max) cr.max_tokens = (int)eng.ctx_limit - pos - (natural ? 1 : 64) - 8;
                if (!natural) {
                    // Qwen's closing sentence, </think> and the blank line; the answer is drawn after them
                    uint32_t ids[64];
                    int k = encode_into(ids, 0, 56, THINK_CLOSE);
                    ids[k++] = THINK_END;
                    k = encode_into(ids, k, 64, "\n\n");
                    cut = 1;
                    if (engine_run(&eng, ids, k, pos, (float)cr.temperature, seed++, &next_token)) {
                        error = 1;
                        break;
                    }
                    pos += k;
                    q[0] = next_token;
                    nq = 1, iq = 0;
                    continue;
                }
                // (</think> as the model wrote it goes in below like any token, and the answer is drawn after it)
            } else {
                int nb = tok_piece(&tok, t, &bytes);
                think_n++;
                if (nb > 0) {
                    sb_add(&rtext, (const char*)bytes, nb);
                    sb_add(&rpend, (const char*)bytes, nb);
                }
                if (streamed && api_stream_piece(fd, id, model, created, &rpend, rpend.n, 1)) client_gone = 1;
            }
        } else {
            if (t == IM_END || t == eng.eos2 || t == eng.eos) {
                finish = "stop";
                break;
            }
            int nb;
            if (has_tools && (t == TOOL_CALL || t == TOOL_CALL_END)) {   // (special tokens: their text, to be read)
                bytes = (const uint8_t*)(t == TOOL_CALL ? "<tool_call>" : "</tool_call>");
                nb = t == TOOL_CALL ? 11 : 12;
            } else {
                nb = tok_piece(&tok, t, &bytes);
            }
            // (after reasoning the answer begins with a blank line: not part of it, and not counted)
            int lead = think && !text.n && nb > 0 && api_is_blank(bytes, nb);
            if (!lead) produced++;
            if (nb > 0 && !lead) {
                sb_add(&text, (const char*)bytes, nb);
                if (!in_call) sb_add(&pend, (const char*)bytes, nb);
            }
            if (has_tools && !in_call && pend.n) {
                const char* call = strstr(pend.p, "<tool_call>");
                if (call) {   // the text before the call goes out (without the blank line); the call is read at the end
                    size_t before_call = api_trim_end(pend.p, (size_t)(call - pend.p));
                    if (streamed && api_stream_piece(fd, id, model, created, &pend, before_call, 0)) client_gone = 1;
                    pend.n = 0;
                    in_call = 1;
                }
            }
            if (streamed && !in_call && pend.n) {
                // with tools, spaces and line breaks at the end wait for what comes next: before a call they are left
                // out, as from the content of a whole reply
                size_t safe = has_tools ? api_trim_end(pend.p, api_tool_prefix(pend.p, pend.n)) : pend.n;
                if (api_stream_piece(fd, id, model, created, &pend, safe, 0)) client_gone = 1;
            }
            if (produced == cr.max_tokens) break;
        }
        if (client_gone) break;
        // a reply that is not streamed (or a tool call, whose text is not) sends nothing until it ends: a client that
        // has closed its connection meanwhile is seen by a look at it now and then
        if ((!streamed || in_call) && produced + think_n >= next_look) {
            next_look = produced + think_n + API_LOOK;
            if (api_client_gone(fd)) {
                client_gone = 1;
                break;
            }
        }
        if (iq < nq) continue;   // (the next token is one the last step kept)
        if (api_chat_step(produced + think_n)) {   // Circle: the reply ends here, cut short as by max_tokens
            stopped = 1;
            break;
        }
        float temp = in_think ? THINK_T : (float)cr.temperature;
        int ne = 0;   // tokens of the draft this step keeps
#ifdef GARLIC_SPEC
        if (spec_active(&eng)) {
            int left = in_think ? think_budget - think_n : cr.max_tokens - produced;   // (a step never keeps more)
            if (engine_spec_step(&eng, t, pos, temp, seed++, left > 0 ? left + 1 : 1, q, &ne, &next_token)) {
                error = 1;
                break;
            }
        } else
#endif
        if (engine_run(&eng, &t, 1, pos, temp, seed++, &next_token)) {
            error = 1;
            break;
        }
        pos += 1 + ne;
        q[ne] = next_token;
        nq = ne + 1, iq = 0;
    }
    if (error) api_snap_n = 0;   // (the state may be half written)

    // The calls the reply ended with (only when the model ended it: a call cut short is not one), and the text
    // before them.
    ApiToolCall calls[API_MAX_TOOL_CALLS];
    Sb before;
    sb_init(&before);
    int n_calls = 0;
    if (has_tools && !strcmp(finish, "stop") && text.p && strstr(text.p, "<tool_call>")) {
        // the tools' schemas say which arguments are not strings (without them, out of memory: all are text)
        ChatRequest again;
        pool = api_chat_parse_pool(job->body, job->body_len, &again, &err);
        n_calls = api_parse_tool_calls(pool, pool ? &again : NULL, text.p, text.n, &before, calls,
                                       cr.parallel_tool_calls ? API_MAX_TOOL_CALLS : 1);
        free(pool);
        for (int i = 0; i < n_calls; i++) snprintf(calls[i].id, sizeof(calls[i].id), "call_%s_%d", id + 9, i + 1);
    }
    if (n_calls > 0) finish = "tool_calls";
    Sb audio_bin;
    sb_init(&audio_bin);
    char audio_id[32];
    snprintf(audio_id, sizeof(audio_id), "audio-%u", api_count);
#ifdef GARLIC_API_AUDIO
    if (cr.want_audio && !error && !stopped && !client_gone && n_calls == 0 && text.n > 0) {
        api_synthesize_audio(&cr, text.p, text.n, produced, &audio_bin);
    }
#endif
    double dt = now() - t0;
    if (think)
        say("api: reasoning %d tokens%s", think_n, cut ? " (cut: closing sentence fed)" : in_think ? " (unfinished)" : "");
    say("api: %d tokens in %.1f s (%.1f tokens/s), %d tool calls, %s", produced + think_n, dt,
        dt > 0 ? (produced + think_n) / dt : 0.0, n_calls,
        error ? "engine error" : client_gone ? "the client left" : stopped ? "stopped" : finish);

    if (streamed) {
        if (!client_gone && !error) {
            // what was held back: text that looked like the start of a call (or a call that could not be read)
            if (!n_calls && in_call && text.p) {
                const char* call = strstr(text.p, "<tool_call>");
                size_t from = call ? (size_t)(call - text.p) : text.n;
                sb_add(&pend, text.p + from, text.n - from);
            }
            if (!n_calls && api_stream_piece(fd, id, model, created, &pend, pend.n, 0)) client_gone = 1;
            for (int i = 0; i < n_calls && !client_gone; i++) {
                Sb ev;
                sb_init(&ev);
                api_json_chunk_tool(&ev, id, model, created, i, &calls[i]);
                if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                sb_free(&ev);
            }
            if (audio_bin.p && audio_bin.n > 0 && !client_gone) {
                Sb ev;
                sb_init(&ev);
                api_json_chunk_audio(&ev, id, model, created, audio_id, (const uint8_t*)audio_bin.p, audio_bin.n,
                                     text.p ? text.p : "", text.n);
                if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                sb_free(&ev);
            }
            Sb last;
            sb_init(&last);
            api_json_chunk(&last, id, model, created, NULL, "", 0, "", 0, finish);
            if (!client_gone && last.p && api_send_event(fd, last.p)) client_gone = 1;
            sb_free(&last);
            if (!client_gone && cr.include_usage) {   // (stream_options.include_usage: the usage, before [DONE])
                Sb u;
                sb_init(&u);
                api_json_chunk_usage(&u, id, model, created, n, produced + think_n, think_n);
                if (u.p && api_send_event(fd, u.p)) client_gone = 1;
                sb_free(&u);
            }
            if (!client_gone) api_send(fd, "data: [DONE]\n\n", 14);
        }
        if (error && !client_gone) {   // the headers are out: the error goes in the stream
            Sb e;
            sb_init(&e);
            api_json_error(&e, eng.error, "server_error", NULL);
            if (e.p) api_send_event(fd, e.p);
            sb_free(&e);
        }
    } else if (client_gone) {
        // (no one to answer)
    } else if (error) {
        api_error(fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
    } else {
        Sb body, head_s;
        sb_init(&body);
        sb_init(&head_s);
        // a reply cut short (max_tokens, Circle) may end inside a character: those bytes are left out, as in a stream,
        // so the JSON stays valid UTF-8; the reasoning's last line break goes too
        const Sb* answer = n_calls ? &before : &text;
        size_t tn = answer->p ? api_utf8_whole((const uint8_t*)answer->p, answer->n) : 0;
        size_t rn = rtext.p ? api_utf8_whole((const uint8_t*)rtext.p, rtext.n) : 0;
        while (rn && (rtext.p[rn - 1] == '\n' || rtext.p[rn - 1] == ' ')) rn--;
        api_json_completion_audio(&body, id, model, created, rtext.p ? rtext.p : "", rn, answer->p ? answer->p : "", tn,
                                  calls, n_calls, audio_id, (const uint8_t*)audio_bin.p, audio_bin.n, finish, n,
                                  produced + think_n, think_n);
        sb_str(&head_s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
        sb_int(&head_s, (long long)body.n);
        sb_str(&head_s, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if (head_s.p && body.p && api_send(fd, head_s.p, head_s.n) == 0) api_send(fd, body.p, body.n);
        sb_free(&body);
        sb_free(&head_s);
    }
    sb_free(&audio_bin);
    api_free_tool_calls(calls, n_calls);
    sb_free(&before);
    sb_free(&text);
    sb_free(&pend);
    sb_free(&rtext);
    sb_free(&rpend);
    // the engine's state is this request's; the conversation on screen starts again at position 0
    pos = 0;
    need_close = 0;
    api_chat_end();
}

#ifdef GARLIC_API_AUDIO
// Runs one durable queued task (task_queue.h) on the model and persists its result/error to /data/garlic/tasks.journal.
static void api_run_task(const GarlicTask* task) {
    if (!task) return;
    int n = 0;
    api_ids[n++] = IM_START;
    n = encode_into(api_ids, n, API_IDS - API_TAIL, "user\n");
    n = encode_into(api_ids, n, API_IDS - API_TAIL, task->prompt);
    api_ids[n++] = IM_END;
    n = encode_into(api_ids, n, API_IDS - API_TAIL, "\n");
    api_ids[n++] = IM_START;
    n = encode_into(api_ids, n, API_IDS, "assistant\n");
    if (eng.qwen35) {
        api_ids[n++] = THINK;
        n = encode_into(api_ids, n, API_IDS, "\n\n");
        api_ids[n++] = THINK_END;
        n = encode_into(api_ids, n, API_IDS, "\n\n");
    }
    if (n + 64 > (int)eng.ctx_limit) {
        api_task_finish(task->id, 0, "Task prompt exceeds context window.");
        return;
    }
    pos = 0;
    need_close = 0;
    hist_n = hist_done = 0;
    thinking = think_turn = 0;
    sys_pre = 0;
    view.scroll = 0;
    api_snap_n = 0;
    api_sys_n = 0;
    engine_reset(&eng);
    api_chat_begin();
    say("tasks: running task-%08u (%d prompt tokens)", task->id, n);
    if (engine_run(&eng, api_ids, n, 0, 0.7f, seed++, &next_token)) {
        api_task_finish(task->id, 0, eng.error[0] ? eng.error : "engine error");
        api_chat_end();
        return;
    }
    pos = n;
    Sb out;
    sb_init(&out);
    int max_tok = (int)eng.ctx_limit - n - 2;
    if (max_tok > 1024) max_tok = 1024;
    int err = 0;
    for (int produced = 0; produced < max_tok; produced++) {
        uint32_t t = next_token;
        if (t == eng.eos || t == eng.eos2) break;
        const uint8_t* s = NULL;
        int slen = tok_piece(&tok, t, &s);
        if (s && slen > 0 && out.n + (size_t)slen < TASK_RESULT_MAX - 1) sb_add(&out, (const char*)s, (size_t)slen);
        if (out.n >= TASK_RESULT_MAX - 16 || api_chat_step(produced)) break;
        if (engine_run(&eng, &t, 1, pos, 0.7f, seed++, &next_token)) {
            err = 1;
            break;
        }
        pos++;
    }
    if (err) {
        api_task_finish(task->id, 0, eng.error[0] ? eng.error : "engine error");
    } else {
        size_t tn = out.p ? api_utf8_whole((const uint8_t*)out.p, out.n) : 0;
        if (out.p) out.p[tn] = 0;
        api_task_finish(task->id, 1, out.p ? out.p : "");
        say("tasks: completed task-%08u (%zu bytes)", task->id, tn);
    }
    sb_free(&out);
    pos = 0;
    need_close = 0;
    api_chat_end();
}
#endif
