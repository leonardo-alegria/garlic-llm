// OpenAI-compatible API jobs (api.c queues them; included in main.c, so it sees the engine and the conversation).
// A job runs on the model between the chat's turns: the request's messages are the whole conversation, so the
// conversation on screen is cleared, and the reply is written to the connection (whole, or as server-sent events).

static uint32_t api_ids[16384];   // the prompt of a job (byte-level tokens: at most one per byte of text)
#define API_MAX_TEXT 15000        // bytes of message text a request may carry (the prompt must fit api_ids)

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

// Builds the prompt: each message as <|im_start|>role\ncontent<|im_end|>\n, then the assistant's turn with the empty
// thought block (as the chat does, enable_thinking=false). Returns the number of ids, or -1 if it does not fit.
static int api_prompt(const JNode* pool, const ChatRequest* cr) {
    int n = 0;
    size_t total = 0;
    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next) {
        Sb content;
        sb_init(&content);
        char role[16] = "user";
        int ok = api_message(pool, m, role, sizeof(role), &content) == 0;
        if (ok) total += content.n;
        if (!ok || total > API_MAX_TEXT) {
            sb_free(&content);
            return -1;
        }
        if (strcmp(role, "system") && strcmp(role, "assistant") && strcmp(role, "user") && strcmp(role, "tool"))
            strcpy(role, "user");
        api_ids[n++] = IM_START;
        n = encode_into(api_ids, n, 16384, role);
        n = encode_into(api_ids, n, 16384, "\n");
        if (content.p) n = encode_into(api_ids, n, 16384, content.p);
        api_ids[n++] = IM_END;
        n = encode_into(api_ids, n, 16384, "\n");
        sb_free(&content);
    }
    api_ids[n++] = IM_START;
    n = encode_into(api_ids, n, 16384, "assistant\n");
    if (eng.qwen35) {
        api_ids[n++] = THINK;
        n = encode_into(api_ids, n, 16384, "\n\n");
        api_ids[n++] = THINK_END;
        n = encode_into(api_ids, n, 16384, "\n\n");
    }
    return n;
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

static void api_job(ApiJob* job) {
    static unsigned api_count;
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    ChatRequest cr;
    const char* err = NULL;
    if (!pool || api_chat_parse(job->body, job->body_len, pool, API_MAX_NODES, &cr, &err)) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", err ? err : "Out of memory.", "invalid_request_error",
                  "invalid_body");
        return;
    }
    // the model answers this request: the conversation on screen starts again from nothing
    msgs_clear();
    pos = 0;
    need_close = 0;
    hist_n = hist_done = 0;
    thinking = think_turn = 0;
    sys_pre = 0;
    view.scroll = 0;

    int n = api_prompt(pool, &cr);
    if (n < 0) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages are malformed or too long.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    free(pool);
    if (n + cr.max_tokens + 2 > (int)eng.ctx_limit) {
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages and max_tokens do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    if (cr.has_seed) seed = cr.seed;
    char id[32];
    snprintf(id, sizeof(id), "chatcmpl-%u", ++api_count);
    const char* model = api_model();
    long long created = (long long)time(NULL);
    say("api: prompt of %d tokens, up to %d new, T %.2f%s", n, cr.max_tokens, cr.temperature,
        cr.stream ? ", streamed" : "");
    engine_reset(&eng);
    if (engine_run(&eng, api_ids, n, 0, (float)cr.temperature, seed++, &next_token)) {
        api_error(job->fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
        pos = 0;
        return;
    }
    pos = n;

    int fd = job->fd, streamed = 0, client_gone = 0, error = 0;
    if (cr.stream) {
        const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        if (api_send(fd, h, strlen(h))) client_gone = 1;
        streamed = 1;
        Sb first;
        sb_init(&first);
        api_json_chunk(&first, id, model, created, "assistant", "", 0, NULL);
        if (!client_gone && first.p && api_send_event(fd, first.p)) client_gone = 1;
        sb_free(&first);
    }

    Sb text, pend;   // the reply (all of it), and the bytes of the last tokens not yet a whole character
    sb_init(&text);
    sb_init(&pend);
    int produced = 0;
    const char* finish = "length";
    uint32_t t = next_token;
    while (produced < cr.max_tokens) {
        if (t == IM_END || t == eng.eos2 || t == eng.eos) {
            finish = "stop";
            break;
        }
        const uint8_t* bytes = NULL;
        int nb = tok_piece(&tok, t, &bytes);
        produced++;
        if (nb > 0) {
            sb_add(&text, (const char*)bytes, nb);
            sb_add(&pend, (const char*)bytes, nb);
        }
        if (streamed && pend.n) {
            size_t whole = api_utf8_whole((const uint8_t*)pend.p, pend.n);
            if (whole) {
                Sb ev;
                sb_init(&ev);
                api_json_chunk(&ev, id, model, created, NULL, pend.p, whole, NULL);
                if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                sb_free(&ev);
                memmove(pend.p, pend.p + whole, pend.n - whole);
                pend.n -= whole;
                pend.p[pend.n] = 0;
            }
        }
        if (client_gone || produced == cr.max_tokens) break;
        if (engine_run(&eng, &t, 1, pos, (float)cr.temperature, seed++, &next_token)) {
            error = 1;
            break;
        }
        pos++;
        t = next_token;
    }
    say("api: %d tokens, %s", produced, error ? "engine error" : finish);

    if (streamed) {
        if (!client_gone && !error) {
            Sb last;
            sb_init(&last);
            api_json_chunk(&last, id, model, created, NULL, "", 0, finish);
            if (last.p) api_send_event(fd, last.p);
            sb_free(&last);
            api_send(fd, "data: [DONE]\n\n", 14);
        }
        if (error && !client_gone) {   // the headers are out: the error goes in the stream
            Sb e;
            sb_init(&e);
            api_json_error(&e, eng.error, "server_error", NULL);
            if (e.p) api_send_event(fd, e.p);
            sb_free(&e);
        }
    } else if (error) {
        api_error(fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
    } else {
        Sb body, head;
        sb_init(&body);
        sb_init(&head);
        api_json_completion(&body, id, model, created, text.p ? text.p : "", text.n, finish, n, produced);
        sb_str(&head, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
        sb_int(&head, (long long)body.n);
        sb_str(&head, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if (head.p && body.p && api_send(fd, head.p, head.n) == 0) api_send(fd, body.p, body.n);
        sb_free(&body);
        sb_free(&head);
    }
    sb_free(&text);
    sb_free(&pend);
    // the engine's state is this request's; the conversation on screen starts again at position 0
    pos = 0;
    need_close = 0;
}
