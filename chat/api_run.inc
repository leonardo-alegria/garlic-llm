// OpenAI-compatible API jobs (api.c queues them; included in main.c, so it sees the engine and the conversation).
// A job runs on the model between the chat's turns: the request's messages are the whole conversation, so the
// conversation on screen is cleared, and the reply is written to the connection (whole, or as server-sent events).
// The reply is generated as the chat generates its own: several tokens per step with the MTP head where the app has
// it (GARLIC_SPEC), and Circle on the controller stops it. What the screen shows meanwhile is main.c's part
// (api_chat_begin / _reply / _step / _end).

#define API_IDS 24576             // tokens a job's prompt may have: the longest context of the app
#define API_TAIL 32               // kept free for the assistant's turn after the messages (at most 17 tokens)
static uint32_t api_ids[API_IDS];   // the prompt of a job

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
// thought block (as the chat does, enable_thinking=false). Returns the number of ids, -1 if a message is malformed,
// or -2 if the messages do not fit API_IDS. The limit counts tokens, not bytes: a long conversation may use the
// whole context. (Every token stands for one byte or more, so the tail after the messages always fits API_TAIL.)
static int api_prompt(const JNode* pool, const ChatRequest* cr, int skip) {
    const int cap = API_IDS - API_TAIL;   // where the messages must end
    int n = 0;
    for (int m = pool[cr->messages].first, i = 0; m >= 0; m = pool[m].next, i++) {
        if (i < skip) continue;   // (already in the engine's state: only the new messages are fed)
        Sb content;
        sb_init(&content);
        char role[16] = "user";
        int ok = api_message(pool, m, role, sizeof(role), &content) == 0;
        if (!ok || content.oom) {
            sb_free(&content);
            return -1;
        }
        if (strcmp(role, "system") && strcmp(role, "assistant") && strcmp(role, "user") && strcmp(role, "tool"))
            strcpy(role, "user");
        if (n >= cap) {
            sb_free(&content);
            return -2;
        }
        api_ids[n++] = IM_START;
        n = encode_into(api_ids, n, cap, role);
        n = encode_into(api_ids, n, cap, "\n");
        if (content.p) n = encode_into(api_ids, n, cap, content.p);
        sb_free(&content);
        if (n + 2 > cap) return -2;   // the text filled what was left (it may have been cut): too long
        api_ids[n++] = IM_END;
        n = encode_into(api_ids, n, cap, "\n");
    }
    api_ids[n++] = IM_START;
    n = encode_into(api_ids, n, API_IDS, "assistant\n");
    if (eng.qwen35) {
        api_ids[n++] = THINK;
        n = encode_into(api_ids, n, API_IDS, "\n\n");
        api_ids[n++] = THINK_END;
        n = encode_into(api_ids, n, API_IDS, "\n\n");
    }
    return n;
}

// Upper bound on the tokens the messages from `skip` on will take (a token is at least one byte;
    // plus the wrapper per message and the assistant header).
static int api_prompt_max(const JNode* pool, const ChatRequest* cr, int skip) {
    int n = 0, i = 0;
    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next, i++) {
        if (i < skip) continue;
        Sb c;
        sb_init(&c);
        char role[16] = "user";
        if (api_message(pool, m, role, sizeof(role), &c) || c.oom) return API_IDS;   // (unknown: no continuation)
        n += (int)c.n + (int)strlen(role) + 8;
        sb_free(&c);
    }
    return n + 64;
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
    static int api_pos, api_msgs_fed;   // the conversation on the engine: where it ends, messages fed
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    ChatRequest cr;
    const char* err = NULL;
    if (!pool || api_chat_parse(job->body, job->body_len, pool, API_MAX_NODES, &cr, &err)) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", err ? err : "Out of memory.", "invalid_request_error",
                  "invalid_body");
        return;
    }
    int mcount = 0;
    for (int m = pool[cr.messages].first; m >= 0; m = pool[m].next) mcount++;
    // keep the conversation: the engine's state holds the earlier requests when nothing typed on the
    // screen took the room since. the client's whole history is then fed from where it stopped.
    int cont = !api_break && api_pos > 0 && api_pos == pos && api_msgs_fed > 0 && api_msgs_fed < mcount &&
               pos + api_prompt_max(pool, &cr, api_msgs_fed) + cr.max_tokens + 2 <= (int)eng.ctx_limit;
    int n = api_prompt(pool, &cr, cont ? api_msgs_fed : 0);
    if (n == -1) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "A message is malformed: it needs a 'role', and 'content' must "
                  "be a string or an array of parts.", "invalid_request_error", "invalid_body");
        return;
    }
    if (n < 0) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    free(pool);
    if (!cont && n + cr.max_tokens + 2 > (int)eng.ctx_limit) {
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages and max_tokens do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    if (cont && pos + n + cr.max_tokens + 2 > (int)eng.ctx_limit) {
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The conversation and max_tokens do not fit the context window anymore.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    // the model answers this request: the conversation on screen starts again from nothing (a request refused
    // above leaves it as it was)
    if (!cont) {
        pos = 0;
        need_close = 0;
    }
    hist_n = hist_done = 0;
    thinking = think_turn = 0;
    sys_pre = 0;
    view.scroll = 0;
    api_chat_begin();   // (clears the conversation: a note says that a request is being answered)
    if (cont && need_close) {   // the last answer's end token was sampled but never fed (as the chat does)
        static uint32_t close_ids[2];
        if (!close_ids[0]) {
            close_ids[0] = IM_END;
            uint32_t one[8];
            encode_into(one, 0, 8, "\n");
            close_ids[1] = one[0];
        }
        if (engine_run(&eng, close_ids, 2, pos, 0.0f, 1, &next_token)) {
            api_error(job->fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
            free(pool);
            return;
        }
        pos += 2;
    }
    if (cr.has_seed) seed = cr.seed;
    char id[32];
    snprintf(id, sizeof(id), "chatcmpl-%u", ++api_count);
    const char* model = api_model();
    long long created = (long long)time(NULL);
    say("api: prompt of %d tokens, up to %d new, T %.2f%s", n, cr.max_tokens, cr.temperature,
        cr.stream ? ", streamed" : "");
    if (!cont) engine_reset(&eng);   // (continuing: the Gated DeltaNet states stay)
    api_break = 0;   // a swap's forced fresh start is consumed by this processed request
    if (engine_run(&eng, api_ids, n, pos, (float)cr.temperature, seed++, &next_token)) {
        api_error(job->fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
        pos = cont ? pos : 0;
        api_chat_end();
        return;
    }
    pos += n;
    api_pos = pos, api_msgs_fed = mcount;   // this conversation is now on the engine up to pos
    need_close = 1;   // the answer's end token will be sampled but not fed: the next request closes it

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
    int produced = 0, stopped = 0;
    const char* finish = "length";
    // The tokens to write: the one a step drew, after those it kept from the MTP head's draft (the model has read
    // these already). Without the head a step is one token.
    uint32_t q[9];
    int nq = 1, iq = 0;
    q[0] = next_token;
    double t0 = now();
    api_chat_reply();
    while (produced < cr.max_tokens) {
        uint32_t t = q[iq++];
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
        if (iq < nq) continue;   // (the next token is one the last step kept)
        if (api_chat_step(produced)) {   // Circle: the reply ends here, cut short as by max_tokens
            stopped = 1;
            break;
        }
        int ne = 0;   // tokens of the draft this step keeps
#ifdef GARLIC_SPEC
        if (spec_active(&eng)) {   // (never more than max_tokens: the step keeps at most what is left of them)
            if (engine_spec_step(&eng, t, pos, (float)cr.temperature, seed++, cr.max_tokens - produced + 1, q, &ne,
                                 &next_token)) {
                error = 1;
                break;
            }
        } else
#endif
        if (engine_run(&eng, &t, 1, pos, (float)cr.temperature, seed++, &next_token)) {
            error = 1;
            break;
        }
        pos += 1 + ne;
        q[ne] = next_token;
        nq = ne + 1, iq = 0;
    }
    double dt = now() - t0;
    say("api: %d tokens in %.1f s (%.1f tokens/s), %s", produced, dt, dt > 0 ? produced / dt : 0.0,
        error ? "engine error" : stopped ? "stopped on the console" : finish);

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
        // a reply cut short (max_tokens, Circle) may end inside a character: those bytes are left out, as in a stream,
        // so the JSON stays valid UTF-8
        size_t tn = text.p ? api_utf8_whole((const uint8_t*)text.p, text.n) : 0;
        api_json_completion(&body, id, model, created, text.p ? text.p : "", tn, finish, n, produced);
        sb_str(&head, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
        sb_int(&head, (long long)body.n);
        sb_str(&head, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if (head.p && body.p && api_send(fd, head.p, head.n) == 0) api_send(fd, body.p, body.n);
        sb_free(&body);
        sb_free(&head);
    }
    sb_free(&text);
    sb_free(&pend);
    // keepctx3: the sampling loop left pos at prompt+answer; the engine conversation ends
    //  right there and stays that way. A request that starts over is reset by the !cont block
    //  up front (pos and the engine together); nothing here may touch pos or the end is dead.
    api_pos = pos, api_msgs_fed = mcount;   // refresh: conversation now ends past the sampled answer
    api_chat_end();
}
