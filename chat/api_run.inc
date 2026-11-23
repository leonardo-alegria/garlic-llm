#include <time.h>

// OpenAI-compatible API jobs (api.c queues them; included in main.c, so it sees the engine and the conversation).
// An Ollama job arrives already converted to an OpenAI body; job->wire only changes the reply (NDJSON, Ollama JSON).
// A job runs on the model between the chat's turns: the request's messages are the whole conversation, so the
// conversation on screen is cleared, and the reply is written to the connection (whole, or as server-sent events).
// The reply is generated as the chat generates its own: several tokens per step with the MTP head where the app has
// it (GARLIC_SPEC), and Circle on the controller stops it. What the screen shows meanwhile is main.c's part
// (api_chat_begin / _reply / _step / _end).

#define API_IDS 16384             // tokens a job's prompt may have: the longest context of the app
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

static void api_role(char* role) {
    if (strcmp(role, "system") && strcmp(role, "assistant") && strcmp(role, "user") && strcmp(role, "tool"))
        strcpy(role, "user");
}

static char* api_trim_sb(Sb* s) {
    if (!s->p) return NULL;
    size_t a = 0, b = s->n;
    while (a < b && (s->p[a] == ' ' || s->p[a] == '\t' || s->p[a] == '\n' || s->p[a] == '\r')) a++;
    while (b > a && (s->p[b - 1] == ' ' || s->p[b - 1] == '\t' || s->p[b - 1] == '\n' || s->p[b - 1] == '\r')) b--;
    s->p[b] = 0;
    return s->p + a;
}

static int api_tool_wrapped(const char* s, size_t n) {
    size_t a = 0, b = s ? n : 0;
    if (!s) s = "";
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) b--;
    if (b - a < 15 + 16) return 0;
    return !memcmp(s + a, "<tool_response>", 15) && !memcmp(s + b - 16, "</tool_response>", 16);
}

// Index of the last user message that is not itself a tool response, or -1 when there is none, or -2 if a message
// is malformed. Qwen3.5 puts the empty think block on assistant turns after that message.
static int api_last_query(const JNode* pool, int messages) {
    int last = -1, i = 0;
    for (int m = pool[messages].first; m >= 0; m = pool[m].next, i++) {
        Sb c;
        sb_init(&c);
        char role[16];
        if (api_message(pool, m, role, sizeof(role), &c)) {
            sb_free(&c);
            return -2;
        }
        if (!strcmp(role, "user") && !api_tool_wrapped(c.p, c.n)) last = i;
        sb_free(&c);
    }
    return last;
}

static int api_close_turn(int n, int cap) {
    if (n < 0 || n + 2 > cap) return -2;
    api_ids[n++] = IM_END;
    return encode_into(api_ids, n, cap, "\n");
}

// The tools path follows Qwen3.5's chat template: one system turn, tool results grouped into a user turn, and the
// empty think block on assistant turns after the last real user message.
static int api_prompt_tools(const JNode* pool, const ChatRequest* cr, int cap) {
    int last = api_last_query(pool, cr->messages);
    if (last < 0) return -1;
    Sb sys;
    sb_init(&sys);
    int ate = 0;
    int first = pool[cr->messages].first;
    if (api_tools_system(pool, cr->tools, first, &sys, &ate) || sys.oom) {
        sb_free(&sys);
        return -1;
    }
    int n = 0;
    if (n >= cap) {
        sb_free(&sys);
        return -2;
    }
    api_ids[n++] = IM_START;
    n = encode_into(api_ids, n, cap, "system\n");
    if (sys.p && sys.n) n = encode_into(api_ids, n, cap, sys.p);
    sb_free(&sys);
    n = api_close_turn(n, cap);
    if (n < 0) return n;

    int m = first;
    int mi = 0;
    if (ate && m >= 0) {
        m = pool[m].next;
        mi = 1;
    }
    while (m >= 0) {
        Sb content;
        sb_init(&content);
        char role[16] = "user";
        if (api_message(pool, m, role, sizeof(role), &content) || content.oom) {
            sb_free(&content);
            return -1;
        }
        api_role(role);
        if (!strcmp(role, "system")) {
            sb_free(&content);
            return -1;
        }
        if (!strcmp(role, "tool")) {
            if (n >= cap) {
                sb_free(&content);
                return -2;
            }
            api_ids[n++] = IM_START;
            n = encode_into(api_ids, n, cap, "user");
            while (m >= 0 && !strcmp(role, "tool")) {
                char* body = api_trim_sb(&content);
                n = encode_into(api_ids, n, cap, "\n<tool_response>\n");
                if (body && body[0]) n = encode_into(api_ids, n, cap, body);
                n = encode_into(api_ids, n, cap, "\n</tool_response>");
                sb_free(&content);
                m = pool[m].next;
                mi++;
                if (m < 0) break;
                sb_init(&content);
                strcpy(role, "user");
                if (api_message(pool, m, role, sizeof(role), &content) || content.oom) {
                    sb_free(&content);
                    return -1;
                }
                api_role(role);
                if (!strcmp(role, "system")) {
                    sb_free(&content);
                    return -1;
                }
            }
            n = api_close_turn(n, cap);
            if (n < 0) {
                if (m >= 0) sb_free(&content);
                return n;
            }
            if (m < 0) break;
        }
        if (!strcmp(role, "assistant")) {
            if (n >= cap) {
                sb_free(&content);
                return -2;
            }
            api_ids[n++] = IM_START;
            n = encode_into(api_ids, n, cap, "assistant\n");
            char* body = api_trim_sb(&content);
            int nonempty = body && body[0];
            if (mi > last && eng.qwen35 && !(body && strstr(body, "</think>"))) {
                if (n >= cap) {
                    sb_free(&content);
                    return -2;
                }
                api_ids[n++] = THINK;
                n = encode_into(api_ids, n, cap, "\n\n");
                if (n >= cap) {
                    sb_free(&content);
                    return -2;
                }
                api_ids[n++] = THINK_END;
                n = encode_into(api_ids, n, cap, "\n\n");
            }
            if (nonempty) n = encode_into(api_ids, n, cap, body);
            Sb xml;
            sb_init(&xml);
            if (api_message_tool_xml(pool, m, nonempty, &xml) || xml.oom) {
                sb_free(&xml);
                sb_free(&content);
                return -1;
            }
            if (xml.p && xml.n) n = encode_into(api_ids, n, cap, xml.p);
            sb_free(&xml);
            n = api_close_turn(n, cap);
            sb_free(&content);
            if (n < 0) return n;
        } else if (strcmp(role, "tool")) {
            if (n >= cap) {
                sb_free(&content);
                return -2;
            }
            api_ids[n++] = IM_START;
            n = encode_into(api_ids, n, cap, role);
            n = encode_into(api_ids, n, cap, "\n");
            char* body = api_trim_sb(&content);
            if (body && body[0]) n = encode_into(api_ids, n, cap, body);
            n = api_close_turn(n, cap);
            sb_free(&content);
            if (n < 0) return n;
        } else {
            sb_free(&content);
        }
        m = pool[m].next;
        mi++;
    }
    return n;
}

// Builds the prompt: each message as <|im_start|>role\ncontent<|im_end|>\n, then the assistant's turn with the empty
// thought block (as the chat does, enable_thinking=false). With tools, the first turn is Qwen3.5's tool system
// message and assistant tool_calls are the template's XML. Returns the number of ids, -1 if a message is malformed,
// or -2 if the messages do not fit API_IDS. The limit counts tokens, not bytes: a long conversation may use the
// whole context. (Every token stands for one byte or more, so the tail after the messages always fits API_TAIL.)
static int api_prompt(const JNode* pool, const ChatRequest* cr) {
    const int cap = API_IDS - API_TAIL;   // where the messages must end
    int n = 0;
    if (cr->n_tools) {
        n = api_prompt_tools(pool, cr, cap);
        if (n < 0) return n;
    } else {
    for (int m = pool[cr->messages].first; m >= 0; m = pool[m].next) {
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

static void api_stamp(char* out, size_t n, long long created) {
    time_t t = (time_t)created;
    struct tm* g = gmtime(&t);
    if (!g || !strftime(out, n, "%Y-%m-%dT%H:%M:%SZ", g)) snprintf(out, n, "1970-01-01T00:00:00Z");
}

static void ollama_piece(Sb* ev, int wire, const char* model, const char* created_at, const char* text, size_t tlen,
                         int done, const char* reason, int prompt_n, int eval_n, const ApiToolCall* calls,
                         int ncalls) {
    if (wire == 2)
        api_json_ollama_gen(ev, model, created_at, text, tlen, done, reason, prompt_n, eval_n);
    else
        api_json_ollama_chat(ev, model, created_at, text, tlen, done, reason, prompt_n, eval_n, calls, ncalls);
}

static int api_send_line(int fd, const char* json) {
    Sb s;
    sb_init(&s);
    sb_str(&s, json);
    sb_str(&s, "\n");
    int r = s.p && !s.oom ? api_send(fd, s.p, s.n) : -1;
    sb_free(&s);
    return r;
}

static void api_fail(ApiJob* job, const char* status_line, const char* message, const char* type, const char* code) {
    if (!job->wire) {
        api_error(job->fd, status_line, message, type, code);
        return;
    }
    Sb s, h;
    sb_init(&s);
    api_json_ollama_error(&s, message);
    sb_init(&h);
    sb_str(&h, status_line);
    sb_str(&h, "\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, s.p ? (long long)s.n : 0);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && s.p && api_send(job->fd, h.p, h.n) == 0) api_send(job->fd, s.p, s.n);
    sb_free(&h);
    sb_free(&s);
    (void)type;
    (void)code;
}

static void api_job(ApiJob* job) {
    static unsigned api_count;
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    ChatRequest cr;
    const char* err = NULL;
    if (!pool || api_chat_parse(job->body, job->body_len, pool, API_MAX_NODES, &cr, &err)) {
        free(pool);
        api_fail(job, "HTTP/1.1 400 Bad Request", err ? err : "Out of memory.", "invalid_request_error",
                  "invalid_body");
        return;
    }
    int n = api_prompt(pool, &cr);
    if (n == -1) {
        free(pool);
        api_fail(job, "HTTP/1.1 400 Bad Request", "A message is malformed: it needs a 'role', and 'content' must "
                  "be a string or an array of parts.", "invalid_request_error", "invalid_body");
        return;
    }
    if (n < 0) {
        free(pool);
        api_fail(job, "HTTP/1.1 400 Bad Request", "The messages do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    free(pool);
    if (n + cr.max_tokens + 2 > (int)eng.ctx_limit) {
        api_fail(job, "HTTP/1.1 400 Bad Request", "The messages and max_tokens do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
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
    char id[32];
    snprintf(id, sizeof(id), "chatcmpl-%u", ++api_count);
    const char* model = api_model();
    long long created = (long long)time(NULL);
    char created_at[32];
    api_stamp(created_at, sizeof(created_at), created);
    say("api: prompt of %d tokens, up to %d new, T %.2f%s", n, cr.max_tokens, cr.temperature,
        cr.stream ? ", streamed" : "");
    engine_reset(&eng);
    if (engine_run(&eng, api_ids, n, 0, (float)cr.temperature, seed++, &next_token)) {
        api_fail(job, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
        pos = 0;
        api_chat_end();
        return;
    }
    pos = n;

    int fd = job->fd, streamed = 0, client_gone = 0, error = 0;
    if (cr.stream) {
        const char* h = job->wire
            ? "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-cache\r\n"
              "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
            : "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
              "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        if (api_send(fd, h, strlen(h))) client_gone = 1;
        streamed = 1;
        if (!job->wire) {   // OpenAI sends an empty role chunk before any text; Ollama does not
            Sb first;
            sb_init(&first);
            api_json_chunk(&first, id, model, created, "assistant", "", 0, NULL);
            if (!client_gone && first.p && api_send_event(fd, first.p)) client_gone = 1;
            sb_free(&first);
        }
    }

    Sb text, pend;   // the reply (all of it), and the bytes of the last tokens not yet a whole character
    sb_init(&text);
    sb_init(&pend);
    size_t sent_visible = 0;   // bytes of text already streamed (tools hold back a split <tool_call> marker)
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
        if (streamed && cr.n_tools && text.p) {
            size_t safe = api_tool_stream_end(text.p, text.n);
            size_t whole = api_utf8_whole((const uint8_t*)text.p, safe);
            if (whole > sent_visible && !client_gone) {
                Sb ev;
                sb_init(&ev);
                if (job->wire)
                    ollama_piece(&ev, job->wire, model, created_at, text.p + sent_visible, whole - sent_visible, 0,
                                 NULL, 0, 0, NULL, 0);
                else
                    api_json_chunk(&ev, id, model, created, NULL, text.p + sent_visible, whole - sent_visible, NULL);
                if (ev.p && (job->wire ? api_send_line(fd, ev.p) : api_send_event(fd, ev.p))) client_gone = 1;
                sb_free(&ev);
                sent_visible = whole;
            }
        } else if (streamed && pend.n) {
            size_t whole = api_utf8_whole((const uint8_t*)pend.p, pend.n);
            if (whole) {
                Sb ev;
                sb_init(&ev);
                if (job->wire)
                    ollama_piece(&ev, job->wire, model, created_at, pend.p, whole, 0, NULL, 0, 0, NULL, 0);
                else
                    api_json_chunk(&ev, id, model, created, NULL, pend.p, whole, NULL);
                if (ev.p && (job->wire ? api_send_line(fd, ev.p) : api_send_event(fd, ev.p))) client_gone = 1;
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

    size_t tn = text.p ? api_utf8_whole((const uint8_t*)text.p, text.n) : 0;
    ApiToolCall* calls = NULL;
    int ncalls = 0, cut = 0;
    Sb visible;
    sb_init(&visible);
    if (!error && cr.n_tools && text.p && tn) {
        calls = calloc(API_MAX_CALLS, sizeof(ApiToolCall));
        if (calls)
            api_parse_tool_calls(text.p, tn, cr.schemas, cr.n_schemas, &visible, calls, API_MAX_CALLS, &ncalls, &cut);
    }
    const char* show = text.p ? text.p : "";
    size_t show_n = tn;
    const char* finish_out = finish;
    if (ncalls > 0) {
        show = visible.p ? visible.p : "";
        show_n = visible.n;
        finish_out = job->wire ? "stop" : "tool_calls";
    } else if (cut) {
        show = text.p ? text.p : "";
        show_n = tn;
    }

    if (streamed) {
        if (!client_gone && !error) {
            if (cr.n_tools && text.p) {
                size_t end = ncalls > 0 ? api_tool_stream_end(text.p, tn) : tn;
                size_t whole = api_utf8_whole((const uint8_t*)text.p, end);
                if (whole > sent_visible) {
                    Sb ev;
                    sb_init(&ev);
                    if (job->wire)
                        ollama_piece(&ev, job->wire, model, created_at, text.p + sent_visible, whole - sent_visible, 0,
                                     NULL, 0, 0, NULL, 0);
                    else
                        api_json_chunk(&ev, id, model, created, NULL, text.p + sent_visible, whole - sent_visible,
                                       NULL);
                    if (ev.p && (job->wire ? api_send_line(fd, ev.p) : api_send_event(fd, ev.p))) client_gone = 1;
                    sb_free(&ev);
                }
            }
            if (!client_gone && ncalls > 0 && !job->wire) {
                for (int i = 0; i < ncalls && !client_gone; i++) {
                    char cid[64];
                    snprintf(cid, sizeof(cid), "call_%u_%d", api_count, i);
                    Sb ev;
                    sb_init(&ev);
                    api_json_tool_chunk(&ev, id, model, created, i, cid, calls[i].name, calls[i].arguments);
                    if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                    sb_free(&ev);
                }
            }
            if (!client_gone) {
                Sb last;
                sb_init(&last);
                if (job->wire)
                    ollama_piece(&last, job->wire, model, created_at, "", 0, 1, finish_out, n, produced, calls, ncalls);
                else
                    api_json_chunk(&last, id, model, created, NULL, "", 0, finish_out);
                if (last.p) {
                    if (job->wire) api_send_line(fd, last.p);
                    else api_send_event(fd, last.p);
                }
                sb_free(&last);
                if (!job->wire) api_send(fd, "data: [DONE]\n\n", 14);
            }
        }
        if (error && !client_gone) {   // the headers are out: the error goes in the stream
            Sb e;
            sb_init(&e);
            if (job->wire) api_json_ollama_error(&e, eng.error);
            else api_json_error(&e, eng.error, "server_error", NULL);
            if (e.p) {
                if (job->wire) api_send_line(fd, e.p);
                else api_send_event(fd, e.p);
            }
            sb_free(&e);
        }
    } else if (error) {
        api_fail(job, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
    } else {
        Sb body, head;
        sb_init(&body);
        sb_init(&head);
        // a reply cut short (max_tokens, Circle) may end inside a character: those bytes are left out, as in a stream,
        // so the JSON stays valid UTF-8
        if (job->wire)
            ollama_piece(&body, job->wire, model, created_at, show, show_n, 1, finish_out, n, produced, calls, ncalls);
        else
            api_json_completion(&body, id, model, created, show, show_n, finish_out, n, produced, calls, ncalls,
                                api_count);
        sb_str(&head, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
        sb_int(&head, (long long)body.n);
        sb_str(&head, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if (head.p && body.p && api_send(fd, head.p, head.n) == 0) api_send(fd, body.p, body.n);
        sb_free(&body);
        sb_free(&head);
    }
    free(calls);
    sb_free(&visible);
    sb_free(&text);
    sb_free(&pend);
    // the engine's state is this request's; the conversation on screen starts again at position 0
    pos = 0;
    need_close = 0;
    api_chat_end();
}
