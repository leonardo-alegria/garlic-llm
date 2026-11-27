"""Real HTTP clients against the PC test server (server.c): what chat apps and agent clients send, over a socket.

Standard library only. From the repository's root, after building the server (the build line is in server.c):

    python tests/api_host/test.py [build/api_server.exe]

It starts the server on 127.0.0.1:18080 (its log goes to build/api_server_test.log), runs every check, stops it,
and prints how many failed; the exit status is 1 if any did.
"""

import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

PORT = 18080
KEY = "test-key"
DELAY_MS = 10          # one step of the mock's reply (up to 3 tokens)
LOG = os.path.join("build", "api_server_test.log")

checks = fails = 0


def check(cond, what):
    global checks, fails
    checks += 1
    if not cond:
        fails += 1
        print("FAIL", what)


# ---- HTTP ----------------------------------------------------------------------------------------------------------

def call(method, path, body=None, key=KEY, headers=None, timeout=60):
    """One request on its own connection: (status, headers in lower case, body bytes)."""
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    h = {} if body is None else {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    h.update(headers or {})
    data = json.dumps(body) if isinstance(body, (dict, list)) else body
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, raw


def chat(body, **kw):
    status, headers, raw = call("POST", "/v1/chat/completions", body, **kw)
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, None


def stream(body, key=KEY):
    """A streamed request, read event by event: (status, headers, events, time of the first event, time of the end).
    An event is the JSON of a chunk, or "[DONE]"."""
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body=json.dumps(body),
              headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    r = c.getresponse()
    headers = {k.lower(): v for k, v in r.getheaders()}
    events, first = [], None
    if r.status != 200:
        r.read()
        c.close()
        return r.status, headers, events, None, time.time() - t0
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode("utf-8").rstrip("\r\n")
        if not line.startswith("data: "):
            continue
        if first is None:
            first = time.time() - t0
        data = line[6:]
        events.append("[DONE]" if data == "[DONE]" else json.loads(data))
    c.close()
    return r.status, headers, events, first, time.time() - t0


def assemble(events):
    """What a client makes of a stream: text, reasoning, calls (by index), finish_reason, usage, whether it ended."""
    out = {"content": "", "reasoning": "", "calls": {}, "finish": None, "usage": None, "done": False,
           "role": None, "order": []}
    for e in events:
        if e == "[DONE]":
            out["done"] = True
            continue
        if e.get("usage") is not None:
            out["usage"] = e["usage"]
            out["order"].append("usage")
        for ch in e.get("choices", []):
            d = ch.get("delta", {})
            if d.get("role"):
                out["role"] = d["role"]
            if d.get("reasoning_content"):
                out["reasoning"] += d["reasoning_content"]
                out["order"].append("reasoning")
            if d.get("content"):
                out["content"] += d["content"]
                out["order"].append("content")
            for tc in d.get("tool_calls") or []:
                acc = out["calls"].setdefault(tc["index"], {"id": "", "name": "", "arguments": ""})
                acc["id"] = tc.get("id") or acc["id"]
                fn = tc.get("function") or {}
                acc["name"] += fn.get("name") or ""
                acc["arguments"] += fn.get("arguments") or ""
                out["order"].append("call")
            if ch.get("finish_reason"):
                out["finish"] = ch["finish_reason"]
                out["order"].append("finish")
    return out


def health():
    return json.loads(call("GET", "/health", key=None)[2])


def wait_idle(limit):
    """Seconds until /health says nothing is being answered, or None after limit seconds."""
    t0 = time.time()
    while time.time() - t0 < limit:
        h = health()
        if not h["busy"] and h["waiting"] == 0:
            return time.time() - t0
        time.sleep(0.05)
    return None


def log_text():
    with open(LOG, encoding="utf-8", errors="replace") as f:
        return f.read()


def json_values(v):
    """The JSON values in v, counted as the server's parser counts them (every value, keys not)."""
    if isinstance(v, dict):
        return 1 + sum(json_values(x) for x in v.values())
    if isinstance(v, list):
        return 1 + sum(json_values(x) for x in v)
    return 1


def has_word(text, *words):
    return any(w in text.lower() for w in words)


USER_HI = [{"role": "user", "content": "hi"}]
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "The weather in a city now.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city", "days"]}}}
CLOCK = {"type": "function", "function": {
    "name": "get_time", "description": "The time in a zone.",
    "parameters": {"type": "object", "properties": {"zone": {"type": "string", "enum": ["UTC", "PET"]}},
                   "required": ["zone"]}}}


# ---- the checks ----------------------------------------------------------------------------------------------------

def test_endpoints():
    check(health() == {"status": "ok", "busy": False, "waiting": 0}, "/health when idle, without a key")
    for path in ("/v1/models", "/models"):
        status, headers, raw = call("GET", path)
        m = json.loads(raw)["data"][0]
        check(status == 200 and m["id"] == "Qwen3.5-9B" and m["object"] == "model", path)
        check(m.get("context_length") == 16384, path + " reports the context window")
        check(headers.get("access-control-allow-origin") == "*", path + " CORS header")
    status, _, raw = call("GET", "/v1/models/Qwen3.5-9B")
    check(status == 200 and json.loads(raw)["context_length"] == 16384, "/v1/models/<id>")
    status, _, raw = call("GET", "/v1/models/another-model")
    check(status == 404 and json.loads(raw)["error"]["code"] == "model_not_found", "/v1/models/<unknown id>")
    status, _, raw = call("GET", "/openapi.json", key=None)
    spec = json.loads(raw) if status == 200 else {}
    check("/v1/chat/completions" in spec.get("paths", {}), "/openapi.json is the API description")
    check("context_length" in spec.get("components", {}).get("schemas", {}).get("Model", {}).get("properties", {}),
          "the description has context_length")
    check(call("GET", "/nothing")[0] == 404, "an unknown path is 404")
    check(call("GET", "/v1/chat/completions")[0] == 404, "GET on the chat endpoint is 404")


def test_key():
    status, headers, raw = call("POST", "/v1/chat/completions", {"messages": USER_HI}, key=None)
    check(status == 401 and json.loads(raw)["error"]["code"] == "invalid_api_key", "no key: 401")
    check(headers.get("access-control-allow-origin") == "*", "a 401 carries the CORS header (a browser reads it)")
    check(call("POST", "/v1/chat/completions", {"messages": USER_HI}, key="wrong")[0] == 401, "a wrong key: 401")
    check(call("GET", "/v1/models", key="wrong")[0] == 401, "a wrong key on /v1/models: 401")
    check(call("GET", "/health", key="wrong")[0] == 200, "/health needs no key")
    check("request refused: wrong API key" in log_text(), "a wrong key is logged")


def test_cors():
    status, headers, raw = call("OPTIONS", "/v1/chat/completions", key=None, headers={
        "Origin": "https://chat.example", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type,x-client-os,x-client-retry-count"})
    allow = headers.get("access-control-allow-headers", "")
    check(status == 204 and raw == b"", "preflight: 204")
    check(headers.get("access-control-allow-origin") == "*", "preflight: any origin")
    check("Authorization" in allow and "*" in allow, "preflight: Authorization and any other header: " + allow)
    check("POST" in headers.get("access-control-allow-methods", ""), "preflight: POST")
    check(headers.get("access-control-allow-private-network") == "true", "preflight: a page on the internet may ask")
    status, headers, _ = call("POST", "/v1/chat/completions", {"messages": USER_HI})
    check(status == 200 and headers.get("access-control-allow-origin") == "*", "a reply carries the CORS header")


def test_plain():
    status, r = chat({"model": "any name", "messages": USER_HI, "temperature": 0.2})
    m = r["choices"][0]
    check(status == 200 and r["object"] == "chat.completion" and r["id"].startswith("chatcmpl-"), "plain chat")
    check(m["message"] == {"role": "assistant", "content": "Hello! You said: hi"}, "plain chat: the answer")
    check(m["finish_reason"] == "stop", "plain chat: stop")
    u = r["usage"]
    check(u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and u["completion_tokens"] > 0, "usage")
    status, r = chat({"messages": USER_HI, "max_tokens": 3})
    check(r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 3, "max_tokens cuts")
    status, r = chat({"messages": [{"role": "system", "content": "Be brief."},
                                   {"role": "user", "content": [{"type": "text", "text": "hi "},
                                                                {"type": "text", "text": "there"}]}]})
    check(r["choices"][0]["message"]["content"] == "Hello! You said: hi there", "content as an array of parts")


def test_stream():
    _, whole = chat({"messages": USER_HI})
    status, headers, events, first, total = stream({"messages": USER_HI, "stream": True,
                                                   "stream_options": {"include_usage": True}})
    a = assemble(events)
    check(status == 200 and headers.get("content-type") == "text/event-stream", "stream: event stream")
    check(a["role"] == "assistant" and events[0]["choices"][0]["delta"] == {"role": "assistant"}, "stream: role first")
    check(a["content"] == "Hello! You said: hi" and a["finish"] == "stop" and a["done"], "stream: text, stop, [DONE]")
    check(a["order"][-2:] == ["finish", "usage"] and events[-1] == "[DONE]", "stream: usage last, before [DONE]")
    last = events[-2]
    check(last.get("choices") == [] and a["usage"] == whole["usage"], "stream: the usage chunk has no choices")
    _, _, events, _, _ = stream({"messages": USER_HI, "stream": True})
    check(assemble(events)["usage"] is None, "stream: no usage unless asked for")
    check(len({e["id"] for e in events if e != "[DONE]"}) == 1, "stream: one id for every chunk")


def test_tools():
    msgs = [{"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "What is the weather in Lima? Please call the tool."}]
    status, r = chat({"messages": msgs, "tools": [WEATHER, CLOCK], "tool_choice": "auto"})
    m = r["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    check(status == 200 and r["choices"][0]["finish_reason"] == "tool_calls", "tools: finish_reason tool_calls")
    check(m["content"] == "Let me check." and len(calls) == 1, "tools: the text before the call, one call")
    c = calls[0] if calls else {"function": {"name": "", "arguments": "{}"}, "id": ""}
    check(c["function"]["name"] == "get_weather" and c["type"] == "function" and c["id"].startswith("call_"),
          "tools: the call")
    check(json.loads(c["function"]["arguments"]) == {"city": "Lima", "days": 3}, "tools: arguments, integer as number")
    # the results back, as an agent client sends them: the assistant turn without content, then role "tool"
    history = msgs + [{"role": "assistant", "tool_calls": [
        {"id": c["id"], "type": "function", "function": c["function"]}]},
        {"role": "tool", "tool_call_id": c["id"], "content": "18 C and sunny"}]
    status, r2 = chat({"messages": history, "tools": [WEATHER, CLOCK], "tool_choice": "auto"})
    check(r2["choices"][0]["message"]["content"] == "The tool said: 18 C and sunny." and
          r2["choices"][0]["finish_reason"] == "stop", "tools: the result read back")
    history[2]["content"] = None   # (the same turn as other clients send it: content null)
    status, r3 = chat({"messages": history, "tools": [WEATHER, CLOCK]})
    check(status == 200 and r3["choices"][0]["message"]["content"] == "The tool said: 18 C and sunny.",
          "tools: an assistant turn with content null")
    # streamed both ways, with usage
    status, _, events, _, _ = stream({"messages": msgs, "tools": [WEATHER, CLOCK], "stream": True,
                                      "stream_options": {"include_usage": True}})
    a = assemble(events)
    sc = a["calls"].get(0, {})
    check(a["content"] == "Let me check." and a["finish"] == "tool_calls" and a["done"], "tools streamed")
    check(sc.get("name") == "get_weather" and json.loads(sc.get("arguments") or "null") == {"city": "Lima", "days": 3},
          "tools streamed: the call, whole")
    check(a["order"][-3:] == ["call", "finish", "usage"], "tools streamed: text, call, finish, usage")
    check(sc.get("id") and sc["id"] != c["id"], "every reply has its own call ids")
    history[2] = {"role": "assistant", "content": "Let me check.", "tool_calls": [
        {"id": sc["id"], "type": "function", "function": {"name": sc["name"], "arguments": sc["arguments"]}}]}
    history[3]["tool_call_id"] = sc["id"]
    _, _, events, _, _ = stream({"messages": history, "tools": [WEATHER, CLOCK], "stream": True})
    check(assemble(events)["content"] == "The tool said: 18 C and sunny.", "tools streamed: the result read back")
    # two calls in one reply; parallel_tool_calls false keeps the first
    two = [{"role": "user", "content": "Make two calls: weather and time."}]
    _, r = chat({"messages": two, "tools": [WEATHER, CLOCK]})
    names = [t["function"]["name"] for t in r["choices"][0]["message"].get("tool_calls") or []]
    check(names == ["get_weather", "get_time"], "two calls in one reply: " + str(names))
    _, _, events, _, _ = stream({"messages": two, "tools": [WEATHER, CLOCK], "stream": True})
    check(sorted(assemble(events)["calls"]) == [0, 1], "two calls streamed, index 0 and 1")
    _, r = chat({"messages": two, "tools": [WEATHER, CLOCK], "parallel_tool_calls": False})
    check(len(r["choices"][0]["message"].get("tool_calls") or []) == 1, "parallel_tool_calls false: one call")


def test_tool_choice():
    ask = [{"role": "user", "content": "Please call the weather tool."}]
    _, r = chat({"messages": ask, "tools": [WEATHER, CLOCK], "tool_choice": "none"})
    check("tool_calls" not in r["choices"][0]["message"] and r["choices"][0]["finish_reason"] == "stop",
          "tool_choice none: no call")
    _, r = chat({"messages": USER_HI, "tools": [WEATHER, CLOCK], "tool_choice": "required"})
    calls = r["choices"][0]["message"].get("tool_calls") or []
    check(r["choices"][0]["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather",
          "tool_choice required: a call even to hi")
    named = {"type": "function", "function": {"name": "get_time"}}
    _, r = chat({"messages": USER_HI, "tools": [WEATHER, CLOCK], "tool_choice": named})
    calls = r["choices"][0]["message"].get("tool_calls") or []
    check(calls and calls[0]["function"]["name"] == "get_time" and
          json.loads(calls[0]["function"]["arguments"]) == {"zone": "UTC"}, "tool_choice names a function")
    check(r["choices"][0]["message"]["content"] is None, "a reply of calls only: content null")
    _, _, events, _, _ = stream({"messages": USER_HI, "tools": [WEATHER, CLOCK], "tool_choice": named,
                                 "stream": True})
    a = assemble(events)
    check(a["content"] == "" and a["calls"].get(0, {}).get("name") == "get_time" and a["finish"] == "tool_calls",
          "tool_choice named, streamed: the call only")
    status, r = chat({"messages": USER_HI, "tools": [WEATHER], "tool_choice": named})
    check(status == 400 and "tool_choice" in r["error"]["message"], "tool_choice naming a function not in tools")
    status, r = chat({"messages": USER_HI, "tool_choice": "required"})
    check(status == 400 and r["error"]["message"] == "'tool_choice' needs 'tools'.", "required without tools")
    status, r = chat({"messages": USER_HI, "tool_choice": "auto"})
    check(status == 200 and r["choices"][0]["message"]["content"] == "Hello! You said: hi", "auto without tools")


def agent_tools(n):
    """n tools as a coding agent sends them: a description, three parameters, an enum, what is required. With
    Qwen's tokenizer each takes about 100 tokens, so 100 of them take 10.5K of the 16K context."""
    out = []
    for i in range(n):
        out.append({"type": "function", "function": {
            "name": "tool_%03d" % i, "description": "Reads or changes part %d of the project." % i,
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string", "description": "The file."},
                "offset": {"type": "integer"},
                "mode": {"type": "string", "enum": ["read", "write", "append"]}},
                "required": ["path"]}}})
    return out


def agent_history(turns):
    """A long agent conversation: each turn a request, a call (no content), its result and an answer; the text of
    the messages as arrays of parts, as some clients send it (about 75 tokens a turn)."""
    msgs = [{"role": "system", "content": "You build what the user asks for, one step at a time."}]
    for i in range(turns):
        cid = "call_old_%d" % i
        msgs += [{"role": "user", "content": [{"type": "text", "text": "Step %d." % i}]},
                 {"role": "assistant", "tool_calls": [{"id": cid, "type": "function", "function": {
                     "name": "tool_%03d" % (i % 100), "arguments": json.dumps({"path": "f%d" % i, "mode": "read"})}}]},
                 {"role": "tool", "tool_call_id": cid, "content": [{"type": "text", "text": "ok %d" % i}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "Done %d." % i}]}]
    return msgs


def test_agent_request():
    tools = agent_tools(100)
    msgs = agent_history(30) + [{"role": "user", "content": "Now call the first tool."}]
    body = {"model": "PS4 Qwen3.5", "stream": True, "stream_options": {"include_usage": True},
            "messages": msgs, "tools": tools, "tool_choice": "auto"}
    values = json_values(body)
    status, _, events, _, _ = stream(body)
    a = assemble(events)
    check(status == 200 and a["done"] and a["finish"] == "tool_calls",
          "100 tools, %d messages, %d JSON values: answered" % (len(msgs), values))
    c = a["calls"].get(0, {})
    check(c.get("name") == "tool_000" and json.loads(c.get("arguments") or "{}") == {"path": "Lima"},
          "100 tools: the call, its arguments from the schema")
    pt = (a["usage"] or {}).get("prompt_tokens", 0)
    check(12000 < pt < 16384, "100 tools: prompt_tokens %d counts the tools" % pt)
    # the next request continues the conversation: only its new messages are read
    call = {"id": c.get("id"), "type": "function", "function": {"name": c.get("name"), "arguments": c.get("arguments")}}
    msgs += [{"role": "assistant", "tool_calls": [call]},
             {"role": "tool", "tool_call_id": c.get("id"), "content": "it worked"}]
    body["messages"] = msgs
    status, _, events, _, _ = stream(body)
    a2 = assemble(events)
    check(a2["content"] == "The tool said: it worked." and a2["finish"] == "stop", "100 tools: the result read back")
    kept = [l for l in log_text().splitlines() if "api: prompt of" in l][-1]
    n_kept = int(kept.split("(")[1].split(" kept")[0])
    check(n_kept > 12000, "the continued conversation read only what is new: " + kept.split("api: ")[1])
    # more JSON values than the first pool holds (4096): the pool grows (logit_bias is ignored, as documented)
    body = {"messages": agent_history(10) + [{"role": "user", "content": "Please call the first tool."}],
            "tools": tools, "logit_bias": {str(i): 0 for i in range(6000)}}
    values = json_values(body)
    status, r = chat(body)
    check(status == 200 and r["choices"][0]["finish_reason"] == "tool_calls",
          "a request of %d JSON values is answered" % values)


def test_limits():
    # tools and messages that do not fit: the numbers, without the words that make some clients drop their tools
    big = agent_tools(60)
    for t in big:
        t["function"]["description"] = "A long description of what this does. " * 40
    status, r = chat({"messages": USER_HI, "tools": big})
    msg = r["error"]["message"] if r else ""
    check(status == 400 and r["error"]["code"] == "context_length_exceeded", "too long with tools: 400")
    check(msg.startswith("This model's maximum context length is 16384 tokens. However, your messages take more than"),
          "too long: as OpenAI words it: " + msg)
    check("60 schemas sent with the request take" in msg and not has_word(msg, "tool", "function"),
          "too long: the tools' share, not called tools")
    long_history = [{"role": "user", "content": "word " * 9000}, {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "more " * 1000}]
    status, r = chat({"messages": long_history, "max_tokens": 8000})
    msg = r["error"]["message"] if r else ""
    check(status == 400 and r["error"]["code"] == "context_length_exceeded" and "in the completion" in msg,
          "messages and max_tokens over the window: " + msg)
    # bodies the API cannot read, each with its reason
    status, r = chat({"messages": USER_HI, "extra": [0] * 70000})
    check(status == 400 and "65536 JSON values" in r["error"]["message"], "more than 65536 values")
    status, r = chat("{\"messages\": [")
    check(status == 400 and r["error"]["message"] == "The body is not valid JSON.", "not JSON")
    status, r = chat("")
    check(status == 400 and "no body" in r["error"]["message"], "no body")
    status, r = chat("[1, 2]")
    check(status == 400 and r["error"]["message"] == "The body must be a JSON object.", "not an object")
    status, r = chat({"messages": USER_HI, "n": 2})
    check(status == 400, "n = 2")
    status, r = chat({"messages": [{"role": "user", "content": "z" * (1536 * 1024)}]})
    check(status == 413 and r and r["error"]["code"] == "request_too_large", "a body over 1 MiB: 413, read whole")
    head = '{"messages": [{"role": "user", "content": "'
    exact = head + "z " * ((1048576 - len(head) - 4) // 2) + '"}]}'
    exact = exact[:-4] + " " * (1048576 - len(exact)) + '"}]}'
    status, r = chat(exact)
    check(len(exact) == 1048576 and status == 400 and r["error"]["code"] == "context_length_exceeded",
          "a body of 1 MiB exactly is read (and is too long for the context)")
    log = log_text()
    check("request refused (" in log and "JSON values" in log and "over 1048576" in log, "refusals are logged")


def test_think():
    _, r = chat({"messages": USER_HI, "think": True})
    m = r["choices"][0]["message"]
    check(m.get("reasoning_content") == "The user wants a short answer." and m["content"] == "Hello! You said: hi",
          "reasoning apart from the answer")
    check(r["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0) > 0, "reasoning tokens")
    _, _, events, _, _ = stream({"messages": USER_HI, "stream": True, "think": True,
                                 "stream_options": {"include_usage": True}})
    a = assemble(events)
    check(a["reasoning"] == "The user wants a short answer." and a["content"] == "Hello! You said: hi", "streamed")
    check(a["order"].index("content") > max(i for i, o in enumerate(a["order"]) if o == "reasoning"),
          "streamed: the reasoning before the answer")
    check(a["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0, "streamed: reasoning tokens in usage")


def test_long_default():
    # without max_tokens a reply may be longer than 512 tokens (an agent's tool call often is)
    _, r = chat({"messages": [{"role": "user", "content": "long 400"}]})
    check(r["choices"][0]["finish_reason"] == "stop" and r["usage"]["completion_tokens"] > 512,
          "no max_tokens: %d tokens, not cut" % r["usage"]["completion_tokens"])
    check(r["choices"][0]["message"]["content"].endswith("word400 The end."), "the whole long reply")


def test_busy():
    result = {}

    def long_stream():
        result["a"] = stream({"messages": [{"role": "user", "content": "long 900"}], "stream": True})

    def second():
        result["b"] = (chat({"messages": USER_HI}), time.time())

    ta = threading.Thread(target=long_stream)
    ta.start()
    t0 = time.time()
    while not health()["busy"] and time.time() - t0 < 5:
        time.sleep(0.02)
    time.sleep(0.2)
    h = health()
    check(h["busy"] and h["waiting"] == 0, "/health while answering: busy, none waiting: %s" % h)
    t1 = time.time()
    status, _, raw = call("GET", "/v1/models")
    check(status == 200 and time.time() - t1 < 0.5, "/v1/models answers at once while the model is busy")
    tb = threading.Thread(target=second)
    tb.start()
    time.sleep(0.3)
    h = health()
    check(h["busy"] and h["waiting"] == 1, "/health with a request in line: waiting 1: %s" % h)
    ta.join()
    a_end = time.time()
    tb.join()
    (status, r), b_end = result["b"]
    check(status == 200 and r["choices"][0]["message"]["content"] == "Hello! You said: hi", "the request in line")
    check(b_end >= a_end - 0.05, "requests are answered in the order they came")
    a = assemble(result["a"][2])
    check(a["finish"] == "stop" and a["content"].endswith("The end."), "the long stream ended whole")
    check(wait_idle(2) is not None, "idle again afterwards")


def test_queue_full():
    results = []
    lock = threading.Lock()

    def one():
        s, r = chat({"messages": [{"role": "user", "content": "long 60"}]})
        with lock:
            results.append((s, r))

    threads = [threading.Thread(target=one) for _ in range(10)]
    for t in threads:
        t.start()
        time.sleep(0.01)
    for t in threads:
        t.join()
    busy = [r for s, r in results if s == 503]
    check(len(busy) >= 1 and all(r["error"]["code"] == "busy" for r in busy), "more than 8 at once: 503 busy")
    check(sum(1 for s, _ in results if s == 200) >= 8, "8 in line are answered")
    check(wait_idle(3) is not None, "idle again after the line")


def raw_request(body):
    s = socket.create_connection(("127.0.0.1", PORT), timeout=30)
    data = json.dumps(body).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
              b"Authorization: Bearer " + KEY.encode() + b"\r\nContent-Length: " + str(len(data)).encode() +
              b"\r\n\r\n" + data)
    return s


def test_disconnect():
    # a stream whose client leaves after a few events (a stop button): the console ends the reply at once
    s = raw_request({"messages": [{"role": "user", "content": "long 3000"}], "stream": True})
    got = b""
    while got.count(b"\"content\"") < 5:
        chunk = s.recv(4096)
        if not chunk:
            break
        got += chunk
    s.close()
    t = wait_idle(3)
    check(t is not None and t < 1.5, "a client that leaves a stream: the reply ends (idle after %s s)" % t)
    check(chat({"messages": USER_HI})[0] == 200, "and the next request is answered")
    # a whole reply (nothing sent until it ends) whose client gives up: seen within a few tokens
    s = raw_request({"messages": [{"role": "user", "content": "long 3000"}]})
    time.sleep(0.5)
    s.close()
    t = wait_idle(3)
    check(t is not None and t < 1.5, "a client that leaves a whole reply: the reply ends (idle after %s s)" % t)
    check(chat({"messages": USER_HI})[0] == 200, "and the next request is answered")
    # a request that waited in line, whose client left before its turn: not answered at all
    a = raw_request({"messages": [{"role": "user", "content": "long 300"}], "stream": True})
    time.sleep(0.2)
    b = raw_request({"messages": [{"role": "user", "content": "long 3000"}]})
    time.sleep(0.2)
    b.close()
    while a.recv(65536):
        pass
    a.close()
    check(wait_idle(3) is not None, "idle after the waiting request was dropped")
    log = log_text()
    check("the client of a waiting request left" in log, "the waiting request is dropped unread")
    check(log.count("the client left") >= 2, "the replies whose clients left are logged")


def test_curl():
    curl = shutil.which("curl")
    if not curl:
        print("(no curl: skipped)")
        return
    out = subprocess.run([curl, "-s", "-H", "Authorization: Bearer " + KEY, "-H", "Content-Type: application/json",
                          "-d", '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":128}',
                          "http://127.0.0.1:%d/v1/chat/completions" % PORT], capture_output=True, timeout=30)
    r = json.loads(out.stdout or b"{}")
    check(r.get("choices", [{}])[0].get("message", {}).get("content") == "Hello! You said: Hello", "curl: a reply")
    out = subprocess.run([curl, "-s", "-N", "-H", "Authorization: Bearer " + KEY,
                          "-d", '{"messages":[{"role":"user","content":"hi"}],"stream":true}',
                          "http://127.0.0.1:%d/v1/chat/completions" % PORT], capture_output=True, timeout=30)
    check(out.stdout.rstrip().endswith(b"data: [DONE]"), "curl: a stream")
    big = json.dumps({"messages": [{"role": "user", "content": "word " * 180000}], "max_tokens": 1})
    t0 = time.time()
    out = subprocess.run([curl, "-s", "-H", "Authorization: Bearer " + KEY, "--data-binary", "@-",
                          "http://127.0.0.1:%d/v1/chat/completions" % PORT], input=big.encode(), capture_output=True,
                         timeout=30)
    r = json.loads(out.stdout or b"{}")
    check(r.get("error", {}).get("code") == "context_length_exceeded" and time.time() - t0 < 2,
          "curl: a 900 KB body is read and refused as too long (%.2f s)" % (time.time() - t0))


def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join("build", "api_server.exe")
    os.makedirs("build", exist_ok=True)
    log = open(LOG, "w")
    server = subprocess.Popen([exe, "--key", KEY, "--delay", str(DELAY_MS)], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                health()
                break
            except OSError:
                time.sleep(0.05)
        tests = [test_endpoints, test_key, test_cors, test_plain, test_stream, test_tools, test_tool_choice,
                 test_agent_request, test_limits, test_think, test_long_default, test_busy, test_queue_full,
                 test_disconnect, test_curl]
        for t in tests:
            before = fails
            t0 = time.time()
            try:
                t()
            except Exception as e:   # (a check that could not even run counts as failed)
                check(False, "%s raised %r" % (t.__name__, e))
            print("%-20s %s (%.1f s)" % (t.__name__, "ok" if fails == before else "FAILED", time.time() - t0))
    finally:
        server.kill()
        server.wait()
        log.close()
    print("%d checks, %d failed" % (checks, fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
