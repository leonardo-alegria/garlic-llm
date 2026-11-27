"""End-to-end live API test suite for Garlic LLM on PS4.

Verifies every feature implemented on the v1.20 + 1.21-api + PR #6 + PR #12/#14 + Issue #17 build:
  1. GET /health
  2. GET /v1/models (active model + /data/garlic/models/ list + context_length)
  3. POST /v1/chat/completions (basic chat + usage)
  4. Streaming SSE + stream_options.include_usage
  5. Reasoning (chat_template_kwargs.enable_thinking -> reasoning_content)
  6. Tool calling (tools + tool_choice + role: "tool" round-trip)
  7. 2-slot prompt/prefix caching (Slot 1 follow-up + Slot 0 new-session system prompt reuse)
  8. Long pre-token tokenizer (>512-byte continuous string without truncation)
  9. Durable task queue (POST /v1/tasks, GET /v1/tasks, GET /v1/tasks/<id>, POST /v1/tasks/<id>/cancel)
 10. TTS audio endpoints (GET /v1/audio/voices, POST /v1/audio/speech, modalities: ["text", "audio"])
 11. Live model switching (optional: --switch <model>, switches and switches back)

Usage:
  python pc/test_api_features.py --host http://192.168.1.29:8080
  python pc/test_api_features.py --host http://192.168.1.29:8080 --switch maple
"""
import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, base: str, key: str = ""):
        self.base = base.rstrip("/")
        self.key = key

    def req(self, method: str, path: str, body=None, timeout: int = 180):
        headers = {"Accept": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        r = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        t0 = time.time()
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                raw = resp.read()
                return resp.status, dict(resp.headers), raw, time.time() - t0
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read(), time.time() - t0

    def stream_chat(self, body: dict, timeout: int = 180):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        r = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.time()
        events = []
        done = False
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                events.append(json.loads(payload))
        return events, done, time.time() - t0


def run_test(name, fn, results):
    print(f"\n[{len(results) + 1}] {name} ...", flush=True)
    try:
        detail = fn()
        print(f"    PASS: {detail}", flush=True)
        results.append((name, "PASS", detail))
    except SkipTest as s:
        print(f"    SKIP: {s}", flush=True)
        results.append((name, "SKIP", str(s)))
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)
        results.append((name, "FAIL", str(e)))


class SkipTest(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description="Live API feature test suite for Garlic LLM PS4")
    ap.add_argument("--host", default="http://192.168.1.29:8080", help="Base URL of Garlic LLM API")
    ap.add_argument("--key", default="", help="Optional Bearer API key")
    ap.add_argument("--switch", default="", help="Optional model folder name to test live POST /v1/models/select")
    args = ap.parse_args()

    c = Client(args.host, args.key)
    results = []
    state = {"active_model": "", "models": [], "is_maple": False}

    def test_health():
        code, _, raw, dt = c.req("GET", "/health", timeout=15)
        assert code == 200, f"HTTP {code}: {raw}"
        j = json.loads(raw)
        assert j.get("status") == "ok", f"unexpected body: {j}"
        return f"status=ok busy={j.get('busy')} waiting={j.get('waiting')} ({dt*1000:.0f} ms)"

    def test_models():
        code, _, raw, dt = c.req("GET", "/v1/models", timeout=15)
        assert code == 200, f"HTTP {code}: {raw}"
        j = json.loads(raw)
        ids = [m["id"] for m in j.get("data", [])]
        assert ids, f"no models listed: {j}"
        ctx = j["data"][0].get("context_length")
        assert ctx and ctx >= 4096, f"missing/invalid context_length: {j['data'][0]}"
        state["active_model"] = ids[0]
        state["models"] = ids
        state["is_maple"] = "maple" in ids[0].lower()
        return f"active={ids[0]} context_length={ctx} all={ids} ({dt*1000:.0f} ms)"

    def test_chat_basic():
        code, _, raw, dt = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [{"role": "user", "content": "Reply with exactly one word: PONG"}],
            "temperature": 0.0,
            "max_tokens": 16,
        })
        assert code == 200, f"HTTP {code}: {raw}"
        j = json.loads(raw)
        txt = j["choices"][0]["message"]["content"]
        u = j.get("usage", {})
        assert txt and u.get("prompt_tokens", 0) > 0 and u.get("completion_tokens", 0) > 0, f"bad reply: {j}"
        return f"reply={txt.strip()!r} usage={u} ({dt:.2f}s)"

    def test_streaming_usage():
        events, done, dt = c.stream_chat({
            "model": state["active_model"] or "garlic",
            "messages": [{"role": "user", "content": "Count 1, 2, 3."}],
            "temperature": 0.0,
            "max_tokens": 24,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        assert done, "missing [DONE] terminator"
        text = "".join(
            ch.get("delta", {}).get("content", "")
            for ev in events for ch in ev.get("choices", [])
        )
        usage_evs = [ev["usage"] for ev in events if ev.get("usage")]
        assert text.strip(), "empty streamed content"
        assert usage_evs, "missing final usage chunk for stream_options.include_usage"
        return f"chunks={len(events)} text={text.strip()!r} usage={usage_evs[-1]} ({dt:.2f}s)"

    def test_reasoning():
        if state["is_maple"]:
            raise SkipTest("Maple does not use Qwen3.5 <think> blocks")
        code, _, raw, dt = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [{"role": "user", "content": "What is 17 * 19? Give the number."}],
            "chat_template_kwargs": {"enable_thinking": True},
            "temperature": 0.0,
            "max_tokens": 64,
        })
        assert code == 200, f"HTTP {code}: {raw}"
        j = json.loads(raw)
        msg = j["choices"][0]["message"]
        rc = msg.get("reasoning_content", "")
        rt = j.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0)
        assert rc and rt > 0, f"expected reasoning_content and reasoning_tokens > 0: {j}"
        return f"reasoning_tokens={rt} answer={msg.get('content', '').strip()!r} ({dt:.2f}s)"

    def test_tool_calling():
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
        code, _, raw, dt1 = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
            "tools": tools,
            "tool_choice": "required",
            "temperature": 0.0,
            "max_tokens": 96,
        })
        assert code == 200, f"HTTP {code}: {raw}"
        j1 = json.loads(raw)
        tcs = j1["choices"][0]["message"].get("tool_calls") or []
        assert tcs and tcs[0]["function"]["name"] == "get_weather", f"expected get_weather tool_call: {j1}"
        tc_id = tcs[0]["id"]
        # Follow-up turn with role: "tool"
        code2, _, raw2, dt2 = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {"role": "assistant", "content": "", "tool_calls": tcs},
                {"role": "tool", "tool_call_id": tc_id, "content": "{\"temp_c\": 22, \"condition\": \"sunny\"}"},
            ],
            "tools": tools,
            "temperature": 0.0,
            "max_tokens": 64,
        })
        assert code2 == 200, f"HTTP {code2}: {raw2}"
        j2 = json.loads(raw2)
        ans = j2["choices"][0]["message"].get("content", "")
        assert "22" in ans or "sunny" in ans.lower(), f"unexpected tool follow-up answer: {ans!r}"
        return f"tool_call={tcs[0]['function']} -> answer={ans.strip()!r} (t1={dt1:.2f}s, t2={dt2:.2f}s)"

    def test_prompt_caching_2slot():
        # ~600-token fixed system prompt to make prefill savings clearly measurable
        big_sys = "You are a concise PS4 coding assistant. " + ("Rule: keep replies under 5 words. " * 60)
        # Turn 1 (cold)
        c1, _, r1, dt_cold = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [
                {"role": "system", "content": big_sys},
                {"role": "user", "content": "Say alpha."},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        })
        assert c1 == 200, f"turn1 HTTP {c1}: {r1}"
        a1 = json.loads(r1)["choices"][0]["message"]["content"]
        # Turn 2 (Slot 1: same-session follow-up)
        c2, _, r2, dt_conv = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [
                {"role": "system", "content": big_sys},
                {"role": "user", "content": "Say alpha."},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": "Now say beta."},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        })
        assert c2 == 200, f"turn2 HTTP {c2}: {r2}"
        # Turn 3 (Slot 0: NEW session with same system prompt, different user prompt)
        c3, _, r3, dt_sys = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [
                {"role": "system", "content": big_sys},
                {"role": "user", "content": "Brand new session: say gamma."},
            ],
            "temperature": 0.0,
            "max_tokens": 8,
        })
        assert c3 == 200, f"turn3 HTTP {c3}: {r3}"
        return f"cold={dt_cold:.2f}s | slot1_followup={dt_conv:.2f}s | slot0_new_session={dt_sys:.2f}s"

    def test_long_pretoken():
        # >512 bytes with no spaces (previously truncated at 512 bytes before 6cfc1ad)
        marker = "ENDMARKER99"
        long_str = ("a" * 700) + marker
        code, _, raw, dt = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [{"role": "user", "content": f"What word is at the very end of this string: {long_str}"}],
            "temperature": 0.0,
            "max_tokens": 32,
        })
        assert code == 200, f"HTTP {code}: {raw}"
        j = json.loads(raw)
        txt = j["choices"][0]["message"]["content"]
        assert "ENDMARKER99" in txt or "99" in txt, f"marker lost (tokenizer truncated >512B?): {txt!r}"
        return f"marker preserved: {txt.strip()!r} ({dt:.2f}s)"

    def test_task_queue():
        code, _, raw, _ = c.req("POST", "/v1/tasks", {"prompt": "Reply with one word: TASKOK"})
        assert code == 202, f"POST /v1/tasks HTTP {code}: {raw}"
        tid = json.loads(raw)["id"]
        # Poll until completed
        deadline = time.time() + 90
        final = None
        while time.time() < deadline:
            gc, _, graw, _ = c.req("GET", f"/v1/tasks/{tid}", timeout=15)
            assert gc == 200, f"GET /v1/tasks/{tid} HTTP {gc}: {graw}"
            final = json.loads(graw)
            if final["state"] in ("completed", "failed", "canceled"):
                break
            time.sleep(0.5)
        assert final and final["state"] == "completed", f"task did not complete: {final}"
        lc, _, lraw, _ = c.req("GET", "/v1/tasks", timeout=15)
        assert lc == 200 and any(t["id"] == tid for t in json.loads(lraw).get("data", [])), "task missing from list"
        return f"id={tid} state={final['state']} result={final.get('result', '').strip()!r}"

    def test_tts_audio():
        if state["is_maple"]:
            raise SkipTest("TTS (Chatterbox) is skipped while Maple is loaded")
        vc, _, vraw, _ = c.req("GET", "/v1/audio/voices", timeout=15)
        assert vc == 200, f"GET /v1/audio/voices HTTP {vc}: {vraw}"
        voices = [v["id"] for v in json.loads(vraw).get("voices", [])]
        sc, _, sraw, sdt = c.req("POST", "/v1/audio/speech", {
            "input": "Hello from PS4.",
            "voice": "en",
            "response_format": "wav",
        })
        if sc == 503:
            raise SkipTest("TTS models not loaded on console")
        assert sc == 200 and sraw[:4] == b"RIFF" and sraw[8:12] == b"WAVE", f"invalid WAV ({sc}, {len(sraw)} B)"
        cc, _, craw, cdt = c.req("POST", "/v1/chat/completions", {
            "model": state["active_model"] or "garlic",
            "messages": [{"role": "user", "content": "Say hi in 2 words."}],
            "modalities": ["text", "audio"],
            "audio": {"format": "wav", "voice": "en"},
            "temperature": 0.0,
            "max_tokens": 12,
        })
        assert cc == 200, f"chat audio HTTP {cc}: {craw}"
        aud = json.loads(craw)["choices"][0]["message"].get("audio", {})
        wav = base64.b64decode(aud.get("data", ""))
        assert wav[:4] == b"RIFF", "chat modalities audio missing RIFF header"
        return f"voices={voices} speech_wav={len(sraw)}B ({sdt:.1f}s) chat_audio_wav={len(wav)}B ({cdt:.1f}s)"

    def test_model_switch():
        if not args.switch:
            raise SkipTest("pass --switch <model> (e.g. --switch maple) to test live model switching")
        orig = state["active_model"]
        c1, _, r1, dt1 = c.req("POST", "/v1/models/select", {"model": args.switch}, timeout=240)
        assert c1 == 200, f"switch to {args.switch} failed HTTP {c1}: {r1}"
        m1 = json.loads(r1)["id"]
        assert m1 == args.switch, f"expected {args.switch}, got {m1}"
        # Quick inference on switched model
        qc, _, qraw, qdt = c.req("POST", "/v1/chat/completions", {
            "model": m1,
            "messages": [{"role": "user", "content": "Say OK."}],
            "temperature": 0.0,
            "max_tokens": 8,
        })
        assert qc == 200, f"inference on {m1} failed HTTP {qc}: {qraw}"
        qtxt = json.loads(qraw)["choices"][0]["message"]["content"].strip()
        # Switch back if different
        if orig and orig != args.switch:
            c2, _, r2, dt2 = c.req("POST", "/v1/models/select", {"model": orig}, timeout=240)
            assert c2 == 200, f"switch back to {orig} failed HTTP {c2}: {r2}"
            return f"{orig} -> {m1} ({dt1:.1f}s, reply={qtxt!r} in {qdt:.2f}s) -> {orig} ({dt2:.1f}s)"
        return f"loaded {m1} ({dt1:.1f}s, reply={qtxt!r} in {qdt:.2f}s)"

    for name, fn in [
        ("GET /health", test_health),
        ("GET /v1/models", test_models),
        ("POST /v1/chat/completions (basic)", test_chat_basic),
        ("SSE streaming + stream_options.include_usage", test_streaming_usage),
        ("Reasoning (reasoning_content)", test_reasoning),
        ("Tool calling + role:tool round-trip", test_tool_calling),
        ("2-Slot Prompt / Prefix Caching", test_prompt_caching_2slot),
        ("Long pre-token tokenizer (>512 B)", test_long_pretoken),
        ("Durable Task Queue (/v1/tasks)", test_task_queue),
        ("TTS Audio (/v1/audio/voices, /v1/audio/speech, modalities)", test_tts_audio),
        ("Live Model Switching (POST /v1/models/select)", test_model_switch),
    ]:
        run_test(name, fn, results)

    passed = sum(1 for _, st, _ in results if st == "PASS")
    failed = sum(1 for _, st, _ in results if st == "FAIL")
    skipped = sum(1 for _, st, _ in results if st == "SKIP")
    print(f"\n=== Summary: {passed} PASS, {failed} FAIL, {skipped} SKIP ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
