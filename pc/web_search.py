"""Chat with Garlic LLM's OpenAI API with keyless web search (DuckDuckGo + Wikipedia) and optional audio output.

No API keys or third-party packages are needed (uses Python's standard library).

Usage:
  python pc/web_search.py "What is the latest GoldHEN release for PS4?"
  python pc/web_search.py --audio reply.wav "Tell me a short fact about the PS4"
  python pc/web_search.py                 # interactive REPL
  PS4_IP=192.168.1.5 python pc/web_search.py "Who won the 2026 World Cup?"
"""

import base64
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HOST = os.environ.get("GARLIC_API_URL") or f"http://{os.environ.get('PS4_IP') or '127.0.0.1'}:8080/v1"
API_KEY = os.environ.get("GARLIC_API_KEY", "")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet (DuckDuckGo and Wikipedia) for up-to-date information, facts, news, or URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {"type": "integer", "description": "Maximum number of results (1 to 8).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Fetch and read the plain text content of a webpage URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The http/https URL to read."},
                },
                "required": ["url"],
            },
        },
    },
]


def strip_tags(s):
    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def ddg_search(query, max_results=5):
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=12) as resp:
        page = resp.read().decode("utf-8", "replace")
    results = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        page,
        re.S,
    ):
        href, title, snippet = m.groups()
        u = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("uddg", [href])[0]
        results.append({"title": strip_tags(title), "url": u, "snippet": strip_tags(snippet)})
        if len(results) >= max_results:
            break
    return results


def wiki_search(query, max_results=3):
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": max_results,
        "format": "json",
    })
    req = urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{params}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    out = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        out.append({"title": f"{title} (Wikipedia)", "url": url, "snippet": strip_tags(item.get("snippet", ""))})
    return out


def web_search(query, max_results=5):
    max_results = max(1, min(int(max_results or 5), 8))
    results = []
    try:
        results.extend(ddg_search(query, max_results))
    except Exception as e:
        results.append({"error": f"DuckDuckGo search failed: {e}"})
    if len(results) < max_results:
        try:
            results.extend(wiki_search(query, max(1, max_results - len(results))))
        except Exception:
            pass
    return json.dumps(results[:max_results], ensure_ascii=False)


def read_webpage(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read(200_000).decode("utf-8", "replace")
        text = strip_tags(raw)
        return text[:4000] if text else "No readable text found."
    except Exception as e:
        return f"Failed to read {url}: {e}"


def chat_completion(messages, want_audio=False, voice="auto", use_tools=True):
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {"messages": messages, "temperature": 0.3, "max_tokens": 768}
    if use_tools:
        payload["tools"] = TOOLS
    if want_audio:
        payload["modalities"] = ["text", "audio"]
        payload["audio"] = {"format": "wav", "voice": voice}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{HOST.rstrip('/')}/chat/completions", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ask(question, messages=None, audio_file=None, voice="auto"):
    if messages is None:
        messages = []
    messages.append({"role": "user", "content": question})
    for _ in range(4):
        res = chat_completion(messages, want_audio=bool(audio_file), voice=voice, use_tools=True)
        msg = res["choices"][0]["message"]
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            if audio_file and msg.get("audio", {}).get("data"):
                wav_bytes = base64.b64decode(msg["audio"]["data"])
                with open(audio_file, "wb") as f:
                    f.write(wav_bytes)
                print(f"  [audio] saved {len(wav_bytes)} bytes to {audio_file}", flush=True)
            return msg.get("content") or ""
        for tc in tool_calls:
            fn = tc["function"]["name"]
            args = json.loads(tc["function"].get("arguments") or "{}")
            print(f"  [tool] {fn}({args})", flush=True)
            if fn == "web_search":
                out = web_search(args.get("query", ""), args.get("max_results", 5))
            elif fn == "read_webpage":
                out = read_webpage(args.get("url", ""))
            else:
                out = json.dumps({"error": f"Unknown tool: {fn}"})
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    res = chat_completion(messages, want_audio=bool(audio_file), voice=voice, use_tools=False)
    msg = res["choices"][0]["message"]
    if audio_file and msg.get("audio", {}).get("data"):
        wav_bytes = base64.b64decode(msg["audio"]["data"])
        with open(audio_file, "wb") as f:
            f.write(wav_bytes)
        print(f"  [audio] saved {len(wav_bytes)} bytes to {audio_file}", flush=True)
    return msg.get("content") or ""


def main():
    global HOST
    args = sys.argv[1:]
    audio_file = None
    voice = "auto"
    if "--ip" in args:
        i = args.index("--ip")
        HOST = f"http://{args[i + 1]}:8080/v1"
        del args[i : i + 2]
    if "--audio" in args:
        i = args.index("--audio")
        audio_file = args[i + 1]
        del args[i : i + 2]
    if "--voice" in args:
        i = args.index("--voice")
        voice = args[i + 1]
        del args[i : i + 2]
    if args:
        print(ask(" ".join(args), audio_file=audio_file, voice=voice))
        return
    print(f"Connected to {HOST} (Ctrl+C to exit)")
    messages = []
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print("\n" + ask(q, messages, audio_file=audio_file, voice=voice))


if __name__ == "__main__":
    main()
