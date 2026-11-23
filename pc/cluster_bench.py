"""Check and time DarkGarlic over its OpenAI-compatible API (L3 on the chat screen turns it on; port 8080).

  python pc/cluster_bench.py --bench http://192.168.1.20:8080
      a few fixed prompts, streamed: time to first token and tokens per second of each

  python pc/cluster_bench.py --compare http://192.168.1.5:8080 http://192.168.1.20:8080
      the same prompts at temperature 0 (greedy) on a single console and on a cluster head: the replies must be the
      same, word for word. This is the cluster's correctness test (docs/CLUSTER.md, milestone M1): the 9B split over
      two consoles has to answer exactly as the 9B on one.

Only the standard library. Set DG_API_KEY if the console has /data/garlic_api_key.txt.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List five uses for an old games console.",
    "What is 17 * 23? Answer with the number only.",
]


def chat(base, prompt, temperature, max_tokens, stream):
    body = json.dumps({"model": "darkgarlic", "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": max_tokens, "stream": stream}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    key = os.environ.get("DG_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t0 = time.time()
    first, pieces = None, []
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            j = json.loads(r.read())
            return j["choices"][0]["message"]["content"], None, time.time() - t0, j.get("usage", {})
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
            if delta:
                if first is None:
                    first = time.time() - t0
                pieces.append(delta)
    return "".join(pieces), first, time.time() - t0, {}


def bench(base, max_tokens):
    print(f"{base}: {len(PROMPTS)} prompts, up to {max_tokens} tokens each")
    for p in PROMPTS:
        text, ttft, total, _ = chat(base, p, 0.7, max_tokens, True)
        # (streamed deltas are about one token each)
        n = max(1, len(text.split()) * 4 // 3)
        gen = total - (ttft or 0)
        print(f"  first token {ttft or 0:5.2f} s, ~{n / gen if gen > 0 else 0:5.1f} tokens/s  | {p[:48]}")


def compare(a, b, max_tokens):
    bad = 0
    for p in PROMPTS:
        ta = chat(a, p, 0.0, max_tokens, False)[0]
        tb = chat(b, p, 0.0, max_tokens, False)[0]
        same = ta == tb
        bad += not same
        print(f"  {'SAME' if same else 'DIFF'}  {p[:60]}")
        if not same:
            k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
            print(f"        first difference at character {k}:\n        A: {ta[max(0, k - 40):k + 40]!r}\n"
                  f"        B: {tb[max(0, k - 40):k + 40]!r}")
    print("all replies match" if not bad else f"{bad} of {len(PROMPTS)} differ")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", metavar="URL")
    ap.add_argument("--compare", nargs=2, metavar=("SINGLE_URL", "CLUSTER_URL"))
    ap.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    if a.bench:
        bench(a.bench, a.max_tokens)
    if a.compare:
        sys.exit(compare(*a.compare, a.max_tokens))
    if not (a.bench or a.compare):
        ap.error("--bench or --compare")


if __name__ == "__main__":
    main()
