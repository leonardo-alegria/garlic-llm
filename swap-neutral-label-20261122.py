#!/usr/bin/env python3
# swap-neutral-label-20261122.py — one patch: the advertised model id stops tracking the weights.
#
# Before: MODEL_NAME was derived from the loaded engine (family flag + hidden width), so a model
# swap on /data could change the id that /v1/models and the chat UI advertise, and every client
# on another machine had to be edited to match.
# After:  MODEL_NAME is a constant for this build. The truth is still in the log (the "weights:"
# line names the actual /data file).
#
# Idempotent: refuses if the target form is already in place.

import sys, pathlib

MAIN = pathlib.Path(__file__).resolve().parent / "chat" / "main.c"

OLD = ('#define MODEL_NAME (eng.qwen35 ? (eng.hidden >= 4096 ? "Qwen3.5-9B" : "Qwen3.5-4B") : "Qwen3-4B")\n')

NEW = ('''// Swap-neutral label. The weights on /data can be changed under the app (see the engine log
// line "weights:"), and the id it advertises must NOT follow that change: clients on the other
// machines (OpenWebUI, the LAN tools) are configured for one model name and stay working across
// swaps. This build is the 9B one, so it always says 9B even when another 9B-class set is loaded.
#define MODEL_NAME "Qwen3.5-9B"
''')

def main():
    src = MAIN.read_text()
    if NEW in src:
        print("main.c: already swap-neutral, nothing to do")
        return 0
    if src.count(OLD) != 1:
        print(f"main.c: anchor miss (count={src.count(OLD)}) — refusing")
        return 1
    MAIN.write_text(src.replace(OLD, NEW))
    print(f"patched {MAIN} (MODEL_NAME now constant, {len(NEW)} bytes)")
    # the other two places that name a model must agree; check they only read MODEL_NAME
    for f, pat in (("chat/assets/openapi.json", '"Qwen3.5-9B"'), ("chat/ui.h", '"Qwen3.5-9B"')):
        p = MAIN.parent.parent / f
        if p.exists() and pat in p.read_text():
            print(f"  {f}: already carries the same constant string, no change needed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
