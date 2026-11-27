"""keepctx3: replace the keepctx2 end block. The end of api_job must never touch pos on any
path: the top !cont block owns resets (pos + engine together), and the sampling loop already
left pos at prompt+answer. So the end only refreshes the keep bookkeeping and restarts the view.
keepctx2's `if (!cont) { pos = 0; ... }` arm zeroed after every fresh job, so the NEXT request's
`api_pos == pos` check (10827 vs 0) always failed -> cont false -> full re-feed (proved on-console
2026-11-23 08:2x: prompt of 10826/10850 back to back on the keepctx2 build)."""
import pathlib
p = pathlib.Path("chat/api_run.inc")
s = p.read_text()

if "keepctx3" in s:
    print("keepctx3 already applied")
    raise SystemExit(0)
assert "keepctx2" in s, "not a keepctx2 tree"

old = (
    "    // keepctx2: a continuing job leaves the engine conversation intact: only the screen restarts.\n"
    "    // (a job that started over still clears, as the chat does; api_pos/api_msgs_fed already mark\n"
    "    //  where this conversation ends, set right after the prompt was fed)\n"
    "    if (!cont) { pos = 0; need_close = 0; }\n"
    "    else api_pos = pos, api_msgs_fed = mcount;   // (the finished conversation: prompt + sampled answer)\n"
    "    api_chat_end();"
)
new = (
    "    // keepctx3: the sampling loop left pos at prompt+answer; the engine conversation ends\n"
    "    //  right there and stays that way. A request that starts over is reset by the !cont block\n"
    "    //  up front (pos and the engine together); nothing here may touch pos or the end is dead.\n"
    "    api_pos = pos, api_msgs_fed = mcount;   // refresh: conversation now ends past the sampled answer\n"
    "    api_chat_end();"
)
assert old in s, "keepctx2 end-block anchor missing"
s = s.replace(old, new, 1)
p.write_text(s)
print("keepctx3 applied: end block is pos-free, bookkeeping refreshed")
