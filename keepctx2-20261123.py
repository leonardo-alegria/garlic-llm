"""keepctx2: the end-of-job block in api_run.inc still reset pos unconditionally, so
`api_pos == pos` could never hold across requests: keep ctx missed on EVERY request
(the 2026-11-22 marmoset proof was too small to tell a 46-token re-feed from a keep).
Guard the end reset: on a continuing job the engine conversation and its position stay;
only the on-screen view restarts (api_chat_end). need_close stays 1: the next request
closes the sampled-but-unfed answer end like the chat loop does."""
import pathlib
p = pathlib.Path("chat/api_run.inc")
s = p.read_text()

if "keepctx2" in s:
    print("keepctx2 already applied")
    raise SystemExit(0)

old = (
    "    // the engine's state is this request's; the conversation on screen starts again at position 0\n"
    "    pos = 0;\n"
    "    need_close = 0;\n"
    "    api_chat_end();"
)
new = (
    "    // keepctx2: a continuing job leaves the engine conversation intact: only the screen restarts.\n"
    "    // (a job that started over still clears, as the chat does; api_pos/api_msgs_fed already mark\n"
    "    //  where this conversation ends, set right after the prompt was fed)\n"
    "    if (!cont) { pos = 0; need_close = 0; }\n"
    "    else api_pos = pos, api_msgs_fed = mcount;   // (the finished conversation: prompt + sampled answer)\n"
    "    api_chat_end();"
)
assert old in s, "end-block anchor missing"
s = s.replace(old, new, 1)

p.write_text(s)
print("keepctx2 applied: end-of-job reset is now cont-aware")
