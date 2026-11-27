"""Strip speech, grow the context, auto-enable the API, /data model slot (2026-11-22).
Run on a clean tree; every op asserts its anchor."""
import pathlib, re, struct
R = pathlib.Path(".")

def lines(p): return R.joinpath(p).read_text().split("\n")
def putl(p, L): R.joinpath(p).write_text("\n".join(L))
def wr(p, s): R.joinpath(p).write_text(s)
def idx(p, needle, frm=0):
    L = lines(p)
    for i in range(frm, len(L)):
        if needle in L[i]: return i
    raise SystemExit(f"{p}: anchor {needle!r} missing")
def guard(p, a, b):   # inclusive 0-based line range
    L = lines(p); L[a:a] = ["#if 0  // speech removed 2026-11-22"]; L[b+2:b+2] = ["#endif"]
    putl(p, L)
def cut(p, a, b):
    L = lines(p); del L[a:b+1]; putl(p, L)
def rep(p, old, new, n=1):
    s = R.joinpath(p).read_text(); assert old in s, (p, old[:70]); wr(p, s.replace(old, new, n))

# ---------- engine.h ----------
p = "chat/engine.h"
L = lines(p)
a = next(k for k, x in enumerate(L) if x.startswith("// Speech to text: Parakeet"))
b = next(k for k, x in enumerate(L) if x == "} S3G;")
L[a:a] = ["#if 0  // speech structs removed 2026-11-22"]; L[b+2:b+2] = ["#endif"]
s = "\n".join(L)
s = s.replace("    Asr asr;\n    T3 t3;\n    S3G s3;\n", "")
s = s.replace("int t3_reload(Engine* e);\n", "")
s = s.replace("    uint64_t tsc_freq;",
              "    uint64_t tsc_freq;\n"
              "    uint64_t ws_begin, ws_end;   // GPU slab the thinking turn copies the DeltaNet state into "
              "(the room the speech models used to take)")
wr(p, s)

# ---------- engine.c ----------
p = "chat/engine.c"
a = idx(p, "static const AsrTensor* at(Engine* e, const char* name) {")
b = idx(p, "int engine_flex_test(Engine* e) {", a) - 1
guard(p, a, b)
a = idx(p, "void asr_release(Engine* e) {")
b = idx(p, '#include "engine_s3g.inc"', a)
guard(p, a, b)
rep(p,
    '    snprintf(path, sizeof(path), "%s/model.idx", dir);\n'
    '    uint8_t* d = read_file(path, &size);\n'
    '    if (!d || (memcmp(d, "PMDL", 4) && memcmp(d, "PMQ5", 4))) {',
    '    uint8_t* d = read_file("/data/model.idx", &size);   // a model dropped on the HDD wins\n'
    '    if (!d || (memcmp(d, "PMDL", 4) && memcmp(d, "PMQ5", 4))) {\n'
    '        snprintf(path, sizeof(path), "%s/model.idx", dir);\n'
    '        d = read_file(path, &size);\n'
    '    }\n'
    '    if (!d || (memcmp(d, "PMDL", 4) && memcmp(d, "PMQ5", 4))) {')
rep(p, "    if (e->tq_split && pos0 + n > (int)e->lend_ctx && e->s3.loaded) e->s3.loaded = 0;",
    "    // (voice borrowing gone: nothing to unload past the lend point)")
rep(p, "    mm_tune(e);   // the batched prompt's product kernels, fastest per shape\n"
    "    swar_tune(e);\n    engine_reset(e);\n    return 0;",
    "    mm_tune(e);   // the batched prompt's product kernels, fastest per shape\n"
    "    swar_tune(e);\n"
    "    // with the speech models gone their GPU room is free again: the thinking turn keeps a slab of it\n"
    "    uint64_t ws = engine_state_size(e);\n"
    "    if (ws && ws < e->mem_size - e->next) {\n"
    "        e->ws_begin = gpu_alloc(e, ws);\n"
    "        if (e->ws_begin) e->ws_end = e->ws_begin + ws;\n"
    "    }\n"
    "    engine_reset(e);\n    return 0;")

# ---------- engine_spec.inc ----------
p = "chat/engine_spec.inc"
s = R.joinpath(p).read_text()
s = s.replace("int spec_load(Engine* e, uint64_t base, uint64_t size, int start) {\n"
              "    Spec* s = &g_spec;\n    if (!s->idx_ok) return -1;\n"
              "    s->ready = s->on = s->drafted = 0;\n"
              "    uint64_t need = spec_layout(e, base);",
              "int spec_load(Engine* e, uint64_t base, uint64_t size, int start) {\n"
              "    Spec* s = &g_spec;\n    if (!s->idx_ok) return -1;\n"
              "    s->ready = s->on = s->drafted = 0;\n"
              "    if (!base) {   // no T3 memory to borrow: the head gets a block of its own\n"
              "        uint64_t want = spec_layout(e, 0);\n"
              "        base = gpu_alloc(e, want);\n"
              "        if (!base) { say(\"spec: no GPU memory left for the MTP head (%u MiB)\", (unsigned)(want >> 20)); return -1; }\n"
              "        size = want;\n"
              "    }\n"
              "    uint64_t need = spec_layout(e, base);", 1)
a = s.index("// T3's weights read again into their memory")
b = s.index("// The draft length cap", a)
wr(p, s[:a] + s[b:])

# ---------- main.c ----------
p = "chat/main.c"
rep(p, "    int on = f && sz && f[0] == '1';",
    "    int on = !f || !sz || f[0] == '1';   // auto-on: only a written 0 turns it off")
# dictation function (with the comment block above it)
a = idx(p, "static int voice_to_text(uint32_t button, DrawUnder under, char* text, int max) {")
L = lines(p)
end = next(i for i in range(a + 1, len(L)) if L[i] == "    return 1;" and L[i+1] == "}")
c = a
while L[c-1].startswith("//"): c -= 1
putl(p, L[:c] + ["#if 0  // dictation removed 2026-11-22"] + L[c:end+2] + ["#endif"] + L[end+2:])
# the asr background thread block
a = idx(p, "static void* asr_reload_main(void* arg) {")
b = idx(p, "static void run_python(void) {", a) - 1
L = lines(p)
putl(p, L[:a] + ["#if 0  // dictation weights gone 2026-11-22"] + L[a:b+1] + ["#endif"] + L[b+1:])
# voice mode section, through push_to_talk
L = lines(p)
a = next(k for k, x in enumerate(L) if x.startswith("// ---- voice mode: hold Square"))
h = next(k for k in range(a + 1, len(L)) if L[k].startswith("static void push_to_talk(void) {"))
b = next(k for k in range(h + 1, len(L)) if L[k] == "}")
putl(p, L[:a] + ["#if 0  // voice mode gone 2026-11-22"] + L[a:b+1] + ["#endif"] + L[b+1:])
for old, new in [
    ("eng.asr.ws_end", "eng.ws_end"), ("eng.asr.ws_begin", "eng.ws_begin"),
    ("    asr_wait();   // (a reload still running from the last program)\n    asr_release(&eng);\n    py_lent = (uint8_t*)asr_lent(&eng, &py_lent_size);\n",
     "    py_lent = NULL;   // (no dictation weights to lend into)\n"),
    ("    asr_reload_bg();   // (it first gives back the screen copies the run may have taken)\n", ""),
    ("            voice_mode();", "            // (voice mode gone)"),
    ("            push_to_talk();", "            // (dictation gone)"),
    ("    static uint32_t ids[16384];", "    static uint32_t ids[24576];"),
    ("    int n = 0, max = 16384;", "    int n = 0, max = 24576;"),
    ("(the 9B's dictation weights live there; Python borrows them)", "(Python borrows what is free)"),
]:
    s = R.joinpath(p).read_text(); assert old in s, old[:60]; wr(p, s.replace(old, new))
s = R.joinpath(p).read_text()
m = re.search(r"    if \(t3_init\(&eng, ASSETS\) == 0\) \{.*?say\(\"speech to text not available \(missing or no GPU memory left\)\"\);\n", s, re.S)
assert m; s = s[:m.start()] + s[m.end():]
wr(p, s)
s = R.joinpath(p).read_text()
m = re.search(r"    tts_boot\(\);\n    size_t flex1 = 0;.*?say\(\"voice %s; system RAM free: %u MiB\", tts_ready \? \"ready\" : \"not available\", \(unsigned\)\(flex1 >> 20\)\);\n", s, re.S)
assert m; s = s[:m.start()] + s[m.end():]
wr(p, s)
s = R.joinpath(p).read_text()
m = re.search(r"static void spec_lend\(void\) \{.*?\n\}", s, re.S)
assert m
s = s[:m.start()] + ("static void spec_lend(void) {\n"
                     "    if (spec_lent) return;\n"
                     "    double t0 = now();\n"
                     "    if (spec_load(&eng, 0, 0, pos) == 0) {   // (it follows the conversation from here)\n"
                     "        spec_lent = 1;\n"
                     "        say(\"spec: MTP head in its own memory in %.2f s (following the conversation from position %d)\", now() - t0, pos);\n"
                     "    }\n"
                     "}") + s[m.end():]
wr(p, s)

# ---------- ui.c ----------
p = "chat/ui.c"
L = lines(p)
a = next(k for k, x in enumerate(L) if x.startswith("// ---- voice ---"))
putl(p, L[:a] + ["#if 0  // voice screens removed 2026-11-22"] + L[a:] + ["#endif"])

# ---------- api_run.inc: API_IDS only (the context keep is a later patch) ----------
rep("chat/api_run.inc", "#define API_IDS 16384", "#define API_IDS 24576")

# ---------- shaders, exporter, shipped model.idx ----------
rep("shaders/qwen35.cl", "#define MAX_SEQ 8192", "#define MAX_SEQ 12288")
rep("pc/export_chat.py", "QWEN35_MAX_SEQ = 8192", "QWEN35_MAX_SEQ = 12288")
d = bytearray(R.joinpath("chat/assets_9b/model.idx").read_bytes())
assert struct.unpack_from("<I", d, 32)[0] == 8192
struct.pack_into("<I", d, 32, 12288)
R.joinpath("chat/assets_9b/model.idx").write_bytes(d)

# ---------- build_chat.py: speech weights opt-in ----------
s = R.joinpath("build_chat.py").read_text()
old = ('FULL_MODELS = (["qwen3.5-4b-q4i.bin"] if "--4b" in sys.argv else ["qwen3.5-9b-q4s.bin", "qwen3.5-9b-mtp.bin"]) + '
      '    ["parakeet-v3-q4i.bin", "chatterbox-t3.bin", "chatterbox-s3gen-mf.bin"]')
assert old in s
wr("build_chat.py", s.replace(old, '# the speech weights are opt-in now: text-only is the default (--speech adds them back)\n'
   'FULL_MODELS = ["qwen3.5-4b-q4i.bin"] if "--4b" in sys.argv else ["qwen3.5-9b-q4s.bin", "qwen3.5-9b-mtp.bin"]\n'
   'if "--speech" in sys.argv:\n'
   '    FULL_MODELS += ["parakeet-v3-q4i.bin", "chatterbox-t3.bin", "chatterbox-s3gen-mf.bin"]', 1))
print("patch set applied")
