# patch-20261127-ctxfit.py — phase B on the model-dir build (run after patch-20261127-modeldir.py).
# 1) engine.c: before the RoPE alloc, re-derive e->max_seq from the GPU memory actually left at that point:
#    per position = full-attention layers x (TQ 4-bit halves or fp16 k+v) + the RoPE row; kernel ceiling respected
#    (attn_tq split design = 16384 with tq16, else 8192). The .idx's max_seq stays a wish; free room decides.
# 2) engine.c: no boot pick -> display name = the weights' basename without the extension (the honest weight name shows).
# 3) main.c:  view_update: the top-right label follows eng.model_name — boot pick, weights basename, and swaps.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EDITS = []

def edit(path, old, new):
    EDITS.append((path, old, new))

edit("chat/engine.c",
"""    // RoPE table: [max_seq][rope_dim] = cos of the first half, then sin (RoPE on the first rope_dim dims of a head)
    uint32_t half = e->rope_dim / 2;""",
"""    // Live ctx: the .idx's max_seq is a wish; what the GPU memory still left can hold decides. The kernels'
    // ceiling stays (attn_tq split design = 16K with tq16, the fp16 kernels 8K). Slack: the MTP head (~234 MiB),
    // the thinking slab (~50 MiB), and the activations/caches that come after this point (~20 MiB of margin).
    {
        uint32_t n_full = 0;
        for (uint32_t i = 0; i < e->n_layers; i++) n_full += e->full_mask >> i & 1;
        uint64_t per_pos = (uint64_t)e->rope_dim * 4;   // the RoPE row's share
        if (e->tq) per_pos += (uint64_t)n_full * 2 * e->n_kv * (e->head_dim / 2 + 4);   // TQ 4-bit k + v + scales
        else per_pos += (uint64_t)n_full * e->n_kv * e->head_dim * 4;                  // fp16 k + v
        uint64_t kern_max = (e->tq && e->tq16) ? 16384u : 8192u;
        uint64_t free = e->mem_size - e->next;
        uint64_t slack = 304ull << 20;
        if (free <= slack) {
            fail(e, "no GPU room for any context (%u MiB left after the model)", (unsigned)(free >> 20));
            return -1;
        }
        uint64_t fit = (free - slack) / per_pos;
        fit -= fit % ATT_BLOCK;
        if (fit > kern_max) fit = kern_max;
        if (fit < e->max_seq) {
            e->max_seq = (uint32_t)fit;
            say("context trimmed to what fits: %u tokens (%u MiB left)", e->max_seq, (unsigned)(free >> 20));
        } else if (fit > e->max_seq) {
            e->max_seq = (uint32_t)fit;
            say("context grown to what fits: %u tokens (%u MiB left)", e->max_seq, (unsigned)(free >> 20));
        }
    }
    // RoPE table: [max_seq][rope_dim] = cos of the first half, then sin (RoPE on the first rope_dim dims of a head)
    uint32_t half = e->rope_dim / 2;""")

edit("chat/engine.c",
"""    weights_locate(e->weights_path);
    memcpy(&e->weights_size, d + 180, 8);""",
"""    weights_locate(e->weights_path);
    if (!e->model_name[0]) {   // no boot pick: the weights' own name is what the UI and /v1/models show
        const char* b = strrchr(e->weights_path, '/');
        snprintf(e->model_name, sizeof(e->model_name), "%s", b ? b + 1 : e->weights_path);
        char* dot = strrchr(e->model_name, '.');
        if (dot) *dot = 0;
    }
    memcpy(&e->weights_size, d + 180, 8);""")

edit("chat/main.c",
"""    view.model = MODEL_NAME;""",
"""    view.model = eng.model_name[0] ? eng.model_name : MODEL_NAME;   // follows the loaded model, and swaps""")

fail = 0
for path, old, new in EDITS:
    p = ROOT / path
    txt = p.read_text()
    n = txt.count(old)
    if n != 1:
        print(f"ANCHOR {path} count={n} — abort")
        fail = 1
if fail:
    sys.exit(1)
for path, old, new in EDITS:
    p = ROOT / path
    p.write_text(p.read_text().replace(old, new, 1))
    print(f"patched {path}")
print("ctx-fit + display name OK")
