# patch-20261127-final-edits.py — captures today's edits that were applied outside a patch script,
# in chain order after patch-20261127-ctxfit.py. Run on the tree state right after ctxfit.
# 1) api.c     models_json emission fix: {"id": prefix restored (sb_json_str carries the quotes).
# 2) engine.c  swap file gate: idx weights_size = GPU part only (cpu_embed models are bigger files);
#              the honest gate is the validated record table's maxend.
# 3) engine.c  phase C generalization for the 32K build (split blocks from the idx, kernel cap 2x idx,
#              kb/kv_split/lend_ctx by idx_max_seq, tq-shape guard drops the seq equality).
# 4) shaders/qwen35.cl  MAX_SEQ 8192 -> 16384 (pair with pc/export_kernels_9b.py regen; the toolchain
#              needs toolchain-*/zig/zig.exe symlink on Linux for gpu.py's Windows name).
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
EDITS = []
def edit(path, old, new): EDITS.append((path, old, new))

edit("chat/api.c",
'''        sb_json_str(out, names[i], strlen(names[i]));
        sb_str(out, ",\\"object\\":\\"model\\"}");''',
'''        sb_str(out, "{\\"id\\":");
        sb_json_str(out, names[i], strlen(names[i]));
        sb_str(out, ",\\"object\\":\\"model\\"}");''')

edit("chat/api.c",
'''        sb_json_str(out, loaded, strlen(loaded));
        sb_str(out, ",\\"object\\":\\"model\\"}");''',
'''        sb_str(out, "{\\"id\\":");
        sb_json_str(out, loaded, strlen(loaded));
        sb_str(out, ",\\"object\\":\\"model\\"}");''')

edit("chat/engine.c",
'''    {   // the file must be whole before anything is overwritten
        int f = open(wp, O_RDONLY);
        if (f < 0) {
            fail(e, "swap: cannot open %s", wp);
            return -1;
        }
        off_t n = lseek(f, 0, SEEK_END);
        close(f);
        if (n < 0 || (uint64_t)n != e->weights_size) {
            fail(e, "swap: weights file size wrong (%lld)", (long long)n);
            return -1;
        }
    }''',
'''    {   // the file must be whole before anything is overwritten: records (validated equal) end at or before it
        uint64_t maxend = 0;
        for (uint32_t i = 0; i < e->n_tensors; i++) {
            uint64_t end = e->t[i].off + e->t[i].size;
            if (end > maxend) maxend = end;
        }
        int f = open(wp, O_RDONLY);
        if (f < 0) {
            fail(e, "swap: cannot open %s", wp);
            return -1;
        }
        off_t n = lseek(f, 0, SEEK_END);
        close(f);
        if (n < 0 || (uint64_t)n < maxend) {
            fail(e, "swap: weights file too small for the record table (%lld of %llu)", (long long)n,
                 (unsigned long long)maxend);
            return -1;
        }
    }''')

edit("chat/engine.c",
"    if (e->tq && e->tq16) e->max_seq = 2 * 8192;",
"    if (e->tq && e->tq16) e->max_seq = 2 * e->idx_max_seq;   // split design: two blocks of the kernels' MAX_SEQ")

edit("chat/engine.c",
"        if (e->tq && (e->head_dim != 256 || e->max_seq != 8192)) e->tq = 0;   // the kernels' shape (qwen35.cl MAX_SEQ)",
"        if (e->tq && e->head_dim != 256) e->tq = 0;   // the kernels' shape (qwen35.cl); block size comes from the idx now")

edit("chat/engine.c",
"        uint64_t kb = (uint64_t)e->n_kv * 8192 * (e->head_dim / 2 + 4), n_full = 0;",
"        uint64_t kb = (uint64_t)e->n_kv * e->idx_max_seq * (e->head_dim / 2 + 4), n_full = 0;")

edit("chat/engine.c",
"        e->kv_split = n_full * (4096ull * e->n_kv * e->head_dim * 2 + 4096ull * e->n_kv * (e->head_dim + 4));",
"        e->kv_split = n_full * ((e->idx_max_seq / 2) * e->n_kv * e->head_dim * 2 + (e->idx_max_seq / 2) * e->n_kv * (e->head_dim + 4));")

edit("chat/engine.c",
"""                e->tq_split = e->kv_hi - all;
                e->lend_ctx = 8192;""",
"""                e->tq_split = e->kv_hi - all;
                e->lend_ctx = e->idx_max_seq;""")

edit("chat/engine.c",
"        uint64_t kern_max = (e->tq && e->tq16) ? 16384u : 8192u;",
"        uint64_t kern_max = (e->tq && e->tq16) ? 2u * e->idx_max_seq : e->idx_max_seq;")

edit("shaders/qwen35.cl",
"#define MAX_SEQ 8192",
"#define MAX_SEQ 16384   // split blocks from the idx now; the 32K build pairs kernels with idx_max_seq 16384")

fail = 0
for path, old, new in EDITS:
    n = (ROOT / path).read_text().count(old)
    if n != 1:
        print(f"ANCHOR {path} count={n} — abort")
        fail = 1
if fail: sys.exit(1)
for path, old, new in EDITS:
    p = ROOT / path
    p.write_text(p.read_text().replace(old, new, 1))
    print(f"patched {path}")
print("final-edits capture OK")
