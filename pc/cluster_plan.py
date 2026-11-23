"""Plan a DarkGarlic model across 1..N PS4s: size every tensor, check the kernels can run the shape, and split the
layers so that each console's GPU memory holds its share.

The cluster runs the model as a pipeline (docs/CLUSTER.md): node 0 (the head) has the tokenizer, the screen, the API
and the input embedding rows (read from the disk by the CPU, as now); every node runs a contiguous range of layers;
the last node also has the final norm and lm_head and draws the token. Between nodes only one hidden row per token
travels (hidden_size fp32 = 16 KiB for the 9B, 20 KiB for 27B).

  python pc/cluster_plan.py --model qwen3.5-27b --nodes 3                # prints the plan
  python pc/cluster_plan.py --model qwen3.5-27b --nodes 3 --quant mlp3   # 3-bit gate/up AND down
  python pc/cluster_plan.py --config hf/<model>/config.json --nodes 2    # any Qwen3 / Qwen3.5 dense config.json
  python pc/cluster_plan.py --model qwen3-14b --nodes 2 --out cluster/profiles/qwen3-14b-2x.json
  python pc/cluster_plan.py --list

Only the standard library. Sizes follow pc/convert_9b_q4s.py: q4s = 4-bit nibbles + an fp16 scale per row and block
of 32 (0.5625 bytes per weight, rows padded to 128); q3s = 3 bits + the same scales (0.4375). The 9B's numbers come
out within 1 % of the real qwen3.5-9b-q4s.bin GPU part (--model qwen3.5-9b --nodes 1 prints it).
"""
import argparse
import json
import math
import sys
from pathlib import Path

MiB = 1 << 20

# Shapes copied from the official config.json files (text_config for Qwen3.5). keep_vocab: lm_head rows kept by the
# frequency trim of convert_9b_q4s.py (pc/vocab_order.npy has 248 070 for the Qwen3.5 tokenizer).
PRESETS = {
    "qwen3.5-9b": dict(
        arch="qwen35", checkpoint="huihui-ai/Huihui-Qwen3.5-9B-abliterated", layers=32, hidden=4096, inter=12288,
        heads=16, kv_heads=4, head_dim=256, lin_k_heads=16, lin_v_heads=32, lin_dk=128, lin_dv=128,
        full_interval=4, vocab=248320, keep_vocab=248070, mtp=1),
    "qwen3.5-27b": dict(
        arch="qwen35", checkpoint="huihui-ai/Huihui-Qwen3.5-27B-abliterated", layers=64, hidden=5120, inter=17408,
        heads=24, kv_heads=4, head_dim=256, lin_k_heads=16, lin_v_heads=48, lin_dk=128, lin_dv=128,
        full_interval=4, vocab=248320, keep_vocab=248070, mtp=1),
    "qwen3-14b": dict(
        arch="qwen3", checkpoint="Qwen3-14B abliterated (e.g. huihui-ai/Qwen3-14B-abliterated)", layers=40,
        hidden=5120, inter=17408, heads=40, kv_heads=8, head_dim=128, vocab=151936, keep_vocab=151936, mtp=0),
    "qwen3-8b": dict(
        arch="qwen3", checkpoint="Qwen3-8B abliterated", layers=36, hidden=4096, inter=12288, heads=32,
        kv_heads=8, head_dim=128, vocab=151936, keep_vocab=151936, mtp=0),
}

# What each matrix is stored as. "attn" = the mixer's projections (wqkv/wo or the DeltaNet w_in/wout).
QUANTS = {
    "default": dict(attn="q4s", gu="q3s", down="q4s", head="q4s"),   # the shipping 9B layout
    "mlp3": dict(attn="q4s", gu="q3s", down="q3s", head="q4s"),      # 3-bit MLP: the 27B on 3 consoles
    "all3": dict(attn="q3s", gu="q3s", down="q3s", head="q4s"),
    "all4": dict(attn="q4s", gu="q4s", down="q4s", head="q4s"),      # the quality option (more memory)
}
BYTES_PER_WEIGHT = {"q4s": 0.5625, "q3s": 0.4375}

# GPU memory per console (engine.c reserve_garlic): the block is capped at 4608 MiB (4736 on a PS4 Pro in Neo mode)
# and eng_margin_mb (64) goes back to the system.
NODE_MiB = {"base": 4608 - 64, "pro": 4736 - 64}
# Fixed GPU memory that is not weights or caches: command buffer + kernel arguments (8 MiB), kernels.bin, RoPE table,
# activations, prompt batch buffers, SWAR partial sums. Measured on the 9B at ~60 MiB; 96 keeps a margin.
FIXED_MiB = 96
# Kernel limits today (shaders/attn.h GQ_MAX, engine.c engine_init's shape check)
GQ_MAX = 4


def pad(n, m):
    return (n + m - 1) // m * m


def mat_bytes(rows, cols, fmt):
    return pad(rows, 128) * cols * BYTES_PER_WEIGHT[fmt]


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    t = c.get("text_config", c)
    qwen35 = "linear_num_value_heads" in t
    m = dict(arch="qwen35" if qwen35 else "qwen3", checkpoint=str(path), layers=t["num_hidden_layers"],
             hidden=t["hidden_size"], inter=t["intermediate_size"], heads=t["num_attention_heads"],
             kv_heads=t["num_key_value_heads"], head_dim=t.get("head_dim", t["hidden_size"] // t["num_attention_heads"]),
             vocab=t["vocab_size"], mtp=t.get("mtp_num_hidden_layers", 0))
    m["keep_vocab"] = 248070 if m["vocab"] == 248320 else m["vocab"]
    if qwen35:
        m.update(lin_k_heads=t["linear_num_key_heads"], lin_v_heads=t["linear_num_value_heads"],
                 lin_dk=t["linear_key_head_dim"], lin_dv=t["linear_value_head_dim"],
                 full_interval=t.get("full_attention_interval", 4))
        if "layer_types" in t:
            m["layer_types"] = t["layer_types"]
    for k in ("num_experts", "num_local_experts", "moe_intermediate_size"):
        if t.get(k):
            m["moe"] = True
    return m


def layer_types(m):
    if m["arch"] != "qwen35":
        return ["full_attention"] * m["layers"]
    if "layer_types" in m:
        return list(m["layer_types"])
    k = m["full_interval"]
    return ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(m["layers"])]


def layer_bytes(m, q, full):
    """GPU bytes of one layer's weights (norms and the small DeltaNet tensors included)."""
    H, I = m["hidden"], m["inter"]
    b = 2 * H * 4                                                    # attn_norm, mlp_norm (fp32)
    if full:
        qg = m["heads"] * m["head_dim"] * (2 if m["arch"] == "qwen35" else 1)   # q (+ output gate)
        rows = qg + 2 * m["kv_heads"] * m["head_dim"]
        b += mat_bytes(rows, H, q["attn"]) + mat_bytes(H, m["heads"] * m["head_dim"], q["attn"])
        b += 2 * m["head_dim"] * 4                                   # q_norm, k_norm
    else:
        kd, vd = m["lin_k_heads"] * m["lin_dk"], m["lin_v_heads"] * m["lin_dv"]
        rows = 2 * kd + 2 * vd + 2 * m["lin_v_heads"]                # q | k | v | z | b | a
        b += mat_bytes(rows, H, q["attn"]) + mat_bytes(H, vd, q["attn"])
        b += (2 * kd + vd) * 4 * 4 + 2 * m["lin_v_heads"] * 4 + m["lin_dv"] * 4   # conv, A, dt_bias, gnorm
    b += mat_bytes(2 * I, H, q["gu"]) + mat_bytes(H, I, q["down"])
    return b


def layer_cache_bytes(m, full, ctx):
    """KV cache (full attention) or DeltaNet state of one layer, as engine_init lays it out."""
    if m["arch"] == "qwen35":
        if full:   # TurboQuant 4-bit: k and v for 8192 positions + the 16K block (kv_hi)
            hd, kv = m["head_dim"], m["kv_heads"]
            lo = 2 * kv * 8192 * (hd // 2 + 4)
            hi = 4096 * kv * hd * 2 + 4096 * kv * (hd + 4)
            return lo + (hi if ctx > 8192 else 0)
        conv_n = 2 * m["lin_k_heads"] * m["lin_dk"] + m["lin_v_heads"] * m["lin_dv"]
        return conv_n * 12 + m["lin_v_heads"] * m["lin_dk"] * m["lin_dv"] * 4
    return 2 * ctx * m["kv_heads"] * m["head_dim"] * 2                # Qwen3: fp16 k and v, max_seq positions


def checks(m):
    """What the current kernels cannot run yet (each is a work item in docs/ROADMAP.md)."""
    out = []
    if m.get("moe"):
        out.append("mixture-of-experts layers: no expert routing kernels yet (ROADMAP: MoE)")
    gq = m["heads"] // m["kv_heads"]
    if gq > GQ_MAX:
        out.append(f"{gq} query heads per KV head: shaders/attn.h GQ_MAX is {GQ_MAX} (ROADMAP: GQA > 4)")
    want_hd = 256 if m["arch"] == "qwen35" else 128
    if m["head_dim"] != want_hd:
        out.append(f"head_dim {m['head_dim']}: the {m['arch']} kernels are built for {want_hd}")
    if m["arch"] == "qwen35" and (m["lin_dk"] != 128 or m["lin_dv"] != 128):
        out.append("DeltaNet head dims other than 128 (qwen35.cl DK/DV)")
    if m["hidden"] % 256:
        out.append(f"hidden {m['hidden']} is not a multiple of 256 (rmsnorm/add dispatch)")
    if m["arch"] == "qwen3":
        out.append("Qwen3 path has no q4s/q3s converter or batched prompt yet (ROADMAP: Qwen3 q4s)")
    return out


def plan(m, nodes, quant="default", console="base", ctx=16384, head_extra_mib=0):
    q = QUANTS[quant]
    types = layer_types(m)
    lw = [layer_bytes(m, q, t == "full_attention") for t in types]
    lc = [layer_cache_bytes(m, t == "full_attention", ctx) for t in types]
    head = mat_bytes(m["keep_vocab"], m["hidden"], q["head"]) + m["hidden"] * 4   # lm_head + final norm
    budget = NODE_MiB[console] * MiB - FIXED_MiB * MiB
    cost = [lw[i] + lc[i] for i in range(m["layers"])]
    # The last node carries lm_head; the head node can be asked to keep room for voice / MTP (head_extra_mib).
    extra = [0.0] * nodes
    extra[-1] += head
    extra[0] += head_extra_mib * MiB
    # Balance: binary search the largest per-node load, then fill nodes greedily in order.
    def split(cap):
        ranges, i = [], 0
        for n in range(nodes):
            load, j = extra[n], i
            while j < m["layers"] and load + cost[j] <= cap:
                load += cost[j]
                j += 1
            ranges.append((i, j))
            i = j
        return ranges if i == m["layers"] else None
    lo, hi = max(extra), sum(cost) + max(extra) + 1
    best = split(hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        r = split(mid)
        if r:
            hi, best = mid, r
        else:
            lo = mid
    out = []
    for n, (a, b) in enumerate(best):
        w = sum(lw[a:b]) + (head if n == nodes - 1 else 0)
        c = sum(lc[a:b])
        used = w + c + FIXED_MiB * MiB + (head_extra_mib * MiB if n == 0 else 0)
        out.append(dict(node=n, layers=[a, b], n_layers=b - a, weights_mib=round(w / MiB, 1),
                        cache_mib=round(c / MiB, 1), used_mib=round(used / MiB, 1),
                        free_mib=round((NODE_MiB[console] * MiB - used) / MiB, 1),
                        has_embed=n == 0, has_head=n == nodes - 1))
    total_w = sum(lw) + head
    # Decode speed: each token reads every weight once; the nodes run one after another (pipeline, one stream).
    # 176 GB/s (base) or 218 GB/s (Pro) at the ~58 % the 9B reaches without MTP, plus ~0.3 ms per network hop.
    bw = (176e9 if console == "base" else 218e9) * 0.58
    t_tok = total_w / bw + 0.0003 * 2 * (nodes - 1)
    return dict(model=m, quant=quant, console=console, nodes=nodes, context=ctx,
                total_weights_mib=round(total_w / MiB, 1), embed_rows_on_disk_mib=round(
                    m["vocab"] * m["hidden"] * 0.5625 / MiB, 1),
                fits=all(n["free_mib"] >= 0 for n in out), node_plan=out,
                est_tokens_per_s_single_stream=round(1 / t_tok, 1), kernel_gaps=checks(m))


def show(p):
    m = p["model"]
    print(f"{m['checkpoint']}  ({m['arch']}, {m['layers']} layers, hidden {m['hidden']}, quant {p['quant']}, "
          f"{p['console']} PS4, context {p['context']})")
    print(f"  weights on the GPUs: {p['total_weights_mib']:.0f} MiB total; embedding rows stay on node 0's disk "
          f"({p['embed_rows_on_disk_mib']:.0f} MiB)")
    for n in p["node_plan"]:
        role = "head" if n["node"] == 0 else "worker"
        tags = (" +embed(disk)" if n["has_embed"] else "") + (" +lm_head" if n["has_head"] else "")
        print(f"  node {n['node']} ({role}): layers {n['layers'][0]:>2}..{n['layers'][1] - 1:<2} "
              f"weights {n['weights_mib']:7.1f}  cache {n['cache_mib']:6.1f}  used {n['used_mib']:7.1f}  "
              f"free {n['free_mib']:7.1f} MiB{tags}")
    print(f"  fits: {'YES' if p['fits'] else 'NO'};  estimated decode ~{p['est_tokens_per_s_single_stream']} "
          f"tokens/s for one chat (before speculative decoding)")
    for g in p["kernel_gaps"]:
        print(f"  needs: {g}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(PRESETS))
    ap.add_argument("--config", help="a Hugging Face config.json instead of a preset")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--quant", choices=sorted(QUANTS), default="default")
    ap.add_argument("--console", choices=sorted(NODE_MiB), default="base")
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--head-extra", type=int, default=0, help="MiB kept free on node 0 (voice, MTP head)")
    ap.add_argument("--out", help="write the plan as a cluster profile (JSON)")
    ap.add_argument("--name", help="display / API model name stored in the profile")
    ap.add_argument("--list", action="store_true", help="every preset on 1..4 consoles")
    a = ap.parse_args(argv)
    if a.list:
        for k in sorted(PRESETS):
            for n in (1, 2, 3, 4):
                for qn in ("default", "mlp3"):
                    p = plan(dict(PRESETS[k]), n, qn, a.console, a.context)
                    print(f"{k:12s} x{n} {qn:8s} {'fits' if p['fits'] else '----'} "
                          f"{p['total_weights_mib']:8.0f} MiB  ~{p['est_tokens_per_s_single_stream']:5.1f} tok/s"
                          f"{'  (kernel work needed)' if p['kernel_gaps'] else ''}")
        return 0
    if not (a.model or a.config):
        ap.error("--model or --config")
    m = load_config(a.config) if a.config else dict(PRESETS[a.model])
    p = plan(m, a.nodes, a.quant, a.console, a.context, a.head_extra)
    show(p)
    if a.out:
        prof = dict(name=a.name or (a.model or Path(a.config).parent.name), preset=a.model, quant=a.quant,
                    console=a.console, context=a.context, nodes=[
                        dict(node=n["node"], host=f"PS4_{n['node']}_IP", layers=n["layers"],
                             has_embed=n["has_embed"], has_head=n["has_head"],
                             weights=f"/data/darkgarlic/node{n['node']}.bin") for n in p["node_plan"]],
                    plan=p)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {a.out}")
    return 0 if p["fits"] else 2


if __name__ == "__main__":
    sys.exit(main())
