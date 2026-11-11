"""Qwen3.5-4B for the PS4, on Modal: conversion to the app's Q4 model file format and the reference run.

Qwen3.5 text model (the vision tower and the MTP head are not used): 32 layers, every 4th full attention
(16 q / 4 kv heads of 256, output gate, RoPE on the first 64 dims), the rest Gated DeltaNet (16 k / 32 v heads of
128, causal conv of 4, gated delta rule). Zero-centred RMSNorms (x * (1 + w)) get 1 + w folded in.

File layout (all tensors on 256-byte boundaries, index in <name>.json):
  embed (Q4), final_norm
  per layer: attn_norm, mlp_norm, then
    full attention: q_norm, k_norm, wqkv = [q heads | k | v | gate heads] (Q4), wo (Q4)
    linear:         w_in = [qkv (2048 + 2048 + 4096) | z (4096) | b (32) | a (32)] (Q4),
                    conv [8192][4] fp32, A = -exp(A_log) [32], dt_bias [32], gnorm [128], wout (Q4)
  wgu = [gate | up] (Q4), wdown (Q4)

Run from pc/:
  python -m modal run modal_qwen35.py::convert           -> in the Modal volume: /out/qwen3.5-4b-q4.bin + .json
  python -m modal run modal_qwen35.py::reference --ids 1,2,3 --n 20
"""

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = "Qwen/Qwen3.5-4B"
NAME = "qwen3.5-4b-q4"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==5.20.1", "safetensors", "numpy<2", "huggingface_hub",
                      "accelerate", "jinja2")
         .env({"HF_HOME": "/cache/huggingface"})
         .add_local_file(HERE / "q4.py", "/root/q4.py"))
cache = modal.Volume.from_name("garlic-hf-cache", create_if_missing=True)
app = modal.App("ps4-qwen35", image=image)


@app.function(cpu=8, memory=32768, timeout=3600, volumes={"/cache": cache})
def convert_remote():
    import json
    import sys
    import numpy as np
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    sys.path.insert(0, "/root")
    from q4 import quantize_q4

    d = Path(snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt"]))
    cfg = json.loads((d / "config.json").read_text())["text_config"]
    wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}

    def get(name):
        f = wmap[name]
        if f not in files:
            files[f] = safe_open(str(d / f), "pt")
        return files[f].get_tensor(name).float().numpy()

    out_path = Path("/cache/out")
    out_path.mkdir(exist_ok=True)
    out = open(out_path / f"{NAME}.bin", "wb")
    index, offset = {}, 0

    def put(key, data, **meta):
        nonlocal offset
        pad = (-offset) % 256
        out.write(bytes(pad))
        offset += pad
        out.write(data)
        index[key] = dict(offset=offset, size=len(data), **meta)
        offset += len(data)

    def put_f32(key, a):
        put(key, np.ascontiguousarray(a, dtype=np.float32).tobytes(), n=int(np.size(a)))

    def put_q4(key, mat):
        packed, d16 = quantize_q4(np.ascontiguousarray(mat, dtype=np.float32))
        put(key, packed.tobytes() + d16.tobytes(), rows=mat.shape[0], cols=mat.shape[1], scales=packed.nbytes,
            fmt="q4")

    P = "model.language_model."
    put_q4("embed", get(P + "embed_tokens.weight"))
    put_f32("final_norm", 1.0 + get(P + "norm.weight"))
    nh, hd = cfg["num_attention_heads"], cfg["head_dim"]
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"{P}layers.{i}."
        put_f32(f"{i}.attn_norm", 1.0 + get(p + "input_layernorm.weight"))
        put_f32(f"{i}.mlp_norm", 1.0 + get(p + "post_attention_layernorm.weight"))
        if kind == "full_attention":
            a = p + "self_attn."
            put_f32(f"{i}.q_norm", 1.0 + get(a + "q_norm.weight"))
            put_f32(f"{i}.k_norm", 1.0 + get(a + "k_norm.weight"))
            qg = get(a + "q_proj.weight").reshape(nh, 2, hd, -1)          # per head: [query | gate]
            q, gate = qg[:, 0].reshape(nh * hd, -1), qg[:, 1].reshape(nh * hd, -1)
            put_q4(f"{i}.wqkv", np.concatenate([q, get(a + "k_proj.weight"), get(a + "v_proj.weight"), gate]))
            put_q4(f"{i}.wo", get(a + "o_proj.weight"))
        else:
            a = p + "linear_attn."
            put_q4(f"{i}.w_in", np.concatenate([get(a + "in_proj_qkv.weight"), get(a + "in_proj_z.weight"),
                                                 get(a + "in_proj_b.weight"), get(a + "in_proj_a.weight")]))
            put_f32(f"{i}.conv", get(a + "conv1d.weight").reshape(-1, cfg["linear_conv_kernel_dim"]))
            put_f32(f"{i}.A", -np.exp(get(a + "A_log")))
            put_f32(f"{i}.dt_bias", get(a + "dt_bias"))
            put_f32(f"{i}.gnorm", get(a + "norm.weight"))
            put_q4(f"{i}.wout", get(a + "out_proj.weight"))
        put_q4(f"{i}.wgu", np.concatenate([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]))
        put_q4(f"{i}.wdown", get(p + "mlp.down_proj.weight"))
        print(f"layer {i} ({kind}) done, {offset / 2**20:.0f} MiB", flush=True)
    out.close()
    (out_path / f"{NAME}.json").write_text(json.dumps(dict(config=cfg, size=offset, tensors=index), indent=1))
    cache.commit()
    return offset


@app.function(gpu=["L40S", "A100-40GB"], timeout=1800, volumes={"/cache": cache})
def reference_remote(ids: list, n_new: int) -> dict:
    """Transformers Qwen3.5 in FP32 with every quantized matrix replaced by exactly the PS4's Q4 weights."""
    import sys
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, "/root")
    from q4 import dequantize_q4, quantize_q4

    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model

    def q4(t):
        return torch.from_numpy(dequantize_q4(*quantize_q4(t.detach().cpu().numpy())))

    with torch.no_grad():
        for name, mod in lm.named_modules():
            if isinstance(mod, torch.nn.Linear):
                mod.weight.copy_(q4(mod.weight))
        lm.embed_tokens.weight.copy_(q4(lm.embed_tokens.weight))
        head = model.get_output_embeddings()
        if head.weight.data_ptr() != lm.embed_tokens.weight.data_ptr():
            head.weight.copy_(lm.embed_tokens.weight)
        model = model.to("cuda")
        x = torch.tensor([ids], device="cuda")
        logits = model(input_ids=x).logits[0, -1].float().cpu().numpy()
        out = model.generate(input_ids=x, max_new_tokens=n_new, do_sample=False)[0, len(ids):].tolist()
    cache.commit()
    return dict(logits=logits.tobytes(), out=out, type=type(model).__name__)


@app.function(timeout=600, volumes={"/cache": cache})
def chat_ids(messages: list, thinking: bool = False) -> list:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO)
    r = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=True)
    r = r["input_ids"] if hasattr(r, "keys") else r
    return [int(v) for v in (r[0] if r and isinstance(r[0], list) else r)]


@app.local_entrypoint()
def convert():
    n = convert_remote.remote()
    print(f"converted: {n / 2**20:.1f} MiB in the volume at /out/{NAME}.bin")


@app.local_entrypoint()
def reference(ids: str = "", n: int = 20, prompt: str = "¿Cuál es la capital del Perú?"):
    import json
    import numpy as np
    id_list = [int(v) for v in ids.split(",")] if ids else chat_ids.remote([{"role": "user", "content": prompt}])
    r = reference_remote.remote(id_list, n)
    out_dir = HERE.parent / "build"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / f"ref_{NAME}.npz", ids=np.array(id_list), logits=np.frombuffer(r["logits"], np.float32),
             out=np.array(r["out"]))
    print(r["type"], "prompt", len(id_list), "tokens ->", r["out"])
