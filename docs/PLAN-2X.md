# Two PS4s (branch `darkgarlic-cluster-2x`)

The cluster code is the same as on `darkgarlic-cluster` (docs/CLUSTER.md). This branch adds the two-console profiles and the plan for getting a bigger model onto them.

## Profiles

| Profile | Model | Split | Planner numbers | Status |
|---|---|---|---|---|
| `cluster/profiles/qwen3.5-9b-2x-test.json` | Abliterated Qwen3.5-9B | 0–18 head, 19–31 + lm_head worker | 3880 MiB, ~2.4 GiB free on each, ~25 tok/s | **Run this first (M1).** It needs no new kernels and proves the pipeline. |
| `cluster/profiles/qwen3-14b-2x.json` | Abliterated Qwen3-14B, 8K context | 0–20 head, 21–39 + lm_head worker | 6656 MiB, 500 / 459 MiB free, ~14 tok/s | Needs kernel work (below). |

### What does not fit on two

Qwen3.5-27B doesn't fit on two PS4s even with every matrix in 3 bits: about 5.7 GiB per console, against a 4.4 GiB budget. It would need ~2-bit weights, and at that point quality drops off a cliff. The 27B is the three-console branch.

### Is the 14B worth it?

Be honest with yourself about this before spending the kernel work. Qwen3-14B is the older Qwen3 generation, and the Qwen3.5-9B already matches or beats it on many benchmarks. Its real advantages are its size (40 layers, 5120 wide) and a different style of answer. It is the largest dense model that fits two PS4s.

The other way to use two consoles is a **better 9B** rather than a bigger model:

| Option | What you get |
|---|---|
| 9B `all4` (no 3-bit gate/up) | 4264 MiB of weights, with 1.5 GiB still free on each console. Gate and up go back to 4 bits, for better answers. |
| Voice + MTP + Python all loaded at once | Today voice mode borrows the MTP head's memory (22 tok/s in voice mode). With half the layers on each console, both fit, so voice runs at full speed. |
| 32K context | The TurboQuant cache costs about 21 MiB per full-attention layer per 16K. With 8 of those layers split over two consoles, 32K costs ~170 MiB per console. |

Try `python pc/cluster_plan.py --model qwen3.5-9b --nodes 2 --quant all4 --head-extra 1200`.

## Work list for Qwen3-14B on two consoles

1. **M1 first:** the 9B two-way test must answer like one console (CLUSTER.md).
2. **GQA 5.** Raise `GQ_MAX` in `shaders/attn.h` to 5 or 8 for the Qwen3 kernels (`shaders/qwen3.cl`), and change the `n_heads / n_kv > 4` check in `engine_init`. LDS use is `acc[4][GQ_MAX][128]`, so 8 KiB per extra head at head_dim 128. Check occupancy with the `speed_case` start-up test (`/data/garlic_kvtest.txt` = 1).
3. **Qwen3 q4s converter.** Write `pc/convert_q4s.py --arch qwen3`: q4s attention/down, q3s gate/up, the lm_head trimmed by token frequency, embedding rows read by the CPU (`q4row`, fmt 4), and a `PMDL` `model.idx` with flags. The engine's Qwen3 path also needs the cpu_embed and SWAR flags read from `PMDL` (today only `PMQ5` has the flags word).
4. **Batched prompt for Qwen3.** `e->pf` requires `qwen35`. The full-attention half of `prefill()` already exists; the Qwen3 path needs q/k norm without the output gate (`mul_sigmoid_rows` skipped).
5. **fp16 KV cache → int8 / TurboQuant.** The Qwen3 path keeps an fp16 cache: 33.5 MiB per layer at 8K, which is 670 MiB per console. With int8 v (the Qwen3.5 layout), 16K fits in the same memory.
6. Shard, deploy, then `cluster_bench.py --bench`.

## Commands

```bash
python pc/cluster_plan.py --model qwen3-14b --nodes 2 --context 8192
python pc/shard_model.py cluster/profiles/qwen3.5-9b-2x-test.json --idx chat/assets_9b/model.idx \
       --bin models/qwen3.5-9b-q4s.bin --out models/cluster-9b-2x
python pc/cluster_deploy.py cluster/profiles/qwen3.5-9b-2x-test.json models/cluster-9b-2x --hosts <head> <worker>
python pc/cluster_bench.py --compare http://<single-console>:8080 http://<head>:8080
```
