# DarkGarlic cluster: one model across several PS4s

**Status: experimental. The code is written, but it hasn't run on hardware yet.** The protocol, the shard tool and the planner pass their host tests. The engine and network code compile against stub headers on a PC. None of it has run on a PS4. Milestone M1 below is the first real test.

## Why a cluster, and what it does not do

One PS4 has 8 GB of GDDR5, shared by the CPU and the GPU. DarkGarlic gets about 4.5 GiB of it for the model. That is enough for the 9B and nothing bigger. Clustering adds **room**, so a bigger model fits. It does **not** make a model faster.

Generating a token reads every weight once, and memory bandwidth is the limit. With the model split across consoles, each token still has to pass through every console in turn. A 27B model on three PS4s reads three times the bytes the 9B reads, so it runs at about a third of the 9B's speed.

What you gain is quality: Qwen3.5-27B against 9B. Speed only goes up when several chats run at once (see the batching item in ROADMAP.md).

## How it works: a pipeline over a star

```
             cluster.txt                     shard: layers 23..44           shard: layers 45..63 + lm_head
  +--------------------------+  rows  +----------------------+     +----------------------------+
  | node 0  (head)           |------->| node 1  (worker)     |     | node 2  (worker, last)     |
  | screen, pad, API :8080   |<-------|                      |     |                            |
  | tokenizer, embed rows    |  rows  +----------------------+     |                            |
  | layers 0..22             |-------------------------------------->  final norm, lm_head,     |
  |                          |<---------------------------------------  sampling -> token row   |
  +--------------------------+               token                 +----------------------------+
```

- **Pipeline parallelism.** Each console holds a contiguous range of layers. Between consoles, the only thing that travels is the hidden row of each token: `hidden` fp32 values, which is 16 KiB for the 9B and 20 KiB for the 27B. Gigabit Ethernet moves that in about 0.2 ms.
- **A star, not a ring.** The head opens one TCP connection to each worker (port 8091). It relays the rows: it sends them to node 1, reads node 1's output, sends that to node 2, and so on. The last node returns the sampled lm_head row, which the head turns into a token id. Workers never talk to each other and need no settings: their shard says which node they are.
- **No changes to the chat or the API code.** `engine_run` on the head runs its own layers, then hands the rows to `EngineLink.rows` (`cluster.c`), which drives the workers. Every caller in `main.c` and `api_run.inc` works unchanged. `engine_reset` and `engine_state_copy` (used by Think mode) are forwarded the same way.
- **Prompts stay batched.** A prompt goes through in the same batches as on one console: 32 rows, or 64 on a Pro. Each batch is relayed as one message.

### The pieces

| File | What it does |
|---|---|
| `pc/cluster_plan.py` | Sizes every tensor from a preset or a Hugging Face `config.json`, checks the shape against the kernels, and splits the layers so each console fits. Writes a profile JSON. Its 9B estimate is within 0.1 MiB of the real `qwen3.5-9b-q4s.bin` GPU part (3880 MiB). |
| `pc/shard_model.py` | Splits a converted `model.idx` + `.bin` into one shard per console, using a profile. `--self-test` builds a fake model and checks every byte. |
| `pc/cluster_deploy.py` | Copies the shards to the consoles over GoldHEN FTP and writes the head's `cluster.txt`. `--remove` puts every console back on the packaged model. |
| `pc/cluster_bench.py` | `--compare` sends the same prompts, greedy, to a single console and to a cluster; the replies must match. `--bench` reports time to first token and tokens per second. |
| `chat/cluster_proto.[ch]` | Wire protocol, `cluster.txt` parser and chain check. Plain C99; `tests/cluster_proto_test.c` tests it. |
| `chat/cluster.[ch]` | Sockets, the head's link and the worker's loop. |
| `chat/engine.[ch]` | Reads the shard trailer, keeps caches only for local layers, takes rows in and hands rows out, and runs `engine_run_rows` for workers. |

### Shard format

`pc/shard_model.py` writes, for each console:

- `/data/darkgarlic/node<k>.bin`, laid out as `[GPU part: final_norm, this node's layers, lm_head on the last node][embedding rows on node 0][ids]`.
- `/data/darkgarlic/model.idx`. It keeps every tensor record in the original order, because the engine finds a layer's tensors by position. Tensors held by another console get offset 0 and size 0; their shape and format stay. After the records comes the trailer: `"PSHD", version 1, node, nodes, layer_first, layer_end, flags (1 embed, 2 lm_head), name[32]`.

If `/data/darkgarlic/model.idx` exists, the engine uses it instead of the packaged one, and it keeps the weights path that index names (it is not replaced by `/app0/models/<same file name>`). Without the trailer it is treated as a whole model. That also lets you try any other converted model without rebuilding the package.

### Wire protocol (`cluster_proto.h`)

Every message is a 40-byte little-endian header followed by a payload. The header holds `magic "DGCL", version, op, seq, pos0, n, flags, temperature, seed, payload, check`. `check` is the FNV-1a hash of the payload, so a framing bug produces an error message instead of a wrong answer.

| Op | Direction | Payload |
|---|---|---|
| `HELLO` | head → worker → head | 68-byte `ClusterHello`: node, nodes, layer range, model shape, name, fingerprint. A 64-byte reply (no fingerprint) is still accepted. |
| `RUN` | head → worker | `n` rows × `hidden` fp32 for positions `pos0..`; `flags & SAMPLE` asks for the token |
| `ROWS` | middle worker → head | the worker's `n` output rows |
| `TOKEN` | last worker → head | the u32 lm_head row (nothing if no token was asked for) |
| `RESET`, `STATE` | head → workers | clear, save or restore the DeltaNet states; the answer is `OK` |
| `ERROR` | worker → head | a message, shown on the head's error screen |

## Which models, on how many consoles

These are planner numbers (`python pc/cluster_plan.py --list`) for base PS4s with 16K context. Speeds are single-chat estimates **before** speculative decoding. The planner gives the 9B about 25 tokens/s; the measured figure without MTP is about 22, and 31–39 with MTP.

| Setup | Model | Weights | Free per console | Est. tokens/s | Kernel work first |
|---|---|---:|---:|---:|---|
| 1× (today) | Qwen3.5-9B | 3880 MiB | 355 MiB | ~25 (31–39 with MTP) | none |
| **2× test rig** | Qwen3.5-9B split 19 + 13 | 3880 MiB | ~2.4 GiB | ~25 | none: this is milestone M1 |
| **2×** (branch `darkgarlic-cluster-2x`) | Qwen3-14B, 8K context | 6656 MiB | ~460–500 MiB | ~14 | GQA 5 (`GQ_MAX`), a Qwen3 q4s/q3s converter |
| **3×** (branch `darkgarlic-cluster-3x`) | Qwen3.5-27B, `mlp3` quant | 11719 MiB | 320–485 MiB | ~8 (≈11–13 with MTP) | GQA 6 (`GQ_MAX`), hidden 5120 / 48 DeltaNet value heads in the converter |
| 3× | Qwen3.5-27B, default quant | 12399 MiB | 75 MiB on node 0 | ~8 | the same, but too tight to recommend |
| 2× | Qwen3.5-27B | — | does not fit | — | it would need ~2-bit weights |

`mlp3` puts the down projection in 3 bits as well as gate and up. That saves 680 MiB across the three consoles, and it is what makes the 27B comfortable. Check the quality cost with the PC reference before shipping it.

## Bring-up plan

**M0: host checks (done).** `tests/cluster_proto_test.c` (43 checks), `tests/layer_mask_test.c`, `tests/api_proto_test.c`, `tests/api_run_test.c`, `python pc/shard_model.py --self-test`, and the planner's 9B size matching the real file.

**M1: the 9B split over two consoles answers exactly like one console.**

1. Build the package from this branch (`python build_chat.py`, no `--full` needed) and install it on both consoles.
2. `python pc/shard_model.py cluster/profiles/qwen3.5-9b-2x-test.json --idx chat/assets_9b/model.idx --bin models/qwen3.5-9b-q4s.bin --out models/cluster-9b-2x`
3. `python pc/cluster_deploy.py cluster/profiles/qwen3.5-9b-2x-test.json models/cluster-9b-2x --hosts <head-ip> <worker-ip>`
4. Open DarkGarlic on the worker. It shows `waiting for the head on port 8091`. Then open it on the head, which connects and shows the chat.
5. Turn the API on (L3) on the head and on a third, single console with the normal package. Then run `python pc/cluster_bench.py --compare http://<single>:8080 http://<head>:8080`.

   The replies should match. A difference late in a long reply can come from `swar_tune` picking a different, equally valid kernel variant per console, which changes float rounding. A difference in the **first** tokens is a bug.
6. Check `/data/ps4ai_log.txt` on both consoles (FTP). Look for the `cluster:` lines, the GPU memory used, and errors.

**M2: kernels for the bigger shapes.** See ROADMAP.md: `GQ_MAX` 6, the 27B converter, the Qwen3 converter. Test each against the PC reference (`pc/qwen3_ref.py`) on one console with a short model, before sharding.

**M3: the 3× 27B (or 2× 14B) for real.** Shard, deploy, `--bench`, and long conversations.

**M4: speed.** Batched serving for the API (several chats in one pass), overlapping consoles during the prompt, and the MTP head on the last node.

## Known limits of this first version

- Speculative decoding (the MTP head) is off in a cluster. It needs the last hidden row and the embedding on the same console, so it belongs on the last node, which also needs the embedding rows (ROADMAP).
- If a worker drops mid-reply, the head shows an error screen. Restart the head. Reconnecting is on the roadmap.
- Each relayed message is copied once out of write-combined GPU memory and hashed (FNV-1a). That costs ~1–3 ms per 32-row prompt batch and nothing noticeable per decoded token.
- Voice mode on the head is untested in a cluster. It should have more room than on a single console, because the head holds fewer weights.
- The head connects to workers in `cluster.txt` order. Start the workers first. A worker address may end in `:8091`. Any other port is rejected: workers listen only on 8091.
- The head refuses a model whose hidden size is above 8192, and a shard whose local layers are missing weights. HELLO carries a fingerprint of the tensor shapes; mixed conversions are rejected when both sides send one.
