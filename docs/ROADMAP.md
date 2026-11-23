# DarkGarlic roadmap

Plans to make DarkGarlic more powerful, more useful and easier to adapt. Status labels:
**done**, **in this branch** (written, not yet tested on a PS4), **next**, **later**.

The branches:

| Branch | What it is for |
|---|---|
| `darkgarlic` | The shipping app: abliterated Qwen3.5-9B, OpenAI-compatible API. |
| `darkgarlic-cluster` | Base for larger models: the model-override folder, cluster shards and protocol, planner and tools (docs/CLUSTER.md). |
| `darkgarlic-cluster-2x` | Two consoles: profile and plan for Qwen3-14B, plus the 9B two-way test rig. |
| `darkgarlic-cluster-3x` | Three consoles: profile and plan for Qwen3.5-27B (abliterated). |

---

## 1. Adaptability: any model, any number of consoles

| Item | Status | Notes |
|---|---|---|
| Model override folder `/data/darkgarlic/` | **in this branch** | When `model.idx` is there, the engine uses it and the weights it names instead of the package's. You can swap models without rebuilding the 5.6 GB package. The MTP head is also read from there, never mixed with the package's. A weights file whose name matches the packaged model is not redirected to `/app0/models`. |
| Model name from the model | **in this branch** | The screen and API name comes from `cluster.txt`'s `name =`, then the shard's name, then the old guess from the model's shape. |
| Planner for any Qwen3 / Qwen3.5 dense `config.json` | **in this branch** | `pc/cluster_plan.py --config <config.json> --nodes N`: memory per console, fit, speed estimate, and which kernel limits the shape hits. |
| Cluster shards + protocol + tools | **in this branch** | docs/CLUSTER.md. |
| A generic converter `pc/convert_q4s.py` | **next** | `convert_9b_q4s.py` hard-codes the 9B (file names, the 32 layers of `Q3_LAYERS`, HF repo). One script should take `--hf <dir> --quant default/mlp3/all4` for any Qwen3.5 dense model, and a Qwen3 mode (§3). The planner already has the byte maths. |
| One `kernels.bin` for all shapes | **later** | `HEAD_DIM`, `ROPE_DIM` and `MAX_SEQ` are compile-time in `qwen35.cl`. That is fine for the Qwen3.5 family (all head_dim 256), but a model-specific `kernels.bin` in `/data/darkgarlic` should be allowed too. |

## 2. Kernel work for bigger models

| Item | Needed by | Status | Notes |
|---|---|---|---|
| GQA ratio > 4 (`GQ_MAX` in `shaders/attn.h`, the shape check in `engine_init`) | 27B (24/4 = 6), 14B (40/8 = 5) | **next** | `attn_part` keeps `GQ_MAX × HEAD_DIM` partials in LDS: 6 × 256 × 4 waves × 4 B = 24 KiB for `acc`, about 34 KiB in total. That is under GCN's 64 KiB, but fewer workgroups fit per CU. Benchmark both a `GQ_MAX 6` build and a two-pass variant that handles 3 query heads per pass. The TurboQuant `attn_tq.h` needs the same change. |
| hidden 5120 and 48 DeltaNet value heads | 27B | **next (check)** | The kernels take `n_kh` and `n_vh` as arguments (`kh = h / (n_vh / n_kh)` works for 48/16 = 3). Check that `delta_rows`, `conv_silu_rows` and the SWAR split-K (`SWAR_SPLIT_ROWS 12416`) handle the 27B's 16 480-row DeltaNet projection (padded to 16 512) and the 17 408 × 5120 MLP. Then test against the PC reference on one console with 2–4 layers of the real 27B. |
| Qwen3 (not 3.5) q4s/q3s converter + batched prompt | 14B | **later** | The engine's Qwen3 path (`PMDL`) is the old 4B one: q4/q4i, no SWAR, no batched prompt (`e->pf` requires `qwen35`), and an fp16 KV cache. It needs a converter plus the batched prompt for full attention only. |
| Mixture-of-experts layers | Qwen3.5-35B-A3B, 122B-A10B | **later** | The best fit for a cluster: only ~3B parameters are read per token. A 35B-A3B over 4–5 consoles could run near today's 9B speed with a much stronger model. It needs a router kernel and expert matvecs, and experts can be split across consoles (expert parallelism). |

## 3. Speed

| Item | Status | Notes |
|---|---|---|
| PS4 Pro native mode | upstream 1.20 (experimental) | Users report up to 56 tokens/s. A Pro as the last node of a cluster helps everyone. |
| MTP head on the last cluster node | **later** | Speculative decoding is about +50 % on the 9B. In a cluster the draft and verify both happen on the last node. That node needs the embedding rows (another ~0.7 GB from disk, read-only) and a CL op that verifies k drafted tokens in one RUN. |
| Prompt-lookup decoding | **next** | Drafts come from n-grams already in the context. No memory, no model. It works well for code edits, summaries and RAG, stacks with MTP, and it is the only speculation available in a cluster until the MTP head moves. |
| Batched serving (several chats per pass) | **later** | Decoding is limited by memory bandwidth, so 2–4 concurrent API requests cost nearly the same per pass as one. Total throughput goes up roughly with batch size. In a cluster this is also how all the consoles stay busy at once. |
| Pipelined prompt in the cluster | **later** | Today the head waits for each worker in turn. With two messages in flight, node 0 can work on batch i+1 while node 1 does batch i, giving up to N× faster prompt processing on N consoles. |
| Onion (coherent) transfer buffers | **later** | The CPU reads relayed rows out of write-combined Garlic memory. A small Onion buffer for `ostage` and the last `pf_x` copy would make that read cheap. |

## 4. Usefulness

| Item | Status | Notes |
|---|---|---|
| OpenAI-compatible API | **done** (`darkgarlic`) | `/v1/chat/completions` with streaming, `/v1/models`, `/health`, optional API key. |
| Ollama-compatible API | **in this branch** | `/api/chat`, `/api/generate`, `/api/tags`, `/api/show` and `/api/version` on port 8080, the same socket as the OpenAI API. Point a client at `OLLAMA_HOST=http://<ps4>:8080`. Streaming is NDJSON and `stream` defaults to true. `options.num_predict` of -1 is capped at 4096. Port 11434 is not opened: the console accepts on one blocking socket. |
| Tool / function calling in the API | **in this branch** | Qwen3.5 XML tool calls on port 8080. A request with `tools` renders the chat template and the reply returns OpenAI `tool_calls` (`finish_reason` `tool_calls`) or Ollama `tool_calls` with arguments as an object. The tool XML is not streamed. |
| Anthropic-compatible `/v1/messages` | **later** | A thin mapping on top of the same job queue. |
| Embeddings endpoint | **later** | The last hidden row, mean-pooled. That's enough for small local RAG. |
| Documents from USB (RAG) | **later** | Index text files from `/mnt/usb0` and put the relevant passages into the prompt. |
| Save and restore conversations | **later** | DeltaNet states plus the KV cache to `/data` (the 9B's state is small). Long system prompts are prefilled once. |
| Repetition control for the abliterated weights | **next** | The abliterated 9B loops more than the original (DARKGARLIC.md). Add a presence/frequency penalty to the `sample` kernel and expose it in the API and on L2/R2. |
| Temperature and fan readout | **later** | Long API sessions keep the GPU busy for hours. |

## 5. Robustness (cluster)

| Item | Status | Notes |
|---|---|---|
| Worker reconnect | **later** | The head shows an error screen if a worker drops. It should reconnect, then RESET and replay the conversation. |
| Model fingerprint in HELLO | **in this branch** | HELLO is 68 bytes: the previous 64, plus a hash of `n_layers`, `hidden`, `vocab` and each tensor's rows, cols and fmt. Offsets and file sizes are not hashed, so every shard of one conversion matches. A 64-byte HELLO (fingerprint 0) still chains. |
| Layer kind past layer 31 | **in this branch** | `full_mask` is 32 bits. Qwen3 (every layer full) and Qwen3.5 (full every 4th layer) stay correct through layer 63. A shift of 32 or more is undefined and is not used. The sharder uses the same rule. |
| Head-side status on screen | **later** | Show each worker's node, layers and round-trip time in the chat header next to the API address. |
