# DarkGarlic LLM

A fork of [Garlic LLM](https://github.com/leonardo-alegria/garlic-llm) by Leonardo Alegría. The app code is GPLv3. The converted checkpoint is [`huihui-ai/Huihui-Qwen3.5-9B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated) (Apache-2.0), the same `Qwen3_5ForConditionalGeneration` layout as `Qwen/Qwen3.5-9B`: 32 layers, hidden size 4096, vocab 248320, one MTP layer.

The screen is black, white and red. The mark is a white garlic in a black cape with a red lining and two fangs. The home-screen title is **DarkGarlic LLM**. The title id is still `PSAI00001`, so this package installs over Garlic LLM.

Future plans: [docs/FUTURE_PLANS.md](docs/FUTURE_PLANS.md) (Stage 2: multi-PS4 clusters, bigger models, speed and API features, once funded).

## Weights

Git does not contain the weights (`models/` is ignored, and GitHub will not take a 4.6 GB file). A converted pair that matches the `model.idx` shipped in this branch:

| File | Bytes | SHA-256 |
|---|---:|---|
| `qwen3.5-9b-q4s.bin` | 4,641,752,088 | `4e15041f0ded0ccb2cfd55c24644bae3947b35af9db7a39809c6a9ddea1b2b31` |
| `qwen3.5-9b-mtp.bin` | 212,420,608 | `881482286945e0b3252a5dda58f4e7a2f22f41b1733b2aff0bdb408a8c2987fc` |

Gate and up projections are 3-bit on all 32 layers. The output head keeps 248,070 token rows, the same `pc/vocab_order.npy` as upstream, which is why `chat/assets_9b/model.idx` and `mtp.idx` did not have to change.

Rebuild them:

```bash
hf download huihui-ai/Huihui-Qwen3.5-9B-abliterated --local-dir hf/Huihui-Qwen3.5-9B-abliterated
python pc/local_qwen35_9b.py --snapshot hf/Huihui-Qwen3.5-9B-abliterated
mkdir models\Huihui-Qwen3.5-9B-abliterated
copy /Y hf\Huihui-Qwen3.5-9B-abliterated\model.safetensors.index.json models\Huihui-Qwen3.5-9B-abliterated\
python pc/convert_9b_q4s.py
python pc/convert_9b_q4s.py --mtp
```

`local_qwen35_9b.py` needs `numpy` only. It reads BF16 and the few F32 tensors (the `A_log` vectors) straight out of the safetensors shards and quantizes in row chunks, so it does not need the 64 GB Modal machine. Do not pass `--q3=none` to `convert_9b_q4s.py`: that makes a larger file than the console has room for. Do not start from a GGUF.

## Install

`python build_chat.py --full` puts the chat weights and, when they are in `models/`, the three voice files inside the package at `/app0/models`. The current all-in-one package includes Parakeet and Chatterbox. Nothing else has to be copied to `/data`.

A build of this branch with the abliterated 9B weights inside produced:

| File | Bytes | SHA-256 |
|---|---:|---|
| `IV0000-PSAI00001_00-PS4LOCALAICHAT01.pkg` | 5,662,638,080 | `c387e47009a2a2aeb1a36bf1952ec044a44af7c55751e33e02e6c4000a67bdd3` |

Public download: https://huggingface.co/bemor187/darkgarlic-llm-ps4

Direct file: https://huggingface.co/bemor187/darkgarlic-llm-ps4/resolve/main/DarkGarlic-LLM-1.20-full.pkg

That file is too large for a GitHub release. The release asset `DarkGarlic-LLM-1.20.pkg` (41,091,072 bytes, SHA-256 `0354ce449ba522d4c590f0c781e0f877eaa2246c88ee227ec4831c5e1379af32`) is the same app with no weights, for people who upload the bins to `/data` themselves.

Title `DarkGarlic LLM`, version 1.20, title id `PSAI00001`. Install from an exFAT USB drive with *Settings → Debug Settings → Game → Package Installer* on a console that already runs GoldHEN. The all-in-one package needs about 10 GB free if the console copies it to the internal disk before installing.

```bash
set PS4_IP=192.168.1.5
python pc/upload_model.py models/qwen3.5-9b-q4s.bin models/qwen3.5-9b-mtp.bin
```

The upstream all-in-one package will not load these files. `chat/engine.c` opens `/app0/models/<name>` first and only then `/data`. Copying the bins to `/data` on a console that already has the official 1.20 package leaves you on the original Qwen weights.

The published package also contains `parakeet-v3-q4i.bin` (355,340,032 bytes), `chatterbox-t3.bin` (308,844,544 bytes), and `chatterbox-s3gen-mf.bin` (99,145,088 bytes), copied from the upstream 1.20 package. Their sizes match `asr.idx`, `t3.idx`, and `s3g.idx`. These scripts do not convert Parakeet or Chatterbox. A `--full` build packs whatever of those three files is already in `models/`.

Press L3 on the chat screen. The header shows `API · <console-ip>:8080`. `GET /health` returns `{"status":"ok"}`. The on-screen model line is `Abliterated Qwen3.5-9B on your PS4's GPU.` The API model id is `Abliterated Qwen3.5-9B`.

## Known limits

- The OpenOrbis toolchain, Zig and the .NET runtime are not in this repository. `python build_chat.py` cannot run until `toolchain/` is filled in as the README describes.
- The abliterated weights drop the refusal direction. They also loop more often than the original. L2 and R2 change the temperature.
- A cold load on a base PS4 is on the order of 45 seconds. That figure is from upstream issue 9, which ran an abliterated conversion of this same format on firmware 13.52.
- `/data/ps4ai_log.txt` should contain `weights: /data/qwen3.5-9b-q4s.bin` for a non-packed build. `Couldn't start` means a weight file is missing or short.
- Russian speech turns the word `DarkGarlic` into `Дарк Гарлик` so the voice model does not spell the Latin name.

## Bigger models and clusters (branch `darkgarlic-cluster`)

This branch adds:

- **A model override folder.** `/data/darkgarlic/model.idx` and the weights it names replace the packaged model, with no rebuild.
- **An experimental cluster.** One model is split by layers across 2 or more PS4s on the same network. The head keeps the screen, the API and the first layers; the workers run the rest.

The design, the shard format and the bring-up plan are in [docs/CLUSTER.md](docs/CLUSTER.md). The plans for improvements are in [docs/ROADMAP.md](docs/ROADMAP.md). The 2-console and 3-console setups live on `darkgarlic-cluster-2x` and `darkgarlic-cluster-3x`.

```bash
python pc/cluster_plan.py --list                                    # what fits on 1-4 consoles
python pc/cluster_plan.py --model qwen3.5-27b --nodes 3 --quant mlp3
gcc -std=c99 -Wall -Wextra -I chat tests/cluster_proto_test.c chat/cluster_proto.c -o build/cluster_proto_test && build/cluster_proto_test
python pc/shard_model.py --self-test
```
