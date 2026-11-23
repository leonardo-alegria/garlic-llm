# Brief for promoting DarkGarlic LLM

Hand this file to the account that will post. Use only the facts in this file. If a fact is missing, say it is missing. Do not invent a download URL, a benchmark, a quote, or a screenshot.

## One-sentence pitch

DarkGarlic LLM is a fork of Leonardo Alegría's Garlic LLM that packs Huihui's abliterated Qwen3.5-9B into one PS4 package, so a GoldHEN console shows "Abliterated Qwen3.5-9B on your PS4's GPU."

## What this is

Garlic LLM is an open-source homebrew app by Leonardo Alegría. It runs a 9-billion-parameter language model on the GPU of a PlayStation 4, PS4 Slim, or PS4 Pro that already has GoldHEN. The app contains no jailbreak. The engine, the GPU kernels, and the package format are his. Source upstream: https://github.com/leonardo-alegria/garlic-llm. Site: https://garlicllm.dev.

DarkGarlic LLM is a fork of that app. It keeps his engine and swaps in an abliterated (refusal-reduced) build of the same Qwen3.5-9B architecture, then restyles the app red, black, and white. The mark is a white garlic in a black Dracula cape with a red lining and two fangs. The home-screen title is DarkGarlic LLM.

The on-screen line, on the empty chat screen, is exactly:

Abliterated Qwen3.5-9B on your PS4's GPU.

The OpenAI-compatible API reports the same model id: `Abliterated Qwen3.5-9B`.

## Who to credit

- Engine, kernels, original app, speed figures below: Leonardo Alegría.
- Abliterated weights: huihui-ai, checkpoint `huihui-ai/Huihui-Qwen3.5-9B-abliterated`, Apache-2.0, based on `Qwen/Qwen3.5-9B` by Alibaba Qwen.
- This fork's changes: the DarkGarlic name and artwork, the converters pointed at that checkpoint, and the all-in-one package that contains those converted weights.
- App license: GNU GPLv3. Source of this fork: https://github.com/bemor187/garlic-llm/tree/darkgarlic

Do not say this fork wrote the PS4 GPU path, invented running a language model on a PS4, or trained the model.

## The file people install

| | |
|---|---|
| Local path | `D:\garlic-build\DarkGarlic-LLM-1.20-full.pkg` |
| Also at | `C:\Users\NewAdmin\garlic-llm\build\chat\IV0000-PSAI00001_00-PS4LOCALAICHAT01.pkg` |
| Bytes | 5,662,638,080 |
| SHA-256 | `c387e47009a2a2aeb1a36bf1952ec044a44af7c55751e33e02e6c4000a67bdd3` |
| Title | DarkGarlic LLM |
| Version | 1.20 |
| Title id | `PSAI00001` (installs over Garlic LLM) |
| Inside | `qwen3.5-9b-q4s.bin`, `qwen3.5-9b-mtp.bin`, `parakeet-v3-q4i.bin`, `chatterbox-t3.bin`, `chatterbox-s3gen-mf.bin` |

Public download: https://huggingface.co/bemor187/darkgarlic-llm-ps4

Direct file: https://huggingface.co/bemor187/darkgarlic-llm-ps4/resolve/main/DarkGarlic-LLM-1.20-full.pkg

The repo is public. GitHub will not host this file. Do not link the GitHub release as the download. That release, https://github.com/bemor187/garlic-llm/releases/tag/darkgarlic-1.20, is a 41,091,072-byte package with no weights (SHA-256 `0354ce449ba522d4c590f0c781e0f877eaa2246c88ee227ec4831c5e1379af32`). Installing it leaves the console without the model.

Loose weights, already inside the full package:

| File | Bytes | SHA-256 |
|---|---:|---|
| `D:\garlic-build\models\qwen3.5-9b-q4s.bin` | 4,641,752,088 | `4e15041f0ded0ccb2cfd55c24644bae3947b35af9db7a39809c6a9ddea1b2b31` |
| `D:\garlic-build\models\qwen3.5-9b-mtp.bin` | 212,420,608 | `881482286945e0b3252a5dda58f4e7a2f22f41b1733b2aff0bdb408a8c2987fc` |

## Install, once the package is public

1. PS4, PS4 Slim, or PS4 Pro already running GoldHEN.
2. Copy the pkg to an exFAT USB drive. FAT32 cannot hold it.
3. On the console: Settings → Debug Settings → Game → Package Installer.
4. About 10 GB free if the console copies the package to the internal disk before installing.
5. Open DarkGarlic LLM from the home screen. The model is inside the package. Nothing else has to be copied to `/data` for text chat.
6. Press L3 (left-stick click, stick and D-pad at rest). The header shows `API · <console-ip>:8080`.
7. From another device on the same network: `GET http://<console-ip>:8080/health` returns `{"status":"ok"}`. Chat is `POST http://<console-ip>:8080/v1/chat/completions`.

## What the package does, and what it leaves out

- Text chat and the OpenAI-compatible API, offline, on the console GPU.
- Speculative decoding with the converted MTP head.
- Dictation and spoken replies are in the package: Parakeet TDT 0.6B v3 and Chatterbox, the same files as upstream Garlic LLM. Hold square to talk.
- The owner has run this build and says it is fully functional. This brief has no new tokens-per-second measurement for the abliterated weights.

Upstream Garlic LLM, measured by Leonardo Alegría on the original Qwen weights, reports about 31–39 tokens/s in chat and up to 43 tokens/s on code, on a base or Slim PS4. Treat those as his figures for his build. The quant layout matches, so the speed should be in that neighborhood, but do not present them as a new DarkGarlic benchmark.

A cold load of an abliterated conversion of this format was about 45 seconds on a base PS4 (CUH-1001A, firmware 13.52, GoldHEN 2.4b18.13), reported on upstream issue 9: https://github.com/leonardo-alegria/garlic-llm/issues/9

## Why post it now

Garlic LLM has been in the press since 16 November 2026. Tom's Hardware, TweakTown, and others covered Leonardo's release. Those threads are still being read. The hook that fits them is one sentence those articles do not have: the same console package, with the abliterated weights, and the screen line above.

Reply where that coverage already is. One reply per thread, with the same text. A second post in the same thread reads as spam.

- Tom's Hardware comments: https://www.tomshardware.com/video-games/playstation/jailbroken-ps4-runs-9-billion-parameter-ai-model-at-up-to-43-tokens-per-second-open-source-garlic-llm-bypasses-drivers-to-talk-directly-to-the-2013-era-gpu
- r/LocalLLaMA and r/ps4homebrew, and Leonardo's own Reddit post about Garlic LLM if it is still open.
- X posts still circulating the story, including https://x.com/TeksEdge/status/2122502480919396559 and https://x.com/ITConnect_fr/status/2123831495630463357

Attach a photo of the TV showing the line "Abliterated Qwen3.5-9B on your PS4's GPU." Do not attach a fake screenshot.

## Paste-ready reply

> DarkGarlic is a fork of Leonardo Alegría's Garlic LLM. Same PS4 package format, with Huihui's abliterated Qwen3.5-9B packed inside, plus Parakeet dictation and Chatterbox speech. The screen says "Abliterated Qwen3.5-9B on your PS4's GPU." One pkg, GoldHEN console, Package Installer: https://huggingface.co/bemor187/darkgarlic-llm-ps4/resolve/main/DarkGarlic-LLM-1.20-full.pkg. Source: https://github.com/bemor187/garlic-llm/tree/darkgarlic. SHA-256 c387e47009a2a2aeb1a36bf1952ec044a44af7c55751e33e02e6c4000a67bdd3

## Rules for the posts

- Name Leonardo Alegría in the first or second sentence.
- Link the fork: https://github.com/bemor187/garlic-llm/tree/darkgarlic
- Include the SHA-256 of the full package.
- Say GoldHEN is required and that DarkGarlic does not contain an exploit.
- Say the weights are Huihui's abliteration of Qwen3.5-9B, Apache-2.0.
- Say voice is included: hold square to dictate, Chatterbox speaks the reply in seven languages.
- Say the model can answer things the original Qwen build refuses, because the refusal direction was removed from the weights. Do not demonstrate that with a criminal, exploit, or self-harm request.
- Do not promise a tokens-per-second number for this build.
- Do not tell anyone how to jailbreak a console.
- Do not open a pull request that replaces the branding on leonardo-alegria/garlic-llm.
