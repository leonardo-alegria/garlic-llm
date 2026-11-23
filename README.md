<p align="center"><img src="docs/logo.svg" width="112" alt="DarkGarlic LLM logo: a fanged garlic in a Dracula cape"></p>

# DarkGarlic LLM

This tree is a fork of [leonardo-alegria/garlic-llm](https://github.com/leonardo-alegria/garlic-llm). The engine, the kernels and the package format are his. This branch restyles the app and points the 9B converters at [`huihui-ai/Huihui-Qwen3.5-9B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-9B-abliterated), which has the same architecture as `Qwen/Qwen3.5-9B`. How to build the weights and what does not ship in git is in [DARKGARLIC.md](DARKGARLIC.md).

**A 9-billion-parameter language model running on a PS4. Offline.**

DarkGarlic LLM is a homebrew app for the PlayStation 4. It runs [Qwen3.5-9B](https://huggingface.co/Qwen) on the console's own GPU, with no PC and no internet while it runs. It also listens, talks, and runs the Python it writes.

[Website](https://garlicllm.dev) · [Download](#install) · [Demo video](https://garlicllm.dev/#demo) · [Support the project](#support)

> 🙏 **DarkGarlic needs hardware to reach Stage 2** (bigger models across several PS4s): PS4 Pros, Dell PowerEdge servers, compute, or any amount through Cash App [$bemor187](https://cash.app/$bemor187) or PayPal [kultivatedkustoms@gmail.com](https://www.paypal.com/donate/?business=kultivatedkustoms%40gmail.com&currency_code=USD). [How to help](#help-darkgarlic-grow-).

![DarkGarlic LLM: a fanged garlic in a cape on a black and red field](docs/chat.png)

## What it does

| Feature | Details |
|---|---|
| **Text chat** | Qwen3.5-9B at 31–39 tokens/s, and up to 43 tokens/s on code. 16K-token context. |
| **Voice conversation** | Speak and it answers aloud (Chatterbox) in Spanish, English, Portuguese, Russian, Italian, French and German. |
| **Dictation** | Hold a button and talk instead of typing (Parakeet TDT 0.6B v3), in 25 European languages. |
| **Python mode** | Ask for a program. It writes MicroPython and the console runs it. |
| **Think mode** | The model reasons before it answers. Off by default; a thought takes about half a minute. |

Everything runs on the console. Nothing is sent anywhere.

Speeds were measured on a base/Slim PS4. On a PS4 Pro, chat runs at up to 45 tokens/s, and a native PS4 Pro mode reaches up to 60 tokens/s; that mode comes with the [experimental version 1.20](https://garlicllm.dev/download/#h-exp). In voice mode the chat model runs at about 22 tokens/s, because the voice model takes the memory that speculative decoding uses.

## Requirements

- A PS4, PS4 Slim or PS4 Pro running [GoldHEN](https://github.com/GoldHEN/GoldHEN). It works on firmware 9.00, 9.04, 11.00, 11.52, 12.02, 12.50, 13.52 and 14.00.
- About 6 GB of free space, or 12 GB if you copy the package to the console's own disk before installing it.
- To use your voice: any ordinary microphone the console recognises.
- To type on a real keyboard: any USB keyboard. Press OPTIONS in the chat to open the console's keyboard.

Other firmware versions are not tested yet. If you try one, please [open an issue](https://github.com/leonardo-alegria/garlic-llm/issues) with the result.

## Install

1. Download the all-in-one package (about 5.7 GB, models inside) from [Hugging Face](https://huggingface.co/garlic-llm/garlic-llm-ps4).
2. Copy the PKG to the console and install it like any other homebrew package (*Settings → Debug Settings → Game → Package Installer*).
3. Open **DarkGarlic LLM** from the home screen. The first screen loads the model; after that it needs no PC and no connection.

**Updating.** Download the new package and install it over the old one. The models are inside, so there is nothing else to copy.

## Controls

**Chat**

| Button | Action |
|---|---|
| ✕ | Type a message |
| OPTIONS | Type with the console's keyboard, which also takes a USB keyboard |
| □ (hold) | Dictate a message; release to finish |
| ○ | Stop the answer |
| △ | New chat |
| ↑ ↓ or left stick | Scroll |
| L1 | Voice mode |
| R1 | Run the Python program in the last answer |
| R3 or touchpad | Think mode on / off |
| L3 | [API](#api) on / off |
| L2 / R2 | Temperature down / up |

After dictating, choose what to do with the text: ✕ send, △ edit, □ record again, ○ discard.

**Keyboard**

| Button | Action |
|---|---|
| ✕ | Type the selected key |
| □ | Delete (hold to repeat) |
| △ | Space |
| L1 | Shift |
| R1 | Emoji |
| L2 (hold) | Dictate and add the text |
| R2 | Send |
| ○ | Close |

**Voice mode:** hold □ to talk (this also interrupts the answer), △ starts a new chat, ○ leaves voice mode.

The interface follows the console's language: English, Spanish, Portuguese or Russian.

## How it works

A PS4 gives homebrew no compute API and no machine-learning library, so DarkGarlic LLM talks to the GPU directly.

- **Compute kernels.** The model's operations (matrix products over 4-bit weights, attention, normalisation) are OpenCL C kernels compiled by LLVM for the console's GCN architecture. They are in [`shaders/`](shaders/).
- **Hand-written command buffers.** The app dispatches those kernels by writing PM4 packets itself, the low-level commands a graphics driver would normally generate.
- **Speculative decoding.** Qwen3.5's own MTP head drafts up to two tokens ahead, and the full model verifies them in one pass.
- **Custom Q4 and Q3 weights.** The 9B is stored in two formats made for these kernels. Q4 holds 4-bit weights and is used for most matrices and the output head. Q3 holds 3-bit weights (about 3.5 bits each with their scales) and is used for the two largest matrices of each layer, the MLP's gate and up projections. Storing everything in Q4 would take about 384 MiB more, which the console does not have free.
- **4-bit attention cache.** The attention cache is quantized to 4 bits too, which is how a 16K context fits.
- **GPU-visible memory.** The weights live in "Garlic" memory, the PS4's name for the memory the GPU reads directly. That is where the project's name comes from.
- **The rest is plain C.** The tokenizer, the user interface, the text renderer and the audio path run on the CPU. Python programs run on an embedded [MicroPython](https://micropython.org).

The app is built with the open-source [OpenOrbis](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain) toolchain. The techniques, the measurements and the dead ends are written up in a 147-page guide, [*Nine Billion Parameters on a PS4*](https://ko-fi.com/s/5d8050b149).

## API

DarkGarlic LLM can serve an OpenAI-compatible API on your network, so that your own programs and apps use the model on the console. It is off until you turn it on.

> The API is new in version 1.20.

**Turn it on.** On the chat screen, press **L3** (click the left stick). The top of the screen then shows where the API listens, for example `API · 192.168.1.5:8080`. L3 again turns it off, and the app remembers the choice for the next time.

**Use it.** Point any OpenAI client at that address, with `http://192.168.1.5:8080/v1` as its base URL:

```bash
curl http://192.168.1.5:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

- `GET /v1/models`, `POST /v1/chat/completions` (`stream: true` for server-sent events), `GET /health`.
- The OpenAPI 3.1 description is at `GET /openapi.json` (also in [`chat/assets/openapi.json`](chat/assets/openapi.json)).
- Replies are generated the way the chat generates its own, speculative decoding included.
- One request runs at a time, and the chat screen is busy while it does: it says that a request is being answered, and ○ stops the reply (the client gets what was written so far, with `finish_reason: "length"`). Requests wait while the keyboard, voice mode or a Python program is open.
- A request is a conversation of its own, so the one on the console's screen is cleared.
- Reasoning is off. Only `n = 1` is supported; `tools`, `logprobs` and similar parameters are ignored.
- For an API key, put it in `/data/garlic_api_key.txt` on the console. Requests then need `Authorization: Bearer <key>`. Without that file, anyone on your network can use the API while it is on, and clients that insist on a key accept any text.
- If the screen shows `API · :8080`, the app could not read the console's address: it is in *Settings → Network → View Connection Status*.

## Build from source

You only need this to change the app. The build scripts are written for Windows.

**Tools**

- Python 3 with `numpy` and `Pillow`.
- The [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain) in `toolchain/OpenOrbis/PS4Toolchain/`.
- [Zig](https://ziglang.org) in `toolchain/zig/`. Its bundled clang compiles both the C code and the GCN kernels.
- The .NET runtime in `toolchain/dotnet/`, for OpenOrbis' package tool.

To keep the tools somewhere else, set the `GARLIC_TOOLCHAIN` environment variable to that folder.

**Build the app**

```bash
python build_chat.py
```

This writes the PKG to `build/chat/`. Add `--upload` to also copy the package to the console over FTP. For `--upload`, set the `PS4_IP` environment variable to your console's address and start GoldHEN's FTP server.

The app's data files (fonts, tokenizer tables, compiled kernels) are already in `chat/assets*`, so a build does not need the scripts in `pc/`.

**Model files**

A plain build keeps the weights outside the package. The app looks for each of these files first inside its package (`/app0/models`) and then in `/data` on the console:

| File | Size | Contents |
|---|---|---|
| `qwen3.5-9b-q4s.bin` | 4.32 GiB | Qwen3.5-9B |
| `qwen3.5-9b-mtp.bin` | 0.20 GiB | The MTP head for speculative decoding |
| `parakeet-v3-q4i.bin` | 0.33 GiB | Speech to text |
| `chatterbox-t3.bin` | 0.29 GiB | Text to speech, first stage |
| `chatterbox-s3gen-mf.bin` | 0.09 GiB | Text to speech, second stage |

The code can also build a smaller app with Qwen3.5-4B (`--4b`, weights in `qwen3.5-4b-q4i.bin`). It has no speculative decoding and is not released as a package.

Download them from [Hugging Face](https://huggingface.co/garlic-llm/garlic-llm-ps4). `python pc/upload_model.py <file>` copies a file to `/data` on the console. To build the all-in-one package instead, put the files in `models/` and add `--full` to the build command.

The scripts that made them from the original checkpoints are in `pc/`, and each one's header describes its inputs and the file format it writes. On a PC, `pc/local_qwen35_9b.py` writes the 9B mix file from a local safetensors checkout (it quantizes in row chunks). `pc/modal_qwen35_9b.py` is the same conversion on [Modal](https://modal.com), which asks for 64 GB of memory. Both are pointed at the abliterated checkpoint. `pc/convert_9b_q4s.py` then writes `qwen3.5-9b-q4s.bin` and, with `--mtp`, `qwen3.5-9b-mtp.bin`.

**Repository layout**

| Path | Contents |
|---|---|
| `chat/` | The app: engine, user interface, tokenizer, voice, Python runner |
| `chat/py/` | MicroPython and its port to the console |
| `shaders/` | The GPU kernels |
| `pc/` | Converters for the models, reference implementations in NumPy, and the scripts that build the app's data files |
| `build_chat.py` | Builds the PKG |

## Troubleshooting

- **"Couldn't start".** A model file is missing or incomplete. Install the package again. With a build from source, check the files in `/data` against the table above.
- **Dictation hears nothing.** Check that the console sees your microphone, and keep □ held while you speak.
- **Anything else.** The app writes a log to `/data/ps4ai_log.txt`. Attach it when you [open an issue](https://github.com/leonardo-alegria/garlic-llm/issues).

## Questions

**Does it need internet?** No. The model, the voices and the speech recognition run on the console.

**Does it run on a PS4 Pro?** Yes, at up to 45 tokens/s. The Pro runs Garlic as if it were a base PS4. A native PS4 Pro mode, which uses the Pro's full GPU, reaches up to 60 tokens/s; it comes with the [experimental version 1.20](https://garlicllm.dev/download/#h-exp).

**And on a PS5?** No. DarkGarlic LLM is written for the PS4's GPU and system software.

**Can it harm my console?** It is an ordinary homebrew app and contains no exploit. It needs GoldHEN, which you install at your own risk.

**Is this the first AI on a PS4's GPU?** As far as I know, yes. I found no earlier public project that runs a language model on the PS4's GPU. If you know of one, please [open an issue](https://github.com/leonardo-alegria/garlic-llm/issues).

## Support

### Help DarkGarlic grow 🙏

I build DarkGarlic on my own, with spare time and old hardware. Stage 2 is planned and partly written: bigger models such as Qwen3.5-27B, run across several PS4s ([future plans](docs/FUTURE_PLANS.md)). It is stuck until I have the hardware to test it on. If DarkGarlic has been useful or fun for you, anything helps, and I am truly grateful.

**Hardware I am hoping for (used is perfect):**

- **PS4 Pros**, or any PS4 that runs GoldHEN. The cluster needs 2–3 consoles.
- **Dell PowerEdge servers**, or similar rack servers, for converting big models, testing and builds.
- **Compute:** GPU time or cloud credits. Converting the 27B takes far more RAM and GPU than I have.
- **A gigabit network switch** to connect the consoles.

**Money, any amount:**

- Cash App: [$bemor187](https://cash.app/$bemor187)
- PayPal: [kultivatedkustoms@gmail.com](https://www.paypal.com/donate/?business=kultivatedkustoms%40gmail.com&currency_code=USD)

To donate hardware or compute, [open an issue](https://github.com/bemor187/garlic-llm/issues/new) and we will work out the details. A ⭐, a share, or a test report from your console helps too. Thank you.

### Support the original Garlic LLM

DarkGarlic is a fork. The engine, the GPU kernels and the original app are Leonardo Alegría's work, and you can support him too:

- [PayPal](https://www.paypal.com/donate/?business=leonardoalegria2002%40gmail.com&currency_code=USD)
- [Ko-fi](https://ko-fi.com/leonardoalegria)
- Buy the guide, [*Nine Billion Parameters on a PS4*](https://ko-fi.com/s/5d8050b149) (PDF, pay what you want from 19 USD)

Bug reports and test results from other consoles help just as much.

## Licence and credits

The app's code is licensed under the [GNU General Public License v3.0](LICENSE). The models and third-party components keep their own licences:

| Component | Author | Licence |
|---|---|---|
| Qwen3.5-9B and Qwen3.5-4B | Alibaba Qwen | Apache-2.0 |
| Chatterbox (text to speech) | Resemble AI | MIT |
| Parakeet TDT 0.6B v3 (speech to text) | NVIDIA | CC-BY-4.0 |
| MicroPython | The MicroPython project | MIT |
| Twemoji | Twitter, Inc. and other contributors | CC-BY-4.0 |
| Fonts: Selawik, Noto, Arimo, Inconsolata, Cascadia Mono | Their authors | SIL OFL 1.1 |
| DejaVu Sans | Its authors | Bitstream Vera licence and public domain |

The complete notices are in [`CREDITS.md`](CREDITS.md) and ship inside the app.

DarkGarlic LLM is built with the [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain) and runs on consoles with [GoldHEN](https://github.com/GoldHEN/GoldHEN). Both are third-party projects by their own authors, and DarkGarlic LLM would not exist without them.

## Contact

The Garlic LLM engine and app are built by [Leonardo Alegría](https://www.linkedin.com/in/leonardoalegria/) (questions about Garlic LLM: [leo@garlicllm.dev](mailto:leo@garlicllm.dev), [upstream issues](https://github.com/leonardo-alegria/garlic-llm/issues)). The DarkGarlic fork is maintained by bemor187: DarkGarlic bugs, test results and hardware offers go to [this fork's issues](https://github.com/bemor187/garlic-llm/issues).

---

Not affiliated with or endorsed by Sony Interactive Entertainment. PlayStation and PS4 are trademarks of their owners.
