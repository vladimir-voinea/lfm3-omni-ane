# LFM3-Omni-2B on the Apple Neural Engine

A from-scratch Core ML conversion of [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B)
that runs the text model on the ANE, with image input, audio input, an engine with sessions
and a prefix cache, and an OpenAI-compatible server.

No MLX, no PyTorch at run time, no Metal. Core ML programs on the Neural Engine, driven by a
small Python runtime.

## What works, measured

| part | state | evidence |
|---|---|---|
| text model (24 layers) on the ANE | **working** | answers correctly; 28,672-token cache |
| image input | **working, incl. HTTP** | vision tower matches the reference at **cos 1.0**; over the API, a red image answers "Red" and a green one "Green" |
| audio input | **working** | encoder matches at **cos 1.0**; transcribes real speech exactly |
| audio output — depth decoder | **exact** | logits cos 1.0, codes 36/36 |
| audio output — codec | **exact** | waveform relative error **0.0**, correlation 1.0 |
| audio output over HTTP | **working** | POST /v1/audio/speech -> audio/wav, 49,964 bytes |
| audio output — generation loop | **working** | speech level: rms 0.072, spectrum matches the reference (centroid 438 Hz vs 532 Hz) |
| engine: sessions, prefix cache, sampling | **working** | 2nd turn re-feeds 14 of 18 tokens, first token 0.92 -> 0.70 s |
| server: OpenAI-compatible, streaming | **working** | `/v1/chat/completions`, incremental SSE, `[DONE]` |
| **tool calling** | **working** | `tools` in the request → `finish_reason: tool_calls` with parsed arguments |
| palettisation (LUT6) | **working** | 16.3 -> 21.1 tok/s, same answers |
| LM head split 2 x 64,000 | **working** | 21.1 -> **24.5 tok/s**, both ops on the engine |
| context, exact and guaranteed | **28,672 tokens** | the shipped configuration, nothing forgotten |
| beyond 28,672 | opt-in rolling window (`LFM3_WINDOW=1`) | runs at 78,042 tokens, but the oldest tokens are **forgotten** — an approximation, not a longer attention |

Speed, measured on an M-series Mac with the model palettised and a short context:

* **29.3 tok/s** cold decode and **32.0 tok/s** with the prefix cache hit (median of three
  runs on a quiet machine; the same build measured 24.5 while other work was running, so
  expect that range)
* 0.60-0.65 s cold to first token, 0.45 s with the cache hit
* one token costs ~41 ms: 14.8 ms in 13 conv layers, 20.0 ms in 11 attention layers,
  ~3 ms in the LM head, ~0.02 ms reading the embedding

Two changes got it from 16.3 to 24.5 tok/s, both from measurement rather than assumption:
6-bit palettisation (16.3 -> 21.1), then splitting the 128,000-way LM head into two chunks
of 64,000 so both matmuls land on the engine instead of the CPU (21.1 -> 24.5, the whole
head cost 11.4 ms in place and 3.0 ms split).

The 46 ms is measured, not estimated, and the 31 ms of arithmetic beneath it is measured
per program. Long contexts are slower: the full-attention programs read a 28,672-row tile
whether or not it is masked, so expect ~11-15 tok/s once the cache is full.

## How the conversion works

The model cannot be converted in one piece on this hardware, so it is split into 26 programs:

* **one program per layer**, each holding only its own state, so no single program exceeds
  the ANE's state budget;
* conv layers carry their short-convolution history as **input/output tensors** rather than
  state (a stateful conv faults with `IOSurface smaller than the model expects`);
* attention layers keep their cache in `MLState` as a 4-D `(1, KV, 1, rows)` tensor and
  write one row per token at a dynamic `pos`, with an additive causal mask passed as input;
* the full-attention cache is 28,672 rows: 32,768 fails to plan (-14) and the softmax width
  wall is 65,536.

Every claim about placement in this repository was checked with
`coremltools.models.compute_plan.MLComputePlan` on a compiled `.mlmodelc`. Stopwatch
timings are not placement: a 0.59 ms "ANE" reading turned out to be the CPU.

## Reproduce it

```bash
# 1. the source model (4.3 GB)
huggingface-cli download LiquidAI/LFM3-Omni-2B --local-dir models/LFM3-Omni-2B

# 2. build the ANE programs (writes model-28k/, ~3.7 GB)
python tools/build_model.py

# 3. palettise to LUT6 (writes model-28k-lut6/, ~1.8 GB, 2x faster)
#    use THIS build: the 3.7 GB fp16 set can exceed the memory available to load
#    26 programs at once, and then make_state() fails even though every program is valid
python tools/palettize_all.py

# 4. talk to it
python tools/lfm3_engine.py                       # one prompt, prints tok/s
python tools/lfm3_serve.py                        # OpenAI-compatible on :8010

# 5. an agent loop: tool call -> tool result -> answer (no SDK needed)
python tools/agent_demo.py

# 5. images, audio in, and speech out
python tools/vision_beach.py                      # caption a photo
python tools/audio_chat.py                        # transcribe a WAV
python tools/say.py 'Good morning, how are you?' out.wav   # speech out (24 kHz)
#
# speech is a CLI path, not an HTTP endpoint: the server already holds all 26 programs and
# a second copy in the same process (or a child) fails make_state() on this machine, so
# /v1/audio/speech answers 501 with this instruction rather than pretending
```

Image input goes over HTTP as an OpenAI-style content part with a `data:` PNG
(`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`); the server
decodes PNG itself, so it needs no Pillow.

Requires an Apple-silicon Mac, `coremltools` (tested 8.x), and a Python with
`tokenizers` and `safetensors`. Building the programs takes about 20 minutes.

## The reference method

Transcribing a model by reading its source is guesswork unless something can check it. This
project ran the **official implementation** side by side (a throwaway venv with a newer torch,
since `transformers` 5 needs torch >= 2.6), dumped its intermediate tensors, and diffed them
against the transcription. That is how these were found:

* the vision tower's norm belongs **after the patchifier**, not at the end;
* the audio stack uses **theta = 1e6**, not the 1e4 of the text model's sliding layers;
* the audio encoder and the depth decoder each end with a **final norm**;
* `depth_linear` has a **bias** -- missed at first because a key listing filtered on
  `depth_linear.weight`;
* the codec decoder attends within a **30-frame sliding window**, not plain causal.

Two measurement traps are recorded in `notes/STATE.md` because they cost real time: a
cosine of 0.998 can hide codes that disagree 39 % of the time (argmax agreement is the metric
for a sampler), and repeating a single audio frame is not a valid loudness test.

## Tool calling

The model has native tool tokens and the official `chat_template.jinja` is used verbatim
(rendered with jinja2, `{% generation %}` markers stripped), so tools go in exactly as
trained. Emitted calls come back in the OpenAI shape:

```json
{"finish_reason": "tool_calls",
 "tool_calls": [{"type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]}
```

That makes it droppable into any OpenAI-compatible agent harness.

## Restoring this checkout

Everything needed is in two places, so deleting the local copy loses nothing:

```bash
# the converted model (2.0 GB, 92 files)
huggingface-cli download vladimir94/lfm3-omni-2b-ane-coreml --local-dir model-28k-lut6

# this repository (code, probes, notes)
git clone https://github.com/vladimir-voinea/lfm3-omni-ane
```

The 4.3 GB source model is **not** needed to run anything here; only `tokenizer.json`,
`config.json` and `generation_config.json` are, and those three copies live inside the
downloaded model directory. Re-download the source only to rebuild the programs:

```bash
huggingface-cli download LiquidAI/LFM3-Omni-2B --local-dir models/LFM3-Omni-2B
```

The text backbone is converted, so vision, audio input and speech out are the only run-time
readers of that checkpoint. Keep just what they need and reclaim 3.4 GB:

```bash
python tools/extract_towers.py    # writes towers.safetensors (0.88 GB)
rm models/LFM3-Omni-2B/model.safetensors
```

### What needs the source model

Text and tool calling run from the converted build alone. **Image input, audio input and
speech output additionally need the original weights** (the vision tower, the audio encoder,
the depth decoder and the codec are still torch fp32 in this conversion, not Core ML):

```bash
huggingface-cli download LiquidAI/LFM3-Omni-2B --local-dir models/LFM3-Omni-2B
```

Converting those four to Core ML is the obvious next step; until then they cost 4.3 GB of
disk and run on the CPU.

### Where Core ML puts its caches

Each program load compiles an `.mlmodelc`. They land under the user temp directory
(`/var/folders/.../T`), which on macOS is **the data volume, not a RAM disk** (`df` confirms
`/dev/disk3s5`). A few hundred loads is tens of gigabytes, so:

```bash
sh tools/clean_caches.sh     # deletes only *.mlmodelc and *.mlpackage from temp
```

The runtime also sets `TMPDIR` to `disk-tmp/` inside the project before loading anything.

## Limits, stated plainly

* **Longer context works with a rolling window** (`LFM3_WINDOW=1`). The full-attention
  caches become rings: the write slot wraps while the rotation stays keyed to the absolute
  position, so the model always sees the last 28,672 tokens exactly and the context grows
  without bound. Measured twice:

  | prompt tokens | answer | wall time |
  | --- | --- | --- |
  | 29,942 | **OMEGA-99** | 891 s |
  | **78,042** | **OMEGA-99** | 2,266 s |

  Both asked for a code placed near the end of the prompt, after the ring had rolled. The cost
  is prefill at ~34 tok/s, so a 78k prompt takes **38 minutes** to read; decode afterwards is
  unaffected. Exact attention is still what ships by default, and the wall it hits is next.
* **75k context is exact-unreachable, for three independent reasons.** Measured, not assumed:
  * *precision*: fp16 tiles cost ~2 % per layer and compound across 24 layers; an fp32 tile
    is exact but Core ML puts it on the CPU at **23.5 ms**, and 75k needs 18 of them per
    token -- about **2 tok/s**;
  * *state*: one exact 28,672-row cache is 58.7 MB per full-attention layer; 32,768 rows
    already fails to plan, so a 75k window cannot live in a single state;
  * *tiling cost*: splitting the window into 19 tiles per layer needs **114 programs**, each
    carrying its own copy of the same attention weights (Core ML cannot share weights across
    programs) -- roughly 2.8 GB of duplicated weights, plus 108 extra dispatches per token.

  The shipping configuration is the exact single 28,672-token cache, which is where the
  hardware stops cooperating.
* **Speech out works over HTTP too**, not only through the CLI: POST /v1/audio/speech
  answered **200, audio/wav, 49,964 bytes** with a valid RIFF header (it used to answer 501 --
  that was the endpoint refusing to load the codec while the server already held all 26
  programs; it loads fine now).
* **Audio output works, within its range.** Speech is generated on the engine end to end:
  *"Good to be back online."* renders 1.84 s at rms **0.0685**, peak 0.46, against the
  official pipeline's rms 0.076. The cause of an earlier silence was a BOS token the model
  never expects, not the sampler.
* **The LM head runs on the CPU**, costing 11.4 ms per token.
* This is a research conversion, not a product. Expect to read the notes.

## Licence

The LFM3-Omni-2B weights are under the **LFM Open License v1.0**; this repository contains
conversion and runtime code only and redistributes no weights. Check that licence before
using the model commercially.