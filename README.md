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
| image input | **working** | vision tower matches the reference at **cos 1.0**; correct captions on a real photo |
| audio input | **working** | encoder matches at **cos 1.0**; transcribes real speech exactly |
| audio output — depth decoder | **exact** | logits cos 1.0, codes 36/36 |
| audio output — codec | **exact** | waveform relative error **0.0**, correlation 1.0 |
| audio output — generation loop | **working** | speech level: rms 0.072, spectrum matches the reference (centroid 438 Hz vs 532 Hz) |
| engine: sessions, prefix cache, sampling | **working** | 2nd turn re-feeds 14 of 18 tokens, first token 0.92 -> 0.70 s |
| server: OpenAI-compatible, streaming | **working** | `/v1/chat/completions`, incremental SSE, `[DONE]` |
| palettisation (LUT6) | **working** | 16.3 -> 21.1 tok/s, same answers |
| LM head split 2 x 64,000 | **working** | 21.1 -> **24.5 tok/s**, both ops on the engine |
| 75k context | **not usable** | mechanism works, 2.1 % per-layer error |

Speed, measured on an M-series Mac with the model palettised and a short context:

* **24.4 tok/s** cold decode, **24.5 tok/s** with the prefix cache hit
* 0.49 s to first token
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

# 5. images, audio in, and speech out
python tools/vision_beach.py                      # caption a photo
python tools/audio_chat.py                        # transcribe a WAV
python tools/say.py 'Good morning, how are you?' out.wav   # speech out (24 kHz)
#
# speech is a CLI path, not an HTTP endpoint: the server already holds all 26 programs and
# a second copy in the same process (or a child) fails make_state() on this machine, so
# /v1/audio/speech answers 501 with this instruction rather than pretending
```

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

## Limits, stated plainly

* **75k context is not reachable.** Measured both ways: fp16 tiles cost ~2 % per layer
  (compounds across 24 layers); an fp32 tile of the same size is exact but Core ML puts it
  on the CPU at **23.5 ms** each, and 75k needs 18 of them per token -- about **2 tok/s**.
  The shipping configuration is the exact single 28,672-token cache.
* **Audio output does not yet produce good speech.** Both halves are numerically exact; the
  generation loop produces quiet frames, so the sampler is still being diagnosed.
* **The LM head runs on the CPU**, costing 11.4 ms per token.
* This is a research conversion, not a product. Expect to read the notes.

## Licence

The LFM3-Omni-2B weights are under the **LFM Open License v1.0**; this repository contains
conversion and runtime code only and redistributes no weights. Check that licence before
using the model commercially.