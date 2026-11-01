---
license: other
license_name: lfm-open-license-v1.0
license_link: https://huggingface.co/LiquidAI/LFM3-Omni-2B/blob/main/LICENSE
base_model: LiquidAI/LFM3-Omni-2B
library_name: coreml
pipeline_tag: text-generation
tags:
  - coreml
  - apple-neural-engine
  - ane
  - lfm3
  - omni
  - palettized
---

# LFM3-Omni-2B — Core ML / Apple Neural Engine

[LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B) converted to Core ML
and running on the Apple Neural Engine. This repository holds the **converted programs
only**; the runtime, the conversion code and the measurements live in
[vladimir-voinea/lfm3-omni-ane](https://github.com/vladimir-voinea/lfm3-omni-ane).

This is a **quantized conversion** of the base model (LUT8 palettization of the text programs),
not a merge and not a fine-tune. No weights were retrained.

## Contents

| file | what |
|---|---|
| `layer{N}_conv.mlpackage` | 13 short-convolution layers, history passed as I/O tensors |
| `layer{N}_full.mlpackage` | 6 full-attention layers, 28,672-row `MLState` cache |
| `layer{N}_slide.mlpackage` | 5 sliding-attention layers, 256-row cache |
| `final_norm.mlpackage` | the final norm |
| `lm_head.mlpackage` | the 128,000-way head (whole) |
| `lm_head_a/b.mlpackage` | the same head split into 2 x 64,000 rows, which is what runs on the engine |
| `vision.mlpackage` | vision tower and projector, 224 x 224 |
| `audio_encoder.mlpackage` | audio encoder and projector, up to 30 s |
| `depth_decoder.mlpackage`, `codec.mlpackage` | speech output, up to 10.2 s through the codec program |
| `speech_tables.npz` | tables the host needs for speech |
| `embed_tokens.npy` | the embedding table |
| `rope.npz`, `layer_types.json` | rotary tables and the layer map |

~3.5 GB. Text programs are palettized to 8 bits; the multimodal programs are fp16.

## Use

```bash
huggingface-cli download vladimir94/lfm3-omni-2b-ane-coreml --local-dir lfm3-omni-2b-ane-coreml
git clone https://github.com/vladimir-voinea/lfm3-omni-ane
LFM3_MODEL=$PWD/lfm3-omni-2b-ane-coreml python lfm3-omni-ane/tools/lfm3_serve.py
```

## Context

Context length is fixed when the model is built, because the full-attention caches hold a
fixed number of rows inside the compiled programs. These programs hold **28,672 tokens**, all
attended exactly. A request needing more is refused with `400 context_exceeded`.

## Measured

* **about 30 tok/s** decode at short context (29.3-29.8 measured over 88 tokens), falling as the
  cache fills. The ~11-15 tok/s long-context figure is from the earlier 6-bit build
* against the official model on a 241-token prompt mixing English, Romanian, code and arithmetic:
  top-1 agreement **96.3 %**, top-5 overlap 4.83 of 5, mean KL **0.0036**, logit cosine 1.0000
  (the earlier 6-bit build here measured 81.7 % and 0.2034)
* an end-to-end smoke test passes: text, tool call, tool result, image in, audio in, speech out and
  the context guard

## What changed the fidelity

The Neural Engine loses accuracy on a large projection whose input values are small, and on silu.
These programs avoid both with forms that equal the originals. The measurements are in `notes/STATE.md`
in the code repository.

## Limits

* **Exact attention at 78k is not reachable at speed on this hardware.** The arithmetic does
  not close: exact 78k needs
  ~109 GFLOPS for attention alone at 20 tok/s, and the best measured ANE tile delivers 92.
* **Long contexts have not been re-measured on this build.** The longest run with the fixes is
  337 tokens.
* **Speech quality has only been compared numerically** (waveform correlation 0.996 against the
  PyTorch codec on random codes).
* **Multimodal limits:** audio input over 30 s and speech over 10.2 s fall back to PyTorch, which
  needs the source checkpoint's towers. PyTorch is still imported by the runtime.
* **Audio output works.** The depth decoder, sampler and codec are numerically exact
  (rms and spectrum checked against the reference decode over 24 frames); the loop runs on
  the engine and produces speech-level audio.
* Core ML will not place a 128,000-way matmul on the engine whole, so the runtime uses the
  split head (`lm_head_a/b`).