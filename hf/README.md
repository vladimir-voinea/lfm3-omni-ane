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

This is a **quantized conversion** of the base model (LUT6 palettization of the converted
programs), not a merge and not a fine-tune. No weights were retrained.

## Contents

| file | what |
|---|---|
| `layer{N}_conv.mlpackage` | 13 short-convolution layers, history passed as I/O tensors |
| `layer{N}_full.mlpackage` | 6 full-attention layers, 28,672-row `MLState` cache |
| `layer{N}_slide.mlpackage` | 5 sliding-attention layers, 256-row cache |
| `final_norm.mlpackage` | the final norm |
| `lm_head.mlpackage` | the 128,000-way head (whole) |
| `lm_head_a/b.mlpackage` | the same head split into 2 x 64,000 rows — **both ops place on the engine**, 11.4 ms -> 3.0 ms |
| `embed_tokens.npy` | the embedding table |
| `rope.npz`, `layer_types.json` | rotary tables and the layer map |

~1.8 GB, palettized 6-bit. The pre-palettization build is ~3.7 GB and marginally more
accurate.

## Use

```bash
huggingface-cli download vladimir94/lfm3-omni-2b-ane-coreml --local-dir model-28k-lut6
git clone https://github.com/vladimir-voinea/lfm3-omni-ane
LFM3_ANE=$PWD/model-28k-lut6 python lfm3-omni-ane/tools/lfm3_serve.py
```

## Measured

* **24.4 tok/s** cold decode, **24.5 tok/s** with a prefix-cache hit, 0.49 s to first token
* 28,672-token context
* palettisation 16.3 -> 21.1 tok/s, then the split head 21.1 -> 24.5 tok/s, answers unchanged throughout ("Japan -> Tokyo", "Italy -> Rome")

## Limits

* **75k context is not usable**: the tiled-attention mechanism runs but carries ~2 % error
  per layer, which compounds. Use the 28,672-token configuration.
* **Audio output works.** The depth decoder, sampler and codec are numerically exact
  (rms and spectrum checked against the reference decode over 24 frames); the loop runs on
  the engine and produces speech-level audio.
* The LM head runs on the CPU (11.4 ms/token); Core ML will not place a 128,000-way matmul
  on the engine whole.