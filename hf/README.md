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
| `final_norm.mlpackage`, `lm_head.mlpackage` | final norm and the 128,000-way head |
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

* **21.1 tok/s** cold decode, **21.5 tok/s** with a prefix-cache hit, 0.648 s to first token
* 28,672-token context
* palettization: 16.3 -> 21.1 tok/s, answers unchanged ("Japan -> Tokyo", "Italy -> Rome")

## Limits

* **75k context is not usable**: the tiled-attention mechanism runs but carries ~2 % error
  per layer, which compounds. Use the 28,672-token configuration.
* **Audio output does not produce good speech yet** — the depth decoder and codec are
  numerically exact but the generation loop yields quiet frames.
* The LM head runs on the CPU (11.4 ms/token); Core ML will not place a 128,000-way matmul
  on the engine whole.

