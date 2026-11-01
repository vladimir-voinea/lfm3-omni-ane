# Published programs against the official model

Run from an empty working directory, in this order. `official.py` and `compare.py` need a venv
with torch and transformers 5; the others use the converter venv (coremltools, torch 2.5).

    python ane_side.py . idsonly
    python official.py <LFM3-Omni-2B dir> official.npz ids.json
    python ane_side.py .        # logits and per-layer states from the published programs
    python compare.py           # logits and per-layer error against the official model
    python threeway.py          # build code, shipped programs and official, layer by layer
    python variants.py          # layer 0 as fp16, LUT8 and LUT6, on the ANE and the CPU
