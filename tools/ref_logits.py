import os

from paths import ANE, MODEL, PROJECT, SOURCE
# Is the ANE model producing the same logits as the real one?  Load the released
# model on the CPU and compare the top-5 for the same token ids.
from __future__ import annotations

import json
import sys

# appended, not prepended: the venv's numpy 1.26 must win over whatever numpy the
# newer transformers pulled in, or coremltools' stack breaks
sys.path.insert(0, os.path.join(PROJECT, 'pylibs'))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = SOURCE
IDS = [1, 1000, 2000, 3000]

tok = None
try:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
except Exception as exc:
    print({'tokenizer_unavailable': str(exc)[:80]}, flush=True)

model = AutoModel.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.float32,
    low_cpu_mem_usage=True)
model.eval()
with torch.no_grad():
    out = model(input_ids=torch.tensor([IDS]), use_cache=False)
logits = out.logits[0, -1].float().numpy()
top = np.argsort(logits)[-5:][::-1]
print({'ref_top5': top.tolist(),
       'ref_values': [round(float(logits[i]), 3) for i in top]}, flush=True)
if tok is not None:
    print({'ref_top5_text': [tok.decode([int(i)]) for i in top]}, flush=True)