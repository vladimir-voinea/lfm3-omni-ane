import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import depth_ref as D

w = {}
with safe_open(D.P, framework='pt') as f:
    for k in f.keys():
        if k.startswith(D.PRE):
            w[k] = f.get_tensor(k).float()

hidden = torch.from_numpy(np.load('/tmp/official_frame0_hidden.npy')).float()
ref = np.load('/tmp/official_frame0_codes.npy')

codes = torch.zeros(1, 12, dtype=torch.long)
revealed = torch.zeros(1, 12, dtype=torch.bool)
for start in range(0, 12, 4):
    pick = list(range(start, start + 4))
    logits = D.forward(hidden, codes, ~revealed, w)
    for index in pick:
        codes[0, index] = int(logits[0, index].argmax())
    revealed[0, pick] = True

mine = codes[0].numpy()
print({'mine': mine.tolist()}, flush=True)
print({'ref': ref.tolist()}, flush=True)
print({'agree': round(float((mine == ref).mean()), 4)}, flush=True)