import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import codec_ref as C
import depth_ref as D

dw = {}
with safe_open(D.P, framework='pt') as f:
    for k in f.keys():
        if k.startswith(D.PRE):
            dw[k] = f.get_tensor(k).float()
cw = {}
with safe_open(C.CODEC, framework='pt') as f:
    for k in f.keys():
        cw[k] = f.get_tensor(k).float()
hidden = torch.from_numpy(np.load('/tmp/official_frame0_hidden.npy')).float()
ref = np.load('/tmp/official_frame0_codes.npy')

codes = torch.zeros(1, 12, dtype=torch.long)
revealed = torch.zeros(1, 12, dtype=torch.bool)
for start in range(0, 12, 4):
    pick = list(range(start, start + 4))
    logits = D.forward(hidden, codes, ~revealed, dw)
    for index in pick:
        codes[0, index] = int(logits[0, index].argmax())
    revealed[0, pick] = True
greedy = codes[0].numpy()
print({'greedy': greedy.tolist(), 'official': ref.tolist()}, flush=True)

def rms_of(frame, reps=8):
    c = torch.tensor([frame] * reps).unsqueeze(0).transpose(1, 2)
    audio = C.decode(c, cw, ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv'])
    x = audio[0].numpy()
    return round(float(np.sqrt((x ** 2).mean())), 5)

print({'rms_greedy': rms_of(greedy.tolist()), 'rms_official': rms_of(ref.tolist())}, flush=True)
probs = (D.forward(hidden, torch.zeros(1, 12, dtype=torch.long),
                     torch.ones(1, 12, dtype=torch.bool), dw)[0, 0] / 0.4).softmax(-1)
print({'top5': probs.topk(5).indices.tolist(), 'top5_p': [round(float(v), 3) for v in probs.topk(5).values]},
      flush=True)