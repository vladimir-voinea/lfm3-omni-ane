import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import codec_ref as C
import depth_ref as D
from audio_out import sample_codes

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
torch.manual_seed(0)
frames = []
for _ in range(24):
    frames.append(sample_codes(hidden, dw, history=frames[-64:])[0].tolist())
codes = torch.tensor(frames)
audio = C.decode(codes.unsqueeze(0).transpose(1, 2), cw,
                 ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv'])
x = audio[0].numpy()
print({'sampler_on_official_hidden': {'rms': round(float(np.sqrt((x ** 2).mean())), 5),
                                      'peak': round(float(np.abs(x).max()), 4)}}, flush=True)