import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import codec_ref as C

w = {}
with safe_open(C.CODEC, framework='pt') as f:
    for k in f.keys():
        w[k] = f.get_tensor(k).float()
codes = torch.from_numpy(np.load('/tmp/official_audio_codes.npy')).long()
audio = C.decode(codes, w, ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv'])
samples = audio[0].numpy()
print({'frames': int(codes.shape[-1]), 'samples': len(samples),
       'seconds': round(len(samples) / 24000, 2),
       'rms_float': round(float(np.sqrt((samples ** 2).mean())), 5),
       'peak': round(float(np.abs(samples).max()), 4)}, flush=True)
np.save('/tmp/official_codes_audio.npy', samples)