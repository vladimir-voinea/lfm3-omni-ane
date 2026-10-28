# Was the sampler ever actually at fault?  The earlier test repeated one hidden state 24
# times, and repeating a frame decodes to silence by construction.  Here the frames get
# DISTINCT inputs, which is the only fair test of the sampler.
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

official = torch.from_numpy(np.load('/tmp/official_audio_codes.npy')).long()


def diversity(codes):
    codes = codes.reshape(12, -1)
    return round(float(np.mean([len(set(codes[i].tolist())) for i in range(12)])), 2)


print({'official_unique_per_codebook': diversity(official)}, flush=True)

base = torch.from_numpy(np.load('/tmp/official_frame0_hidden.npy')).float()
torch.manual_seed(0)
frames = []
for step in range(24):
    hidden = base + torch.randn_like(base) * 0.6      # distinct but nearby states
    frames.append(sample_codes(hidden, dw, history=frames[-64:])[0].tolist())
mine = torch.tensor(frames)
print({'synthetic_unique_per_codebook': diversity(mine)}, flush=True)

audio = C.decode(mine.unsqueeze(0).transpose(1, 2), cw,
                 ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv'])[0].numpy()
print({'synthetic_rms': round(float(np.sqrt((audio ** 2).mean())), 5),
       'peak': round(float(np.abs(audio).max()), 4)}, flush=True)
np.save('/tmp/synth_codes.npy', mine.numpy())