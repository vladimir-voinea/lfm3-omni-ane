# If my pipeline (depth decoder + sampler + codec) receives the OFFICIAL hidden states, it
# should reproduce good speech.  That separates 'my audio stack is right' from 'the fp16
# text model is the limiter'.
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

hiddens = torch.from_numpy(np.load('/tmp/official_hidden_traj.npy')).float()
print({'frames': len(hiddens)}, flush=True)
torch.manual_seed(0)
frames = []
for hidden in hiddens:
    frames.append(sample_codes(hidden.reshape(1, -1), dw, history=frames[-64:])[0].tolist())
codes = torch.tensor(frames)
audio = C.decode(codes.unsqueeze(0).transpose(1, 2), cw,
                 ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv'])[0].numpy()
steps = np.abs(np.diff(codes.numpy().T.astype(float), axis=1)).mean()
print({'frames': len(frames), 'unique_per_codebook': [len(set(codes[:, i].tolist())) for i in range(12)]},
      flush=True)
print({'rms': round(float(np.sqrt((audio ** 2).mean())), 5),
       'peak': round(float(np.abs(audio).max()), 4),
       'mean_frame_jump': round(float(steps), 1),
       'official_jump_reference': 554.0}, flush=True)
np.save('/tmp/mypipeline_official_hidden.npy', audio)