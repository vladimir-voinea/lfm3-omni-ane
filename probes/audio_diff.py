import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

import audio_ref as A

P = A.P
w = {}
with safe_open(P, framework='pt') as f:
    for k in f.keys():
        if 'audio_encoder' in k or 'audio_projector' in k:
            w[k] = f.get_tensor(k).float()
data = np.load('/tmp/official_audio.npz')
feats = torch.from_numpy(data['input_features']).float()
official = np.load('/tmp/official_audio_layers.npz')

mine_ds = A.downsample(feats, w)
ref_ds = torch.from_numpy(official['downsample']).transpose(1, 2)
print({'downsample_cos': round(float(F.cosine_similarity(mine_ds, ref_ds, dim=-1).mean()), 5),
       'maxdiff': round(float((mine_ds - ref_ds).abs().max()), 5),
       'mine_absmean': round(float(mine_ds.abs().mean()), 5),
       'ref_absmean': round(float(ref_ds.abs().mean()), 5)}, flush=True)