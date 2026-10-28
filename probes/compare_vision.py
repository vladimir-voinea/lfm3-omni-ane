# Same patches, both towers: cosine agreement per token tells me whether my
# transcription is right, and the norm gap says what kind of error it is.
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from vision_ref import Vision, P

w = {}
with safe_open(P, framework='pt') as f:
    for k in f.keys():
        if 'vision_tower' in k or 'multi_modal_projector' in k:
            w[k] = f.get_tensor(k).float()
types = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))['vision_config']['layer_types']

data = np.load('/tmp/official_inputs.npz')
pixels = torch.from_numpy(data['pixel_values']).float()
grid = data['grid'][0].tolist()
_, h, wd = grid
mine = Vision(w, types).forward(pixels, h, wd)
official = torch.from_numpy(np.load('/tmp/official_features.npy'))

cos = torch.nn.functional.cosine_similarity(mine, official, dim=-1)
print({'shape_mine': list(mine.shape), 'shape_official': list(official.shape),
       'cos_mean': round(float(cos.mean()), 4), 'cos_min': round(float(cos.min()), 4),
       'norm_mine': round(float(mine.norm(dim=-1).mean()), 3),
       'norm_official': round(float(official.norm(dim=-1).mean()), 3),
       'official_absmax': round(float(official.abs().max()), 2)}, flush=True)
np.save('/tmp/mine_features.npy', mine.detach().numpy())