import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from vision_ref import Vision, projector, P

w = {}
with safe_open(P, framework='pt') as f:
    for k in f.keys():
        if 'vision_tower' in k or 'multi_modal_projector' in k:
            w[k] = f.get_tensor(k).float()
types = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))['vision_config']['layer_types']
data = np.load('/tmp/official_inputs.npz')
pixels = torch.from_numpy(data['pixel_values']).float()
_, h, wd = data['grid'][0].tolist()
mine = Vision(w, types).forward(pixels, h, wd)
official = torch.from_numpy(np.load('/tmp/official_features.npy'))
cos = torch.nn.functional.cosine_similarity(mine, official, dim=-1)
print({'cos_mean': round(float(cos.mean()), 5), 'cos_min': round(float(cos.min()), 5),
       'norm_mine': round(float(mine.norm(dim=-1).mean()), 2),
       'norm_official': round(float(official.norm(dim=-1).mean()), 2)}, flush=True)
proj_mine = projector(mine, w)
proj_official = torch.from_numpy(np.load('/tmp/official_projected.npy'))
pc = torch.nn.functional.cosine_similarity(proj_mine, proj_official, dim=-1)
print({'proj_cos_mean': round(float(pc.mean()), 5),
       'proj_rel': round(float((proj_mine - proj_official).norm() / proj_official.norm()), 5)}, flush=True)