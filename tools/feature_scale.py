import sys
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from vision_ref import Vision, projector, P

# how big is a text embedding row, and how big is a projected image vector?
emb = np.load('/Users/vladimir/dev/lfm3-ane/model-28k/embed_tokens.npy', mmap_mode='r')
for tok in (1, 1000, 124899):
    row = np.asarray(emb[tok], np.float32)
    print('embedding', tok, 'norm', round(float(np.linalg.norm(row)), 3),
          'absmax', round(float(np.abs(row).max()), 3))

w = {}
with safe_open(P, framework='pt') as f:
    for k in f.keys():
        if 'vision_tower' in k or 'multi_modal_projector' in k:
            w[k] = f.get_tensor(k).float()
import json
types = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))['vision_config']['layer_types']

def patchify(img, ps=16, merge=2, temporal=2):
    c, hgt, wid = img.shape
    gh, gw = hgt // ps, wid // ps
    p = img.reshape(c, gh // merge, merge, ps, gw // merge, merge, ps)
    p = p.permute(1, 4, 2, 5, 0, 3, 6).unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
    return p.reshape(gh * gw, c * temporal * ps * ps), gh, gw

arr = torch.from_numpy(np.load('/tmp/beach.npy')).float().permute(2, 0, 1) / 255.0
pixels, gh, gw = patchify(arr)
pixels = ((pixels - 0.5) / 0.5).float()
feats = Vision(w, types).forward(pixels, gh, gw)
proj = projector(feats, w)
print({'feature_norm_per_token': round(float(feats.norm(dim=-1).mean()), 3),
       'projected_norm_per_token': round(float(proj.norm(dim=-1).mean()), 3),
       'projected_absmax': round(float(proj.abs().max()), 3)})