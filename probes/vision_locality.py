# Does the tower keep spatial order?  Paint one 32x32 block and check that the feature
# change lands on the tokens that cover that block, rather than spreading at random.
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from vision_ref import Vision, P


def patchify(img, ps=16, merge=2, temporal=2):
    c, hgt, wid = img.shape
    gh, gw = hgt // ps, wid // ps
    p = img.reshape(c, gh // merge, merge, ps, gw // merge, merge, ps)
    p = p.permute(1, 4, 2, 5, 0, 3, 6).unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
    return p.reshape(gh * gw, c * temporal * ps * ps), gh, gw


w = {}
with safe_open(P, framework='pt') as f:
    for k in f.keys():
        if 'vision_tower' in k or 'multi_modal_projector' in k:
            w[k] = f.get_tensor(k).float()
types = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))['vision_config']['layer_types']
tower = Vision(w, types)

base = torch.from_numpy(np.load('/tmp/beach.npy')).float().permute(2, 0, 1) / 255.0
hits = base.clone()
hits[:, 64:96, 64:96] = 1.0        # a 32x32 block at patch rows 4-5, cols 4-5

def feats(img):
    pixels, gh, gw = patchify(img)
    pixels = ((pixels - 0.5) / 0.5).float()
    return tower.forward(pixels, gh, gw)

a, b = feats(base), feats(hits)
delta = (a - b).norm(dim=-1).detach().numpy()
gh = 16
order = np.argsort(-delta)
top = order[:8]
print({'top_tokens': top.tolist()})
print({'as_hw': [[int(i // (gh // 2)), int(i % (gh // 2))] for i in top]})
inside = [i for i in top if 2 <= i // (gh // 2) <= 3 and 2 <= i % (gh // 2) <= 3]
print({'expected_block_tokens': [8, 9, 10, 11], 'hits_in_top8': len(inside),
       'delta_mean_inside': round(float(delta[[8, 9, 10, 11]].mean()), 4),
       'delta_mean_outside': round(float(np.delete(delta, [8, 9, 10, 11]).mean()), 4)})