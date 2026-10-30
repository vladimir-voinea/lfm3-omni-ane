# Where does my tower start to diverge from the official one?
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

import vision_ref
from vision_ref import Vision, rms, P

w = {}
with safe_open(P, framework='pt') as f:
    for k in f.keys():
        if 'vision_tower' in k or 'multi_modal_projector' in k:
            w[k] = f.get_tensor(k).float()
types = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))['vision_config']['layer_types']

data = np.load('/tmp/official_inputs.npz')
pixels = torch.from_numpy(data['pixel_values']).float()
_, h, wd = data['grid'][0].tolist()
official = np.load('/tmp/official_layers.npz')

# my tower, capturing the same points
tower = Vision(w, types)
x = pixels @ w['model.vision_tower.patchifier.weight'].T
print({'patchifier_cos': round(float(torch.nn.functional.cosine_similarity(
    x, torch.from_numpy(official['patchifier']), dim=-1).mean()), 4),
    'patchifier_maxdiff': round(float((x - torch.from_numpy(official['patchifier'])).abs().max()), 5)})

hi, wi = vision_ref.position_ids(h, wd)
cos, sin = vision_ref.axial_rope(hi, wi)
for i, kind in enumerate(types):
    p = 'model.vision_tower.layers.%d.' % i
    hn = rms(x, w[p + 'operator_norm.weight'])
    if kind == 'full_attention':
        y = tower.attn_block(hn, i, h, wd, 1, cos, sin)
    else:
        y = tower.conv_block(hn, i, h, wd, 1)
    x = x + y
    pre = 'model.vision_tower.layers.%d.feed_forward.' % i
    f = rms(x, w[p + 'ffn_norm.weight'])
    x = x + (torch.nn.functional.silu(f @ w[pre + 'w1.weight'].T)
             * (f @ w[pre + 'w3.weight'].T)) @ w[pre + 'w2.weight'].T
    ref = torch.from_numpy(official['layer%d' % i])
    c = float(torch.nn.functional.cosine_similarity(x, ref, dim=-1).mean())
    print({'layer': i, 'kind': kind, 'cos': round(c, 4),
           'mine_absmean': round(float(x.abs().mean()), 4),
           'official_absmean': round(float(ref.abs().mean()), 4)}, flush=True)