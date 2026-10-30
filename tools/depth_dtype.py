# Is the official depth_linear effectively a bf16 matmul?
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import depth_ref as D

w = {}
with safe_open(D.P, framework='pt') as f:
    for k in f.keys():
        if k.startswith(D.PRE):
            w[k] = f.get_tensor(k).float()
data = np.load('/tmp/depth_input.npz')
hidden = torch.from_numpy(data['hidden']).float()
ref = torch.from_numpy(np.load('/tmp/depth_parts.npz')['depth_linear'])
W = w[D.PRE + 'depth_linear.weight']

for label, tensor, weight in (
        ('fp32', hidden, W),
        ('bf16-both', hidden.bfloat16(), W.bfloat16()),
        ('bf16-weight-only', hidden.bfloat16(), W.bfloat16())):
    out = (tensor.to(weight.dtype) @ weight.T).float()
    print({label: round(float(F.cosine_similarity(out, ref, dim=-1).mean()), 6),
           'maxdiff': round(float((out - ref).abs().max()), 6)}, flush=True)