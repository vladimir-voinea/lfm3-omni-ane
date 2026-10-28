# For a sampler the discrete outcome is what matters: do the argmax codes agree?
import sys

import numpy as np
import torch
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
codes = torch.from_numpy(data['codes']).long()
masked = torch.from_numpy(data['masked'])
mine = D.forward(hidden, codes, masked, w)
ref = torch.from_numpy(np.load('/tmp/official_depth_logits.npy'))

mine_arg = mine.argmax(-1)
ref_arg = ref.argmax(-1)
agree = (mine_arg == ref_arg).float().mean().item()
print({'argmax_agreement': round(agree, 4), 'codes': mine_arg.shape,
       'mine': mine_arg[0].tolist(), 'ref': ref_arg[0].tolist()}, flush=True)
gap = (mine.max(-1).values - ref.max(-1).values).abs()
print({'top_logit_gap_max': round(float(gap.max()), 4),
       'top_logit_scale': round(float(ref.max(-1).values.abs().mean()), 3)}, flush=True)