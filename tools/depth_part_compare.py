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
codes = torch.from_numpy(data['codes']).long()
masked = torch.from_numpy(data['masked'])
parts = np.load('/tmp/depth_parts.npz')
frames = hidden.shape[0]

mine_proj = hidden @ w[D.PRE + 'depth_linear.weight'].T
ref_proj = torch.from_numpy(parts['depth_linear'])
print({'depth_linear_cos': round(float(F.cosine_similarity(mine_proj, ref_proj, dim=-1).mean()), 6),
       'maxdiff': round(float((mine_proj - ref_proj).abs().max()), 6)}, flush=True)

for i in (0, 5):
    mine_emb = w[D.PRE + 'depth_embeddings.%d.embedding.weight' % i][codes[:, i]]
    ref_emb = torch.from_numpy(parts['emb%d' % i])
    print({'emb': i, 'cos': round(float(F.cosine_similarity(mine_emb, ref_emb, dim=-1).mean()), 6),
           'maxdiff': round(float((mine_emb - ref_emb).abs().max()), 6)}, flush=True)

x = mine_proj.view(frames, D.CB, D.H)
embeds = torch.stack([w[D.PRE + 'depth_embeddings.%d.embedding.weight' % i][codes[:, i]]
                      for i in range(D.CB)], dim=1)
x = x + torch.where(masked[..., None], w[D.PRE + 'mask_embedding'], embeds)
ref_in = torch.from_numpy(parts['layer0_in'])
print({'layer0_in_cos': round(float(F.cosine_similarity(x, ref_in, dim=-1).mean()), 6),
       'maxdiff': round(float((x - ref_in).abs().max()), 5)}, flush=True)