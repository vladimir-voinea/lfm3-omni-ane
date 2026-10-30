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
norms = np.load('/tmp/depth_norms.npz')
frames = hidden.shape[0]

x = (hidden @ w[D.PRE + 'depth_linear.weight'].T).view(frames, D.CB, D.H)
embeds = torch.stack([w[D.PRE + 'depth_embeddings.%d.embedding.weight' % i][codes[:, i]]
                      for i in range(D.CB)], dim=1)
x = x + torch.where(masked[..., None], w[D.PRE + 'mask_embedding'], embeds)
ref_in = torch.from_numpy(norms['opnorm_in'])
print({'layer0_input_cos': round(float(F.cosine_similarity(x, ref_in, dim=-1).mean()), 5),
       'mine_absmean': round(float(x.abs().mean()), 5),
       'ref_absmean': round(float(ref_in.abs().mean()), 5)}, flush=True)
p = D.PRE + 'model.layers.0.'
hn = D.rms(x, w[p + 'operator_norm.weight'])
ref_hn = torch.from_numpy(norms['opnorm_out'])
print({'opnorm_out_cos': round(float(F.cosine_similarity(hn, ref_hn, dim=-1).mean()), 6)}, flush=True)
q = (hn @ w[p + 'self_attn.q_proj.weight'].T).view(frames, D.CB, D.NH, D.HD)
q = D.rms(q, w[p + 'self_attn.q_norm.weight'])
ref_q = torch.from_numpy(norms['qnorm_out'])
print({'qnorm_out_cos': round(float(F.cosine_similarity(q, ref_q, dim=-1).mean()), 6),
       'qnorm_ref_absmean': round(float(ref_q.abs().mean()), 5)}, flush=True)