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
official = np.load('/tmp/depth_layers.npz')

frames = hidden.shape[0]
x = (hidden @ w[D.PRE + 'depth_linear.weight'].T).view(frames, D.CB, D.H)
embeds = torch.stack([w[D.PRE + 'depth_embeddings.%d.embedding.weight' % i][codes[:, i]]
                      for i in range(D.CB)], dim=1)
x = x + torch.where(masked[..., None], w[D.PRE + 'mask_embedding'], embeds)
cos, sin = D.rope(D.CB)

# attention output of layer 0 alone, to compare against the attn0 hook
p = D.PRE + 'model.layers.0.'
hn = D.rms(x, w[p + 'operator_norm.weight'])
q = (hn @ w[p + 'self_attn.q_proj.weight'].T).view(frames, D.CB, D.NH, D.HD)
k = (hn @ w[p + 'self_attn.k_proj.weight'].T).view(frames, D.CB, D.NKV, D.HD)
v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(frames, D.CB, D.NKV, D.HD)
q = D.rms(q, w[p + 'self_attn.q_norm.weight'])
k = D.rms(k, w[p + 'self_attn.k_norm.weight'])
q = q * cos[None, :, None, :] + D.rotate_half(q) * sin[None, :, None, :]
k = k * cos[None, :, None, :] + D.rotate_half(k) * sin[None, :, None, :]
qp = q.permute(0, 2, 1, 3)
kp = k.repeat_interleave(D.NH // D.NKV, dim=2).permute(0, 2, 1, 3)
vp = v.repeat_interleave(D.NH // D.NKV, dim=2).permute(0, 2, 1, 3)
scores = torch.matmul(qp, kp.transpose(2, 3)) * (D.HD ** -0.5)
attn = torch.matmul(torch.softmax(scores, dim=-1), vp).permute(0, 2, 1, 3).reshape(frames, D.CB, D.NH * D.HD)
gate = torch.sigmoid(hn @ w[p + 'self_attn.gate_proj.weight'].T)
attn = attn * gate.repeat_interleave(D.HD, dim=-1)

ref_attn = torch.from_numpy(official['attn0'])
ref_layer0 = torch.from_numpy(official['layer0'])
print({'attn_cos': round(float(F.cosine_similarity(attn, ref_attn, dim=-1).mean()), 4),
       'attn_absmean_mine': round(float(attn.abs().mean()), 4),
       'attn_absmean_ref': round(float(ref_attn.abs().mean()), 4)}, flush=True)
o = attn @ w[p + 'self_attn.o_proj.weight'].T
x0 = x + o
print({'o_absmean': round(float(o.abs().mean()), 4), 'x0_absmean': round(float(x0.abs().mean()), 4),
       'ref_layer0_absmean': round(float(ref_layer0.abs().mean()), 4)}, flush=True)