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
official = torch.from_numpy(np.load('/tmp/depth_layers.npz')['attn0'])
frames = hidden.shape[0]
x = (hidden @ w[D.PRE + 'depth_linear.weight'].T).view(frames, D.CB, D.H)
embeds = torch.stack([w[D.PRE + 'depth_embeddings.%d.embedding.weight' % i][codes[:, i]]
                      for i in range(D.CB)], dim=1)
x = x + torch.where(masked[..., None], w[D.PRE + 'mask_embedding'], embeds)
p = D.PRE + 'model.layers.0.'
hn = D.rms(x, w[p + 'operator_norm.weight'])


def run(rope_mode):
    if rope_mode == 'normal':
        cosr, sinr = D.rope(D.CB)
    elif rope_mode == 'none':
        cosr = torch.ones(D.CB, D.HD)
        sinr = torch.zeros(D.CB, D.HD)
    else:
        cosr, sinr = D.rope(D.CB)
        cosr = torch.ones_like(cosr)
        sinr = torch.zeros_like(sinr)
    q = (hn @ w[p + 'self_attn.q_proj.weight'].T).view(frames, D.CB, D.NH, D.HD)
    k = (hn @ w[p + 'self_attn.k_proj.weight'].T).view(frames, D.CB, D.NKV, D.HD)
    v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(frames, D.CB, D.NKV, D.HD)
    q = D.rms(q, w[p + 'self_attn.q_norm.weight'])
    k = D.rms(k, w[p + 'self_attn.k_norm.weight'])
    q = q * cosr[None, :, None, :] + D.rotate_half(q) * sinr[None, :, None, :]
    k = k * cosr[None, :, None, :] + D.rotate_half(k) * sinr[None, :, None, :]
    qp = q.permute(0, 2, 1, 3)
    kp = k.repeat_interleave(D.NH // D.NKV, dim=2).permute(0, 2, 1, 3)
    vp = v.repeat_interleave(D.NH // D.NKV, dim=2).permute(0, 2, 1, 3)
    scores = torch.matmul(qp, kp.transpose(2, 3)) * (D.HD ** -0.5)
    out = torch.matmul(torch.softmax(scores, dim=-1), vp).permute(0, 2, 1, 3)
    out = out.reshape(frames, D.CB, D.NH * D.HD)
    gate = torch.sigmoid(hn @ w[p + 'self_attn.gate_proj.weight'].T)
    return (out * gate.repeat_interleave(D.HD, dim=-1)) @ w[p + 'self_attn.o_proj.weight'].T


for mode in ('normal', 'none'):
    out = run(mode)
    c = float(F.cosine_similarity(out, official, dim=-1).mean())
    print({'rope': mode, 'cos': round(c, 5),
           'rel': round(float((out - official).norm() / official.norm()), 5)}, flush=True)