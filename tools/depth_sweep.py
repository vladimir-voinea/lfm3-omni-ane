# Which head/norm ordering reproduces the official attention output?
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
cosr, sinr = D.rope(D.CB)
p = D.PRE + 'model.layers.0.'
hn = D.rms(x, w[p + 'operator_norm.weight'])


def attend(rope_then_norm, interleave_heads, use_gate):
    q = (hn @ w[p + 'self_attn.q_proj.weight'].T)
    k = (hn @ w[p + 'self_attn.k_proj.weight'].T)
    v = (hn @ w[p + 'self_attn.v_proj.weight'].T)
    if interleave_heads:
        q = q.view(frames, D.CB, D.HD, D.NH).transpose(2, 3)
        k = k.view(frames, D.CB, D.HD, D.NKV).transpose(2, 3)
    else:
        q = q.view(frames, D.CB, D.NH, D.HD)
        k = k.view(frames, D.CB, D.NKV, D.HD)
    v = v.view(frames, D.CB, D.NKV, D.HD)
    q = D.rms(q, w[p + 'self_attn.q_norm.weight'])
    k = D.rms(k, w[p + 'self_attn.k_norm.weight'])
    spin = D.rotate_half(q) if interleave_heads else D.rotate_half(q)
    q = q * cosr[None, :, None, :] + spin * sinr[None, :, None, :]
    k = k * cosr[None, :, None, :] + D.rotate_half(k) * sinr[None, :, None, :]
    qp = q.permute(0, 2, 1, 3)
    kp = k.repeat_interleave(D.NH // D.NKV, dim=2).permute(0, 2, 1, 3)
    vp = v.repeat_interleave(D.NH // D.NKV, dim=2).permute(0, 2, 1, 3)
    scores = torch.matmul(qp, kp.transpose(2, 3)) * (D.HD ** -0.5)
    out = torch.matmul(torch.softmax(scores, dim=-1), vp).permute(0, 2, 1, 3)
    out = out.reshape(frames, D.CB, D.NH * D.HD)
    if use_gate:
        gate = torch.sigmoid(hn @ w[p + 'self_attn.gate_proj.weight'].T)
        out = out * gate.repeat_interleave(D.HD, dim=-1)
    return out @ w[p + 'self_attn.o_proj.weight'].T


for interleave, gate in ((False, True), (True, True), (False, False), (True, False)):
    out = attend(False, interleave, gate)
    c = float(F.cosine_similarity(out, official, dim=-1).mean())
    print({'interleave_heads': interleave, 'gate': gate, 'cos': round(c, 4)}, flush=True)