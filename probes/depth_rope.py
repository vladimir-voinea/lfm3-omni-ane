# Llama-style (half) vs GPT-NeoX-style (interleaved) rotary: which does this model use?
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

inv = 1.0 / (D.THETA ** (torch.arange(0, D.HD, 2, dtype=torch.float) / D.HD))
freqs = torch.arange(D.CB, dtype=torch.float)[:, None] * inv[None, :]
half_cos, half_sin = torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()
int_cos = freqs.cos().repeat_interleave(2, dim=-1)
int_sin = freqs.sin().repeat_interleave(2, dim=-1)


def rot_half(t):
    return torch.cat([-t[..., t.shape[-1] // 2:], t[..., :t.shape[-1] // 2]], dim=-1)


def rot_int(t):
    even, odd = t[..., 0::2], t[..., 1::2]
    out = torch.stack([-odd, even], dim=-1)
    return out.reshape(t.shape)


def run(cosr, sinr, rotate):
    q = (hn @ w[p + 'self_attn.q_proj.weight'].T).view(frames, D.CB, D.NH, D.HD)
    k = (hn @ w[p + 'self_attn.k_proj.weight'].T).view(frames, D.CB, D.NKV, D.HD)
    v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(frames, D.CB, D.NKV, D.HD)
    q = D.rms(q, w[p + 'self_attn.q_norm.weight'])
    k = D.rms(k, w[p + 'self_attn.k_norm.weight'])
    q = q * cosr[None, :, None, :] + rotate(q) * sinr[None, :, None, :]
    k = k * cosr[None, :, None, :] + rotate(k) * sinr[None, :, None, :]
    qp = q.permute(0, 2, 1, 3)
    kp = k.repeat_interleave(D.NH // D.NKV, dim=2).permute(0, 2, 1, 3)
    vp = v.repeat_interleave(D.NH // D.NKV, dim=2).permute(0, 2, 1, 3)
    s = torch.matmul(qp, kp.transpose(2, 3)) * (D.HD ** -0.5)
    out = torch.matmul(torch.softmax(s, dim=-1), vp).permute(0, 2, 1, 3)
    out = out.reshape(frames, D.CB, D.NH * D.HD)
    gate = torch.sigmoid(hn @ w[p + 'self_attn.gate_proj.weight'].T)
    return (out * gate.repeat_interleave(D.HD, dim=-1)) @ w[p + 'self_attn.o_proj.weight'].T


for name, cosr, sinr, fn in (('half', half_cos, half_sin, rot_half),
                             ('interleaved', int_cos, int_sin, rot_int)):
    out = run(cosr, sinr, fn)
    print({name: {'cos': round(float(F.cosine_similarity(out, official, dim=-1).mean()), 5),
                  'rel': round(float((out - official).norm() / official.norm()), 5)}}, flush=True)