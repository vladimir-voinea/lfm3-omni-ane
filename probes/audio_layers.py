import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

import audio_ref as A

w = {}
with safe_open(A.P, framework='pt') as f:
    for k in f.keys():
        if 'audio_encoder' in k or 'audio_projector' in k:
            w[k] = f.get_tensor(k).float()
data = np.load('/tmp/official_audio.npz')
feats = torch.from_numpy(data['input_features']).float()
official = np.load('/tmp/official_audio_layers.npz')

x = A.downsample(feats, w)
seq = x.shape[1]
cos, sin = A.rope(seq)
causal = torch.full((seq, seq), float('-inf')).triu(1)
idx = torch.arange(seq)
far = (idx[:, None] - idx[None, :]) >= A.WINDOW

for i in range(17):
    p = '%slayers.%d.' % (A.PRE, i)
    hn = A.rms(x, w[p + 'operator_norm.weight'])
    if (p + 'conv.conv.weight') in w:
        proj = hn @ w[p + 'conv.in_proj.weight'].T
        b, c, gate = proj.chunk(3, dim=-1)
        mixed = b * gate
        padded = F.pad(mixed.transpose(1, 2), (A.CONV_L - 1, 0))
        conv = F.conv1d(padded, w[p + 'conv.conv.weight'], groups=A.H).transpose(1, 2)
        y = (c * conv) @ w[p + 'conv.out_proj.weight'].T
        kind = 'conv'
    else:
        q = (hn @ w[p + 'self_attn.q_proj.weight'].T).view(seq, A.NH, A.HD)
        k = (hn @ w[p + 'self_attn.k_proj.weight'].T).view(seq, A.NKV, A.HD)
        v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(seq, A.NKV, A.HD)
        q = A.rms(q, w[p + 'self_attn.q_layernorm.weight'])
        k = A.rms(k, w[p + 'self_attn.k_layernorm.weight'])
        q = q * cos[:, None, :] + A.rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + A.rotate_half(k) * sin[:, None, :]
        q = q.permute(1, 0, 2)
        k = k.repeat_interleave(A.NH // A.NKV, dim=1).permute(1, 0, 2)
        v = v.repeat_interleave(A.NH // A.NKV, dim=1).permute(1, 0, 2)
        scores = torch.matmul(q, k.transpose(1, 2)) * (A.HD ** -0.5) + causal
        scores = scores.masked_fill(far[None], float('-inf'))
        y = torch.matmul(torch.softmax(scores, dim=-1), v).permute(1, 0, 2)
        y = y.reshape(seq, A.NH * A.HD) @ w[p + 'self_attn.out_proj.weight'].T
        kind = 'attn'
    x = x + y
    f = A.rms(x, w[p + 'ffn_norm.weight'])
    x = x + (F.silu(f @ w[p + 'feed_forward.w1.weight'].T)
             * (f @ w[p + 'feed_forward.w3.weight'].T)) @ w[p + 'feed_forward.w2.weight'].T
    ref = torch.from_numpy(official['layer%d' % i])
    print({'layer': i, 'kind': kind,
           'cos': round(float(F.cosine_similarity(x, ref, dim=-1).mean()), 4),
           'mine': round(float(x.abs().mean()), 4), 'ref': round(float(ref.abs().mean()), 4)},
          flush=True)