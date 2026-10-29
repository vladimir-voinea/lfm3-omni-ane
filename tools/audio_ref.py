
# Audio tower, transcribed: mel -> downsampler -> 17-layer LFM2 -> projector.
from __future__ import annotations
import os
from paths import SOURCE, TOWERS

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

P = TOWERS
H, NH, NKV, HD, FF = 768, 8, 4, 96, 1792
EPS, WINDOW, CONV_L = 1e-5, 32, 3
PRE = 'model.audio_encoder.lfm.'


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)


def rope(seq, theta=1e6):
    inv = 1.0 / (theta ** (torch.arange(0, HD, 2, dtype=torch.float) / HD))
    freqs = torch.arange(seq, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def downsample(feats, w):
    """(B, 128, T) mel -> (B, T/8, 768)."""
    d = 'model.audio_encoder.downsample.'
    x = feats.unsqueeze(1)
    x = F.silu(F.conv2d(x, w[d + 'conv_in.weight'], w[d + 'conv_in.bias'], stride=2, padding=1))
    x = F.conv2d(x, w[d + 'conv_mid.weight'], w[d + 'conv_mid.bias'], stride=2, padding=1,
                 groups=256)
    x = F.silu(F.conv2d(x, w[d + 'lin_mid.weight'], w[d + 'lin_mid.bias']))
    x = F.conv2d(x, w[d + 'conv_end.weight'], w[d + 'conv_end.bias'], stride=2, padding=1,
                 groups=256)
    x = F.silu(F.conv2d(x, w[d + 'lin_end.weight'], w[d + 'lin_end.bias']))
    x = x.flatten(1, 2)
    return F.conv1d(x, w[d + 'lin_out.weight'], w[d + 'lin_out.bias']).transpose(1, 2)


def forward(feats, w):
    """Mel -> encoder hidden states -> projected language vectors."""
    x = downsample(feats, w)
    seq = x.shape[1]
    cos, sin = rope(seq)
    causal = torch.full((seq, seq), float('-inf')).triu(1)
    idx = torch.arange(seq)
    far = (idx[:, None] - idx[None, :]) >= WINDOW
    for i in range(17):
        p = '%slayers.%d.' % (PRE, i)
        hn = rms(x, w[p + 'operator_norm.weight'])
        if (p + 'conv.conv.weight') in w:
            proj = hn @ w[p + 'conv.in_proj.weight'].T
            b, c, gate = proj.chunk(3, dim=-1)
            mixed = b * gate
            padded = F.pad(mixed.transpose(1, 2), (CONV_L - 1, 0))
            conv = F.conv1d(padded, w[p + 'conv.conv.weight'], groups=H).transpose(1, 2)
            y = (c * conv) @ w[p + 'conv.out_proj.weight'].T
        else:
            q = (hn @ w[p + 'self_attn.q_proj.weight'].T).view(seq, NH, HD)
            k = (hn @ w[p + 'self_attn.k_proj.weight'].T).view(seq, NKV, HD)
            v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(seq, NKV, HD)
            q = rms(q, w[p + 'self_attn.q_layernorm.weight'])
            k = rms(k, w[p + 'self_attn.k_layernorm.weight'])
            q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
            k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
            q = q.permute(1, 0, 2)
            k = k.repeat_interleave(NH // NKV, dim=1).permute(1, 0, 2)
            v = v.repeat_interleave(NH // NKV, dim=1).permute(1, 0, 2)
            scores = torch.matmul(q, k.transpose(1, 2)) * (HD ** -0.5) + causal
            scores = scores.masked_fill(far[None], float('-inf'))
            y = torch.matmul(torch.softmax(scores, dim=-1), v).permute(1, 0, 2)
            y = y.reshape(seq, NH * HD) @ w[p + 'self_attn.out_proj.weight'].T
        x = x + y
        f = rms(x, w[p + 'ffn_norm.weight'])
        x = x + (F.silu(f @ w[p + 'feed_forward.w1.weight'].T)
                 * (f @ w[p + 'feed_forward.w3.weight'].T)) @ w[p + 'feed_forward.w2.weight'].T
    # a final norm sits between the stack and the projector, same naming as the vision
    # tower; without it the encoder output is off by a per-token scale
    x = rms(x, w['model.audio_encoder.lfm.embedding_norm.weight'])
    pre = 'model.audio_projector.'
    y = F.layer_norm(x, (H,), w[pre + 'layer_norm.weight'], w[pre + 'layer_norm.bias'], 1e-5)
    y = F.gelu(y @ w[pre + 'linear_1.weight'].T + w[pre + 'linear_1.bias'])
    return x, y @ w[pre + 'linear_2.weight'].T + w[pre + 'linear_2.bias']


if __name__ == '__main__':
    w = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if 'audio_encoder' in k or 'audio_projector' in k:
                w[k] = f.get_tensor(k).float()
    data = np.load('/tmp/official_audio.npz')
    feats = torch.from_numpy(data['input_features']).float()
    enc, proj = forward(feats, w)
    ref_enc = torch.from_numpy(np.load('/tmp/official_audio_enc.npy'))
    ref_proj = torch.from_numpy(np.load('/tmp/official_audio_proj.npy'))
    print({'mine': list(enc.shape), 'official': list(ref_enc.shape),
           'cos': round(float(F.cosine_similarity(enc, ref_enc, dim=-1).mean()), 5),
           'rel': round(float((enc - ref_enc).norm() / ref_enc.norm()), 5),
           'proj_cos': round(float(F.cosine_similarity(proj, ref_proj, dim=-1).mean()), 5)},
          flush=True)