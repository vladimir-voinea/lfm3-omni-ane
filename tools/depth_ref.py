import os

from paths import SOURCE
# Depth decoder transcribed: 3 bidirectional LFM3 layers over the 12 codebook slots.
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

P = os.path.join(SOURCE, 'model.safetensors')
PRE = 'model.depth_decoder.'
H, NH, NKV, HD, FF, CB = 1024, 32, 8, 32, 2816, 12
EPS, THETA = 1e-5, 1e6


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)


def rope(seq):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float) / HD))
    freqs = torch.arange(seq, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def forward(hidden, codes, masked, w):
    frames = hidden.shape[0]
    # depth_linear has a bias -- easy to miss, because a key listing filtered on
    # 'depth_linear.weight' silently drops it
    x = (hidden @ w[PRE + 'depth_linear.weight'].T
         + w[PRE + 'depth_linear.bias']).view(frames, CB, H)
    embeds = torch.stack([w[PRE + 'depth_embeddings.%d.embedding.weight' % i][codes[:, i]]
                          for i in range(CB)], dim=1)
    mask_emb = w[PRE + 'mask_embedding']
    x = x + torch.where(masked[..., None], mask_emb, embeds)
    cos, sin = rope(CB)
    for layer in range(3):
        p = PRE + 'model.layers.%d.' % layer
        # per frame the 12 slots attend to each other bidirectionally
        out = torch.zeros_like(x)
        hn = rms(x, w[p + 'operator_norm.weight'])
        q = (hn @ w[p + 'self_attn.q_proj.weight'].T).view(frames, CB, NH, HD)
        k = (hn @ w[p + 'self_attn.k_proj.weight'].T).view(frames, CB, NKV, HD)
        v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(frames, CB, NKV, HD)
        q = rms(q, w[p + 'self_attn.q_norm.weight'])
        k = rms(k, w[p + 'self_attn.k_norm.weight'])
        q = q * cos[None, :, None, :] + rotate_half(q) * sin[None, :, None, :]
        k = k * cos[None, :, None, :] + rotate_half(k) * sin[None, :, None, :]
        q = q.permute(0, 2, 1, 3)
        k = k.repeat_interleave(NH // NKV, dim=2).permute(0, 2, 1, 3)
        v = v.repeat_interleave(NH // NKV, dim=2).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(2, 3)) * (HD ** -0.5)
        attn = torch.matmul(torch.softmax(scores, dim=-1), v).permute(0, 2, 1, 3)
        attn = attn.reshape(frames, CB, NH * HD)
        gate = torch.sigmoid(hn @ w[p + 'self_attn.gate_proj.weight'].T)
        attn = attn * gate.repeat_interleave(HD, dim=-1)
        x = x + attn @ w[p + 'self_attn.o_proj.weight'].T
        f = rms(x, w[p + 'ffn_norm.weight'])
        x = x + (F.silu(f @ w[p + 'feed_forward.w1.weight'].T)
                 * (f @ w[p + 'feed_forward.w3.weight'].T)) @ w[p + 'feed_forward.w2.weight'].T
    # the trunk ends with its own norm, exactly like the audio encoder's stack
    x = rms(x, w[PRE + 'model.embedding_norm.weight'])
    logits = []
    for i in range(CB):
        emb = w[PRE + 'depth_embeddings.%d.embedding.weight' % i]
        norm = w[PRE + 'depth_embeddings.%d.embedding_norm.weight' % i]
        logits.append(rms(x[:, i], norm) @ emb.T)
    return torch.stack(logits, dim=1)


if __name__ == '__main__':
    w = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if k.startswith(PRE):
                w[k] = f.get_tensor(k).float()
    data = np.load('/tmp/depth_input.npz')
    hidden = torch.from_numpy(data['hidden']).float()
    codes = torch.from_numpy(data['codes']).long()
    masked = torch.from_numpy(data['masked'])
    mine = forward(hidden, codes, masked, w)
    ref = torch.from_numpy(np.load('/tmp/official_depth_logits.npy'))
    print({'mine': list(mine.shape), 'ref': list(ref.shape),
           'cos': round(float(F.cosine_similarity(mine.reshape(-1, mine.shape[-1]),
                                                 ref.reshape(-1, ref.shape[-1]), dim=-1).mean()), 5),
           'rel': round(float((mine - ref).norm() / ref.norm()), 5)}, flush=True)