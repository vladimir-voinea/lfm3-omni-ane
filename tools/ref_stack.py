import os

from paths import MODEL, PROJECT, SOURCE
# The same 24-layer stack in pure PyTorch, all tensors (1, T, H).
# Exposes forward(ids) -> logits for the last position, with the weights loaded once.
from __future__ import annotations

import json

import numpy as np
import torch
from safetensors import safe_open

P = os.path.join(SOURCE, 'model.safetensors')
CFG = os.path.join(MODEL, 'config.json')
H, D, NH, NKV, K = 2048, 128, 16, 4, 7
EPS = 1e-5

_W = None


def weights():
    global _W
    if _W is None:
        with safe_open(P, framework='pt') as f:
            _W = {k: f.get_tensor(k).float() for k in f.keys()}
    return _W


def rotate_half(x):
    return torch.cat([-x[..., D // 2:], x[..., :D // 2]], dim=-1)


def rope(T, theta, partial):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    angles = int(partial * D) // 2
    if angles < D // 2:
        inv = torch.cat([inv[:angles], torch.zeros(D // 2 - angles)])
    freq = torch.arange(T, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat([freq, freq], dim=-1)
    return emb.cos(), emb.sin()


def forward(ids):
    w = weights()
    g = lambda k: w[k]
    emb = g('model.language_model.embed_tokens.weight')
    final_w = g('model.language_model.embedding_norm.weight')
    layer_types = json.load(open(CFG))['text_config']['layer_types']
    T = len(ids)
    fc, fs = rope(T, 1e6, 0.5)
    sc, ss = rope(T, 1e4, 1.0)
    h = emb[torch.tensor(ids)].view(1, T, H)
    for i, kind in enumerate(layer_types):
        pre = 'model.language_model.layers.%d.' % i
        x = h
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * g(pre + 'operator_norm.weight')
        if kind == 'conv':
            ww = g(pre + 'conv.in_proj.weight').view(3, H, H)
            b = hn @ ww[0].T
            c = hn @ ww[1].T
            t = hn @ ww[2].T
            gated = b * t
            pad = torch.cat([torch.zeros(1, K - 1, H), gated], dim=1)
            win = pad.unfold(1, K, 1)
            dw = g(pre + 'conv.conv.weight').view(H, K)
            conv = (win * dw.view(1, 1, H, K)).sum(-1)
            y = (c * conv) @ g(pre + 'conv.out_proj.weight').T
        else:
            q = (hn @ g(pre + 'self_attn.q_proj.weight').T).view(1, T, NH, D)
            k = (hn @ g(pre + 'self_attn.k_proj.weight').T).view(1, T, NKV, D)
            v = (hn @ g(pre + 'self_attn.v_proj.weight').T).view(1, T, NKV, D)
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * g(pre + 'self_attn.q_norm.weight')
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * g(pre + 'self_attn.k_norm.weight')
            cos, sin = (sc, ss) if kind == 'sliding_attention' else (fc, fs)
            q = q * cos.view(1, T, 1, D) + rotate_half(q) * sin.view(1, T, 1, D)
            k = k * cos.view(1, T, 1, D) + rotate_half(k) * sin.view(1, T, 1, D)
            mask = torch.full((T, T), float('-inf')).triu(1)
            if kind == 'sliding_attention':
                idx = torch.arange(T)
                mask = mask.masked_fill(idx[None, :] < idx[:, None] - 255, float('-inf'))
            out = torch.zeros(T, NH, D)
            qh, kh, vh = q[0].permute(1, 0, 2), k[0].permute(1, 0, 2), v[0].permute(1, 0, 2)
            for hh in range(NH):
                kv = hh // (NH // NKV)
                s = qh[hh] @ kh[kv].T * 0.0884 + mask
                out[:, hh] = torch.softmax(s, dim=-1) @ vh[kv]
            gate = torch.sigmoid(hn @ g(pre + 'self_attn.gate_proj.weight').T)
            out = out * gate.view(T, NH, 1)
            y = out.reshape(1, T, NH * D) @ g(pre + 'self_attn.o_proj.weight').T
        h = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS) * g(pre + 'post_operator_norm.weight') + x
        hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * g(pre + 'ffn_norm.weight')
        up = torch.nn.functional.silu(hn @ g(pre + 'feed_forward.w1.weight').T) * (hn @ g(pre + 'feed_forward.w3.weight').T)
        ffn = up @ g(pre + 'feed_forward.w2.weight').T
        h = ffn * torch.rsqrt(ffn.pow(2).mean(-1, keepdim=True) + EPS) * g(pre + 'post_ffn_norm.weight') + h
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * final_w
    return (h[:, -1] @ emb.T)[0].detach().numpy()


if __name__ == '__main__':
    logits = forward([1, 1000, 2000, 3000])
    top = np.argsort(logits)[-5:][::-1]
    print({'ref_top5': top.tolist(), 'ref_values': [round(float(logits[i]), 3) for i in top]}, flush=True)
    np.save(os.path.join(PROJECT, 'ref_logits.npy'), logits.astype(np.float32))