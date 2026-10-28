import os

from paths import SOURCE
# Tiled full attention, verified before it is scaled to 75k.
#
# A tile program owns one slice of the sequence (its own KV cache) and returns an
# unnormalised partial: weighted values, the row max and the sum of exps.  An
# epilogue merges the tiles with the usual log-sum-exp, then runs gate, o_proj and
# the MLP.  Here the sequence is 40 tokens over 3 tiles of 16 so the whole thing is
# checkable against plain full attention.
from __future__ import annotations

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

P = os.path.join(SOURCE, 'model.safetensors')
H, D, NH, NKV, FF = 2048, 128, 16, 4, 7680
EPS, SCALE = 1e-5, 0.0884


def rotate_half(x):
    return torch.cat([-x[..., D // 2:], x[..., :D // 2]], dim=-1)


class Tile(torch.nn.Module):
    def __init__(self, t, rows):
        super().__init__()
        self.rows = rows
        self.pre = torch.nn.Parameter(t['operator_norm.weight'].clone())
        self.q = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.q.weight.data = t['self_attn.q_proj.weight'].view(NH * D, H, 1, 1).clone()
        self.k = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.k.weight.data = t['self_attn.k_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.v = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.v.weight.data = t['self_attn.v_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.qn = torch.nn.Parameter(t['self_attn.q_norm.weight'].clone())
        self.kn = torch.nn.Parameter(t['self_attn.k_norm.weight'].clone())
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, rows, D), persistent=False)

    def forward(self, x, pos, cos, sin, mask):
        h = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + EPS) * self.pre.view(1, -1, 1, 1)
        q = self.q(h).view(1, NH, 1, D)
        k = self.k(h).view(1, NKV, 1, D)
        v = self.v(h).view(1, NKV, 1, D)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * self.qn.view(1, 1, 1, D)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * self.kn.view(1, 1, 1, D)
        c, s = cos.view(1, 1, 1, D), sin.view(1, 1, 1, D)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q.view(1, NKV, 4, D), key) * SCALE + mask
        m = scores.max(dim=-1, keepdim=True).values
        e = torch.exp(scores - m)
        ssum = e.sum(dim=-1)
        wv = torch.matmul(e, val)
        return wv, m.squeeze(-1), ssum


class Epilogue(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.post = torch.nn.Parameter(t['post_operator_norm.weight'].clone())
        self.ffn_norm = torch.nn.Parameter(t['ffn_norm.weight'].clone())
        self.ffn_post = torch.nn.Parameter(t['post_ffn_norm.weight'].clone())
        self.o = torch.nn.Conv2d(H, H, 1, bias=False)
        self.o.weight.data = t['self_attn.o_proj.weight'].view(H, H, 1, 1).clone()
        self.gate = torch.nn.Conv2d(H, NH, 1, bias=False)
        self.gate.weight.data = t['self_attn.gate_proj.weight'].view(NH, H, 1, 1).clone()
        self.w1 = torch.nn.Conv2d(H, FF, 1, bias=False)
        self.w1.weight.data = t['feed_forward.w1.weight'].view(FF, H, 1, 1).clone()
        self.w3 = torch.nn.Conv2d(H, FF, 1, bias=False)
        self.w3.weight.data = t['feed_forward.w3.weight'].view(FF, H, 1, 1).clone()
        self.w2 = torch.nn.Conv2d(FF, H, 1, bias=False)
        self.w2.weight.data = t['feed_forward.w2.weight'].view(H, FF, 1, 1).clone()

    def forward(self, x, h, wv0, m0, s0, wv1, m1, s1, wv2, m2, s2):
        m = torch.maximum(torch.maximum(m0, m1), m2)
        def scale_of(mi):
            return torch.exp(mi - m).unsqueeze(-1)
        num = wv0 * scale_of(m0) + wv1 * scale_of(m1) + wv2 * scale_of(m2)
        den = (s0 * torch.exp(m0 - m).squeeze(-1) + s1 * torch.exp(m1 - m).squeeze(-1)
               + s2 * torch.exp(m2 - m).squeeze(-1)).unsqueeze(-1)
        attn = (num / den).reshape(1, NH, D)
        g = torch.sigmoid(self.gate(h)).view(1, NH, 1)
        y = self.o((attn * g).reshape(1, NH * D, 1, 1))
        h1 = y * torch.rsqrt(y.pow(2).mean(dim=1, keepdim=True) + EPS) * self.post.view(1, -1, 1, 1) + x
        hn = h1 * torch.rsqrt(h1.pow(2).mean(dim=1, keepdim=True) + EPS) * self.ffn_norm.view(1, -1, 1, 1)
        f = self.w2(torch.nn.functional.silu(self.w1(hn)) * self.w3(hn))
        return f * torch.rsqrt(f.pow(2).mean(dim=1, keepdim=True) + EPS) * self.ffn_post.view(1, -1, 1, 1) + h1


def rope(T, theta, partial):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    angles = int(partial * D) // 2
    if angles < D // 2:
        inv = torch.cat([inv[:angles], torch.zeros(D // 2 - angles)])
    freq = torch.arange(T, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat([freq, freq], dim=-1)
    return emb.cos(), emb.sin()