# Tiled full attention: tile programs return (weighted values, row max, sum of exps);
# the host merges them with log-sum-exp; a lean epilogue applies gate, o_proj and MLP.
from __future__ import annotations

import numpy as np
import torch

H, D, NH, NKV, FF = 2048, 128, 16, 4, 7680
EPS, SCALE = 1e-5, 0.0884


def rotate_half(x):
    return torch.cat([-x[..., D // 2:], x[..., :D // 2]], dim=-1)


def rms1(x, w):
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + EPS) * w.view(1, -1, 1, 1)


class Tile(torch.nn.Module):
    def __init__(self, t, rows, write=True):
        super().__init__()
        # one extra row: the scratch slot a full tile writes into, always masked out
        # of the attention.  Writing something is what keeps this ONE program able to
        # serve both roles -- two programs cannot share one MLState, and a gated
        # read-modify-write of the slot kills the ANE plan (-14).
        self.rows = rows + 1
        self.write = write
        self.pre = torch.nn.Parameter(t['operator_norm.weight'].clone())
        self.wq = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.wq.weight.data = t['self_attn.q_proj.weight'].view(NH * D, H, 1, 1).clone()
        self.wk = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.wk.weight.data = t['self_attn.k_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.wv = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.wv.weight.data = t['self_attn.v_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.qn = torch.nn.Parameter(t['self_attn.q_norm.weight'].clone())
        self.kn = torch.nn.Parameter(t['self_attn.k_norm.weight'].clone())
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, self.rows, D), persistent=False)

    def forward(self, x, pos, cos, sin, mask):
        h = rms1(x, self.pre)
        q = self.wq(h).view(1, NH, 1, D)
        k = self.wk(h).view(1, NKV, 1, D)
        v = self.wv(h).view(1, NKV, 1, D)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * self.qn.view(1, 1, 1, D)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * self.kn.view(1, 1, 1, D)
        c, s = cos.view(1, 1, 1, D), sin.view(1, 1, 1, D)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        # A tile that is already full must still be READ but must not write, and a
        # gated read-modify-write of the slot kills the ANE plan (-14).  So there are
        # two programs: one that writes its row, one that is read-only.
        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q.view(1, NKV, 4, D), key) * SCALE + mask
        # fp16 exp underflows below 6e-5: over a long cache most weights are smaller
        # than that, and each one is a lost contribution to the denominator.  The
        # softmax phase therefore runs in fp32; the partials handed back are tiny
        # (4x4x128 and 4x4 numbers), so the transfer cost of fp32 is nothing.
        s32 = scores.float()
        m = s32.max(dim=-1, keepdim=True).values
        e = torch.exp(s32 - m)
        return (torch.matmul(e, val.float()), m, e.sum(dim=-1))


class Epilogue(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        # the per-head gate is a function of the NORMALISED hidden state, not the raw
        # one -- the tile program normalises before projecting, so the epilogue has to
        # normalise the same way or the gate is applied at the wrong scale
        self.pre = torch.nn.Parameter(t['operator_norm.weight'].clone())
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

    def forward(self, x, attn):
        # attn arrives flat as (1, NH*D, 1, 1); the per-head gate is built by
        # repeating each of the NH gate scalars across its D channels
        hn = rms1(x, self.pre)
        g = torch.sigmoid(self.gate(hn)).expand(1, NH, 1, D).reshape(1, NH * D, 1, 1)
        y = self.o(attn * g)
        h1 = rms1(y, self.post) + x
        hn = rms1(h1, self.ffn_norm)
        f = self.w2(torch.nn.functional.silu(self.w1(hn)) * self.w3(hn))
        return rms1(f, self.ffn_post) + h1


def rope(T, theta, partial):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    angles = int(partial * D) // 2
    if angles < D // 2:
        inv = torch.cat([inv[:angles], torch.zeros(D // 2 - angles)])
    freq = torch.arange(T, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat([freq, freq], dim=-1)
    return emb.cos(), emb.sin()