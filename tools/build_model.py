# Build the whole text backbone: 24 Core ML programs (one per layer, attention or
# conv, each with its MLP fused in), plus a final norm and the LM head.
#
# Verified pieces this rests on: layer 0 conv+MLP 0.315% error, layer 2 attention
# 0.326% error, both against PyTorch on real weights.
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
OUT = '/Users/vladimir/dev/lfm3-ane/model-28k'
H, FF, K, D = 2048, 7680, 7, 128
NH, NKV = 16, 4
EPS = 1e-5
FULL_ROWS = 28672   # the tile width the engine accepts
SLIDING_ROWS = 256
ROPE_THETA_FULL, ROPE_THETA_SLIDE = 1e6, 1e4
PARTIAL_FULL = 0.5


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class Rms(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w.clone())

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + EPS) * self.w.view(1, -1, 1, 1)


class Mlp(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.w1 = torch.nn.Conv2d(H, FF, 1, bias=False)
        self.w1.weight.data = t['feed_forward.w1.weight'].view(FF, H, 1, 1).clone()
        self.w3 = torch.nn.Conv2d(H, FF, 1, bias=False)
        self.w3.weight.data = t['feed_forward.w3.weight'].view(FF, H, 1, 1).clone()
        self.w2 = torch.nn.Conv2d(FF, H, 1, bias=False)
        self.w2.weight.data = t['feed_forward.w2.weight'].view(H, FF, 1, 1).clone()

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class ConvBlock(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.pre = Rms(t['operator_norm.weight'])
        self.post = Rms(t['post_operator_norm.weight'])
        self.ffn_norm = Rms(t['ffn_norm.weight'])
        self.ffn_post = Rms(t['post_ffn_norm.weight'])
        w = t['conv.in_proj.weight'].view(3, H, H)
        self.in_b = torch.nn.Conv2d(H, H, 1, bias=False)
        self.in_b.weight.data = w[0].view(H, H, 1, 1).clone()
        self.in_c = torch.nn.Conv2d(H, H, 1, bias=False)
        self.in_c.weight.data = w[1].view(H, H, 1, 1).clone()
        self.in_x = torch.nn.Conv2d(H, H, 1, bias=False)
        self.in_x.weight.data = w[2].view(H, H, 1, 1).clone()
        self.dw = torch.nn.Conv2d(H, H, (1, K), groups=H, bias=False)
        self.dw.weight.data = t['conv.conv.weight'].view(H, 1, 1, K).clone()
        self.out_proj = torch.nn.Conv2d(H, H, 1, bias=False)
        self.out_proj.weight.data = t['conv.out_proj.weight'].view(H, H, 1, 1).clone()
        self.mlp = Mlp(t)

    def forward(self, x, hist):
        h = self.pre(x)
        gated = self.in_b(h) * self.in_x(h)
        full = torch.cat([hist, gated], dim=3)
        new_hist = torch.narrow(full, 3, 1, K - 1)
        y = self.out_proj(self.in_c(h) * self.dw(full))
        h = self.post(y) + x
        return self.ffn_post(self.mlp(self.ffn_norm(h))) + h, new_hist


class AttnBlock(torch.nn.Module):
    def __init__(self, t, rows):
        super().__init__()
        self.rows = rows
        self.pre = Rms(t['operator_norm.weight'])
        self.post = Rms(t['post_operator_norm.weight'])
        self.ffn_norm = Rms(t['ffn_norm.weight'])
        self.ffn_post = Rms(t['post_ffn_norm.weight'])
        self.q_proj = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.q_proj.weight.data = t['self_attn.q_proj.weight'].view(NH * D, H, 1, 1).clone()
        self.k_proj = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.k_proj.weight.data = t['self_attn.k_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.v_proj = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.v_proj.weight.data = t['self_attn.v_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.o_proj = torch.nn.Conv2d(H, H, 1, bias=False)
        self.o_proj.weight.data = t['self_attn.o_proj.weight'].view(H, H, 1, 1).clone()
        self.q_norm = torch.nn.Parameter(t['self_attn.q_norm.weight'].clone())
        self.k_norm = torch.nn.Parameter(t['self_attn.k_norm.weight'].clone())
        self.gate = torch.nn.Conv2d(H, NH, 1, bias=False)
        self.gate.weight.data = t['self_attn.gate_proj.weight'].view(NH, H, 1, 1).clone()
        self.mlp = Mlp(t)
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, rows, D), persistent=False)

    def forward(self, x, pos, cos, sin, mask):
        h = self.pre(x)
        q = self.q_proj(h).view(1, NH, 1, D)
        k = self.k_proj(h).view(1, NKV, 1, D)
        v = self.v_proj(h).view(1, NKV, 1, D)

        def rms(z, w):
            return (z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + EPS)
                    * w.view(1, 1, 1, D))

        q = rms(q, self.q_norm)
        k = rms(k, self.k_norm)
        c, s = cos.view(1, 1, 1, D), sin.view(1, 1, 1, D)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q.view(1, NKV, 4, D), key) * 0.0884 + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), val).reshape(1, NH, D)
        attn = attn * torch.sigmoid(self.gate(h)).view(1, NH, 1)
        y = self.o_proj(attn.reshape(1, NH * D, 1, 1))
        h2 = self.post(y) + x
        return self.ffn_post(self.mlp(self.ffn_norm(h2))) + h2


class FinalNorm(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.n = Rms(w)

    def forward(self, x):
        return self.n(x)


class LmHead(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.p = torch.nn.Conv2d(H, w.shape[0], 1, bias=False)
        self.p.weight.data = w.view(w.shape[0], H, 1, 1).clone()

    def forward(self, x):
        return self.p(x)


def rope_tables(rows: int, theta: float, partial: float):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    angles = int(partial * D) // 2
    if angles < D // 2:
        inv = torch.cat([inv[:angles], torch.zeros(D // 2 - angles)])
    freqs = torch.arange(rows, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def save(model, inputs, states, path):
    example = [torch.zeros(*s, dtype=torch.int32 if s == (1,) else torch.float16)
               for s in inputs]
    traced = torch.jit.trace(model.half().eval(), tuple(example))
    ct.convert(traced,
               inputs=[ct.TensorType(name='in%d' % i, shape=tuple(s),
                                     dtype=np.int32 if s == (1,) else np.float16)
                       for i, s in enumerate(inputs)],
               states=states, convert_to='mlprogram',
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    return path


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    with safe_open(P, framework='pt') as f:
        keys = [k for k in f.keys() if k.startswith('model.language_model')]
        emb = f.get_tensor('model.language_model.embed_tokens.weight').to(torch.float32)
        final_w = f.get_tensor('model.language_model.embedding_norm.weight').to(torch.float32)
        types = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))
        layer_types = types['text_config']['layer_types']
        for i, kind in enumerate(layer_types):
            t = {k.split('layers.%d.' % i)[1]: f.get_tensor(k).to(torch.float32)
                 for k in keys if ('layers.%d.' % i) in k}
            if kind == 'conv':
                path = os.path.join(OUT, 'layer%d_conv.mlpackage' % i)
                if not os.path.exists(path):
                    save(ConvBlock(t), [(1, H, 1, 1), (1, H, 1, K - 1)], [], path)
            else:
                rows = SLIDING_ROWS if kind == 'sliding_attention' else FULL_ROWS
                theta = ROPE_THETA_SLIDE if kind == 'sliding_attention' else ROPE_THETA_FULL
                partial = 1.0 if kind == 'sliding_attention' else PARTIAL_FULL
                name = 'layer%d_%s.mlpackage' % (i, 'slide' if rows == SLIDING_ROWS else 'full')
                path = os.path.join(OUT, name)
                save(AttnBlock(t, rows),
                     [(1, H, 1, 1), (1,), (D,), (D,), (1, NKV, 1, rows)],
                     [ct.StateType(name='cache',
                                   wrapped_type=ct.TensorType(
                                       shape=(2 * NKV, 1, rows, D), dtype=np.float16))],
                   path)
            print({'built': name if kind != 'conv' else ('layer%d_conv.mlpackage' % i)}, flush=True)

    save(FinalNorm(final_w), [(1, H, 1, 1)], [], os.path.join(OUT, 'final_norm.mlpackage'))
    save(LmHead(emb), [(1, H, 1, 1)], [], os.path.join(OUT, 'lm_head.mlpackage'))
    np.save(os.path.join(OUT, 'embed_tokens.npy'), emb.half().numpy())
    f_cos, f_sin = rope_tables(FULL_ROWS, ROPE_THETA_FULL, PARTIAL_FULL)
    s_cos, s_sin = rope_tables(SLIDING_ROWS, ROPE_THETA_SLIDE, 1.0)
    np.savez(os.path.join(OUT, 'rope.npz'),
             full_cos=f_cos.half().numpy(), full_sin=f_sin.half().numpy(),
             slide_cos=s_cos.half().numpy(), slide_sin=s_sin.half().numpy())
    json.dump(layer_types, open(os.path.join(OUT, 'layer_types.json'), 'w'))
    print('done', flush=True)