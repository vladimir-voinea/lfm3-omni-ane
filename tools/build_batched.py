# Sequence-batched layer programs: T tokens per dispatch, for prefill.
# Same math as build_model's blocks, with the position axis generalised from 1 to T.
import json
import os
import sys

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from paths import SOURCE, PROJECT

H, NH, NKV, D, K = 2048, 16, 4, 128, 7
FULL_ROWS, SLIDE_ROWS = 28672, 256
T = int(os.environ.get('LFM3_T', '8'))
OUT = os.path.join(PROJECT, 'model-28k-b%d' % T)


def rms(z, w, eps=1e-5):
    """Channel-first normalisation: the hidden axis is dim 1, as in build_model."""
    return z * torch.rsqrt(z.pow(2).mean(dim=1, keepdim=True) + eps) * w.view(1, -1, 1, 1)


def rms_head(z, w, eps=1e-5):
    """Per-head normalisation for q and k after the reshape: head dim is last."""
    return (z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + eps) * w.view(1, 1, 1, -1))


def rotate_half(z):
    half = z.shape[-1] // 2
    return torch.cat([-z[..., half:], z[..., :half]], dim=-1)


class Rms(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w.clone())

    def forward(self, x):
        return rms(x, self.w)


class Mlp(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.gate = torch.nn.Conv2d(H, 7680, 1, bias=False)
        self.up = torch.nn.Conv2d(H, 7680, 1, bias=False)
        self.down = torch.nn.Conv2d(7680, H, 1, bias=False)
        self.gate.weight.data = t['feed_forward.w1.weight'].view(7680, H, 1, 1).clone()
        self.up.weight.data = t['feed_forward.w3.weight'].view(7680, H, 1, 1).clone()
        self.down.weight.data = t['feed_forward.w2.weight'].view(H, 7680, 1, 1).clone()

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class SeqConv(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.pre = Rms(t['operator_norm.weight'])
        self.post = Rms(t['post_operator_norm.weight'])
        self.ffn_norm = Rms(t['ffn_norm.weight'])
        self.ffn_post = Rms(t['post_ffn_norm.weight'])
        w = t['conv.in_proj.weight'].view(3, H, H)
        self.in_b = torch.nn.Conv2d(H, H, 1, bias=False)
        self.in_c = torch.nn.Conv2d(H, H, 1, bias=False)
        self.in_x = torch.nn.Conv2d(H, H, 1, bias=False)
        self.in_b.weight.data = w[0].view(H, H, 1, 1).clone()
        self.in_c.weight.data = w[1].view(H, H, 1, 1).clone()
        self.in_x.weight.data = w[2].view(H, H, 1, 1).clone()
        # the original single-position build parks the history in the W axis because the
        # sequence axis has length 1; with T positions the window has to run along the
        # sequence (dim 2) instead
        self.dw = torch.nn.Conv2d(H, H, (K, 1), groups=H, bias=False)
        self.dw.weight.data = t['conv.conv.weight'].view(H, 1, 1, K).permute(0, 1, 3, 2).contiguous().clone()
        self.out_proj = torch.nn.Conv2d(H, H, 1, bias=False)
        self.out_proj.weight.data = t['conv.out_proj.weight'].view(H, H, 1, 1).clone()
        self.mlp = Mlp(t)

    def forward(self, x, hist):
        h = self.pre(x)
        gated = self.in_b(h) * self.in_x(h)
        full = torch.cat([hist, gated], dim=2)          # (1, H, T + K - 1, 1)
        new_hist = torch.narrow(full, 2, T, K - 1)
        y = self.out_proj(self.in_c(h) * self.dw(full))
        h = self.post(y) + x
        return self.ffn_post(self.mlp(self.ffn_norm(h))) + h, new_hist


class SeqAttn(torch.nn.Module):
    def __init__(self, t, rows):
        super().__init__()
        self.rows = rows
        self.pre = Rms(t['operator_norm.weight'])
        self.post = Rms(t['post_operator_norm.weight'])
        self.ffn_norm = Rms(t['ffn_norm.weight'])
        self.ffn_post = Rms(t['post_ffn_norm.weight'])
        self.q_proj = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.k_proj = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.v_proj = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.o_proj = torch.nn.Conv2d(H, H, 1, bias=False)
        self.q_proj.weight.data = t['self_attn.q_proj.weight'].view(NH * D, H, 1, 1).clone()
        self.k_proj.weight.data = t['self_attn.k_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.v_proj.weight.data = t['self_attn.v_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.o_proj.weight.data = t['self_attn.o_proj.weight'].view(H, H, 1, 1).clone()
        self.q_norm = torch.nn.Parameter(t['self_attn.q_norm.weight'].clone())
        self.k_norm = torch.nn.Parameter(t['self_attn.k_norm.weight'].clone())
        self.gate = torch.nn.Conv2d(H, NH, 1, bias=False)
        self.gate.weight.data = t['self_attn.gate_proj.weight'].view(NH, H, 1, 1).clone()
        self.mlp = Mlp(t)
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, rows, D), persistent=False)

    def forward(self, x, pos, cos, sin, mask):
        h = self.pre(x)
        q = self.q_proj(h).view(1, NH, T, D)
        k = self.k_proj(h).view(1, NKV, T, D)
        v = self.v_proj(h).view(1, NKV, T, D)
        q = rms_head(q, self.q_norm)
        k = rms_head(k, self.k_norm)
        c, s = cos.view(1, 1, T, D), sin.view(1, 1, T, D)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        self.cache[0:NKV, :, pos:pos + T, :] = k.view(NKV, 1, T, D)
        self.cache[NKV:2 * NKV, :, pos:pos + T, :] = v.view(NKV, 1, T, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        # the grouped-query trick flattens (head, position) into one axis, so the mask has to
        # carry the same layout: each position's row repeated once per head in the group
        scores = torch.matmul(q.view(1, NKV, T * 4, D), key) * 0.0884 + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), val).reshape(1, NH, T, D)
        attn = attn * torch.sigmoid(self.gate(h)).view(1, NH, T, 1)
        y = self.o_proj(attn.reshape(1, NH * D, T, 1))
        h2 = self.post(y) + x
        return self.ffn_post(self.mlp(self.ffn_norm(h2))) + h2


def save(model, inputs, states, path):
    example = [torch.zeros(*s, dtype=torch.int32 if s == (1,) else torch.float16) for s in inputs]
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
    types = json.load(open(os.path.join(PROJECT, 'model-28k-lut6', 'layer_types.json')))
    tensors = {}
    with safe_open(os.path.join(SOURCE, 'model.safetensors'), framework='pt') as f:
        for key in f.keys():
            if key.startswith('model.language_model.layers.'):
                index = int(key.split('layers.')[1].split('.')[0])
                tensors.setdefault(index, {})[key.split('layers.%d.' % index)[1]] = f.get_tensor(key).float()
    print({'layers': len(tensors), 'T': T}, flush=True)
    for i in sorted(tensors):
        kind = types[i]
        if kind == 'conv':
            path = os.path.join(OUT, 'layer%d_conv.mlpackage' % i)
            if not os.path.exists(path):
                save(SeqConv(tensors[i]), [(1, H, T, 1), (1, H, K - 1, 1)], [], path)
        else:
            rows = SLIDE_ROWS if kind == 'sliding_attention' else FULL_ROWS
            tag = 'slide' if rows == SLIDE_ROWS else 'full'
            path = os.path.join(OUT, 'layer%d_%s.mlpackage' % (i, tag))
            if not os.path.exists(path):
                save(SeqAttn(tensors[i], rows),
                     [(1, H, T, 1), (1,), (T, D), (T, D), (1, NKV, T * 4, rows)],
                     [ct.StateType(name='cache', wrapped_type=ct.TensorType(
                         shape=(2 * NKV, 1, rows, D), dtype=np.float16))], path)
        print('built', i, kind, flush=True)
    print('done', flush=True)