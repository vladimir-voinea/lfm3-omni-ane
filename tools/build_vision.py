# The vision tower and projector as one Core ML program, at the fixed 224 x 224 the server feeds
# it (196 patches, 49 tokens for the text model).  Transcribed from vision_ref.py, which matches
# the official model at cosine 1.0, with the layout changed to suit the engine.
from __future__ import annotations

import json
import os
import sys

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from safetensors import safe_open

from paths import MODEL, TOWERS
from vision_ref import FF, H, HD, MERGE, NH, NKV, axial_rope, position_ids, rotate_half

GRID = 14
N = GRID * GRID
PATCH = 3 * 2 * 16 * 16
EPS = 1e-5
# the engine loses accuracy on silu and on a projection with small inputs; see notes/STATE.md
PROJECT_SCALE = 8.0


def silu_engine(x):
    return 0.5 * x * (1 + torch.tanh(0.5 * x))


def project(layer, x):
    return layer(x * PROJECT_SCALE) / PROJECT_SCALE


def linear(weight, bias=None):
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    layer.weight.data = weight.clone()
    if bias is not None:
        layer.bias.data = bias.clone()
    return layer


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


class Attn(torch.nn.Module):
    def __init__(self, t, cos, sin):
        super().__init__()
        self.q, self.k, self.v = (linear(t[n + '_proj.weight']) for n in 'qkv')
        self.out = linear(t['out_proj.weight'])
        self.q_norm = torch.nn.Parameter(t['q_layernorm.weight'].clone())
        self.k_norm = torch.nn.Parameter(t['k_layernorm.weight'].clone())
        self.register_buffer('cos', cos[:, None, :])
        self.register_buffer('sin', sin[:, None, :])

    def forward(self, x):
        q = rms(self.q(x).view(N, NH, HD), self.q_norm)
        k = rms(self.k(x).view(N, NKV, HD), self.k_norm)
        v = self.v(x).view(N, NKV, HD)
        q = q * self.cos + rotate_half(q) * self.sin
        k = k * self.cos + rotate_half(k) * self.sin
        rep = NH // NKV
        q = q.permute(1, 0, 2).view(NKV, rep, N, HD)
        k = k.permute(1, 0, 2).view(NKV, 1, N, HD)
        v = v.permute(1, 0, 2).view(NKV, 1, N, HD)
        scores = torch.matmul(q, k.transpose(2, 3)) * (HD ** -0.5)
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)
        return project(self.out, attn.view(NH, N, HD).permute(1, 0, 2).reshape(N, NH * HD))


class Conv(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.in_proj = linear(t['in_proj.weight'])
        self.spatial = torch.nn.Conv2d(H, H, 3, padding=1, groups=H, bias=False)
        self.spatial.weight.data = t['spatial_conv.weight'].clone()
        self.out = linear(t['out_proj.weight'])

    def forward(self, x):
        proj = self.in_proj(x)
        b, c, g = proj[:, :H], proj[:, H:2 * H], proj[:, 2 * H:]
        half = GRID // MERGE
        grid = (b * g).view(half, half, MERGE, MERGE, H).permute(4, 0, 2, 1, 3).reshape(1, H, GRID, GRID)
        grid = self.spatial(grid).view(H, half, MERGE, half, MERGE).permute(1, 3, 2, 4, 0)
        return project(self.out, c * grid.reshape(N, H))


class Layer(torch.nn.Module):
    def __init__(self, op, t):
        super().__init__()
        self.op = op
        self.op_norm = torch.nn.Parameter(t['operator_norm.weight'].clone())
        self.ffn_norm = torch.nn.Parameter(t['ffn_norm.weight'].clone())
        self.w1 = linear(t['feed_forward.w1.weight'])
        self.w2 = linear(t['feed_forward.w2.weight'])
        self.w3 = linear(t['feed_forward.w3.weight'])

    def forward(self, x):
        x = x + self.op(rms(x, self.op_norm))
        f = rms(x, self.ffn_norm)
        return x + project(self.w2, silu_engine(self.w1(f)) * self.w3(f))


class Tower(torch.nn.Module):
    def __init__(self, w, layer_types):
        super().__init__()
        cos, sin = axial_rope(*position_ids(GRID, GRID))
        self.patchify = linear(w['model.vision_tower.patchifier.weight'])
        self.norm = torch.nn.Parameter(w['model.vision_tower.embedding_norm.weight'].clone())
        layers = []
        for i, kind in enumerate(layer_types):
            prefix = 'model.vision_tower.layers.%d.' % i
            t = {k[len(prefix):]: v for k, v in w.items() if k.startswith(prefix)}
            sub = {k.split('.', 1)[1]: v for k, v in t.items()
                   if k.startswith('self_attn.') or k.startswith('conv.')}
            layers.append(Layer(Attn(sub, cos, sin) if kind == 'full_attention' else Conv(sub), t))
        self.layers = torch.nn.ModuleList(layers)
        pre = 'model.multi_modal_projector.'
        self.ln_w = torch.nn.Parameter(w[pre + 'layer_norm.weight'].clone())
        self.ln_b = torch.nn.Parameter(w[pre + 'layer_norm.bias'].clone())
        self.l1 = linear(w[pre + 'linear_1.weight'], w[pre + 'linear_1.bias'])
        self.l2 = linear(w[pre + 'linear_2.weight'], w[pre + 'linear_2.bias'])

    def forward(self, patches):
        x = rms(self.patchify(patches[0]), self.norm)
        for layer in self.layers:
            x = layer(x)
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + 1e-5) * self.ln_w + self.ln_b
        x = self.l1(x.view(N // 4, H * 4))
        return self.l2(torch.nn.functional.gelu(x))


def load():
    w = {}
    with safe_open(TOWERS, framework='pt') as f:
        for k in f.keys():
            if 'vision_tower' in k or 'multi_modal_projector' in k:
                w[k] = f.get_tensor(k).float()
    types = json.load(open(os.path.join(MODEL, 'config.json')))['vision_config']['layer_types']
    return w, types


if __name__ == '__main__':
    out = sys.argv[1]
    bits = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    w, types = load()
    tower = Tower(w, types).eval()
    example = torch.zeros(1, N, PATCH)
    traced = torch.jit.trace(tower, (example,))
    model = ct.convert(traced, inputs=[ct.TensorType(name='patches', shape=example.shape, dtype=np.float32)],
                       outputs=[ct.TensorType(name='tokens')], convert_to='mlprogram',
                       compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.iOS18,
                       compute_units=ct.ComputeUnit.CPU_AND_NE)
    if bits:
        config = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=bits))
        model = cto.palettize_weights(model, config)
    model.save(out)
    print({'saved': out}, flush=True)
