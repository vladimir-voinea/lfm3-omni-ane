# Where does the conversion diverge from PyTorch?  Compare stage by stage
# (norm only, then conv block, then the MLP), each converted and run.
from __future__ import annotations

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
H, FF, K = 2048, 7680, 7
EPS = 1e-5


def load():
    tensors = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if 'language_model.layers.0.' in k:
                tensors[k.split('layers.0.')[1]] = f.get_tensor(k).to(torch.float32)
    return tensors


def convert_and_compare(model, x, name, compare_at='out'):
    with torch.no_grad():
        ref = model(x)
    if isinstance(ref, tuple):
        ref = ref[0]
    half = model.half()
    traced = torch.jit.trace(half, x.half())
    core = ct.convert(traced,
                      inputs=[ct.TensorType(name='x', shape=tuple(x.shape),
                                            dtype=np.float16)],
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
    got = np.asarray(list(core.predict({'x': x.half().numpy()}).values())[0]).astype(np.float32)
    r = ref.detach().numpy().astype(np.float32)
    rel = float(np.linalg.norm(got - r) / max(np.linalg.norm(r), 1e-9))
    print({'stage': name, 'rel': round(rel, 6),
           'max_abs': round(float(np.abs(got - r).max()), 5)}, flush=True)
    return rel


class OnlyNorm(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone())

    def forward(self, x):
        v = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + EPS)
        return v * self.weight.view(1, -1, 1, 1)


class ConvBlock(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.norm = OnlyNorm(t['operator_norm.weight'])
        self.in_proj = torch.nn.Conv2d(H, 3 * H, 1, bias=False)
        self.in_proj.weight.data = t['conv.in_proj.weight'].view(3 * H, H, 1, 1).clone()
        self.dw = torch.nn.Conv2d(H, H, (1, K), groups=H, bias=False)
        self.dw.weight.data = t['conv.conv.weight'].view(H, 1, 1, K).clone()
        self.out_proj = torch.nn.Conv2d(H, H, 1, bias=False)
        self.out_proj.weight.data = t['conv.out_proj.weight'].view(H, H, 1, 1).clone()

    def forward(self, x):
        h = self.norm(x)
        proj = self.in_proj(h).view(1, 3, H, 1)
        b, c, t_ = proj[:, 0:1], proj[:, 1:2], proj[:, 2:3]
        gated = (b * t_).view(1, H, 1, 1).expand(1, H, 1, K)   # (1, H, 1, K)
        out = self.dw(gated.contiguous())
        y = c * out.view(1, 1, H, 1)
        return self.out_proj(y.view(1, H, 1, 1))


class MlpBlock(torch.nn.Module):
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


if __name__ == '__main__':
    t = load()
    torch.manual_seed(0)
    x = torch.randn(1, H, 1, 1) * 0.5
    convert_and_compare(OnlyNorm(t['operator_norm.weight']).eval(), x, 'norm')
    convert_and_compare(ConvBlock(t).eval(), x, 'conv_block')
    convert_and_compare(MlpBlock(t).eval(), x, 'mlp_block')