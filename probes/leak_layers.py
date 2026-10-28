# Return every intermediate of the real layer as a separate output and compare each,
# so the first divergence names the op that converts wrong.
from __future__ import annotations

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
H, FF, K = 2048, 7680, 7
EPS = 1e-5


class Probe(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.nw = torch.nn.Parameter(t['operator_norm.weight'].clone())
        w = t['conv.in_proj.weight'].view(3, H, H)
        self.b = torch.nn.Conv2d(H, H, 1, bias=False)
        self.b.weight.data = w[0].view(H, H, 1, 1).clone()
        self.c = torch.nn.Conv2d(H, H, 1, bias=False)
        self.c.weight.data = w[1].view(H, H, 1, 1).clone()
        self.xp = torch.nn.Conv2d(H, H, 1, bias=False)
        self.xp.weight.data = w[2].view(H, H, 1, 1).clone()
        self.dw = torch.nn.Conv2d(H, H, (1, K), groups=H, bias=False)
        self.dw.weight.data = t['conv.conv.weight'].view(H, 1, 1, K).clone()
        self.op = torch.nn.Conv2d(H, H, 1, bias=False)
        self.op.weight.data = t['conv.out_proj.weight'].view(H, H, 1, 1).clone()

    def forward(self, x, hist):
        v = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + EPS)
        h = v * self.nw.view(1, -1, 1, 1)               # norm
        bb = self.b(h)
        cc = self.c(h)
        xx = self.xp(h)
        gated = bb * xx
        full = torch.cat([hist, gated], dim=3)
        out = self.dw(full)
        new_hist = torch.narrow(full, 3, 1, K - 1)
        y = self.op(cc * out)
        return h, gated, out, y, new_hist


if __name__ == '__main__':
    t = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if 'language_model.layers.0.' in k:
                t[k.split('layers.0.')[1]] = f.get_tensor(k).to(torch.float32)
    torch.manual_seed(0)
    model = Probe(t).eval()
    x = torch.randn(1, H, 1, 1) * 0.5
    hist = torch.randn(1, H, 1, K - 1) * 0.5
    with torch.no_grad():
        refs = [r for r in model(x, hist)]
    half = model.half()
    traced = torch.jit.trace(half, (x.half(), hist.half()))
    core = ct.convert(traced,
                      inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                              ct.TensorType(name='hist', shape=(1, H, 1, K - 1),
                                            dtype=np.float16)],
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
    out = core.predict({'x': x.half().numpy(), 'hist': hist.half().numpy()})
    names = ['norm', 'gated', 'conv', 'out_proj', 'new_hist']
    for name, ref, got in zip(names, refs, out.values()):
        r = ref.float().numpy().astype(np.float32)
        g = np.asarray(got).astype(np.float32)
        if r.shape != g.shape:
            print({'name': name, 'shape_mismatch': (r.shape, g.shape)}, flush=True)
            continue
        rel = float(np.linalg.norm(g - r) / max(np.linalg.norm(r), 1e-9))
        print({'name': name, 'rel': round(rel, 5),
               'max_abs': round(float(np.abs(g - r).max()), 5),
               'ref_norm': round(float(np.linalg.norm(r)), 2)}, flush=True)