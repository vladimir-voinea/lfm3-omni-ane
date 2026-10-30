# One op, one output: the depthwise (1,7) conv over the time axis, torch vs Core ML.
from __future__ import annotations

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
H, K = 2048, 7


class Dw(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.dw = torch.nn.Conv2d(H, H, (1, K), groups=H, bias=False)
        self.dw.weight.data = w.view(H, 1, 1, K).clone()

    def forward(self, full):
        return self.dw(full)


class Proj(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.p = torch.nn.Conv2d(H, H, 1, bias=False)
        self.p.weight.data = w.view(H, H, 1, 1).clone()

    def forward(self, x):
        return self.p(x)


def check(model, shape, name):
    torch.manual_seed(0)
    x = torch.randn(*shape) * 0.5
    with torch.no_grad():
        ref = model(x).numpy().astype(np.float32)
    half = model.half()
    traced = torch.jit.trace(half, x.half())
    core = ct.convert(traced,
                      inputs=[ct.TensorType(name='x', shape=shape, dtype=np.float16)],
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
    got = np.asarray(list(core.predict({'x': x.half().numpy()}).values())[0]).astype(np.float32)
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))
    print({'op': name, 'rel': round(rel, 5), 'max_abs': round(float(np.abs(got - ref).max()), 5),
           'ref_norm': round(float(np.linalg.norm(ref)), 3)}, flush=True)


if __name__ == '__main__':
    w_dw = w_proj = None
    with safe_open(P, framework='pt') as f:
        w_dw = f.get_tensor('model.language_model.layers.0.conv.conv.weight').float()
        w_proj = f.get_tensor('model.language_model.layers.0.conv.out_proj.weight').float()
    check(Dw(w_dw).eval(), (1, H, 1, K), 'depthwise_conv_1x7')
    check(Proj(w_proj).eval(), (1, H, 1, 1), 'conv1x1')