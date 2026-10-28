# One real decoder layer (LFM3 layer 0: short conv + SwiGLU MLP + sandwich norms),
# converted with its actual weights and checked against a PyTorch reference of the
# same equations.  This is the numerics gate before converting 24 of them.
from __future__ import annotations

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
H, FF, K = 2048, 7680, 7
EPS = 1e-5


class RmsNorm(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone())

    def forward(self, x):
        v = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + EPS)
        return v * self.weight.view(1, -1, 1, 1)


class ConvLayer(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.operator_norm = RmsNorm(t['operator_norm.weight'])
        self.post_operator_norm = RmsNorm(t['post_operator_norm.weight'])
        self.ffn_norm = RmsNorm(t['ffn_norm.weight'])
        self.post_ffn_norm = RmsNorm(t['post_ffn_norm.weight'])
        # three separate 1x1 projections instead of one 3H-wide projection sliced
        # afterwards: every tensor then stays in (1, H, 1, T) layout and no view
        # follows a cat, which is the shape the converter got wrong
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
        self.w1 = torch.nn.Conv2d(H, FF, 1, bias=False)
        self.w1.weight.data = t['feed_forward.w1.weight'].view(FF, H, 1, 1).clone()
        self.w3 = torch.nn.Conv2d(H, FF, 1, bias=False)
        self.w3.weight.data = t['feed_forward.w3.weight'].view(FF, H, 1, 1).clone()
        self.w2 = torch.nn.Conv2d(FF, H, 1, bias=False)
        self.w2.weight.data = t['feed_forward.w2.weight'].view(H, FF, 1, 1).clone()

    def forward(self, x, hist):
        residual = x
        h = self.operator_norm(x)
        gated = self.in_b(h) * self.in_x(h)            # (1, H, 1, 1)
        full = torch.cat([hist, gated], dim=3)          # (1, H, 1, 7)
        out = self.dw(full)                             # (1, H, 1, 1)
        new_hist = torch.narrow(full, 3, 1, K - 1)
        y = self.out_proj(self.in_c(h) * out)
        h = self.post_operator_norm(y) + residual
        ffn = self.w2(torch.nn.functional.silu(self.w1(self.ffn_norm(h)))
                      * self.w3(self.ffn_norm(h)))
        return self.post_ffn_norm(ffn) + h, new_hist


if __name__ == '__main__':
    tensors = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if 'language_model.layers.0.' in k:
                tensors[k.split('layers.0.')[1]] = f.get_tensor(k).to(torch.float32)

    torch.manual_seed(0)
    model = ConvLayer(tensors).eval()
    x = torch.randn(1, H, 1, 1) * 0.5
    hist = torch.randn(1, H, 1, K - 1) * 0.5

    with torch.no_grad():
        ref_y, ref_hist = model(x, hist)

    half = model.half()
    with torch.no_grad():
        ref16_y, _ = half(x.half(), hist.half())
    ref16 = ref16_y.float().numpy().astype(np.float32)
    traced = torch.jit.trace(half, (x.half(), hist.half()))
    conv = ct.convert(traced,
                      inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                              ct.TensorType(name='hist', shape=(1, H, 1, K - 1),
                                            dtype=np.float16)],
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
    out = conv.predict({'x': x.half().numpy(), 'hist': hist.half().numpy()})
    # Core ML does not preserve the tuple order of a traced function's outputs, so
    # pick them by shape: the (1,H,1,K-1) one is the history, the other is y.
    y_arr = h_arr = None
    for value in out.values():
        arr = np.asarray(value)
        if arr.shape == (1, H, 1, K - 1):
            h_arr = arr.astype(np.float32)
        elif arr.shape == (1, H, 1, 1):
            y_arr = arr.astype(np.float32)
    got = y_arr
    ref = ref_y.numpy().astype(np.float32)
    diff = np.abs(got - ref)
    d16 = np.abs(got - ref16)
    print({'vs_torch_fp16_rel': round(float(np.linalg.norm(got - ref16) / max(np.linalg.norm(ref16), 1e-9)), 5),
           'vs_torch_fp16_max_abs': round(float(d16.max()), 5),
           'vs_torch_fp32_rel': round(float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9)), 5)}, flush=True)
    print({'ref_norm': float(np.linalg.norm(ref)),
           'out_norm': float(np.linalg.norm(got)),
           'max_abs_diff': float(diff.max()),
           'mean_abs_diff': float(diff.mean()),
           'rel': float(np.linalg.norm(got - ref) / np.linalg.norm(ref))}, flush=True)
    hist_out = h_arr
    print({'hist_max_diff': float(np.abs(hist_out - ref_hist.numpy()).max())}, flush=True)