# The attention layer with real weights: q/k/v projections, per-head q_norm and
# k_norm, Llama-style rotate_half RoPE in fp32 (partial for full-attention layers),
# the per-head sigmoid gate, o_proj, over a real KV state cache.
from __future__ import annotations

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
H, NH, NKV, D = 2048, 16, 4, 128
ROWS = 512
EPS = 1e-5


def rotate_half(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


class AttnLayer(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
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
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, ROWS, D), persistent=False)

    def forward(self, x, pos, cos, sin, mask):
        q = self.q_proj(x).view(1, NH, 1, D)
        k = self.k_proj(x).view(1, NKV, 1, D)
        v = self.v_proj(x).view(1, NKV, 1, D)

        def rms(z, w):
            n = z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + EPS)
            return n * w.view(1, 1, 1, D)

        q = rms(q, self.q_norm)
        k = rms(k, self.k_norm)
        c = cos.view(1, 1, 1, D)
        s = sin.view(1, 1, 1, D)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s

        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, ROWS, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, ROWS, D)

        scores = torch.matmul(q.view(1, NKV, 4, D), key) * 0.0884 + mask
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, val)                      # (1, NKV, 4, D)
        attn = attn.reshape(1, NH, D)
        g = torch.sigmoid(self.gate(x)).view(1, NH, 1)
        attn = attn * g
        return self.o_proj(attn.reshape(1, NH * D, 1, 1))


def rope_tables(pos: int, theta: float, partial: float):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    angles = int(partial * D) // 2
    inv = torch.cat([inv[:angles], torch.zeros(D // 2 - angles)])
    freqs = torch.arange(pos, pos + 1, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[0], emb.sin()[0]


if __name__ == '__main__':
    t = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if 'language_model.layers.2.' in k:
                t[k.split('layers.2.')[1]] = f.get_tensor(k).to(torch.float32)
    torch.manual_seed(0)
    model = AttnLayer(t).eval()
    x = torch.randn(1, H, 1, 1) * 0.5
    cos, sin = rope_tables(7, 1e6, 0.5)
    mask = torch.zeros(1, NKV, 1, ROWS)
    pos = torch.zeros(1, dtype=torch.int32)

    with torch.no_grad():
        ref = model(x, pos, cos, sin, mask)

    half = model.half()
    traced = torch.jit.trace(half, (x.half(), pos, cos.half(), sin.half(), mask.half()))
    core = ct.convert(traced,
                      inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                              ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                              ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
                              ct.TensorType(name='sin', shape=(D,), dtype=np.float16),
                              ct.TensorType(name='mask', shape=(1, NKV, 1, ROWS),
                                            dtype=np.float16)],
                      states=[ct.StateType(name='cache',
                                           wrapped_type=ct.TensorType(
                                               shape=(2 * NKV, 1, ROWS, D),
                                               dtype=np.float16))],
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = core.make_state()
    out = core.predict({'x': x.half().numpy(), 'pos': pos.numpy(),
                        'cos': cos.half().numpy(), 'sin': sin.half().numpy(),
                        'mask': mask.half().numpy()}, state=state)
    got = np.asarray(list(out.values())[0]).astype(np.float32)
    r = ref.detach().numpy().astype(np.float32)
    print({'rel': round(float(np.linalg.norm(got - r) / max(np.linalg.norm(r), 1e-9)), 5),
           'max_abs': round(float(np.abs(got - r).max()), 5),
           'ref_norm': round(float(np.linalg.norm(r)), 3)}, flush=True)