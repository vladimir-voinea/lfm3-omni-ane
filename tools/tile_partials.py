# Does the TILE program compute the right partials?  Feed it 6 positions, then compare
# its (weighted values, max, sum) against the same numbers computed in torch fp32.
import sys

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from tiled_parts2 import D, EPS, H, NKV, NH, SCALE, Tile, rotate_half, rms1, rope

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
ROWS, STEPS = 16, 6

t = {}
with safe_open(P, framework='pt') as fh:
    for k in fh.keys():
        if 'language_model.layers.2.' in k:
            t[k.split('layers.2.')[1]] = fh.get_tensor(k).float()

torch.manual_seed(0)
x = torch.randn(STEPS, H) * 0.5
cos, sin = rope(64, 1e6, 0.5)

tile = Tile(t, ROWS).eval().half()
ex = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
      torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16),
      torch.zeros(1, NKV, 1, ROWS + 1, dtype=torch.float16))
prog = ct.convert(torch.jit.trace(tile, ex),
                  inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                          ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                          ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
                          ct.TensorType(name='sin', shape=(D,), dtype=np.float16),
                          ct.TensorType(name='mask', shape=(1, NKV, 1, ROWS + 1), dtype=np.float16)],
                  states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
                      shape=(2 * NKV, 1, ROWS + 1, D), dtype=np.float16))],
                  compute_precision=ct.precision.FLOAT16,
                  minimum_deployment_target=ct.target.iOS18,
                  compute_units=ct.ComputeUnit.CPU_AND_NE)
state = prog.make_state()

ks, vs, qs = [], [], []
for pos in range(STEPS):
    hidden = x[pos].view(1, H, 1, 1)
    h = rms1(hidden, t['operator_norm.weight'])
    q = (h.view(1, H) @ t['self_attn.q_proj.weight'].T).view(NH, D)
    k = (h.view(1, H) @ t['self_attn.k_proj.weight'].T).view(NKV, D)
    v = (h.view(1, H) @ t['self_attn.v_proj.weight'].T).view(NKV, D)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * t['self_attn.q_norm.weight']
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * t['self_attn.k_norm.weight']
    q = q * cos[pos] + rotate_half(q) * sin[pos]
    k = k * cos[pos] + rotate_half(k) * sin[pos]
    qs.append(q)
    ks.append(k)
    vs.append(v)
    mask = np.full((1, NKV, 1, ROWS + 1), -1e4, np.float16)
    mask[:, :, :, :pos + 1] = 0
    out = prog.predict({'x': hidden.half().numpy(), 'pos': np.array([pos], np.int32),
                        'cos': cos[pos].half().numpy(), 'sin': sin[pos].half().numpy(),
                        'mask': mask}, state=state)

shapes = {np.asarray(v).shape: np.asarray(v, np.float32) for v in out.values()}
wv = shapes[(1, NKV, 4, D)]
m = shapes[(1, NKV, 4, 1)]
s = shapes[(1, NKV, 4)]

# the same numbers in torch fp32
qr = torch.stack(qs[-1:]).view(NKV, 4, D)
kt = torch.stack(ks)
vt = torch.stack(vs)
sc = torch.zeros(NKV, 4, STEPS)
for kv in range(NKV):
    sc[kv] = torch.stack([qr[kv] @ kt[p, kv] * SCALE for p in range(STEPS)]).T * 1.0
mr = sc.max(dim=-1, keepdim=True).values
e = torch.exp(sc - mr)
wvr = torch.zeros(NKV, 4, D)
for kv in range(NKV):
    wvr[kv] = e[kv] @ vt[:, kv]
sr = e.sum(-1)

print({'wv_rel': round(float((wv[0] - wvr.numpy()).__abs__().mean() / np.abs(wvr.numpy()).mean()), 5),
       'm_maxdiff': round(float(np.abs(m[0].reshape(NKV, 4) - mr.numpy()).max()), 5),
       's_rel': round(float(np.abs(s[0] - sr.numpy()).mean() / np.abs(sr.numpy()).mean()), 5),
       'engine_m': m[0].reshape(-1)[:4].tolist(),
       'torch_m': mr.numpy().reshape(-1)[:4].tolist()}, flush=True)