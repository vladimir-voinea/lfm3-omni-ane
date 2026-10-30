"""Test how the 16 heads are laid out on the way into o_proj.

A permuted layout produces exactly the symptom already measured: the layer's delta has the
right magnitude but is uncorrelated with the shipped one.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_model as M
import build_tiled as B
from lfm3_runtime import Runtime, H, NKV, D

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
index = json.load(open('/tmp/full_layer_index.json'))
start = min(entry['data_offsets'][0] for entry in index.values())
blob = open(B.BIN, 'rb').read()
raw = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        raw[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info).float()

rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16)
mask[..., 0] = 0
x = (np.random.RandomState(5).randn(1, H, 1, 1) * 0.1).astype(np.float16)
session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(
    {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask},
    state=session.states[2]).values())[0]).reshape(-1)


class Layout(M.AttnBlock):
    def __init__(self, tensors, rows, layout):
        super().__init__(tensors, rows)
        self.layout = layout

    def forward(self, x, pos, cos, sin, mask):
        h = self.pre(x)
        q = self.q_proj(h).view(1, NKV, 4, D)
        k = self.k_proj(h).view(1, NKV, 1, D)
        v = self.v_proj(h).view(1, NKV, 1, D)

        def rms(z, w):
            return z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * w.view(1, 1, 1, D)

        q, k = rms(q, self.q_norm), rms(k, self.k_norm)
        c, s = cos.view(1, 1, 1, D), sin.view(1, 1, 1, D)

        def rot(z):
            half = z.shape[-1] // 2
            return z * c + torch.cat([-z[..., half:], z[..., :half]], dim=-1) * s

        q, k = rot(q), rot(k)
        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q, key) * 0.0884 + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), val)          # (1, NKV, 4, D)

        gate = torch.sigmoid(self.gate(h)).view(1, NKV, 4, 1)
        attn = attn * gate

        if self.layout == 'kv_q_d':
            flat = attn.reshape(1, 16 * D)
        elif self.layout == 'q_kv_d':
            flat = attn.permute(0, 2, 1, 3).reshape(1, 16 * D)
        elif self.layout == 'kv_d_q':
            flat = attn.permute(0, 1, 3, 2).reshape(1, 16 * D)
        else:
            flat = attn.reshape(1, 16 * D)

        y = self.o_proj(flat.reshape(1, 16 * D, 1, 1))
        if self.layout == 'gate_after_oproj':
            y = (y.view(1, 16, D) * gate.view(1, 16, 1)).reshape(1, 16 * D, 1, 1)
        h2 = self.post(y) + x
        return self.ffn_post(self.mlp(self.ffn_norm(h2))) + h2


def rel(a):
    return round(float(np.linalg.norm(a - shipped) / (np.linalg.norm(shipped) + 1e-9)) * 100, 2)


for layout in ('kv_q_d', 'q_kv_d', 'kv_d_q', 'gate_after_oproj'):
    model = Layout(raw, M.FULL_ROWS, layout).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                    torch.from_numpy(mask)).numpy().reshape(-1)
    print({layout: rel(out)}, flush=True)