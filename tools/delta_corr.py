"""What is the shipped layer's delta made of?  Correlate it against candidate pieces."""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_model as M
import build_tiled as B
from lfm3_runtime import Runtime, H, NKV, D

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6', mode='28k')
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
flat_x = x.reshape(-1).astype(np.float32)
shipped_delta = shipped - flat_x

layer = M.AttnBlock(raw, M.FULL_ROWS).eval()
with torch.no_grad():
    xt = torch.from_numpy(x)
    c, s = torch.from_numpy(cos), torch.from_numpy(sin)
    m = torch.from_numpy(mask)
    mine = layer(xt, 0, c, s, m).numpy().reshape(-1)
    h = layer.pre(xt)
    v = layer.v_proj(h).view(1, NKV, 1, D)
    gate = torch.sigmoid(layer.gate(h)).view(1, NKV, 4, 1)
    attn = v.repeat(1, 1, 4, 1) * gate
    y = layer.o_proj(attn.reshape(1, 16 * D, 1, 1))
    attention_delta = layer.post(y).numpy().reshape(-1)
    h2 = torch.from_numpy(shipped).view(1, H, 1, 1)
    mlp_delta = layer.ffn_post(layer.mlp(layer.ffn_norm(h2))).numpy().reshape(-1)

def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return round(float((a * b).mean() / (a.std() * b.std() + 1e-12)), 3)

def rms(a):
    return round(float(np.sqrt((a ** 2).mean())), 4)

print({'rms_shipped_delta': rms(shipped_delta), 'rms_mine_delta': rms(mine - flat_x),
       'rms_attention_delta': rms(attention_delta), 'rms_mlp_delta': rms(mlp_delta)}, flush=True)
print({'corr_shipped_mine': corr(shipped_delta, mine - flat_x),
       'corr_shipped_attention': corr(shipped_delta, attention_delta),
       'corr_shipped_mlp': corr(shipped_delta, mlp_delta)}, flush=True)
