# Feed the shipped program different candidate inputs and see which one lines up with torch.
import json, sys
import numpy as np
import torch
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_model as M
import build_tiled as B
from lfm3_runtime import Runtime, H, NKV, D

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
index = json.load(open('/tmp/full_layer_index.json'))
start = min(i['data_offsets'][0] for i in index.values())
blob = open(B.BIN, 'rb').read()
t = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        t[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info).float()

layer = M.AttnBlock(t, M.FULL_ROWS).eval()
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0

x = (np.random.RandomState(5).randn(1, H, 1, 1) * 0.1).astype(np.float16)
session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(
    {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask},
    state=session.states[2]).values())[0]).reshape(-1)

with torch.no_grad():
    pre_x = layer.pre(torch.from_numpy(x)).numpy().astype(np.float16)
    mine_x = layer(torch.from_numpy(x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                   torch.from_numpy(mask)).numpy().reshape(-1)
    mine_pre = layer(torch.from_numpy(pre_x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                     torch.from_numpy(mask)).numpy().reshape(-1)
fresh = runtime._plain_session()
shipped_pre = np.asarray(list(ruby := runtime.programs[2].predict(
    {'in0': pre_x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask},
                       state=fresh.states[2]).values())[0]).reshape(-1)


def rel(a, b):
    return round(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)) * 100, 2)


print({'norm_x': round(float(np.linalg.norm(x)), 3), 'norm_pre_x': round(float(np.linalg.norm(pre_x)), 3),
       'norm_shipped': round(float(np.linalg.norm(shipped)), 3)}, flush=True)
print({'shipped vs mine(x)': rel(mine_x, shipped),
       'shipped vs mine(pre)': rel(mine_pre, shipped)}, flush=True)
print({'shipped_pre vs mine(x)': rel(mine_x, shipped_pre),
       'shipped_pre vs mine(pre)': rel(mine_pre, shipped_pre)}, flush=True)