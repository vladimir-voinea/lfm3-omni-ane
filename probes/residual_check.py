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
raw = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        raw[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info).float()
x = (np.random.RandomState(3).randn(1, H, 1, 1) * 0.1).astype(np.float16)
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0
feeds = {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask}
session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(feeds, state=session.states[2]).values())[0]).reshape(-1)
model = M.AttnBlock(raw, M.FULL_ROWS).eval()
with torch.no_grad():
    mine = model(torch.from_numpy(x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                 torch.from_numpy(mask)).numpy().reshape(-1)
flat_x = x.reshape(-1).astype(np.float32)
def rel(a, b):
    return round(float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9)), 3)
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return round(float((a * b).mean() / (a.std() * b.std() + 1e-9)), 3)
print({'shipped_vs_mine': rel(shipped, mine), 'corr_shipped_mine': corr(shipped, mine)}, flush=True)
print({'shipped_vs_mine_minus_x': rel(shipped, mine - flat_x), 'corr': corr(shipped, mine - flat_x)}, flush=True)
print({'shipped_vs_x': rel(shipped, flat_x), 'corr': corr(shipped, flat_x)}, flush=True)
print({'absmax': {'shipped': round(float(np.abs(shipped).max()), 3),
                  'mine': round(float(np.abs(mine).max()), 3),
                  'x': round(float(np.abs(flat_x).max()), 3),
                  'mine_minus_x': round(float(np.abs(mine - flat_x).max()), 3)}}, flush=True)