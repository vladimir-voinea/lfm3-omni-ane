# The shipped layer changes the hidden state by ~0.33 of its norm; mine changes it by almost
# nothing.  The norm weights are ~0.02, which would shrink my layer's contribution 50x.
# Test the conventions on the L2 metric.
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
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0
x = (np.random.RandomState(5).randn(1, H, 1, 1) * 0.1).astype(np.float16)
session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(
    {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask},
    state=session.states[2]).values())[0]).reshape(-1)

NORMS = ('operator_norm.weight', 'post_operator_norm.weight', 'ffn_norm.weight',
         'post_ffn_norm.weight', 'self_attn.q_norm.weight', 'self_attn.k_norm.weight')


def rel(a):
    return round(float(np.linalg.norm(a - shipped) / (np.linalg.norm(shipped) + 1e-9)) * 100, 2)


def run(transform):
    t = dict(raw)
    for key in NORMS:
        t[key] = transform(raw[key])
    model = M.AttnBlock(t, M.FULL_ROWS).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                    torch.from_numpy(mask)).numpy().reshape(-1)
    return rel(out)


print({'plain': run(lambda w: w), 'one_plus_w': run(lambda w: 1.0 + w),
       'w_as_one': run(lambda w: torch.ones_like(w))}, flush=True)
print({'delta_shipped_rms': round(float(np.sqrt(((shipped - x.reshape(-1)) ** 2).mean())), 4),
       'x_rms': round(float(np.sqrt((x.reshape(-1).astype(np.float32) ** 2).mean())), 4)}, flush=True)