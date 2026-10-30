# Compare rebuild vs shipped on a REAL hidden state, not random noise.
import json, sys, os
import numpy as np
import torch
import coremltools as ct
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_model as M
import build_tiled as B
from lfm3_runtime import Runtime, H, NKV, D

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
captured = {}
original = runtime._run_position


def traced(session, hidden):
    captured.setdefault('inputs', []).append(np.asarray(hidden).copy())
    return original(session, hidden)


runtime._run_position = traced
session = runtime.new_session()
runtime.step(session, [1, 9707, 500, 8000])
real_x = captured['inputs'][2]   # a real hidden state from the 3rd token
print({'real_x_shape': list(real_x.shape), 'real_absmax': round(float(np.abs(real_x).max()), 3),
       'random_absmax': 0.35}, flush=True)

index = json.load(open('/tmp/full_layer_index.json'))
start = min(i['data_offsets'][0] for i in index.values())
blob = open(B.BIN, 'rb').read()
t = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        t[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info).float()
path = '/tmp/real_layer2.mlpackage'
M.save(M.AttnBlock(t, M.FULL_ROWS),
       [(1, H, 1, 1), (1,), (D,), (D,), (1, NKV, 1, M.FULL_ROWS)],
       [ct.StateType(name='cache', wrapped_type=ct.TensorType(
           shape=(2 * NKV, 1, M.FULL_ROWS, D), dtype=np.float16))], path)
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][2].reshape(D).astype(np.float16)
sin = rope['full_sin'][2].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., :3] = 0
feeds = {'in0': real_x.astype(np.float16), 'in1': np.array([2], np.int32), 'in2': cos,
         'in3': sin, 'in4': mask}
fresh_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
fresh = np.asarray(list(fresh_model.predict(feeds, state=fresh_model.make_state()).values())[0]).reshape(-1)
plain = np.asarray(list(runtime.programs[2].predict(feeds, state=runtime._plain_session().states[2]).values())[0]).reshape(-1)
print({'rel_diff_real_input': round(float(np.abs(fresh - plain).max() / (np.abs(plain).max() + 1e-9)), 5)}, flush=True)