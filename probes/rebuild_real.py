# Rebuild layer 2 through build_model's own loader and save(), then compare with the shipped
# program.  If these agree, the drift was in my test harness; if not, the published model was
# built from different code than the repository holds.
import json, os, sys
import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_model as M
import build_tiled as B
from lfm3_runtime import Runtime, H, NKV, D

index = json.load(open('/tmp/full_layer_index.json'))
start = min(i['data_offsets'][0] for i in index.values())
blob = open(B.BIN, 'rb').read()
t = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        t[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info).float()

out_dir = '/tmp/rebuild_layer2'
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, 'layer2_full.mlpackage')
M.save(M.AttnBlock(t, M.FULL_ROWS),
       [(1, H, 1, 1), (1,), (D,), (D,), (1, NKV, 1, M.FULL_ROWS)],
       [ct.StateType(name='cache', wrapped_type=ct.TensorType(
           shape=(2 * NKV, 1, M.FULL_ROWS, D), dtype=np.float16))], path)
print({'built': path}, flush=True)

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
x = (np.random.RandomState(3).randn(1, H, 1, 1) * 0.1).astype(np.float16)
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0
feeds = {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask}
fresh_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
fresh = np.asarray(list(fresh_model.predict(feeds, state=fresh_model.make_state()).values())[0]).reshape(-1)
session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(feeds, state=session.states[2]).values())[0]).reshape(-1)
print({'fresh_scale': round(float(np.abs(fresh).max()), 4),
       'shipped_scale': round(float(np.abs(shipped).max()), 4),
       'rel_diff': round(float(np.abs(fresh - shipped).max() / (np.abs(shipped).max() + 1e-9)), 5)}, flush=True)