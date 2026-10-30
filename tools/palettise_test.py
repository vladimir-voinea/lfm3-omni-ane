# Is the 0.93 difference just 6-bit palettisation?  Palettise the same rebuild the way the
# shipped build was palettised and compare again.
import json, os, sys
import numpy as np
import torch
import coremltools as ct
import coremltools.optimize.coreml as cto

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

plain = '/tmp/plain2.mlpackage'
M.save(M.AttnBlock(t, M.FULL_ROWS),
       [(1, H, 1, 1), (1,), (D,), (D,), (1, NKV, 1, M.FULL_ROWS)],
       [ct.StateType(name='cache', wrapped_type=ct.TensorType(
           shape=(2 * NKV, 1, M.FULL_ROWS, D), dtype=np.float16))], plain)
model = ct.models.MLModel(plain)
config = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=6))
quantised = cto.palettize_weights(model, config)
quantised.save('/tmp/lut6_2.mlpackage')
print({'palettised': True}, flush=True)

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
x = (np.random.RandomState(3).randn(1, H, 1, 1) * 0.1).astype(np.float16)
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0
feeds = {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask}


def run(path):
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    return np.asarray(list(model.predict(feeds, state=model.make_state()).values())[0]).reshape(-1)


session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(feeds, state=session.states[2]).values())[0]).reshape(-1)
fp16_out = run(plain)
lut6_out = run('/tmp/lut6_2.mlpackage')
def rel(a):
    return round(float(np.abs(a - shipped).max() / (np.abs(shipped).max() + 1e-9)), 5)
print({'fp16_vs_shipped': rel(fp16_out), 'lut6_vs_shipped': rel(lut6_out),
       'fp16_vs_lut6': round(float(np.abs(fp16_out - lut6_out).max() / (np.abs(lut6_out).max() + 1e-9)), 5)}, flush=True)