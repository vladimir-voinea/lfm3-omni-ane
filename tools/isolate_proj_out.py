import json, sys
import numpy as np
import torch
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_tiled as B
from lfm3_runtime import Runtime, H, NKV, D, NH

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
runtime._load_tiled()
entry = runtime.tiled_programs[2]
index = json.load(open('/tmp/full_layer_index.json'))
start = min(i['data_offsets'][0] for i in index.values())
blob = open(B.BIN, 'rb').read()
t = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        t[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info).float()

proj = B.Proj(t).eval().half()
out = B.Out(t).eval().half()
x = (np.random.RandomState(5).randn(1, H, 1, 1) * 0.1).astype(np.float16)
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)

result = entry['proj'].predict({'in0': x, 'in1': cos, 'in2': sin})
names = entry['proj_outputs']
with torch.no_grad():
    reference = proj(torch.from_numpy(x), torch.from_numpy(cos), torch.from_numpy(sin))
for label, name, want in zip(('q', 'k', 'v', 'gate'), names, reference):
    got = np.asarray(result[name]).reshape(-1)
    ref = want.numpy().reshape(-1)
    rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9)) * 100
    print({'proj': label, 'rel_l2_pct': round(rel, 3)}, flush=True)

v_raw = np.asarray(result[names[2]]).astype(np.float16).reshape(NKV, D)
v = np.repeat(v_raw, 4, axis=0).reshape(1, NH * D)
gate = np.asarray(result[names[3]]).astype(np.float16)
produced = entry['out'].predict({'in0': x, 'in1': v, 'in2': gate})
produced_array = np.asarray(list(produced.values())[0]).reshape(-1)
with torch.no_grad():
    expected = out(torch.from_numpy(x), torch.from_numpy(v),
                   torch.from_numpy(gate)).numpy().reshape(-1)
rel = float(np.linalg.norm(produced_array - expected) / (np.linalg.norm(expected) + 1e-9)) * 100
print({'out_program_vs_torch_rel_pct': round(rel, 3)}, flush=True)