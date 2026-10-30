# Capture the real hidden state that enters each full-attention layer during an engine run,
# then compare torch rebuild vs shipped program on THAT input.
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
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
full = [i for i, kind in enumerate(runtime.layer_types) if kind == 'full_attention']

captured = {}
for layer in full:
    program = runtime.programs[layer]
    original = program.predict
    def make(original_predict, layer_index):
        def wrapper(feeds, state=None):
            captured.setdefault(layer_index, feeds['in0'].copy())
            return original_predict(feeds, state=state)
        return wrapper
    program.predict = make(original, layer)

session = runtime.new_session()
runtime.step(session, [1, 9707, 500, 8000, 1234, 77])
print({'captured_layers': len(captured)}, flush=True)


def tensors_for(layer):
    result = {}
    for key, info in index.items():
        if int(key.split('layers.')[1].split('.')[0]) == layer:
            result[key.split('layers.%d.' % layer)[1]] = B.load_tensor(blob, start, info).float()
    return result


for layer in full:
    x = captured[layer].astype(np.float16)
    position = 5
    cos = rope['full_cos'][position].reshape(D).astype(np.float16)
    sin = rope['full_sin'][position].reshape(D).astype(np.float16)
    mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16)
    mask[..., :position + 1] = 0
    feeds = {'in0': x, 'in1': np.array([position], np.int32), 'in2': cos, 'in3': sin, 'in4': mask}
    fresh_session = runtime._plain_session()
    shipped = np.asarray(list(runtime.programs[layer].predict(
        feeds, state=fresh_session.states[layer]).values())[0]).reshape(-1)
    module = M.AttnBlock(tensors_for(layer), M.FULL_ROWS).eval()
    with torch.no_grad():
        rebuilt = module(torch.from_numpy(x), position, torch.from_numpy(cos),
                         torch.from_numpy(sin), torch.from_numpy(mask)).numpy().reshape(-1)
    rel = float(np.linalg.norm(rebuilt - shipped) / (np.linalg.norm(shipped) + 1e-9)) * 100
    print({'layer': layer, 'rel_l2_pct_real_input': round(rel, 2)}, flush=True)