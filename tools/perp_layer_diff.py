# For every full-attention layer: step two positions through the shipped program and through
# a torch rebuild, then compare.  Two positions bring the mask and the cache write into play.
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
rng = np.random.RandomState(11)


def tensors_for(layer):
    out = {}
    for key, info in index.items():
        if int(key.split('layers.')[1].split('.')[0]) == layer:
            out[key.split('layers.%d.' % layer)[1]] = B.load_tensor(blob, start, info).float()
    return out


for layer in [i for i, kind in enumerate(runtime.layer_types) if kind == 'full_attention']:
    tensors = tensors_for(layer)
    module = M.AttnBlock(tensors, M.FULL_ROWS).eval()
    session = runtime._plain_session()
    shipped_outputs = []
    torch_outputs = []
    for position in (0, 1):
        x = (rng.randn(1, H, 1, 1) * 0.1).astype(np.float16)
        cos = rope['full_cos'][position].reshape(D).astype(np.float16)
        sin = rope['full_sin'][position].reshape(D).astype(np.float16)
        mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16)
        mask[..., :position + 1] = 0
        feeds = {'in0': x, 'in1': np.array([position], np.int32), 'in2': cos, 'in3': sin,
                 'in4': mask}
        shipped_outputs.append(np.asarray(list(runtime.programs[layer].predict(
            feeds, state=session.states[layer]).values())[0]).reshape(-1))
        with torch.no_grad():
            torch_outputs.append(module(torch.from_numpy(x), position,
                                        torch.from_numpy(cos), torch.from_numpy(sin),
                                        torch.from_numpy(mask)).numpy().reshape(-1))
    a = np.concatenate(torch_outputs)
    b = np.concatenate(shipped_outputs)
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)) * 100
    corr = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-9))
    print({'layer': layer, 'rel_l2_pct': round(rel, 2), 'corr': round(corr, 4)}, flush=True)