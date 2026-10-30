"""Hypothesis: the shipped build reinterpreted the checkpoint's BF16 bits as FP16.

RMSNorm divides by the RMS, so a uniform scale error on every tensor is absorbed by the
network and the model still answers correctly.  Test it by rebuilding the layer with the bits
reinterpreted as fp16 and comparing against the shipped program.
"""
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


def load(layer, reinterpret):
    out = {}
    for key, info in index.items():
        if int(key.split('layers.')[1].split('.')[0]) != layer:
            continue
        begin, end = info['data_offsets']
        raw = blob[begin - start:end - start]
        dtype = np.float16 if reinterpret else torch.bfloat16
        if reinterpret:
            values = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            tensor = torch.from_numpy(values.copy())
        else:
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()
        out[key.split('layers.%d.' % layer)[1]] = tensor.reshape(info['shape'])
    return out


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


def rel(a):
    return round(float(np.linalg.norm(a - shipped) / (np.linalg.norm(shipped) + 1e-9)) * 100, 2)


for reinterpret in (False, True):
    tensors = load(2, reinterpret)
    model = M.AttnBlock(tensors, M.FULL_ROWS).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                    torch.from_numpy(mask)).numpy().reshape(-1)
    label = 'bf16_reinterpreted_as_fp16' if reinterpret else 'bf16_decoded_properly'
    print({label: rel(out)}, flush=True)
