"""Test tensor-assignment swaps that keep every shape valid.

q_proj and o_proj are both 2048x2048, and w1 and w3 are both 7680x2048, so a swapped
assignment produces plausible-looking output with the right shapes -- and a mismatch that is
uncorrelated, which is what the measurements show.
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
raw = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        begin, end = info['data_offsets']
        raw[key.split('layers.2.')[1]] = torch.frombuffer(
            bytearray(blob[begin - start:end - start]), dtype=torch.bfloat16).float().reshape(info['shape'])

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


def run(swaps):
    t = dict(raw)
    for left, right in swaps:
        t[left], t[right] = raw[right], raw[left]
    model = M.AttnBlock(t, M.FULL_ROWS).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                    torch.from_numpy(mask)).numpy().reshape(-1)
    return rel(out)


cases = {
    'baseline': [],
    'swap_w1_w3': [('feed_forward.w1.weight', 'feed_forward.w3.weight')],
    'swap_q_o': [('self_attn.q_proj.weight', 'self_attn.o_proj.weight')],
    'swap_q_k_first_rows': [],
    'swap_norms_pre_post': [('operator_norm.weight', 'post_operator_norm.weight')],
    'swap_ffn_norms': [('ffn_norm.weight', 'post_ffn_norm.weight')],
}
for label, swaps in cases.items():
    print({label: run(swaps)}, flush=True)