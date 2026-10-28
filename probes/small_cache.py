# Is a smaller full-attention cache much cheaper?  Build one 8,192-row program and time it.
import statistics
import sys
import time

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

import build_model as B

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
ROWS = 8192

t = {}
with safe_open(P, framework='pt') as f:
    for k in f.keys():
        if 'language_model.layers.2.' in k:
            t[k.split('layers.2.')[1]] = f.get_tensor(k).float()

block = B.AttnBlock(t, ROWS).eval().half()
ex = (torch.zeros(1, B.H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
      torch.zeros(B.D, dtype=torch.float16), torch.zeros(B.D, dtype=torch.float16),
      torch.zeros(1, B.NKV, 1, ROWS, dtype=torch.float16))
prog = ct.convert(torch.jit.trace(block, ex),
                  inputs=[ct.TensorType(name='in0', shape=(1, B.H, 1, 1), dtype=np.float16),
                          ct.TensorType(name='in1', shape=(1,), dtype=np.int32),
                          ct.TensorType(name='in2', shape=(B.D,), dtype=np.float16),
                          ct.TensorType(name='in3', shape=(B.D,), dtype=np.float16),
                          ct.TensorType(name='in4', shape=(1, B.NKV, 1, ROWS), dtype=np.float16)],
                  states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
                      shape=(2 * B.NKV, 1, ROWS, B.D), dtype=np.float16))],
                  compute_precision=ct.precision.FLOAT16,
                  minimum_deployment_target=ct.target.iOS18,
                  compute_units=ct.ComputeUnit.CPU_AND_NE)
path = '/tmp/layer1_full_8192.mlpackage'
prog.save(path)
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
state = model.make_state()
feed = {'in0': np.zeros((1, B.H, 1, 1), np.float16), 'in1': np.array([8], np.int32),
        'in2': np.zeros(B.D, np.float16), 'in3': np.zeros(B.D, np.float16),
        'in4': np.zeros((1, B.NKV, 1, ROWS), np.float16)}
model.predict(feed, state=state)
times = []
for _ in range(20):
    start = time.perf_counter()
    model.predict(feed, state=state)
    times.append((time.perf_counter() - start) * 1000)
print({'rows': ROWS, 'median_ms': round(statistics.median(times), 3)}, flush=True)