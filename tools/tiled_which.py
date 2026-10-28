# Which of the two new programs actually fails on the engine?
import sys
import traceback

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from tiled_parts import Epilogue, Tile

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
H, D, NH, NKV = 2048, 128, 16, 4
ROWS, T = 16, 40

t = {}
with safe_open(P, framework='pt') as fh:
    for key in fh.keys():
        if 'language_model.layers.2.' in key:
            t[key.split('layers.2.')[1]] = fh.get_tensor(key).float()

tile = Tile(t, ROWS).eval().half()
ex = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
      torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16),
      torch.zeros(1, NKV, 1, ROWS, dtype=torch.float16))
tile_prog = ct.convert(torch.jit.trace(tile, ex),
                       inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                               ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                               ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
                               ct.TensorType(name='sin', shape=(D,), dtype=np.float16),
                               ct.TensorType(name='mask', shape=(1, NKV, 1, ROWS),
                                             dtype=np.float16)],
                       states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
                           shape=(2 * NKV, 1, ROWS, D), dtype=np.float16))],
                       compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.iOS18,
                       compute_units=ct.ComputeUnit.CPU_AND_NE)
try:
    st = tile_prog.make_state()
    out = tile_prog.predict({'x': np.zeros((1, H, 1, 1), np.float16),
                             'pos': np.array([0], np.int32),
                             'cos': np.zeros(D, np.float16),
                             'sin': np.zeros(D, np.float16),
                             'mask': np.zeros((1, NKV, 1, ROWS), np.float16)}, state=st)
    print({'tile': 'ok', 'outputs': {k: list(np.asarray(v).shape) for k, v in out.items()}}, flush=True)
except Exception as exc:
    print({'tile': 'FAILED', 'err': str(exc)[:200]}, flush=True)

epi = Epilogue(t).eval().half()
feed_shapes = [(1, H, 1, 1), (1, H, 1, 1)]
for _ in range(3):
    feed_shapes += [(1, NKV, 4, D), (1, NKV, 4), (1, NKV, 4)]
ex2 = tuple(torch.zeros(*s, dtype=torch.float16) for s in feed_shapes)
epi_prog = ct.convert(torch.jit.trace(epi, ex2),
                      inputs=[ct.TensorType(name='in%d' % i, shape=s, dtype=np.float16)
                              for i, s in enumerate(feed_shapes)],
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
try:
    out = epi_prog.predict({'in%d' % i: np.zeros(s, np.float16)
                            for i, s in enumerate(feed_shapes)})
    print({'epilogue': 'ok', 'shape': list(np.asarray(list(out.values())[0]).shape)}, flush=True)
except Exception as exc:
    print({'epilogue': 'FAILED', 'err': str(exc)[:200]}, flush=True)