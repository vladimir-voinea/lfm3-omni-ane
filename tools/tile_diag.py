import sys
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open
from tiled_parts2 import Tile, D, H, NKV, NH

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
ROWS = 16
t = {}
with safe_open(P, framework='pt') as fh:
    for k in fh.keys():
        if 'language_model.layers.2.' in k:
            t[k.split('layers.2.')[1]] = fh.get_tensor(k).float()
tile = Tile(t, ROWS).eval().half()
ex = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
      torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16),
      torch.zeros(1, NKV, 1, ROWS, dtype=torch.float16))
prog = ct.convert(torch.jit.trace(tile, ex),
                  inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                          ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                          ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
                          ct.TensorType(name='sin', shape=(D,), dtype=np.float16),
                          ct.TensorType(name='mask', shape=(1, NKV, 1, ROWS), dtype=np.float16)],
                  states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
                      shape=(2 * NKV, 1, ROWS, D), dtype=np.float16))],
                  compute_precision=ct.precision.FLOAT16,
                  minimum_deployment_target=ct.target.iOS18,
                  compute_units=ct.ComputeUnit.CPU_AND_NE)
state = prog.make_state()
for label, pos, visible in (('zero_mask_pos0', 0, 1), ('masked_pos5', 5, 6), ('all_hidden', 3, 0)):
    mask = np.full((1, NKV, 1, ROWS), -1e4, np.float16)
    if visible:
        mask[:, :, :, :visible] = 0
    try:
        out = prog.predict({'x': np.zeros((1, H, 1, 1), np.float16),
                            'pos': np.array([pos], np.int32),
                            'cos': np.zeros(D, np.float16), 'sin': np.zeros(D, np.float16),
                            'mask': mask}, state=state)
        print({label: 'ok'}, flush=True)
    except Exception as exc:
        print({label: 'FAILED', 'err': str(exc)[:90]}, flush=True)