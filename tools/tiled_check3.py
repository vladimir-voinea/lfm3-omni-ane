import os

from paths import ANE, PROJECT, SOURCE
# Same tiled path, instrumented: every predict is labelled, so the failing call names
# itself instead of surfacing as a bare ANE error. Short sequence to keep it quick.
import sys

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, PROJECT)

from tiled_parts2 import D, H, NKV, NH, Epilogue, Tile, rope

P = os.path.join(SOURCE, 'model.safetensors')
T, TILES, ROWS = 40, 3, 16

t = {}
with safe_open(P, framework='pt') as fh:
    for key in fh.keys():
        if 'language_model.layers.2.' in key:
            t[key.split('layers.2.')[1]] = fh.get_tensor(key).float()

cos, sin = rope(TILES * ROWS, 1e6, 0.5)

tiles = []
for j in range(TILES):
    tile = Tile(t, ROWS).eval().half()
    ex = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
          torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16),
          torch.zeros(1, NKV, 1, ROWS, dtype=torch.float16))
    prog = ct.convert(torch.jit.trace(tile, ex),
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
    tiles.append((prog, prog.make_state()))

epi = Epilogue(t).eval().half()
epi_prog = ct.convert(torch.jit.trace(epi, (torch.zeros(1, H, 1, 1, dtype=torch.float16),
                                                torch.zeros(1, H, 1, 1, dtype=torch.float16))),
                      inputs=[ct.TensorType(name='in0', shape=(1, H, 1, 1), dtype=np.float16),
                              ct.TensorType(name='in1', shape=(1, H, 1, 1), dtype=np.float16)],
                      compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)

torch.manual_seed(0)
x = torch.randn(1, T, H) * 0.5
for pos in range(T):
    hidden = x[0, pos].view(1, H, 1, 1).half().numpy()
    wvs, ms, ss = [], [], []
    for j, (prog, state) in enumerate(tiles):
        local = pos - j * ROWS
        if local < 0:
            wvs.append(np.zeros((1, NKV, 4, D), np.float32))
            ms.append(np.full((1, NKV, 4, 1), -1e4, np.float32))
            ss.append(np.zeros((1, NKV, 4), np.float32))
            continue
        mask = np.full((1, NKV, 1, ROWS), -1e4, np.float16)
        mask[:, :, :, :local + 1] = 0
        try:
            out = prog.predict({'x': hidden, 'pos': np.array([local], np.int32),
                                'cos': cos[pos].numpy().astype(np.float16),
                                'sin': sin[pos].numpy().astype(np.float16),
                                'mask': mask}, state=state)
        except Exception as exc:
            print({'FAILED_AT': 'tile', 'pos': pos, 'tile': j, 'err': str(exc)[:100]}, flush=True)
            raise SystemExit(1)
        shapes = {np.asarray(v).shape: np.asarray(v, np.float32) for v in out.values()}
        print({'pos': pos, 'tile': j, 'shapes': [list(s) for s in shapes]}, flush=True)
        wvs.append(shapes[(1, NKV, 4, D)])
        ms.append(shapes[(1, NKV, 4, 1)])
        ss.append(shapes[(1, NKV, 4)])
    m = np.maximum(np.maximum(ms[0], ms[1]), ms[2])
    wgt = [np.exp(mi - m) for mi in ms]
    num = sum(a * b for a, b in zip(wvs, wgt))
    den = sum(s * b[..., 0] for s, b in zip(ss, wgt))
    attn = (num / den[..., None]).reshape(1, H, 1, 1).astype(np.float16)
    try:
        epi_prog.predict({'in0': hidden, 'in1': attn})
    except Exception as exc:
        print({'FAILED_AT': 'epilogue', 'pos': pos, 'err': str(exc)[:100],
               'attn_finite': bool(np.isfinite(attn).all())}, flush=True)
        raise SystemExit(1)
print({'result': 'all steps ran'}, flush=True)