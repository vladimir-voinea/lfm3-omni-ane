import sys
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open
from tiled_parts2 import Epilogue, H, NH, D

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
t = {}
with safe_open(P, framework='pt') as fh:
    for k in fh.keys():
        if 'language_model.layers.2.' in k:
            t[k.split('layers.2.')[1]] = fh.get_tensor(k).float()
epi = Epilogue(t).eval().half()
ex = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, H, 1, 1, dtype=torch.float16))
prog = ct.convert(torch.jit.trace(epi, ex),
                  inputs=[ct.TensorType(name='in0', shape=(1, H, 1, 1), dtype=np.float16),
                          ct.TensorType(name='in1', shape=(1, H, 1, 1), dtype=np.float16)],
                  compute_precision=ct.precision.FLOAT16,
                  minimum_deployment_target=ct.target.iOS18,
                  compute_units=ct.ComputeUnit.CPU_AND_NE)
for label, units in (('ne', ct.ComputeUnit.CPU_AND_NE), ('cpu', ct.ComputeUnit.CPU_ONLY)):
    try:
        m = ct.models.MLModel(prog.get_spec().description if False else '/tmp/epi_test.mlpackage', compute_units=units) if False else None
    except Exception:
        pass
prog.save('/tmp/epi_test.mlpackage')
for label, units in (('ne', ct.ComputeUnit.CPU_AND_NE), ('cpu', ct.ComputeUnit.CPU_ONLY)):
    try:
        m = ct.models.MLModel('/tmp/epi_test.mlpackage', compute_units=units)
        out = m.predict({'in0': np.zeros((1, H, 1, 1), np.float16),
                         'in1': np.zeros((1, H, 1, 1), np.float16)})
        print({label: 'ok', 'shape': list(np.asarray(list(out.values())[0]).shape)}, flush=True)
    except Exception as exc:
        print({label: 'FAILED', 'err': str(exc)[:120]}, flush=True)