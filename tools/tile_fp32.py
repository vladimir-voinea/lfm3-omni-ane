# 75k needs accurate tile partials.  fp16 tiles cost ~2 % per layer.  This measures the
# other option: a full-size tile built at FLOAT32, where it lands, and what it costs.
import collections
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from tiled_parts2 import D, H, NKV, Tile, rope

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
ROWS = 28672

t = {}
with safe_open(P, framework='pt') as fh:
    for k in fh.keys():
        if 'language_model.layers.2.' in k:
            t[k.split('layers.2.')[1]] = fh.get_tensor(k).float()


def build(precision, path):
    tile = Tile(t, ROWS - 1).eval().half()
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
                      compute_precision=precision,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
    prog.save(path)
    return path


def placement(path):
    out = '/tmp/cp_tile_' + os.path.basename(path)
    subprocess.run(['rm', '-rf', out], check=False)
    os.makedirs(out, exist_ok=True)
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', path, out], check=True, capture_output=True)
    name = [f for f in os.listdir(out) if f.endswith('.mlmodelc')][0]
    plan = MLComputePlan.load_from_path(os.path.join(out, name),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = collections.Counter()

    def walk(block):
        for op in block.operations:
            try:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                counts[type(usage.preferred_compute_device).__name__.replace('ComputeDevice', '')] += 1
            except Exception:
                counts['other'] += 1
        for nested in getattr(block, 'blocks', []) or []:
            walk(nested)

    for fn in plan.model_structure.program.functions.values():
        walk(fn.block)
    return dict(counts)


if __name__ == '__main__':
    path = '/tmp/tile_fp32_full.mlpackage'
    build(ct.precision.FLOAT32, path)
    print({'built': path}, flush=True)
    print({'placement': placement(path)}, flush=True)
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state()
    cos, sin = rope(64, 1e6, 0.5)
    feed = {'x': np.zeros((1, H, 1, 1), np.float16), 'pos': np.array([100], np.int32),
            'cos': cos[100].numpy().astype(np.float16), 'sin': sin[100].numpy().astype(np.float16),
            'mask': np.zeros((1, NKV, 1, ROWS), np.float16)}
    model.predict(feed, state=state)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        model.predict(feed, state=state)
        times.append((time.perf_counter() - start) * 1000)
    print({'rows': ROWS, 'median_ms': round(statistics.median(times), 3)}, flush=True)