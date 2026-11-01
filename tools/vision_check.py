# The Core ML tower against the torch one (itself cosine 1.0 against official), and where it runs.
import json
import os
import sys
import time

import coremltools as ct
import numpy as np
import torch

import build_vision as B
from vision_ref import Vision, projector

path = sys.argv[1]
w, types = B.load()
state = np.random.RandomState(3)
y, x = np.mgrid[0:B.GRID, 0:B.GRID] / B.GRID
smooth = np.stack([np.sin(3 * x + c) * np.cos(4 * y - c) for c in range(B.PATCH)], -1).reshape(B.N, B.PATCH)
inputs = {'noise': state.randn(B.N, B.PATCH) * 0.5, 'smooth': smooth * 0.8,
          'flat red': np.tile(np.array([0.9, -0.8, -0.8] * (B.PATCH // 3)), (B.N, 1))}
if len(sys.argv) > 2:
    from png_codec import png_decode
    from vision_chat import patchify
    image = png_decode(open(sys.argv[2], 'rb').read()).astype(np.float32).transpose(2, 0, 1) / 255.0
    inputs = {'photo': ((patchify(torch.from_numpy(image[:3]))[0] - 0.5) / 0.5).numpy(), **inputs}
reference = Vision(w, types)
wrapped = B.Tower(w, types).eval()
for cu, label in ((ct.ComputeUnit.CPU_AND_NE, 'NE'), (ct.ComputeUnit.CPU_ONLY, 'CPU')):
    start = time.perf_counter()
    model = ct.models.MLModel(path, compute_units=cu)
    loaded = time.perf_counter() - start
    for name, patches in inputs.items():
        p = torch.from_numpy(patches.astype(np.float32))
        with torch.no_grad():
            want = projector(reference.forward(p, B.GRID, B.GRID), w).numpy()
            module = wrapped(p[None]).numpy()
        start = time.perf_counter()
        got = np.asarray(model.predict({'patches': p[None].numpy()})['tokens'])
        took = time.perf_counter() - start
        cos = float((got * want).sum() / np.linalg.norm(got) / np.linalg.norm(want))
        print({'on': label, 'input': name, 'module_rel_pct': round(100 * float(np.linalg.norm(module - want) / np.linalg.norm(want)), 4),
               'coreml_cos': round(cos, 5), 'rel_pct': round(100 * float(np.linalg.norm(got - want) / np.linalg.norm(want)), 2),
               'ms': round(took * 1000), 'load_s': round(loaded, 1)}, flush=True)
held = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
compiled = held.get_compiled_model_path()
from coremltools.models.compute_plan import MLComputePlan
plan = MLComputePlan.load_from_path(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
counts = {}
ops = plan.model_structure.program.functions['main'].block.operations
for op in ops:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if usage is None:
        continue
    key = type(usage.preferred_compute_device).__name__
    counts[key] = counts.get(key, 0) + 1
    if key != 'MLNeuralEngineComputeDevice' and op.operator_name not in ('const',):
        counts.setdefault('cpu_ops', {})
        counts['cpu_ops'][op.operator_name] = counts['cpu_ops'].get(op.operator_name, 0) + 1
print(counts, flush=True)
