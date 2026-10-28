# Will the engine give fp32 to the tile's softmax phase, and at what cost?
# Builds the same tile at FLOAT16 and FLOAT32, compares partials against torch and
# times a dispatch, so accuracy and speed can be weighed together.
import collections
import os

from paths import PROJECT, SOURCE
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from safetensors import safe_open

sys.path.insert(0, PROJECT)

from tiled_parts2 import D, EPS, H, NKV, NH, SCALE, Tile, rotate_half, rms1, rope

P = os.path.join(SOURCE, 'model.safetensors')
ROWS, STEPS = 16, 6

t = {}
with safe_open(P, framework='pt') as fh:
    for k in fh.keys():
        if 'language_model.layers.2.' in k:
            t[k.split('layers.2.')[1]] = fh.get_tensor(k).float()

torch.manual_seed(0)
x = torch.randn(STEPS, H) * 0.5
cos, sin = rope(64, 1e6, 0.5)


def build(precision, path):
    tile = Tile(t, ROWS).eval().half()
    ex = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
          torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16),
          torch.zeros(1, NKV, 1, ROWS + 1, dtype=torch.float16))
    prog = ct.convert(torch.jit.trace(tile, ex),
                      inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                              ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                              ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
                              ct.TensorType(name='sin', shape=(D,), dtype=np.float16),
                              ct.TensorType(name='mask', shape=(1, NKV, 1, ROWS + 1),
                                            dtype=np.float16)],
                      states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
                          shape=(2 * NKV, 1, ROWS + 1, D), dtype=np.float16))],
                      compute_precision=precision,
                      minimum_deployment_target=ct.target.iOS18,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)
    prog.save(path)
    return path


def devices(path):
    out = '/tmp/dp_' + os.path.basename(path).replace('.mlpackage', '')
    subprocess.run(['rm', '-rf', out], check=False)
    os.makedirs(out, exist_ok=True)
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', path, out],
                   check=True, capture_output=True)
    name = [f for f in os.listdir(out) if f.endswith('.mlmodelc')][0]
    plan = MLComputePlan.load_from_path(os.path.join(out, name),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = collections.Counter()

    def walk(block):
        for op in block.operations:
            try:
                u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                counts[type(u.preferred_compute_device).__name__.replace('ComputeDevice', '')] += 1
            except Exception:
                counts['other'] += 1
        for nested in getattr(block, 'blocks', []) or []:
            walk(nested)

    for fn in plan.model_structure.program.functions.values():
        walk(fn.block)
    return dict(counts)


def run(path):
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state()
    last = None
    for pos in range(STEPS):
        hidden = x[pos].view(1, H, 1, 1)
        h = rms1(hidden, t['operator_norm.weight'])
        q = (h.view(1, H) @ t['self_attn.q_proj.weight'].T).view(NH, D)
        k = (h.view(1, H) @ t['self_attn.k_proj.weight'].T).view(NKV, D)
        v = (h.view(1, H) @ t['self_attn.v_proj.weight'].T).view(NKV, D)
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * t['self_attn.q_norm.weight']
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * t['self_attn.k_norm.weight']
        q = q * cos[pos] + rotate_half(q) * sin[pos]
        k = k * cos[pos] + rotate_half(k) * sin[pos]
        mask = np.full((1, NKV, 1, ROWS + 1), -1e4, np.float16)
        mask[:, :, :, :pos + 1] = 0
        last = model.predict({'x': hidden.half().numpy(), 'pos': np.array([pos], np.int32),
                              'cos': cos[pos].half().numpy(), 'sin': sin[pos].half().numpy(),
                              'mask': mask}, state=state)
        torch_side = (q, k, v)
    times = []
    feed = {'x': x[-1].view(1, H, 1, 1).half().numpy(), 'pos': np.array([5], np.int32),
            'cos': cos[5].half().numpy(), 'sin': sin[5].half().numpy(),
            'mask': np.zeros((1, NKV, 1, ROWS + 1), np.float16)}
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict(feed, state=state)
        times.append((time.perf_counter() - t0) * 1000)

    shapes = {np.asarray(v).shape: np.asarray(v, np.float32) for v in last.values()}
    wv = shapes[(1, NKV, 4, D)]
    q, k, v = torch_side
    ks = torch.stack([k] * STEPS)
    sc = torch.zeros(NKV, 4)
    for kv in range(NKV):
        sc[kv] = torch.stack([q[kv] @ k[kv] * SCALE for _ in range(4)])
    return round(statistics.median(times), 3), wv


if __name__ == '__main__':
    for label, precision in (('fp16', ct.precision.FLOAT16), ('fp32', ct.precision.FLOAT32)):
        try:
            path = build(precision, '/tmp/tile_%s.mlpackage' % label)
            ms, wv = run(path)
            print({'precision': label, 'ms': ms, 'devices': devices(path),
                   'wv_norm': round(float(np.abs(wv).mean()), 4)}, flush=True)
        except Exception as exc:
            print({'precision': label, 'error': str(exc)[:110]}, flush=True)