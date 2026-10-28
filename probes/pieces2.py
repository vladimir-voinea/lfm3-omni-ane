# The rest of a token, each piece measured on the engine.
# The conv layer carries its 6-row history as an input/output pair rather than a
# state buffer: a stateful conv faults on the engine (IOSurface size mismatch) and
# 24 KB of history per token is cheaper to move than to debug.
from __future__ import annotations

import collections
import os
import statistics
import subprocess
import time
import traceback

import numpy as np
import torch
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

H, FF, K, D = 2048, 7680, 7, 128


class ConvLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Conv2d(H, 3 * H, 1, bias=False)
        self.conv = torch.nn.Conv2d(H, H, (1, K), groups=H, bias=False)
        self.out_proj = torch.nn.Conv2d(H, H, 1, bias=False)

    def forward(self, x, hist):
        proj = self.in_proj(x).view(1, 3, H, 1)
        b, c, t = proj[:, 0:1], proj[:, 1:2], proj[:, 2:3]
        gated = b * t                                  # (1, 1, H, 1)
        full = torch.cat([hist, gated], dim=3)         # (1, 1, H, 7)
        out = self.conv(full.view(1, H, 1, K))         # (1, H, 1, 1)
        new_hist = torch.narrow(full, 3, 1, K - 1)
        y = c * out.view(1, 1, H, 1)
        return self.out_proj(y.view(1, H, 1, 1)), new_hist


class Mlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Conv2d(H, FF, 1, bias=False)
        self.up = torch.nn.Conv2d(H, FF, 1, bias=False)
        self.down = torch.nn.Conv2d(FF, H, 1, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class Sliding(torch.nn.Module):
    def __init__(self, window=256):
        super().__init__()
        self.window = window
        self.q = torch.nn.Conv2d(H, 16 * D, 1, bias=False)
        self.k = torch.nn.Conv2d(H, 4 * D, 1, bias=False)
        self.v = torch.nn.Conv2d(H, 4 * D, 1, bias=False)
        self.o = torch.nn.Conv2d(16 * D, H, 1, bias=False)

    def forward(self, x, cache, mask):
        w = self.window
        q = self.q(x).view(1, 4, 4, D)
        k = self.k(x).view(1, 4, 1, D)
        v = self.v(x).view(1, 4, 1, D)
        step = torch.cat([k, v], dim=1)                 # (1, 8, 1, D)
        new_cache = torch.cat([cache[:, :, 1:, :], step], dim=2)
        key = new_cache[:, 0:4].transpose(-1, -2)
        val = new_cache[:, 4:8]
        scores = torch.matmul(q, key) * 0.0884 + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), val)
        return self.o(attn.reshape(1, 16 * D, 1, 1)), new_cache


class LmHead(torch.nn.Module):
    def __init__(self, chunk):
        super().__init__()
        self.proj = torch.nn.Conv2d(H, chunk, 1, bias=False)

    def forward(self, x):
        return self.proj(x)


def convert(model, shapes, path):
    example = [torch.zeros(*s, dtype=torch.float16) for s in shapes]
    traced = torch.jit.trace(model, tuple(example))
    ct.convert(traced,
               inputs=[ct.TensorType(name='in%d' % i, shape=tuple(s), dtype=np.float16)
                       for i, s in enumerate(shapes)],
               convert_to='mlprogram',
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    return path


def bench(path, shapes, iters=20):
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    feed = {'in%d' % i: np.zeros(tuple(s), np.float16) for i, s in enumerate(shapes)}
    model.predict(feed)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 3)


def devices(pkg):
    out = '/tmp/dv_' + os.path.basename(pkg).replace('.mlpackage', '')
    subprocess.run(['rm', '-rf', out], check=False)
    os.makedirs(out, exist_ok=True)
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', pkg, out],
                   check=True, capture_output=True)
    name = [f for f in os.listdir(out) if f.endswith('.mlmodelc')][0]
    plan = MLComputePlan.load_from_path(os.path.join(out, name),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = collections.Counter()

    def walk(block):
        for op in block.operations:
            try:
                u = plan.get_compute_device_usage_for_mlprogram_operation(op)
                counts[type(u.preferred_compute_device).__name__
                       .replace('ComputeDevice', '')] += 1
            except Exception:
                counts['other'] += 1
        for nested in getattr(block, 'blocks', []) or []:
            walk(nested)

    for fn in plan.model_structure.program.functions.values():
        walk(fn.block)
    return dict(counts)


PIECES = [
    ('conv_layer', ConvLayer, [(1, H, 1, 1), (1, 1, H, K - 1)]),
    ('mlp', Mlp, [(1, H, 1, 1)]),
    ('sliding_attn', Sliding, [(1, H, 1, 1), (1, 8, 256, D), (1, 4, 1, 256)]),
    ('lm_head_64000', lambda: LmHead(64000), [(1, H, 1, 1)]),
    ('lm_head_128000', lambda: LmHead(128000), [(1, H, 1, 1)]),
]

if __name__ == '__main__':
    for name, factory, shapes in PIECES:
        try:
            model = factory().eval().half()
            path = convert(model, shapes, '/tmp/pc_%s.mlpackage' % name)
            print({'piece': name, 'ms': bench(path, shapes),
                   'devices': devices(path)}, flush=True)
        except Exception as exc:
            print({'piece': name, 'error': str(exc).replace(chr(10), ' ')[:110]}, flush=True)