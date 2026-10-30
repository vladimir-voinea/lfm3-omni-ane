# The rest of a token: conv layer with its 6-step state, sliding attention,
# one MLP, and the LM head split to stay under the 65,536 axis limit.
# Everything measured on the engine, with a device map for the new shapes.
from __future__ import annotations

import collections
import os
import statistics
import subprocess
import time

import numpy as np
import torch
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

H, FF, K, D = 2048, 7680, 7, 128


class ConvLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Conv2d(H, 3 * H, 1, bias=False)
        # a depthwise conv over the time axis, expressed as a conv2d with a (1,K)
        # kernel: the conv1d/groups path faults on the engine with an IOSurface
        # size error, the 2d path is the one the engine takes cleanly
        self.conv = torch.nn.Conv2d(H, H, (1, K), groups=H, bias=False)
        self.out_proj = torch.nn.Conv2d(H, H, 1, bias=False)
        self.register_buffer('state', torch.zeros(1, 1, H, K - 1, dtype=torch.float16),
                             persistent=False)

    def forward(self, x):
        proj = self.in_proj(x).view(1, 3, H, 1)
        b = proj[:, 0:1]
        c = proj[:, 1:2]
        t = proj[:, 2:3]
        gated = b * t                                   # (1, 1, H, 1)
        hist = torch.cat([self.state, gated], dim=3)    # (1, 1, H, 7)
        out = self.conv(hist.view(1, H, 1, K))          # (1, H, 1, 1)
        self.state[:] = torch.narrow(hist, 3, 1, K - 1)
        y = c * out.view(1, 1, H, 1)
        return self.out_proj(y.view(1, H, 1, 1))


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
        self.register_buffer('cache', torch.zeros(8, 1, window, D, dtype=torch.float16),
                             persistent=False)
        self.q = torch.nn.Conv2d(H, 16 * D, 1, bias=False)
        self.k = torch.nn.Conv2d(H, 4 * D, 1, bias=False)
        self.v = torch.nn.Conv2d(H, 4 * D, 1, bias=False)
        self.o = torch.nn.Conv2d(16 * D, H, 1, bias=False)

    def forward(self, x, pos):
        w = self.window
        q = self.q(x).view(1, 4, 4, D)
        k = self.k(x).view(4, 1, 1, D)
        v = self.v(x).view(4, 1, 1, D)
        self.cache[0:4, :, pos:pos + 1, :] = k
        self.cache[4:8, :, pos:pos + 1, :] = v
        key = self.cache[0:4].view(1, 4, w, D).transpose(-1, -2)
        val = self.cache[4:8].view(1, 4, w, D)
        scores = torch.matmul(q, key) * 0.0884
        attn = torch.matmul(torch.softmax(scores, dim=-1), val)
        return self.o(attn.reshape(1, 16 * D, 1, 1))


class LmHead(torch.nn.Module):
    def __init__(self, chunk):
        super().__init__()
        self.proj = torch.nn.Conv2d(H, chunk, 1, bias=False)

    def forward(self, x):
        return self.proj(x)


def convert(model, inputs, states, path):
    example = tuple(torch.zeros(*s, dtype=torch.float16 if s != (1,) else torch.int32)
                    for s in inputs)
    traced = torch.jit.trace(model, example)
    ct.convert(traced,
               inputs=[ct.TensorType(name='in%d' % i, shape=tuple(s),
                                     dtype=np.float16 if s != (1,) else np.int32)
                       for i, s in enumerate(inputs)],
               states=states, convert_to='mlprogram',
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    return path


def bench(path, inputs, iters=20):
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state() if model.get_spec().description.state else None
    feed = {}
    for spec in model.get_spec().description.input:
        shape = tuple(int(d) for d in spec.type.multiArrayType.shape)
        dtype = np.int32 if shape == (1,) else np.float16
        feed[spec.name] = np.zeros(shape, dtype)
    kwargs = {'state': state} if state is not None else {}
    model.predict(feed, **kwargs)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(feed, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 3)


def devices(pkg):
    out = '/tmp/dev_' + os.path.basename(pkg).replace('.mlpackage', '')
    subprocess.run(['rm', '-rf', out], check=False)
    os.makedirs(out, exist_ok=True)
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', pkg, out],
                   check=True, capture_output=True)
    name = [f for f in os.listdir(out) if f.endswith('.mlmodelc')][0]
    try:
        plan = MLComputePlan.load_from_path(os.path.join(out, name),
                                            compute_units=ct.ComputeUnit.CPU_AND_NE)
    except Exception as exc:
        return {'plan': 'failed: ' + str(exc)[:40]}
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


if __name__ == '__main__':
    results = {}
    conv = ConvLayer().eval().half()
    p = convert(conv, [(1, H, 1, 1)],
                [ct.StateType(name='state', wrapped_type=ct.TensorType(
                    shape=(1, 1, H, K - 1), dtype=np.float16))], '/tmp/conv.mlpackage')
    results['conv_layer'] = {'ms': bench(p, [(1, H, 1, 1)]), 'devices': devices(p)}
    print(results, flush=True)

    mlp = Mlp().eval().half()
    p = convert(mlp, [(1, H, 1, 1)], [], '/tmp/mlp.mlpackage')
    results['mlp'] = {'ms': bench(p, [(1, H, 1, 1)]), 'devices': devices(p)}
    print(results['mlp'], flush=True)

    sl = Sliding().eval().half()
    p = convert(sl, [(1, H, 1, 1), (1,)],
                [ct.StateType(name='cache', wrapped_type=ct.TensorType(
                    shape=(8, 1, 256, D), dtype=np.float16))], '/tmp/sliding.mlpackage')
    results['sliding_attn'] = {'ms': bench(p, [(1, H, 1, 1), (1,)]), 'devices': devices(p)}
    print(results['sliding_attn'], flush=True)

    for chunk in (64000, 128000):
        head = LmHead(chunk).eval().half()
        p = convert(head, [(1, H, 1, 1)], [], '/tmp/lm%d.mlpackage' % chunk)
        results['lm_head_%d' % chunk] = {'ms': bench(p, [(1, H, 1, 1)]), 'devices': devices(p)}
        print(results['lm_head_%d' % chunk], flush=True)