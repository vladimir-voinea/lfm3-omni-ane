# What inside the tile program costs 4.9 ms: the matmuls or the softmax?
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import torch
import coremltools as ct

ROWS, HIDDEN, KV, REP, DIM = 28672, 2048, 4, 4, 128


def build(kind: str, path: str):
    kv, rep, d = KV, REP, DIM

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('cache',
                                 torch.zeros(2 * kv, 1, ROWS, d, dtype=torch.float16),
                                 persistent=False)
            self.q_proj = torch.nn.Conv2d(HIDDEN, kv * rep * d, 1, bias=False)
            self.k_proj = torch.nn.Conv2d(HIDDEN, kv * d, 1, bias=False)
            self.v_proj = torch.nn.Conv2d(HIDDEN, kv * d, 1, bias=False)

        def forward(self, hidden, mask, pos):
            q = self.q_proj(hidden).view(1, kv, rep, d)
            k = self.k_proj(hidden).view(kv, 1, 1, d)
            v = self.v_proj(hidden).view(kv, 1, 1, d)
            self.cache[0:kv, :, pos:pos + 1, :] = k
            self.cache[kv:2 * kv, :, pos:pos + 1, :] = v
            key = self.cache[0:kv, :, :ROWS, :].view(1, kv, ROWS, d).transpose(-1, -2)
            value = self.cache[kv:2 * kv, :, :ROWS, :].view(1, kv, ROWS, d)
            scores = torch.matmul(q, key) * 0.0884 + mask
            if kind == 'no_softmax':
                probs = scores
            else:
                probs = torch.softmax(scores, dim=-1)
            attn = torch.matmul(probs, value)
            return attn.reshape(1, kv * rep * d, 1, 1)

    model = M().eval().half()
    hidden_shape = (1, HIDDEN, 1, 1)
    mask_shape = (1, kv, 1, ROWS)
    traced = torch.jit.trace(model, (torch.zeros(*hidden_shape, dtype=torch.float16),
                                     torch.zeros(*mask_shape, dtype=torch.float16),
                                     torch.zeros(1, dtype=torch.int32)))
    states = [ct.StateType(name='cache',
                           wrapped_type=ct.TensorType(shape=(2 * kv, 1, ROWS, d),
                                                      dtype=np.float16))]
    ct.convert(traced,
               inputs=[ct.TensorType(name='hidden', shape=hidden_shape, dtype=np.float16),
                       ct.TensorType(name='mask', shape=mask_shape, dtype=np.float16),
                       ct.TensorType(name='pos', shape=(1,), dtype=np.int32)],
               states=states, convert_to='mlprogram',
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    return path


def bench(path, units=ct.ComputeUnit.CPU_AND_NE, iters=15):
    model = ct.models.MLModel(path, compute_units=units)
    state = model.make_state()
    feed = {'hidden': np.zeros((1, HIDDEN, 1, 1), np.float16),
            'mask': np.zeros((1, KV, 1, ROWS), np.float16),
            'pos': np.zeros((1,), np.int32)}
    model.predict(feed, state=state)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(feed, state=state)
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 3)


if __name__ == '__main__':
    for kind in ('no_softmax', 'softmax'):
        try:
            path = build(kind, '/tmp/cost_' + kind + '.mlpackage')
            ne = bench(path)
            cpu = bench(path, ct.ComputeUnit.CPU_ONLY)
            print({'kind': kind, 'ne_ms': ne, 'cpu_ms': cpu}, flush=True)
        except Exception as exc:
            print({'kind': kind, 'error': str(exc).replace(chr(10), ' ')[:90]}, flush=True)