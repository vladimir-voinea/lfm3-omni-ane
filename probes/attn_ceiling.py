"""The probe that actually decides LFM3-on-ANE: real attention over a real cache.

The state ceiling measured with a trivial program (293 MiB loaded) says nothing
about a program that has to build a 75,000-wide score tensor inside itself -- that
is the shape Dory found guilty at 4,096 wide.  So: one program per head-group that
scores the whole cache, softmaxes it, and multiplies by V.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import coremltools as ct


class Attention(torch.nn.Module):
    """One KV head group, `heads` query heads sharing it, over `rows` positions."""

    def __init__(self, rows: int, heads: int = 1, dim: int = 128, kv_heads: int = 1):
        super().__init__()
        self.rows, self.heads, self.dim = rows, heads, dim
        self.register_buffer("k", torch.zeros(kv_heads, rows, dim, dtype=torch.float16),
                             persistent=False)
        self.register_buffer("v", torch.zeros(kv_heads, rows, dim, dtype=torch.float16),
                             persistent=False)

    def forward(self, q):
        # q: (heads, 1, dim); cache: (1, rows, dim)
        k = self.k.to(torch.float32).transpose(1, 2)          # (1, dim, rows)
        scores = torch.matmul(q.to(torch.float32), k) / (self.dim ** 0.5)
        probs = torch.softmax(mb_scratch(scores), dim=-1)
        v = self.v.to(torch.float32)                          # (1, rows, dim)
        return torch.matmul(probs, v).to(torch.float16)       # (heads, 1, dim)


def mb_scratch(x):
    return x


def probe(rows: int, heads: int = 1, dim: int = 128) -> dict:
    mib = 2 * rows * dim * 2 / (1024 * 1024)
    started = time.perf_counter()
    try:
        model = Attention(rows, heads, dim).eval()
        example = torch.zeros(heads, 1, dim, dtype=torch.float16)
        traced = torch.jit.trace(model, example)
        states = [ct.StateType(name=name,
                               wrapped_type=ct.TensorType(shape=(1, rows, dim),
                                                          dtype=np.float16))
                  for name in ("k", "v")]
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="q", shape=tuple(example.shape), dtype=np.float16)],
            states=states, convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = converted.make_state()
        feed = {"q": np.zeros(tuple(example.shape), np.float16)}
        converted.predict(feed, state=state)
        t0 = time.perf_counter()
        for _ in range(5):
            out = converted.predict(feed, state=state)
        ms = (time.perf_counter() - t0) / 5 * 1000
        return {"rows": rows, "heads": heads, "state_mib": round(mib, 1), "loaded": True,
                "out_shape": list(np.asarray(list(out.values())[0]).shape),
                "ms": round(ms, 2),
                "seconds": round(time.perf_counter() - started, 1)}
    except Exception as exc:  # noqa: BLE001
        return {"rows": rows, "heads": heads, "state_mib": round(mib, 1),
                "loaded": False, "error": str(exc).replace(chr(10), " ")[:150]}


if __name__ == "__main__":
    for rows in (8192, 75000):
        for heads in (1, 4):
            print(probe(rows, heads), flush=True)