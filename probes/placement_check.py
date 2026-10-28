"""Is that attention probe really on the Neural Engine, or did Core ML fall back?

Timing alone cannot say.  Convert the same program twice, once restricted to the
CPU and once letting Core ML choose, and compare: a matmul over a 75,000-row cache
is the kind of thing the CPU can do too, so the ratio is the evidence.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import coremltools as ct


class Attention(torch.nn.Module):
    def __init__(self, rows: int, heads: int = 4, dim: int = 128):
        super().__init__()
        self.rows, self.heads, self.dim = rows, heads, dim
        self.register_buffer("k", torch.zeros(1, rows, dim, dtype=torch.float16),
                             persistent=False)
        self.register_buffer("v", torch.zeros(1, rows, dim, dtype=torch.float16),
                             persistent=False)

    def forward(self, q):
        k = self.k.to(torch.float32).transpose(1, 2)
        scores = torch.matmul(q.to(torch.float32), k) / (self.dim ** 0.5)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, self.v.to(torch.float32)).to(torch.float16)


def build(rows: int, units):
    model = Attention(rows).eval()
    example = torch.zeros(4, 1, 128, dtype=torch.float16)
    traced = torch.jit.trace(model, example)
    states = [ct.StateType(name=n, wrapped_type=ct.TensorType(shape=(1, rows, 128),
                                                             dtype=np.float16))
              for n in ("k", "v")]
    converted = ct.convert(traced,
                           inputs=[ct.TensorType(name="q", shape=(4, 1, 128),
                                                 dtype=np.float16)],
                           states=states, convert_to="mlprogram",
                           compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=ct.target.iOS18,
                           compute_units=units)
    state = converted.make_state()
    feed = {"q": np.zeros((4, 1, 128), np.float16)}
    converted.predict(feed, state=state)
    t0 = time.perf_counter()
    for _ in range(10):
        out = converted.predict(feed, state=state)
    return (time.perf_counter() - t0) / 10 * 1000, list(np.asarray(list(out.values())[0]).shape)


if __name__ == "__main__":
    rows = 75000
    for label, units in (("cpu_only", ct.ComputeUnit.CPU_ONLY),
                         ("all_units", ct.ComputeUnit.ALL),
                         ("cpu_and_ne", ct.ComputeUnit.CPU_AND_NE)):
        try:
            ms, shape = build(rows, units)
            print({"units": label, "ms": round(ms, 2), "out": shape}, flush=True)
        except Exception as exc:  # noqa: BLE001
            print({"units": label, "error": str(exc)[:120]}, flush=True)