"""Control: a conv the NE is known to like, run under CPU-only and CPU+NE.

If those two differ, the ANE is participating and a program whose timing is
identical under both is falling back to the CPU.  That distinction decides whether
the 75k attention program can be trusted at 0.65 ms.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import coremltools as ct


class BigConv(torch.nn.Module):
    def __init__(self, hidden=2048, inner=7680):
        super().__init__()
        self.a = torch.nn.Conv2d(hidden, inner, 1)
        self.b = torch.nn.Conv2d(inner, hidden, 1)

    def forward(self, x):
        return self.b(torch.nn.functional.silu(self.a(x)))


def run(units, label, tokens=64, hidden=2048):
    model = BigConv(hidden).eval()
    example = torch.zeros(1, hidden, 1, tokens)
    traced = torch.jit.trace(model, example)
    converted = ct.convert(traced,
                           inputs=[ct.TensorType(name="x", shape=tuple(example.shape))],
                           convert_to="mlprogram",
                           compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=ct.target.iOS18,
                           compute_units=units)
    feed = {"x": np.zeros(tuple(example.shape), np.float16)}
    converted.predict(feed)
    t0 = time.perf_counter()
    for _ in range(10):
        converted.predict(feed)
    return {"units": label, "ms": round((time.perf_counter() - t0) / 10 * 1000, 3)}


if __name__ == "__main__":
    for label, units in (("cpu_only", ct.ComputeUnit.CPU_ONLY),
                         ("cpu_and_ne", ct.ComputeUnit.CPU_AND_NE)):
        try:
            print(run(units, label), flush=True)
        except Exception as exc:  # noqa: BLE001
            print({"units": label, "error": str(exc)[:120]}, flush=True)