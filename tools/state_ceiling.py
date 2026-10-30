"""How much state can one Core ML program hold on this Neural Engine?

Dory measured ~34 MiB (33.8 loads, 45.0 fails) for a program that does real
attention.  LFM3 needs a different number: one full-attention head at 75k context
is 75,000 x 128 fp16 for K plus the same for V = 36.6 MiB.  If that loads, the plan
is 24 dispatches for full attention; if not, every head needs splitting and the
dispatch budget doubles.  So: build programs whose whole job is to declare a state
of a given size, touch it, and see which ones load.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import coremltools as ct


class Probe(torch.nn.Module):
    def __init__(self, rows: int, dim: int = 128):
        super().__init__()
        self.rows, self.dim = rows, dim
        self.register_buffer("k", torch.zeros(1, rows, dim, dtype=torch.float16),
                             persistent=False)
        self.register_buffer("v", torch.zeros(1, rows, dim, dtype=torch.float16),
                             persistent=False)

    def forward(self, q):
        return q + self.k[:, :1, :] + self.v[:, :1, :]


def probe(rows: int, dim: int = 128) -> dict:
    mib = 2 * rows * dim * 2 / (1024 * 1024)
    started = time.perf_counter()
    try:
        model = Probe(rows, dim).eval()
        traced = torch.jit.trace(model, torch.zeros(1, 1, dim, dtype=torch.float16))
        states = [ct.StateType(name=name,
                               wrapped_type=ct.TensorType(shape=(1, rows, dim),
                                                          dtype=np.float16))
                  for name in ("k", "v")]
        converted = ct.convert(
            traced, inputs=[ct.TensorType(name="q", shape=(1, 1, dim), dtype=np.float16)],
            states=states, convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=ct.ComputeUnit.CPU_AND_NE)
        state = converted.make_state()
        out = converted.predict({"q": np.zeros((1, 1, dim), np.float16)}, state=state)
        return {"rows": rows, "state_mib": round(mib, 1), "loaded": True,
                "out_shape": list(np.asarray(list(out.values())[0]).shape),
                "seconds": round(time.perf_counter() - started, 1)}
    except Exception as exc:  # noqa: BLE001 - the failure is the datum
        return {"rows": rows, "state_mib": round(mib, 1), "loaded": False,
                "error": str(exc).replace(chr(10), " ")[:150]}


if __name__ == "__main__":
    for n in [int(a) for a in sys.argv[1:]] or [8192, 24576, 75000]:
        print(probe(n), flush=True)