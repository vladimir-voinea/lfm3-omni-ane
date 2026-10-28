"""Confirm the ANE-eligible attention shape at 75k, and time it properly."""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch
import coremltools as ct

ROWS, DIM, HEADS = 75000, 128, 4


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("k", torch.zeros(1, ROWS, DIM, dtype=torch.float16), persistent=False)
        self.register_buffer("v", torch.zeros(1, ROWS, DIM, dtype=torch.float16), persistent=False)

    def forward(self, q, mask):
        k = self.k.view(1, 1, ROWS, DIM).transpose(-1, -2)
        scores = torch.matmul(q, k) * 0.0884 + mask
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, self.v.view(1, 1, ROWS, DIM))


def bench(units, iters=30):
    model = M().eval()
    q_shape, mask_shape = (1, HEADS, 1, DIM), (1, 1, 1, ROWS)
    traced = torch.jit.trace(model, (torch.zeros(*q_shape, dtype=torch.float16),
                                     torch.zeros(*mask_shape, dtype=torch.float16)))
    states = [ct.StateType(name=n,
                           wrapped_type=ct.TensorType(shape=(1, ROWS, DIM), dtype=np.float16))
              for n in ("k", "v")]
    converted = ct.convert(traced,
                           inputs=[ct.TensorType(name="q", shape=q_shape, dtype=np.float16),
                                   ct.TensorType(name="mask", shape=mask_shape, dtype=np.float16)],
                           states=states, convert_to="mlprogram",
                           compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=ct.target.iOS18,
                           compute_units=units)
    state = converted.make_state()
    feed = {"q": np.zeros(q_shape, np.float16), "mask": np.zeros(mask_shape, np.float16)}
    for _ in range(3):
        converted.predict(feed, state=state)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter(); converted.predict(feed, state=state); times.append((time.perf_counter()-t0)*1000)
    return {"median_ms": round(statistics.median(times), 3),
            "p95_ms": round(sorted(times)[int(0.95*len(times))-1], 3),
            "min_ms": round(min(times), 3)}


if __name__ == "__main__":
    cpu = bench(ct.ComputeUnit.CPU_ONLY)
    ne = bench(ct.ComputeUnit.CPU_AND_NE)
    print({"cpu": cpu, "cpu_and_ne": ne}, flush=True)
    print({"speedup": round(cpu["median_ms"] / ne["median_ms"], 2),
           "full_attention_ms_per_token": round(24 * ne["median_ms"], 1),
           "implied_tps_if_only_attention": round(1000 / (24 * ne["median_ms"]), 1)}, flush=True)