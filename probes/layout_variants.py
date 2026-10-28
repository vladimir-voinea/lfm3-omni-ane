"""Which attention layout does the Neural Engine accept at 75k?

Dory's program (proven on the ANE at 24,576 rows) keeps everything 4D: a mask
added to the logits, state read as (2, 1, rows, dim), and matmuls of the form
(1, 1, q, d) x (1, 1, d, rows).  The first probe used 3D tensors and landed on the
CPU.  This walks: 3D vs 4D, with and without the mask add, and reports which one
shows the ANE's advantage (a lower time under CPU_AND_NE than under CPU_ONLY).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import coremltools as ct

ROWS = 75000
DIM = 128
HEADS = 4


def build(kind: str):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("k", torch.zeros(1, ROWS, DIM, dtype=torch.float16),
                                 persistent=False)
            self.register_buffer("v", torch.zeros(1, ROWS, DIM, dtype=torch.float16),
                                 persistent=False)

        def forward(self, q, mask):
            k = self.k.view(1, 1, ROWS, DIM).transpose(-1, -2)   # (1,1,dim,rows)
            scores = torch.matmul(q, k) * 0.0884                 # (1,heads,1,rows)
            if kind in ("4d_mask", "4d_mask_softmax"):
                scores = scores + mask
            if kind in ("3d_softmax", "4d_mask_softmax"):
                probs = torch.softmax(scores, dim=-1)
                return torch.matmul(probs, self.v.view(1, 1, ROWS, DIM))
            return scores

    model = M().eval()
    q_shape = (1, HEADS, 1, DIM)
    mask_shape = (1, 1, 1, ROWS)
    q = torch.zeros(*q_shape, dtype=torch.float16)
    mask = torch.zeros(*mask_shape, dtype=torch.float16)
    return model, q, mask, q_shape, mask_shape


def measure(kind: str, units):
    try:
        model, q, mask, q_shape, mask_shape = build(kind)
        traced = torch.jit.trace(model, (q, mask))
        states = [ct.StateType(name=n,
                               wrapped_type=ct.TensorType(shape=(1, ROWS, DIM),
                                                          dtype=np.float16))
                  for n in ("k", "v")]
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="q", shape=q_shape, dtype=np.float16),
                    ct.TensorType(name="mask", shape=mask_shape, dtype=np.float16)],
            states=states, convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
            compute_units=units)
        state = converted.make_state()
        feed = {"q": np.zeros(q_shape, np.float16), "mask": np.zeros(mask_shape, np.float16)}
        converted.predict(feed, state=state)
        t0 = time.perf_counter()
        for _ in range(5):
            converted.predict(feed, state=state)
        return round((time.perf_counter() - t0) / 5 * 1000, 3)
    except Exception as exc:  # noqa: BLE001
        return "error: " + str(exc).replace(chr(10), " ")[:80]


if __name__ == "__main__":
    for kind in ("4d_scores", "4d_mask", "3d_softmax", "4d_mask_softmax"):
        cpu = measure(kind, ct.ComputeUnit.CPU_ONLY)
        ne = measure(kind, ct.ComputeUnit.CPU_AND_NE)
        verdict = "?"
        if isinstance(cpu, float) and isinstance(ne, float):
            verdict = "ANE" if ne < cpu * 0.8 else "cpu"
        print({"variant": kind, "cpu_ms": cpu, "ne_ms": ne, "verdict": verdict}, flush=True)