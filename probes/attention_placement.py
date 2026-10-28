"""Which part of a 75k attention program knocks it off the Neural Engine?

The conv control shows the NE is real here (1.63 ms -> 0.81 ms).  The attention
program over a 75,000-row state measured the same under CPU_ONLY and CPU_AND_NE, so
it fell back.  This walks the graph: pure fp16 matmuls, then softmax, then the
fp32 casts, to find the op that costs us the NE.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import coremltools as ct

ROWS = 75000
DIM = 128
HEADS = 4


class Variant(torch.nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.register_buffer("k", torch.zeros(1, ROWS, DIM, dtype=torch.float16),
                             persistent=False)
        self.register_buffer("v", torch.zeros(1, ROWS, DIM, dtype=torch.float16),
                             persistent=False)

    def forward(self, q):
        kt = self.k.transpose(1, 2)
        if self.kind == "fp16_mm":
            scores = torch.matmul(q, kt)
            return torch.matmul(scores, self.v)
        if self.kind == "fp16_softmax":
            scores = torch.matmul(q, kt) / (DIM ** 0.5)
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, self.v)
        if self.kind == "fp32_softmax":
            scores = torch.matmul(q.float(), self.k.float().transpose(1, 2)) / (DIM ** 0.5)
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, self.v.float()).half()
        if self.kind == "sliced":
            scores = torch.matmul(q, kt)
            head = scores[:, :, :4096]
            return torch.matmul(head, self.v[:, :4096, :])
        raise ValueError(self.kind)


def measure(kind, units, label):
    try:
        model = Variant(kind).eval()
        example = torch.zeros(HEADS, 1, DIM, dtype=torch.float16)
        traced = torch.jit.trace(model, example)
        states = [ct.StateType(name=n, wrapped_type=ct.TensorType(shape=(1, ROWS, DIM),
                                                               dtype=np.float16))
                  for n in ("k", "v")]
        converted = ct.convert(traced,
                               inputs=[ct.TensorType(name="q",
                                                     shape=(HEADS, 1, DIM),
                                                     dtype=np.float16)],
                               states=states, convert_to="mlprogram",
                               compute_precision=ct.precision.FLOAT16,
                               minimum_deployment_target=ct.target.iOS18,
                               compute_units=units)
        state = converted.make_state()
        feed = {"q": np.zeros((HEADS, 1, DIM), np.float16)}
        converted.predict(feed, state=state)
        t0 = time.perf_counter()
        for _ in range(5):
            converted.predict(feed, state=state)
        return round((time.perf_counter() - t0) / 5 * 1000, 3)
    except Exception as exc:  # noqa: BLE001
        return "error: " + str(exc).replace(chr(10), " ")[:90]


if __name__ == "__main__":
    for kind in ("fp16_mm", "sliced", "fp16_softmax", "fp32_softmax"):
        cpu = measure(kind, ct.ComputeUnit.CPU_ONLY, "cpu")
        ne = measure(kind, ct.ComputeUnit.CPU_AND_NE, "ne")
        verdict = "?"
        if isinstance(cpu, float) and isinstance(ne, float):
            verdict = "NE" if ne < cpu * 0.75 else "cpu-ish"
        print({"variant": kind, "cpu_ms": cpu, "ne_ms": ne, "verdict": verdict}, flush=True)