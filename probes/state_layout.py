"""Which state/attention construction does the ANE accept at 75k?

Decided by MLComputePlan, not by stopwatch: every earlier timing verdict in this
project was a guess about which device ran the work.
"""
from __future__ import annotations

import collections
import os
import subprocess
import sys

import numpy as np
import torch
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

ROWS, DIM = 75000, 128


def build(kind: str, path: str):
    rows, dim = ROWS, DIM

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "state4d":
                self.register_buffer("k", torch.zeros(1, 1, rows, dim, dtype=torch.float16), persistent=False)
                self.register_buffer("v", torch.zeros(1, 1, rows, dim, dtype=torch.float16), persistent=False)
            else:
                self.register_buffer("k", torch.zeros(1, rows, dim, dtype=torch.float16), persistent=False)
                self.register_buffer("v", torch.zeros(1, rows, dim, dtype=torch.float16), persistent=False)

        def forward(self, q, mask):
            if kind == "state4d":
                k = self.k
                v = self.v
            else:
                k = self.k.view(1, 1, rows, dim)
                v = self.v.view(1, 1, rows, dim)
            scores = torch.matmul(q, k.transpose(-1, -2)) * 0.0884 + mask
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)

    model = M().eval()
    q_shape, mask_shape = (1, 4, 1, dim), (1, 1, 1, rows)
    traced = torch.jit.trace(model, (torch.zeros(*q_shape, dtype=torch.float16),
                                     torch.zeros(*mask_shape, dtype=torch.float16)))
    state_shape = (1, 1, rows, dim) if kind == "state4d" else (1, rows, dim)
    states = [ct.StateType(name=n,
                           wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16))
              for n in ("k", "v")]
    converted = ct.convert(traced,
                           inputs=[ct.TensorType(name="q", shape=q_shape, dtype=np.float16),
                                   ct.TensorType(name="mask", shape=mask_shape, dtype=np.float16)],
                           states=states, convert_to="mlprogram",
                           compute_precision=ct.precision.FLOAT16,
                           minimum_deployment_target=ct.target.iOS18,
                           compute_units=ct.ComputeUnit.CPU_AND_NE)
    converted.save(path)
    return path


def devices(pkg: str) -> dict:
    out = "/tmp/plan_v_" + os.path.basename(pkg).replace(".mlpackage", "")
    subprocess.run(["rm", "-rf", out], check=False)
    os.makedirs(out, exist_ok=True)
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out],
                   check=True, capture_output=True)
    name = [f for f in os.listdir(out) if f.endswith(".mlmodelc")][0]
    plan = MLComputePlan.load_from_path(os.path.join(out, name),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = collections.Counter()

    def walk(block):
        for op in block.operations:
            try:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                counts[type(usage.preferred_compute_device).__name__] += 1
            except Exception:  # noqa: BLE001
                counts["unsupported"] += 1
        for nested in getattr(block, "blocks", []) or []:
            walk(nested)

    for fn in plan.model_structure.program.functions.values():
        walk(fn.block)
    return dict(counts)


if __name__ == "__main__":
    for kind in ("state4d", "state3d"):
        pkg = build(kind, f"/tmp/v_{kind}.mlpackage")
        print({"kind": kind, "devices": devices(pkg)}, flush=True)