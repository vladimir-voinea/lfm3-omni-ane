"""Where is the softmax width wall on the ANE?  Dory softmaxes 24,576 on the engine;
the 75,000 LFM3 shape falls to the CPU.  Bisect the width and read the device map."""
from __future__ import annotations

import collections
import os
import subprocess
import sys

import numpy as np
import torch
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

DIM = 128


def build(rows: int, path: str):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("k", torch.zeros(1, rows, DIM, dtype=torch.float16), persistent=False)
            self.register_buffer("v", torch.zeros(1, rows, DIM, dtype=torch.float16), persistent=False)

        def forward(self, q, mask):
            k = self.k.view(1, 1, rows, DIM).transpose(-1, -2)
            v = self.v.view(1, 1, rows, DIM)
            scores = torch.matmul(q, k) * 0.0884 + mask
            return torch.matmul(torch.softmax(scores, dim=-1), v)

    model = M().eval()
    q_shape, mask_shape = (1, 4, 1, DIM), (1, 1, 1, rows)
    traced = torch.jit.trace(model, (torch.zeros(*q_shape, dtype=torch.float16),
                                     torch.zeros(*mask_shape, dtype=torch.float16)))
    states = [ct.StateType(name=n, wrapped_type=ct.TensorType(shape=(1, rows, DIM),
                                                           dtype=np.float16))
              for n in ("k", "v")]
    ct.convert(traced,
               inputs=[ct.TensorType(name="q", shape=q_shape, dtype=np.float16),
                       ct.TensorType(name="mask", shape=mask_shape, dtype=np.float16)],
               states=states, convert_to="mlprogram",
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    return path


def devices(pkg: str) -> dict:
    out = "/tmp/pw_" + os.path.basename(pkg).replace(".mlpackage", "")
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
                counts[type(usage.preferred_compute_device).__name__
                       .replace("ComputeDevice", "")] += 1
            except Exception:  # noqa: BLE001
                counts["unsupported"] += 1
        for nested in getattr(block, "blocks", []) or []:
            walk(nested)

    for fn in plan.model_structure.program.functions.values():
        walk(fn.block)
    return dict(counts)


if __name__ == "__main__":
    for rows in [int(a) for a in sys.argv[1:]] or [32768, 65536, 75000]:
        pkg = build(rows, f"/tmp/pw_{rows}.mlpackage")
        print({"rows": rows, "devices": devices(pkg)}, flush=True)