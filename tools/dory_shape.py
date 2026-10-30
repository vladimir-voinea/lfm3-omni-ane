"""Copy Dory's accepted construction exactly, then ask the device map about it.

Dory's full-attention program is ANE-only at 24,576 rows.  The differences from my
probe were: one 4D state buffer (2, 1, rows, head_dim) read by slicing the slot axis,
q projected by a 1x1 conv rather than fed from outside, and the cache written in the
same program.  This builds that shape and reads the compute plan at three widths.
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

DIM = 128
QHEADS = 4


def build(rows: int, path: str, write: bool = True):
    dim, heads = DIM, QHEADS

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.zeros(2, 1, rows, dim, dtype=torch.float16),
                                 persistent=False)
            self.q_proj = torch.nn.Conv2d(4 * dim, heads * dim, 1, bias=False)
            self.k_proj = torch.nn.Conv2d(4 * dim, dim, 1, bias=False)
            self.v_proj = torch.nn.Conv2d(4 * dim, dim, 1, bias=False)

        def forward(self, hidden, mask, pos):
            b, _, _, t = hidden.shape
            q = self.q_proj(hidden).view(1, heads, 1, dim)
            k = self.k_proj(hidden).view(1, 1, 1, dim)
            v = self.v_proj(hidden).view(1, 1, 1, dim)
            if write:
                self.cache[0:1, :, pos:pos + 1, :] = k.to(self.cache.dtype)
                self.cache[1:2, :, pos:pos + 1, :] = v.to(self.cache.dtype)
            key = self.cache[0:1, :, :rows, :].transpose(-1, -2)
            value = self.cache[1:2, :, :rows, :]
            scores = torch.matmul(q, key) * 0.0884 + mask
            out = torch.matmul(torch.softmax(scores, dim=-1), value)
            return out.view(1, heads * dim, 1, 1)

    model = M().eval().half()
    hidden_shape = (1, 4 * dim, 1, 1)
    mask_shape = (1, 1, 1, rows)
    traced = torch.jit.trace(model, (torch.zeros(*hidden_shape, dtype=torch.float16),
                                     torch.zeros(*mask_shape, dtype=torch.float16),
                                     torch.zeros(1, dtype=torch.int32)))
    states = [ct.StateType(name="cache",
                           wrapped_type=ct.TensorType(shape=(2, 1, rows, dim),
                                                      dtype=np.float16))]
    ct.convert(traced,
               inputs=[ct.TensorType(name="hidden", shape=hidden_shape, dtype=np.float16),
                       ct.TensorType(name="mask", shape=mask_shape, dtype=np.float16),
                       ct.TensorType(name="pos", shape=(1,), dtype=np.int32)],
               states=states, convert_to="mlprogram",
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    return path


def devices(pkg: str) -> dict:
    out = "/tmp/dw_" + os.path.basename(pkg).replace(".mlpackage", "")
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
                counts["other"] += 1
        for nested in getattr(block, "blocks", []) or []:
            walk(nested)

    for fn in plan.model_structure.program.functions.values():
        walk(fn.block)
    return dict(counts)


if __name__ == "__main__":
    for rows in [int(a) for a in sys.argv[1:]] or [24576, 37500, 65536]:
        pkg = build(rows, f"/tmp/dory_shape_{rows}.mlpackage")
        print({"rows": rows, "devices": devices(pkg)}, flush=True)