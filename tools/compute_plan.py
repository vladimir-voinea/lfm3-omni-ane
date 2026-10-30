"""Where does Core ML actually place each op?  MLComputePlan answers it directly."""
from __future__ import annotations

import collections
import sys

import numpy as np
import torch
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def save(rows=75000, dim=128, heads=4, with_mask=True, path="/tmp/probe.mlpackage"):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("k", torch.zeros(1, rows, dim, dtype=torch.float16), persistent=False)
            self.register_buffer("v", torch.zeros(1, rows, dim, dtype=torch.float16), persistent=False)

        def forward(self, q, mask):
            k = self.k.view(1, 1, rows, dim).transpose(-1, -2)
            scores = torch.matmul(q, k) * 0.0884
            if with_mask:
                scores = scores + mask
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, self.v.view(1, 1, rows, dim))

    model = M().eval()
    q_shape, mask_shape = (1, heads, 1, dim), (1, 1, 1, rows)
    traced = torch.jit.trace(model, (torch.zeros(*q_shape, dtype=torch.float16),
                                     torch.zeros(*mask_shape, dtype=torch.float16)))
    states = [ct.StateType(name=n,
                           wrapped_type=ct.TensorType(shape=(1, rows, dim), dtype=np.float16))
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


def report(path):
    plan = MLComputePlan.load_from_path(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = collections.Counter()

    def walk(block):
        for op in block.operations:
            try:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                device = str(usage.preferred_compute_device).split("(")[-1].rstrip(")")
            except Exception as exc:  # noqa: BLE001
                device = "unknown"
            counts[device] += 1
        for nested in getattr(block, "blocks", []) or []:
            walk(nested)

    program = plan.model_structure.program
    for function in program.functions.values():
        walk(function.block)
    return dict(counts)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "lfm3"
    if which == "lfm3":
        path = save()
    else:
        path = ("/Users/vladimir/dev/dory-ane/dane5-artifacts/set24k/"
                "dory_attn_full_0_h0_lut6_infer.mlpackage")
    print({"model": which, "device_usage": report(path)}, flush=True)