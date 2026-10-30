"""One tile, all KV heads in one program -- Dory's exact 4D matmul form.

q is (1, kv, rep, d) with the query-head axis in the sequence slot, the cache is
(2*kv, 1, rows, d) sliced by slot, and the two matmuls are (1,kv,rep,d) x
(1,kv,d,rows) and (1,kv,rep,rows) x (1,kv,rows,d) -- the shape Dory proved the
engine takes.  Question: how many KV heads fit before it falls off?
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Users/vladimir/dev/lfm3-ane/probes")

import numpy as np
import torch
import coremltools as ct

from dory_shape import devices

DIM, REP, HIDDEN = 128, 4, 2048


def build(rows: int, kv_heads: int, path: str):
    kv, rep, d = kv_heads, REP, DIM

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache",
                                 torch.zeros(2 * kv, 1, rows, d, dtype=torch.float16),
                                 persistent=False)
            self.q_proj = torch.nn.Conv2d(HIDDEN, kv * rep * d, 1, bias=False)
            self.k_proj = torch.nn.Conv2d(HIDDEN, kv * d, 1, bias=False)
            self.v_proj = torch.nn.Conv2d(HIDDEN, kv * d, 1, bias=False)

        def forward(self, hidden, mask, pos):
            q = self.q_proj(hidden).view(1, kv, rep, d)
            k = self.k_proj(hidden).view(kv, 1, 1, d)
            v = self.v_proj(hidden).view(kv, 1, 1, d)
            self.cache[0:kv, :, pos:pos + 1, :] = k
            self.cache[kv:2 * kv, :, pos:pos + 1, :] = v
            key = self.cache[0:kv, :, :rows, :].view(1, kv, rows, d).transpose(-1, -2)
            value = self.cache[kv:2 * kv, :, :rows, :].view(1, kv, rows, d)
            scores = torch.matmul(q, key) * 0.0884 + mask
            attn = torch.matmul(torch.softmax(scores, dim=-1), value)
            return attn.reshape(1, kv * rep * d, 1, 1)

    model = M().eval().half()
    hidden_shape = (1, HIDDEN, 1, 1)
    mask_shape = (1, kv, 1, rows)
    traced = torch.jit.trace(model, (torch.zeros(*hidden_shape, dtype=torch.float16),
                                     torch.zeros(*mask_shape, dtype=torch.float16),
                                     torch.zeros(1, dtype=torch.int32)))
    states = [ct.StateType(name="cache",
                           wrapped_type=ct.TensorType(
                               shape=(2 * kv, 1, rows, d), dtype=np.float16))]
    ct.convert(traced,
               inputs=[ct.TensorType(name="hidden", shape=hidden_shape, dtype=np.float16),
                       ct.TensorType(name="mask", shape=mask_shape, dtype=np.float16),
                       ct.TensorType(name="pos", shape=(1,), dtype=np.int32)],
               states=states, convert_to="mlprogram",
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    return path


if __name__ == "__main__":
    for kv in (1, 2, 4):
        try:
            pkg = build(28672, kv, f"/tmp/tile2_kv{kv}.mlpackage")
            print({"kv_heads": kv, "devices": devices(pkg)}, flush=True)
        except Exception as exc:  # noqa: BLE001
            print({"kv_heads": kv, "error": str(exc).replace(chr(10), " ")[:100]}, flush=True)