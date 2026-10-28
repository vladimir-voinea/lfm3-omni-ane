"""One tile, all four KV heads in one program?

At 28,672 rows a tile program is ANE-only.  If the same program can carry the whole
layer (4 KV heads, 16 query heads) the per-token attention cost falls from 72
dispatches to 18.  If it cannot, the fallback is one program per KV head.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/Users/vladimir/dev/lfm3-ane/probes")

import numpy as np
import torch
import coremltools as ct

from dory_shape import devices

DIM, KV = 128, 4


def build(rows: int, kv_heads: int, path: str):
    rep = 4  # query heads per KV head in LFM3

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache",
                                 torch.zeros(2 * kv_heads, 1, rows, DIM,
                                             dtype=torch.float16),
                                 persistent=False)
            self.q_proj = torch.nn.Conv2d(4 * DIM, kv_heads * rep * DIM, 1, bias=False)
            self.k_proj = torch.nn.Conv2d(4 * DIM, kv_heads * DIM, 1, bias=False)
            self.v_proj = torch.nn.Conv2d(4 * DIM, kv_heads * DIM, 1, bias=False)

        def forward(self, hidden, mask, pos):
            q = self.q_proj(hidden).view(1, kv_heads * rep, 1, DIM)
            k = self.k_proj(hidden).view(1, kv_heads, 1, DIM)
            v = self.v_proj(hidden).view(1, kv_heads, 1, DIM)
            self.cache[0:kv_heads, :, pos:pos + 1, :] = k.to(self.cache.dtype)
            self.cache[kv_heads:2 * kv_heads, :, pos:pos + 1, :] = v.to(self.cache.dtype)
            key = self.cache[0:kv_heads, :, :rows, :].transpose(-1, -2)
            value = self.cache[kv_heads:2 * kv_heads, :, :rows, :]
            scores = torch.matmul(q.view(1, kv_heads, rep, DIM), key) * 0.0884 + mask
            attn = torch.matmul(torch.softmax(scores, dim=-1),
                                value.unsqueeze(2).expand(-1, -1, rep, -1, -1))
            return attn.reshape(1, kv_heads * rep * DIM, 1, 1)

    model = M().eval().half()
    hidden_shape = (1, 4 * DIM, 1, 1)
    mask_shape = (1, kv_heads, 1, rows)
    traced = torch.jit.trace(model, (torch.zeros(*hidden_shape, dtype=torch.float16),
                                     torch.zeros(*mask_shape, dtype=torch.float16),
                                     torch.zeros(1, dtype=torch.int32)))
    states = [ct.StateType(name="cache",
                           wrapped_type=ct.TensorType(
                               shape=(2 * kv_heads, 1, rows, DIM), dtype=np.float16))]
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
            pkg = build(28672, kv, f"/tmp/tile_kv{kv}.mlpackage")
            print({"kv_heads": kv, "devices": devices(pkg)}, flush=True)
        except Exception as exc:  # noqa: BLE001
            print({"kv_heads": kv, "error": str(exc).replace(chr(10), " ")[:90]}, flush=True)