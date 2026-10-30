# The head is 11.4 ms because a 128,000-way matmul is not placed on the engine whole.
# Split it into chunks and measure placement and total cost.
import collections
import os
import statistics
import subprocess
import time

import numpy as np
import torch
import coremltools as ct
import coremltools.optimize.coreml as cto
from coremltools.models.compute_plan import MLComputePlan
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
H = 2048
VOCAB = 128000


class Chunk(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return torch.matmul(x.reshape(1, -1), self.weight.t())


def placement(path):
    out = '/tmp/cp_' + os.path.basename(path)
    subprocess.run(['rm', '-rf', out], check=False)
    os.makedirs(out, exist_ok=True)
    subprocess.run(['xcrun', 'coremlcompiler', 'compile', path, out], check=True, capture_output=True)
    name = [f for f in os.listdir(out) if f.endswith('.mlmodelc')][0]
    plan = MLComputePlan.load_from_path(os.path.join(out, name),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    counts = collections.Counter()

    def walk(block):
        for op in block.operations:
            try:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                counts[type(usage.preferred_compute_device).__name__.replace('ComputeDevice', '')] += 1
            except Exception:
                counts['other'] += 1
        for nested in getattr(block, 'blocks', []) or []:
            walk(nested)

    for fn in plan.model_structure.program.functions.values():
        walk(fn.block)
    return dict(counts)


if __name__ == '__main__':
    with safe_open(P, framework='pt') as f:
        emb = f.get_tensor('model.language_model.embed_tokens.weight').float()
    x = np.random.default_rng(0).standard_normal((1, H, 1, 1)).astype(np.float16)
    for chunks in (2, 4, 8):
        size = VOCAB // chunks
        total = 0.0
        note = {}
        for index in range(chunks):
            weight = emb[index * size:(index + 1) * size].clone()
            module = Chunk(weight).eval().half()
            model = ct.convert(torch.jit.trace(module, torch.zeros(1, H, 1, 1, dtype=torch.float16)),
                               inputs=[ct.TensorType(name='in0', shape=(1, H, 1, 1), dtype=np.float16)],
                               compute_precision=ct.precision.FLOAT16,
                               minimum_deployment_target=ct.target.iOS18,
                               compute_units=ct.ComputeUnit.CPU_AND_NE)
            path = '/tmp/head_chunk_%d_of_%d.mlpackage' % (index, chunks)
            config = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=6))
            cto.palettize_weights(model, config).save(path)
            if index == 0:
                note = placement(path)
            loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            loaded.predict({'in0': x})
            times = []
            for _ in range(10):
                start = time.perf_counter()
                loaded.predict({'in0': x})
                times.append((time.perf_counter() - start) * 1000)
            total += statistics.median(times)
            del model, loaded
        print({'chunks': chunks, 'rows_each': size, 'total_ms': round(total, 2),
               'placement_first_chunk': note}, flush=True)