import collections
import os
import subprocess

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

PATHS = ['/Users/vladimir/dev/lfm3-ane/model-28k/lm_head.mlpackage',
         '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/lm_head.mlpackage',
         '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/lm_head_lut4.mlpackage',
         '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/layer10_full.mlpackage']

for path in PATHS:
    out = '/tmp/cp_' + os.path.basename(os.path.dirname(path)) + '_' + os.path.basename(path)
    subprocess.run(['rm', '-rf', out], check=False)
    os.makedirs(out, exist_ok=True)
    result = subprocess.run(['xcrun', 'coremlcompiler', 'compile', path, out],
                            capture_output=True)
    if result.returncode:
        print({'path': path, 'compile_error': result.stderr.decode()[:80]}, flush=True)
        continue
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
    print({os.path.basename(path): dict(counts)}, flush=True)