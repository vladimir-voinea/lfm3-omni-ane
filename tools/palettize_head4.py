# The LM head costs 11.4 ms in the engine but 2.6 ms standalone: it is memory-bound.
# Test a 4-bit head (half the traffic) and see what it does to speed and answers.
import os
import shutil

import coremltools as ct
import coremltools.optimize.coreml as cto

SRC = '/Users/vladimir/dev/lfm3-ane/model-28k/lm_head.mlpackage'
DST = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/lm_head_lut4.mlpackage'

if __name__ == '__main__':
    model = ct.models.MLModel(SRC, compute_units=ct.ComputeUnit.CPU_AND_NE)
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=4))
    optimized = cto.palettize_weights(model, config)
    optimized.save(DST)
    print({'saved': DST, 'mb': round(os.path.getsize(DST + '/Data/com.apple.CoreML/weights/weight.bin') / 1e6, 1)},
          flush=True)