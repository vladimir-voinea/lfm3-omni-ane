# Palettize every program of the built model into a parallel directory, then the engine
# can be pointed at it with LFM3_ANE.
import os

from paths import BUILD
import shutil
import sys

import coremltools as ct
import coremltools.optimize.coreml as cto

SRC = BUILD


def main():
    nbits = int(os.environ.get('LUT_BITS', '6'))
    dst = os.environ.get('DST', '/Users/vladimir/dev/lfm3-ane/model-28k-lut%d' % nbits)
    os.makedirs(dst, exist_ok=True)
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=nbits))
    for name in sorted(os.listdir(SRC)):
        source = os.path.join(SRC, name)
        target = os.path.join(dst, name)
        if os.path.exists(target):
            continue
        if name.endswith('.npy') or name.endswith('.npz') or name.endswith('.json'):
            shutil.copy2(source, target)
            continue
        model = ct.models.MLModel(source, compute_units=ct.ComputeUnit.CPU_AND_NE)
        optimized = cto.palettize_weights(model, config)
        optimized.save(target)
        del model, optimized
        print('done', name, flush=True)
    print('all done', flush=True)


if __name__ == '__main__':
    main()