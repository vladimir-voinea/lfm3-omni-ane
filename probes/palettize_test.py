# Does palettization buy speed on this engine, and what does it cost in accuracy?
# Builds LUT4 and LUT6 variants of one converted conv layer and measures both.
import statistics
import time

import numpy as np
import coremltools as ct
import coremltools.optimize.coreml as cto

SRC = '/Users/vladimir/dev/lfm3-ane/model-28k/layer0_conv.mlpackage'
H = 2048


def build(path, nbits):
    model = ct.models.MLModel(SRC, compute_units=ct.ComputeUnit.CPU_AND_NE)
    if nbits is None:
        model.save(path)
        return path
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=nbits))
    palettized = cto.palettize_weights(model, config)
    palettized.save(path)
    return path


def measure(path, seed=0):
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state()
    rng = np.random.default_rng(seed)
    hidden = (rng.standard_normal((1, H, 1, 1)) * 0.5).astype(np.float16)
    hist = np.zeros((1, H, 1, 6), np.float16)
    out = model.predict({'in0': hidden, 'in1': hist})
    result = {np.asarray(v).shape: np.asarray(v, np.float32) for v in out.values()}
    times = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict({'in0': hidden, 'in1': hist})
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times), result, model


if __name__ == '__main__':
    import os
    baseline_ms, baseline, _ = measure(SRC)
    print({'variant': 'fp16', 'ms': round(baseline_ms, 3),
           'size_mb': round(os.path.getsize(SRC + '/Data/com.apple.CoreML/weights/weight.bin') / 1e6, 1)}, flush=True)
    key = [k for k in baseline if baseline[k].size == H][0]
    for nbits in (4, 6):
        path = '/tmp/layer0_lut%d.mlpackage' % nbits
        try:
            build(path, nbits)
            ms, result, _ = measure(path)
            out = result[key]
            ref = baseline[key]
            rel = float(np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-9))
            print({'variant': 'lut%d' % nbits, 'ms': round(ms, 3), 'rel_error': round(rel, 5),
                   'size_mb': round(os.path.getsize(path + '/Data/com.apple.CoreML/weights/weight.bin') / 1e6, 1)},
                  flush=True)
        except Exception as exc:
            print({'variant': 'lut%d' % nbits, 'error': str(exc)[:120]}, flush=True)