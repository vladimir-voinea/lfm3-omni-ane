# Control: the shipped attention program is known to run on the engine.  If this method
# cannot tell the shipped program apart from CPU, the method is wrong, not the tiles.
import glob, time
import numpy as np
import coremltools as ct

path = sorted(glob.glob('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/*_full.mlpackage'))[0]
print({'program': path.split('/')[-1]}, flush=True)


def timing(units, repeats=20):
    model = ct.models.MLModel(path, compute_units=units)
    state = model.make_state()
    feeds = {'in0': np.zeros((1, 2048, 1, 1), np.float16),
             'in1': np.array([0], np.int32),
             'in2': np.zeros((128,), np.float16), 'in3': np.zeros((128,), np.float16),
             'in4': np.full((1, 4, 1, 28672), -1e4, np.float16)}
    model.predict(feeds, state=state)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(feeds, state=state)
    return (time.perf_counter() - start) / repeats * 1000


ne = timing(ct.ComputeUnit.CPU_AND_NE)
cpu = timing(ct.ComputeUnit.CPU_ONLY)
print({'shipped_ne_ms': round(ne, 3), 'shipped_cpu_ms': round(cpu, 3),
       'engine_clearly_faster': ne < cpu * 0.6}, flush=True)