import statistics
import sys
import time

import numpy as np
import coremltools as ct

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from tiled_parts2 import D, H, NKV, rope

ROWS = 28672
model = ct.models.MLModel('/tmp/tile_fp32_full.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_NE)
state = model.make_state()
cos, sin = rope(32, 1e6, 0.5)
feed = {'x': np.zeros((1, H, 1, 1), np.float16), 'pos': np.array([16], np.int32),
        'cos': cos[16].numpy().astype(np.float16), 'sin': sin[16].numpy().astype(np.float16),
        'mask': np.zeros((1, NKV, 1, ROWS), np.float16)}
model.predict(feed, state=state)
times = []
for _ in range(10):
    start = time.perf_counter()
    model.predict(feed, state=state)
    times.append((time.perf_counter() - start) * 1000)
print({'rows': ROWS, 'precision': 'fp32 (CPU)', 'median_ms': round(statistics.median(times), 2)},
      flush=True)