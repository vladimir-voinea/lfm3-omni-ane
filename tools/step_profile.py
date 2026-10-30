# Breakdown of a decode step: where the time outside the programs goes.
import sys
import time

import numpy as np

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_runtime import Runtime

ANE = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6'
runtime = Runtime(ANE)
session = runtime.new_session()
H = 2048

# warm up
runtime.step(session, [1, 100, 200, 300])

embed_time = 0.0
layer_times = {}
head_time = 0.0
mask_time = 0.0
convert_time = 0.0

for _ in range(10):
    start = time.perf_counter()
    hidden = np.asarray(runtime.emb[int(500)], np.float16).reshape(1, H, 1, 1)
    embed_time += time.perf_counter() - start
    for index, kind in enumerate(runtime.layer_types):
        program = runtime.programs[index]
        start = time.perf_counter()
        if kind == 'conv':
            out = program.predict({'in0': hidden, 'in1': session.hists[index]})
            hidden = None
            for value in out.values():
                arr = np.asarray(value)
                if arr.shape == (1, H, 1, 1):
                    hidden = arr
                else:
                    session.hists[index] = arr
        else:
            slide = kind == 'sliding_attention'
            cos = runtime.rope['slide_cos' if slide else 'full_cos'][session.position]
            sin = runtime.rope['slide_sin' if slide else 'full_sin'][session.position]
            feed = {'in0': hidden, 'in1': np.array([session.position], np.int32),
                    'in2': cos.reshape(128), 'in3': sin.reshape(128),
                    'in4': runtime.mask(session, kind, session.position)}
            out = program.predict(feed, state=session.states[index])
            hidden = np.asarray(list(out.values())[0])
        layer_times.setdefault((index, kind), []).append(time.perf_counter() - start)
        session.position += 1
    start = time.perf_counter()
    normed = runtime.norm.predict({'in0': hidden})
    runtime.head.predict({'in0': np.asarray(list(normed.values())[0], np.float16)})
    head_time += time.perf_counter() - start

print({'embed_ms': round(embed_time / 10 * 1000, 3), 'head_ms': round(head_time / 10 * 1000, 3)}, flush=True)
totals = {}
for (index, kind), values in layer_times.items():
    totals.setdefault(kind, []).append(np.median(values) * 1000)
for kind, values in totals.items():
    print(kind, 'count', len(values), 'median_ms_sum', round(sum(values), 2),
          'per_layer', [round(v, 3) for v in values[:4]], flush=True)