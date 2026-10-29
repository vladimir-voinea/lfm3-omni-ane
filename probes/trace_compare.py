# Same tokens, tiled off and on, compare the hidden state after every layer.  Real caches,
# real masks -- the first layer that diverges is the broken one.
import sys, os
import numpy as np
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_runtime import Runtime

TOKENS = [1, 9707, 500]


def run(tiled):
    os.environ['LFM3_TILED'] = '1' if tiled else '0'
    runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
    runtime.trace = []
    session = runtime.new_session()
    runtime.step(session, TOKENS)
    return runtime.trace


plain = run(False)
tiled = run(True)
print({'steps': [len(plain), len(tiled)]}, flush=True)
for index, (a, b) in enumerate(zip(plain, tiled)):
    layer = index % 24
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9)) * 100
    if index < 8:
        print({'index': index, 'layer': layer, 'rel_pct': round(rel, 2)}, flush=True)