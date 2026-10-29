# Does batched prefill compute the same thing as serial prefill?  Same 128 tokens,
# two sessions, compare the logits the head would see.
import sys
import time

import numpy as np

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from lfm3_runtime import Runtime

TOKENS = (list(range(1000, 1064)) + list(range(2000, 2064)))

if __name__ == '__main__':
    runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
    serial = runtime.new_session()
    start = time.perf_counter()
    runtime.step(serial, TOKENS)
    serial_s = time.perf_counter() - start
    batched = runtime.new_session()
    start = time.perf_counter()
    consumed = runtime.prefill_batch(batched, TOKENS)
    batched_s = time.perf_counter() - start
    rest = TOKENS[consumed:]
    if rest:
        runtime.step(batched, rest)
    a = np.asarray(serial.hidden).reshape(-1).astype(np.float32)
    b = np.asarray(batched.hidden).reshape(-1).astype(np.float32)
    denom = np.abs(a).mean() + 1e-9
    print({'tokens': len(TOKENS), 'batched_consumed': consumed,
           'serial_s': round(serial_s, 1), 'batched_s': round(batched_s, 1),
           'speedup': round(serial_s / batched_s, 1) if batched_s else None,
           'hidden_rel_diff': round(float(np.abs(a - b).mean() / denom), 5),
           'position_serial': serial.position, 'position_batched': batched.position},
          flush=True)