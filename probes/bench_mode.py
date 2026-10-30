"""Measure the decode ceiling of one mode at short context."""
import sys
import time

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_runtime import Runtime

MODE = sys.argv[1]
PROMPT = [9707, 500, 1234, 88, 420, 7, 91, 33] * 8      # 64 tokens
DECODE = 32

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6', mode=MODE)
session = runtime.new_session()
start = time.perf_counter()
logits = runtime.step(session, PROMPT)
prefill = time.perf_counter() - start

start = time.perf_counter()
for _ in range(DECODE):
    logits = runtime.step(session, [int(logits.argmax())])
decode = time.perf_counter() - start

print({'mode': runtime.mode.name, 'exact': runtime.mode.exact,
       'prefill_tps': round(len(PROMPT) / prefill, 1),
       'decode_ms': round(decode / DECODE * 1000, 1),
       'decode_tps': round(DECODE / decode, 1)}, flush=True)
