# Anchor the speed claim at the real limit: fill the shipped 28,672-token cache and measure
# decode speed there.
import sys, time
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_runtime import Runtime

FILLER = list(range(1000, 1000 + 200)) * 145   # ~29,000 token ids that are all valid
runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
session = runtime.new_session()
tokens = FILLER[:28672]
start = time.perf_counter()
logits = runtime.step(session, tokens)
prefill = time.perf_counter() - start
warm = []
start = time.perf_counter()
for _ in range(16):
    token = int(logits.argmax())
    logits = runtime.step(session, [token])
decode = (time.perf_counter() - start) / 16
print({'prefill_tokens': len(tokens), 'prefill_s': round(prefill, 1),
       'prefill_tps': round(len(tokens) / prefill, 1),
       'decode_ms_at_full_context': round(decode * 1000, 1),
       'decode_tps_at_full_context': round(1 / decode, 1)}, flush=True)