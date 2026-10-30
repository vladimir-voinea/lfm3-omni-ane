# Does interleaving two sessions beat running them one after the other?  If the engine
# stalls between the 26 dispatches, a second session should fill the gaps.
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_runtime import Runtime

ANE = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6'


def ids_for(text):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ANE, 'tokenizer.json'))
    return [1] + tok.encode(text).ids


def decode(runtime, prompt, steps=16):
    session = runtime.new_session()
    logits = runtime.step(session, prompt)
    produced = []
    for _ in range(steps):
        token = int(np.argmax(logits))
        produced.append(token)
        logits = runtime.step(session, [token])
    return produced


if __name__ == '__main__':
    runtime = Runtime(ANE)
    prompt = ids_for('Name three colours, comma separated.')
    decode(runtime, prompt, 4)                       # warm up

    start = time.perf_counter()
    decode(runtime, prompt, 16)
    one = 16 / (time.perf_counter() - start)
    print({'single_tps': round(one, 2)}, flush=True)

    collected = {}

    def worker(name):
        collected[name] = decode(runtime, prompt, 16)

    start = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - start
    print({'two_sessions_wall_s': round(wall, 2), 'aggregate_tps': round(32 / wall, 2),
           'per_session_tps': round(16 / wall, 2)}, flush=True)