# How does the engine behave as the cache fills?  The README claims it falls; this measures
# the curve at 2k, 8k and 16k prefill on the shipped 28,672-token build.
import sys
import time

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from lfm3_engine import Engine, render_chat
from sampling import SamplingParams

FILLER = ('The Andes are a long mountain range along the western edge of South America, '
          'and many of its peaks rise above six thousand metres. ')


def prompt_of(engine, target_tokens):
    message = {'role': 'user', 'content': FILLER * 200 + '\n\nQuestion: how tall are the Andes? Answer briefly.'}
    text = render_chat([message])
    ids = [1] + engine.tokenizer.encode(text).ids
    step = max(1, len(ids) // target_tokens)
    return ids[:target_tokens] if len(ids) > target_tokens else ids


if __name__ == '__main__':
    engine = Engine()
    for target in (1024, 4096, 8192):
        ids = prompt_of(engine, target)
        session = engine.runtime.new_session()
        start = time.perf_counter()
        logits = engine.runtime.step(session, ids)
        prefill = time.perf_counter() - start
        produced = 0
        start = time.perf_counter()
        for _ in range(8):
            token = int(logits.argmax())
            produced += 1
            logits = engine.runtime.step(session, [token])
        decode = (time.perf_counter() - start) / produced
        print({'context': len(ids), 'prefill_s': round(prefill, 1),
               'prefill_tps': round(len(ids) / prefill, 1),
               'decode_ms': round(decode * 1000, 1), 'decode_tps': round(1 / decode, 1)},
              flush=True)