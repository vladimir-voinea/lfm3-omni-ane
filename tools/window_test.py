# Beyond the cache: with LFM3_WINDOW=1 the full-attention caches become rings, so a context
# longer than 28,672 tokens runs instead of failing to plan.  Two needles: one at the start
# (outside the window once it rolls) and one near the end (inside it).
import sys
import time

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from lfm3_engine import Engine
from sampling import SamplingParams

FILLER = ('The Andes are a long mountain range along the western edge of South America, '
          'and many of its peaks rise above six thousand metres. ')

if __name__ == '__main__':
    engine = Engine()
    head = 'Remember this: the first code is ALPHA-11. '
    middle = FILLER * 3000
    tail = ' Also remember: the last code is OMEGA-99. '
    question = '\n\nQuestion: what is the last code? Answer with the code only.'
    content = head + middle + tail + question
    messages = [{'role': 'user', 'content': content}]
    tokens = engine.tokenizer.encode(content).ids
    started = time.perf_counter()
    out = list(engine.stream(messages, SamplingParams(), max_tokens=24))
    elapsed = time.perf_counter() - started
    print({'prompt_tokens': len(tokens), 'window_mode': bool(engine.runtime.window),
           'seconds': round(elapsed, 1),
           'answer': engine.tokenizer.decode([t for t, _ in out]).strip()[:60]}, flush=True)