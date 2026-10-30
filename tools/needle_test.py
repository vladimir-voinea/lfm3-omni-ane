# A crash fix is not a correctness fix. Put a fact at the start of a long prompt and ask for
# it at the end: if retrieval works, the cache and the ring buffer are behaving.
import sys
import time

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from lfm3_engine import Engine
from sampling import SamplingParams

FILLER = ('The Andes are a long mountain range along the western edge of South America, '
          'and many of its peaks rise above six thousand metres. ')
NEEDLE = 'Remember this: the secret code is PLATYPUS-42.'


if __name__ == '__main__':
    engine = Engine()
    for repeats in (200,):
        content = (NEEDLE + ' ' + FILLER * repeats +
                   '\n\nQuestion: what is the secret code? Answer with the code only.')
        messages = [{'role': 'user', 'content': content}]
        start = time.perf_counter()
        tokens = list(engine.stream(messages, SamplingParams(), max_tokens=24))
        elapsed = time.perf_counter() - start
        answer = engine.tokenizer.decode([t for t, _ in tokens])
        prompt_len = engine.tokenizer.encode(content).ids.__len__()
        print({'prompt_tokens': prompt_len, 'seconds': round(elapsed, 1),
               'answer': answer.strip()[:60].replace(chr(10), ' ')}, flush=True)