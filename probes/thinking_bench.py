import sys
import time

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
sys.path.insert(0, '/Users/vladimir/dev/dory-ane-worktrees/DANE-5')

from anemll.runtime.dory_sampling import SamplingParams
from lfm3_engine import Engine

QUESTIONS = ['Name three primary colours.', 'What is 17 times 24?', 'Say hello in French.']
engine = Engine()
for thinking in (False, True):
    for question in QUESTIONS:
        messages = [{'role': 'user', 'content': question}]
        start = time.perf_counter()
        tokens = list(engine.stream(messages, SamplingParams(), max_tokens=96,
                                    enable_thinking=thinking))
        elapsed = time.perf_counter() - start
        text = engine.tokenizer.decode([t for t, _ in tokens])
        print({'thinking': thinking, 'q': question[:20], 'tokens': len(tokens),
               'tps': round(len(tokens) / elapsed, 1),
               'head': text[:50].replace(chr(10), ' ')}, flush=True)