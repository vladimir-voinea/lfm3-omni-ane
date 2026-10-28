import os

from paths import ANE, MODEL, PROJECT, SOURCE
# Generate text with the engine model and read it back as words.
# A model whose math is wrong produces gibberish; a correct one completes a sentence.
import sys

import numpy as np

sys.path.insert(0, PROJECT)

from tokenizers import Tokenizer

from run_model import Lfm3Ane

MODEL = SOURCE
ANE = MODEL


def main():
    tok = Tokenizer.from_file(MODEL + '/tokenizer.json')
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'The capital of France is'
    ids = tok.encode(prompt).ids
    print({'prompt': prompt, 'tokens': ids[:20], 'count': len(ids)}, flush=True)

    runtime = Lfm3Ane(ANE)
    logits, seconds = runtime.step(ids)
    print({'prefill_seconds': round(seconds, 2)}, flush=True)

    out_ids = []
    started = seconds
    for _ in range(12):
        nxt = int(np.argmax(logits))
        out_ids.append(nxt)
        logits, _ = runtime.step([nxt])
    text = tok.decode(out_ids)
    print({'completion': text}, flush=True)
    print({'generated_tokens': out_ids}, flush=True)


if __name__ == '__main__':
    main()