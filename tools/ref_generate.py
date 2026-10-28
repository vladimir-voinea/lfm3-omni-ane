import os

from paths import MODEL, PROJECT, SOURCE
# The same prompt through the pure-torch reference, greedy, so the engine's output
# can be compared against it word for word.
import sys

sys.path.insert(0, PROJECT)

import numpy as np
from tokenizers import Tokenizer

from ref_stack import forward

MODEL = SOURCE


def main():
    tok = Tokenizer.from_file(MODEL + '/tokenizer.json')
    prompt = sys.argv[1] if len(sys.argv) > 1 else 'The capital of France is'
    ids = tok.encode(prompt).ids
    for _ in range(12):
        logits = forward(ids)
        nxt = int(np.argmax(logits))
        ids.append(nxt)
    print({'reference_completion': tok.decode(ids[len(tok.encode(prompt).ids):])}, flush=True)


if __name__ == '__main__':
    main()