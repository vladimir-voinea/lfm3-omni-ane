# Does the model need its BOS token, and does a chat-formatted prompt behave better?
import sys

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

import numpy as np
from tokenizers import Tokenizer

from ref_stack import forward

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'


def run(tok, ids, steps=8):
    seq = list(ids)
    out = []
    for _ in range(steps):
        logits = forward(seq)
        nxt = int(np.argmax(logits))
        out.append(nxt)
        seq.append(nxt)
    return tok.decode(out), out


if __name__ == '__main__':
    tok = Tokenizer.from_file(MODEL + '/tokenizer.json')
    base = tok.encode('The capital of France is').ids
    print({'with_bos': run(tok, [1] + base)[0]}, flush=True)
    chat = tok.encode('<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n').ids
    print({'chat_ids': chat[:12], 'chat': run(tok, chat)[0]}, flush=True)