# The engine, with the BOS token the model expects, on both prompt forms.
import sys

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from run_model import Lfm3Ane

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
ANE = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6'


def generate(runtime, tok, ids, steps=8):
    logits, seconds = runtime.step(ids)
    out = []
    for _ in range(steps):
        nxt = int(np.argmax(logits))
        out.append(nxt)
        logits, _ = runtime.step([nxt])
    return tok.decode(out), out, seconds


if __name__ == '__main__':
    tok = Tokenizer.from_file(MODEL + '/tokenizer.json')
    runtime = Lfm3Ane(ANE)
    base = [1] + tok.encode('The capital of France is').ids
    text, ids, seconds = generate(runtime, tok, base, 8)
    print({'base_prompt': text, 'prefill_s': round(seconds, 2), 'ids': ids}, flush=True)

    chat = tok.encode('<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n').ids
    text, ids, seconds = generate(runtime, tok, chat, 10)
    print({'chat_prompt': text, 'prefill_s': round(seconds, 2)}, flush=True)