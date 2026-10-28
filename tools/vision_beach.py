# End to end with the corrected tower, using the official processor's own patches.
import json
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from lfm3_runtime import Runtime
from vision_ref import Vision, projector, P

ANE = '/Users/vladimir/dev/lfm3-ane/model-28k'
EOS = 124900

w = {}
with safe_open(P, framework='pt') as f:
    for k in f.keys():
        if 'vision_tower' in k or 'multi_modal_projector' in k:
            w[k] = f.get_tensor(k).float()
types = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))['vision_config']['layer_types']

data = np.load('/tmp/official_inputs.npz')
pixels = torch.from_numpy(data['pixel_values']).float()
_, h, wd = data['grid'][0].tolist()
tokens = projector(Vision(w, types).forward(pixels, h, wd), w).detach().numpy()
print({'image_tokens': list(tokens.shape)}, flush=True)

# free the torch side before the engine allocates its state, or the ANE programs fail
# to load under memory pressure
del w
import gc
gc.collect()

tok = Tokenizer.from_file('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/tokenizer.json')
runtime = Runtime(ANE)

for question in ('Describe this image in one sentence.', 'Is this a beach or a forest? One word.'):
    head = ('<|im_start|>user\n<|vision_start|><|image|><|vision_end|>%s<|im_end|>\n'
            '<|im_start|>assistant\n') % question
    ids = [1] + tok.encode(head).ids
    session = runtime.new_session()
    at = ids.index(125030)
    runtime.step(session, ids[:at])
    for row in tokens:
        runtime.step_hidden(session, row)
    logits = runtime.step(session, ids[at + 1:])
    out = []
    for _ in range(24):
        nxt = int(np.argmax(logits))
        if nxt == EOS:
            break
        out.append(nxt)
        logits = runtime.step(session, [nxt])
    print({'question': question, 'answer': tok.decode(out)}, flush=True)