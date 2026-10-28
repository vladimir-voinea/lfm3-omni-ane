import os

from paths import ANE, BUILD, MODEL, PROJECT
# A real photograph, so the question has a right answer the model can know.
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from safetensors import safe_open

sys.path.insert(0, PROJECT)

from lfm3_runtime import Runtime
from vision_ref import Vision, projector, P

ANE = BUILD
IMG_START, IMG_END, PLACEHOLDER = 125009, 125010, 125030
EOS = 124900


def patchify(img, ps=16, merge=2, temporal=2):
    c, hgt, wid = img.shape
    gh, gw = hgt // ps, wid // ps
    patches = img.reshape(c, gh // merge, merge, ps, gw // merge, merge, ps)
    patches = patches.permute(1, 4, 2, 5, 0, 3, 6)
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
    return patches.reshape(gh * gw, c * temporal * ps * ps), gh, gw


def ask(w, layer_types, tok, question, path, size=256):
    arr = torch.from_numpy(np.load(path)).float().permute(2, 0, 1) / 255.0
    pixels, gh, gw = patchify(arr)
    pixels = ((pixels - 0.5) / 0.5).float()
    tokens = projector(Vision(w, layer_types).forward(pixels, gh, gw), w)
    head = ('<|im_start|>user\n<|vision_start|><|image|><|vision_end|>%s<|im_end|>\n'
            '<|im_start|>assistant\n') % question
    ids = [1] + tok.encode(head).ids
    runtime = Runtime(ANE)
    session = runtime.new_session()
    at = ids.index(PLACEHOLDER)
    runtime.step(session, ids[:at])
    for row in tokens:
        runtime.step_hidden(session, row.numpy())
    logits = runtime.step(session, ids[at + 1:])
    out = []
    for _ in range(20):
        nxt = int(np.argmax(logits))
        if nxt == EOS:
            break
        out.append(nxt)
        logits = runtime.step(session, [nxt])
    return tok.decode(out), list(tokens.shape)


if __name__ == '__main__':
    import json
    w = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if 'vision_tower' in k or 'multi_modal_projector' in k:
                w[k] = f.get_tensor(k).float()
    layer_types = json.load(open(os.path.join(MODEL, 'config.json')))['vision_config']['layer_types']
    tok = Tokenizer.from_file(os.path.join(MODEL, 'tokenizer.json'))
    for path, question in (('/tmp/beach.npy', 'Describe this image in one sentence.'),
                           ('/tmp/tree.npy', 'Is this a photo of a tree or a beach? One word.')):
        answer, shape = ask(w, layer_types, tok, question, path)
        print({'image': path, 'tokens': shape, 'answer': answer}, flush=True)