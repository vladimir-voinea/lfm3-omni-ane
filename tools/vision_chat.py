import os

from paths import ANE, MODEL, PROJECT
# Image in: build a solid-colour picture, run it through the vision tower and the
# projector, feed the resulting vectors into the text model where the <|image|>
# placeholder sits, and ask the model what it sees. A right colour is the behavioural
# proof that the whole vision path -- patchify order, axial RoPE, merge, projector --
# is wired the way the model expects.
import sys

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, PROJECT)

from lfm3_runtime import Runtime
from vision_ref import Vision, projector, P

from safetensors import safe_open

ANE = MODEL
IDS = {'im_start': 124899, 'im_end': 124900, 'vision_start': 125009,
       'vision_end': 125010, 'image': 125030}


def make_image(size=224, colour=(1.0, 0.0, 0.0)):
    img = torch.zeros(3, size, size)
    for c, value in enumerate(colour):
        img[c] = value
    return img                      # (3, H, W) in 0..1


def patchify(img, ps=16, merge=2, temporal=2):
    c, hgt, wid = img.shape
    gh, gw = hgt // ps, wid // ps
    patches = img.reshape(c, gh // merge, merge, ps, gw // merge, merge, ps)
    patches = patches.permute(1, 4, 2, 5, 0, 3, 6)          # gh/2, gw/2, m, m, c, ps, ps
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
    return patches.reshape(gh * gw, c * temporal * ps * ps), gh, gw


def main():
    w = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if 'vision_tower' in k or 'multi_modal_projector' in k:
                w[k] = f.get_tensor(k).float()
    import json
    layer_types = json.load(open(os.path.join(MODEL, 'config.json')))['vision_config']['layer_types']

    tok = Tokenizer.from_file(os.path.join(MODEL, 'tokenizer.json'))
    question = 'What colour fills this image? Answer with one word.'
    head = ('<|im_start|>user\n<|vision_start|><|image|><|vision_end|>%s<|im_end|>\n'
            '<|im_start|>assistant\n') % question
    ids = [1] + tok.encode(head).ids
    runtime = Runtime(ANE)

    # the same question with three different pictures: if the answer never moves, the
    # image is not reaching the text model at all
    for name, colour in (('red', (1.0, 0.0, 0.0)), ('green', (0.0, 1.0, 0.0)),
                         ('blue', (0.0, 0.0, 1.0))):
        pixels, gh, gw = patchify(make_image(colour=colour))
        pixels = ((pixels - 0.5) / 0.5).float()
        tokens = projector(Vision(w, layer_types).forward(pixels, gh, gw), w)
        session = runtime.new_session()
        image_at = ids.index(IDS['image'])
        runtime.step(session, ids[:image_at])
        for row in tokens:
            runtime.step_hidden(session, row.numpy())
        logits = runtime.step(session, ids[image_at + 1:])
        out = []
        for _ in range(6):
            nxt = int(np.argmax(logits))
            if nxt == IDS['im_end']:
                break
            out.append(nxt)
            logits = runtime.step(session, [nxt])
        print({name: tok.decode(out)}, flush=True)


if __name__ == '__main__':
    main()