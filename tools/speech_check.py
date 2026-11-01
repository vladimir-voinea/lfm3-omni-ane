# The Core ML speech programs against the transcriptions: depth argmax agreement, codec waveform.
import sys
import time

import numpy as np
import torch

import build_speech as B
import codec_ref as C
import depth_ref as D
import lfm3_speech

speech = lfm3_speech.Speech(sys.argv[1])
depth, codec = B.load_depth(), B.load_codec()
torch.manual_seed(0)
hidden = torch.from_numpy(np.load(sys.argv[2])).float() if len(sys.argv) > 2 else torch.randn(2048) * 0.02
# a plausible frame: reveal codes in the model's order
codes = torch.zeros(1, 12, dtype=torch.long); agree = []; start = 0; cos = []
revealed = torch.zeros(1, 12, dtype=torch.bool)
for width in (4, 4, 4):
    want = D.forward(hidden.view(1, -1), codes, ~revealed, depth)
    t = time.perf_counter(); got = speech.depth_logits(hidden, codes, ~revealed); ms = 1000 * (time.perf_counter() - t)
    agree.append(float((got[0].argmax(-1) == want[0].argmax(-1)).float().mean()))
    cos.append(float(torch.nn.functional.cosine_similarity(got[0], want[0], dim=-1).mean()))
    pick = range(start, start + width)
    for i in pick:
        codes[0, i] = want[0, i].argmax()
    revealed[0, list(pick)] = True; start += width
print({'depth_argmax_agree': agree, 'cos': [round(c, 4) for c in cos], 'ms': round(ms, 1)}, flush=True)
frames = torch.randint(0, 2016, (48, 12))
want = C.decode(frames.T[None], codec, B.CODEC_LAYERS)[0].clamp(-1, 1).numpy()
t = time.perf_counter(); got = speech.waveform(frames); ms = 1000 * (time.perf_counter() - t)
print({'codec_rel_pct': round(100 * float(np.linalg.norm(got - want) / np.linalg.norm(want)), 2),
       'corr': round(float(np.corrcoef(got, want)[0, 1]), 5), 'ms': round(ms), 'seconds': round(len(got) / 24000, 2)}, flush=True)
