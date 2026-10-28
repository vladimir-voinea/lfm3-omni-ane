# Teacher forcing: feed the OFFICIAL codes as the next input embedding and compare my ANE
# hidden at each frame to the official one.  Isolates per-step error from feedback drift.
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_runtime import Runtime

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
official = np.load('/tmp/official_hidden_traj.npy')
codes = torch.from_numpy(np.load('/tmp/official_codes_traj.npy')).long()   # (12, frames)
with safe_open(P, framework='pt') as f:
    book = f.get_tensor('model.codebook_embedding.weight').float()
offsets = torch.arange(12) * 2017

runtime = Runtime(os.environ.get('LFM3_ANE', '/Users/vladimir/dev/lfm3-ane/model-28k-lut6'))
session = runtime.new_session()
head = '<|im_start|>user\nSay hello in a friendly voice.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>'
from tokenizers import Tokenizer
tok = Tokenizer.from_file('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/tokenizer.json')
ids = [1] + tok.encode(head).ids
runtime.step(session, ids)

cos = []
for frame in range(min(12, codes.shape[1])):
    normed = runtime.norm.predict({'in0': session.hidden})
    hidden = torch.from_numpy(np.asarray(list(normed.values())[0], np.float16).reshape(-1).astype(np.float32))
    ref = torch.from_numpy(official[frame]).float()
    cos.append(round(float(F.cosine_similarity(hidden, ref, dim=0)), 3))
    embed = book[codes[:, frame] + offsets].sum(0).numpy()
    runtime.step_hidden(session, embed)
print({'teacher_forced_cos': cos}, flush=True)