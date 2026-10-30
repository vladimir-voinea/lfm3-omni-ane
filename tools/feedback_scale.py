import numpy as np
import torch
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
with safe_open(P, framework='pt') as f:
    book = f.get_tensor('model.codebook_embedding.weight').float()
print({'table': list(book.shape)}, flush=True)

codes = torch.from_numpy(np.load('/tmp/official_codes_traj.npy')).long()   # (12, frames)
offsets = torch.arange(12) * 2017
frame0 = codes[:, 0]
summed = book[frame0 + offsets].sum(0)
print({'frame0_embed_norm': round(float(summed.norm()), 3),
       'per_codebook_norm': round(float(book[frame0 + offsets].norm(dim=-1).mean()), 3)}, flush=True)

emb = np.load('/Users/vladimir/dev/lfm3-ane/model-28k/embed_tokens.npy', mmap_mode='r')
for token in (1, 1000, 5000):
    print({'token': token, 'norm': round(float(np.linalg.norm(np.asarray(emb[token], np.float32))), 3)},
          flush=True)