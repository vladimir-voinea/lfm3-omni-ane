import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_runtime import Runtime

ids = np.load('/tmp/prompt_ids.npy')[0].tolist()
ref = torch.from_numpy(np.load('/tmp/official_frame0_hidden.npy')).float().reshape(-1)
runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k')
session = runtime.new_session()
runtime.step(session, ids)
normed = runtime.norm.predict({'in0': session.hidden})
mine_norm = torch.from_numpy(np.asarray(list(normed.values())[0], np.float16).reshape(-1).astype(np.float32))
mine_raw = torch.from_numpy(session.hidden.reshape(-1).astype(np.float32))
print({'prompt_tokens': len(ids), 'cos_normed': round(float(F.cosine_similarity(mine_norm, ref, dim=0)), 4),
       'cos_raw': round(float(F.cosine_similarity(mine_raw, ref, dim=0)), 4),
       'norm_mine': round(float(mine_norm.norm()), 2), 'norm_ref': round(float(ref.norm()), 2)}, flush=True)