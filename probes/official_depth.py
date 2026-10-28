# Official depth-decoder logits for a fixed input, so the transcription can be diffed.
import numpy as np
import torch
from transformers import AutoModel

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
torch.manual_seed(0)
frames, cbs = 3, 12
hidden = torch.randn(frames, 2048) * 0.5
codes = torch.randint(0, 2017, (frames, cbs))
masked = torch.zeros(frames, cbs, dtype=torch.bool)
masked[:, 0::2] = True
np.savez('/tmp/depth_input.npz', hidden=hidden.numpy(), codes=codes.numpy(),
         masked=masked.numpy())

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
model.eval()
with torch.no_grad():
    logits = model.depth_decoder(hidden, codes, masked)
print({'logits': list(logits.shape)}, flush=True)
np.save('/tmp/official_depth_logits.npy', logits.float().numpy())
print('saved', flush=True)