# Reference vision features from the official tower, on the official processor output.
import numpy as np
import torch
from transformers import AutoModel

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
data = np.load('/tmp/official_inputs.npz')
pixels = torch.from_numpy(data['pixel_values']).float()
grid = torch.from_numpy(data['grid'])

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
model.eval()
with torch.no_grad():
    vision = model.vision_tower(pixels, grid_thw=grid, return_dict=True)
    feats = vision.last_hidden_state
    proj = model.multi_modal_projector(feats)
print({'features': list(feats.shape), 'projected': list(proj.shape),
       'feature_norm_per_token': round(float(feats.norm(dim=-1).mean()), 3)}, flush=True)
np.save('/tmp/official_features.npy', feats.float().numpy())
np.save('/tmp/official_projected.npy', proj.float().numpy())
print('saved', flush=True)