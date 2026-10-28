import base64
import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_vision_api import rows_from_data_uri
from png_codec import png_encode
from vision_chat import patchify
from vision_ref import P, Vision, projector

weights = {}
with safe_open(P, framework='pt') as handle:
    for key in handle.keys():
        if 'vision_tower' in key or 'multi_modal_projector' in key:
            weights[key] = handle.get_tensor(key).float()
types = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))['vision_config']['layer_types']
tower = Vision(weights, types)

pixels = np.zeros((224, 224, 3), np.uint8)
pixels[:, :] = (255, 0, 0)
uri = 'data:image/png;base64,' + base64.b64encode(png_encode(pixels)).decode()
mine = rows_from_data_uri(uri)

manual = np.zeros((3, 224, 224), np.float32)
manual[0] = 1.0
patches, gh, gw = patchify(torch.from_numpy(manual))
patches = ((patches - 0.5) / 0.5).float()
with torch.no_grad():
    reference = projector(tower.forward(patches, gh, gw), weights)
print({'api_rows': len(mine), 'reference_rows': len(reference)}, flush=True)
stacked = torch.stack([torch.from_numpy(r).float() for r in mine])
print({'cos': round(float(torch.nn.functional.cosine_similarity(
          stacked.reshape(-1), reference.reshape(-1), dim=0)), 4)}, flush=True)