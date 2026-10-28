import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
print('torch', torch.__version__, 'transformers', '5.21', flush=True)
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
image = Image.open('/tmp/beach.jpg').convert('RGB')
inputs = processor(images=image, text='<|vision_start|><|image|><|vision_end|>Describe.',
                   return_tensors='pt')
print({k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}, flush=True)
np.savez('/tmp/official_inputs.npz',
         pixel_values=inputs['pixel_values'].float().numpy(),
         grid=inputs['image_grid_thw'].numpy(),
         input_ids=inputs['input_ids'].numpy())
print('grid', inputs['image_grid_thw'].tolist(), flush=True)