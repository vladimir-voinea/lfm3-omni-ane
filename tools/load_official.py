# transformers 5 requires torch >= 2.6; this environment pins 2.5.0 for coremltools.
# Patch the availability check and see whether the model loads anyway -- cheaper and
# more honest than downloading a second torch if the code does not actually need 2.6.
import sys

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/pylibs')

import torch  # noqa: E402

import transformers  # noqa: E402
import transformers.utils.import_utils as iu  # noqa: E402

iu.is_torch_available = lambda: True
transformers.utils.is_torch_available = lambda: True
for name in ('transformers.utils.import_utils', 'transformers.utils', 'transformers'):
    module = sys.modules.get(name)
    if module is not None:
        setattr(module, 'is_torch_available', lambda: True)

from transformers import AutoModel  # noqa: E402

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
print({'loading': MODEL}, flush=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
model.eval()
print({'loaded': type(model).__name__}, flush=True)
vision = getattr(model, 'vision_tower', None)
print({'vision_tower': type(vision).__name__ if vision is not None else None}, flush=True)