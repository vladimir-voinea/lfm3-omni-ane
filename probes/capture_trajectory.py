# Every hidden state the official model feeds its depth decoder, frame by frame, so the
# two trajectories can be compared instead of their endpoints.
import numpy as np
import torch
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
cls = get_class_from_dynamic_module('modeling_lfm3_omni.Lfm3OmniForConditionalGeneration', MODEL)
model = cls.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                            low_cpu_mem_usage=True).eval()

text = '<|im_start|>user\nSay hello in a friendly voice.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>'
inputs = processor(text=text, return_tensors='pt')
captured = []


def hook(module, args, kwargs):
    hidden = kwargs.get('hidden_states', args[0] if args else None)
    if hidden is not None:
        captured.append(hidden.detach().float().clone().reshape(-1))


model.model.depth_decoder.register_forward_pre_hook(hook, with_kwargs=True)
torch.manual_seed(0)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=24, generation_mode='audio',
                         return_dict_in_generate=True, use_cache=True)
print({'depth_calls': len(captured), 'frames': int(out.audio_codes.shape[-1])}, flush=True)
unique = []
for hidden in captured:
    if not unique or not torch.allclose(hidden, unique[-1]):
        unique.append(hidden)
print({'unique_hiddens': len(unique)}, flush=True)
np.save('/tmp/official_hidden_traj.npy', torch.stack(unique).numpy())
np.save('/tmp/official_codes_traj.npy', out.audio_codes[0].numpy())
print('saved', flush=True)