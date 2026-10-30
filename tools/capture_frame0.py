# Capture the official frame-0 hidden state and its codes with a FIXED reveal order,
# so my depth decoder can be driven with exactly the same inputs.
import numpy as np
import torch
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
real_randperm = torch.randperm
torch.randperm = lambda n, **kw: torch.arange(n)

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
cls = get_class_from_dynamic_module('modeling_lfm3_omni.Lfm3OmniForConditionalGeneration', MODEL)
model = cls.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                            low_cpu_mem_usage=True).eval()

text = '<|im_start|>user\nSay hello in a friendly voice.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>'
inputs = processor(text=text, return_tensors='pt')
np.save('/tmp/prompt_ids.npy', inputs['input_ids'].numpy())

captured = {}

def hook(module, args, kwargs):
    hidden = kwargs.get('hidden_states', args[0] if args else None)
    if hidden is not None and 'hidden' not in captured:
        captured['hidden'] = hidden.detach().float().clone()

model.model.depth_decoder.register_forward_pre_hook(hook, with_kwargs=True)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=4, generation_mode='audio',
                         return_dict_in_generate=True, use_cache=True)
torch.randperm = real_randperm
codes = out.audio_codes
print({'codes': list(codes.shape), 'first': codes[0, :, 0].tolist()}, flush=True)
print({'hidden': list(captured['hidden'].shape)}, flush=True)
np.save('/tmp/official_frame0_hidden.npy', captured['hidden'].numpy())
np.save('/tmp/official_frame0_codes.npy', codes[0, :, 0].numpy())
print('saved', flush=True)