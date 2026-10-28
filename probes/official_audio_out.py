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
print({'prompt_tokens': inputs['input_ids'].shape[-1]}, flush=True)
torch.manual_seed(0)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=24, generation_mode='audio',
                         return_dict_in_generate=True, use_cache=True)
fields = list(out.keys()) if hasattr(out, 'keys') else []
print({'fields': fields}, flush=True)
for field in fields:
    value = getattr(out, field)
    if isinstance(value, torch.Tensor):
        print(field, list(value.shape), flush=True)
codes = getattr(out, 'audio_out_codes', None)
if codes is None:
    codes = getattr(out, 'audio_codes', None)
print({'codes': None if codes is None else list(codes.shape)}, flush=True)
if codes is not None:
    np.save('/tmp/official_audio_codes.npy', codes.cpu().numpy())
    print('first frame', codes[0].tolist() if codes.ndim == 2 else codes[..., 0].tolist(), flush=True)
audio = getattr(out, 'audio_values', None)
if audio is not None:
    print({'audio': list(audio.shape), 'rms': round(float(audio.std()), 4)}, flush=True)