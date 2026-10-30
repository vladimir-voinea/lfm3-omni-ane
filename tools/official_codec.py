import sys

import numpy as np
import torch
from safetensors.torch import load_file

PARENT = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
CODEC = PARENT + '/audio_tokenizer'
sys.path.insert(0, '/tmp/codecpkg')

from liquid.configuration_liquid_codec import LiquidCodecConfig
from liquid.modeling_liquid_codec import LiquidCodecModel

cfg = LiquidCodecConfig.from_pretrained(CODEC)
model = LiquidCodecModel(cfg)
state = load_file(CODEC + '/model.safetensors')
missing, unexpected = model.load_state_dict(state, strict=False)
print('missing', len(missing), 'unexpected', len(unexpected), flush=True)
model.eval()

torch.manual_seed(0)
codes = torch.randint(0, 2017, (1, 12, 16))   # (batch, codebooks, frames)
trace = {}

def hook(name):
    def inner(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(value, torch.Tensor):
            trace[name] = list(value.shape)
    return inner

for name in ('quantizer',):
    getattr(model, name).register_forward_hook(hook(name))
model.decoder.lfm.register_forward_hook(hook('lfm'))
model.decoder.lin.register_forward_hook(hook('lin'))
model.decoder.istft.register_forward_hook(hook('istft'))
with torch.no_grad():
    out = model.decode(codes)
print({'output_fields': [f for f in out.keys()] if hasattr(out, 'keys') else type(out).__name__}, flush=True)
audio = None
for field in ('audio_values', 'audio', 'waveform', 'samples', 'audios'):
    if hasattr(out, field):
        audio = getattr(out, field)
        break
print({'audio': list(audio.shape), 'absmean': round(float(audio.abs().mean()), 5)}, flush=True)
print(trace, flush=True)
flat = audio.reshape(-1, audio.shape[-1])[0].detach().float().numpy()
np.save('/tmp/official_codec_audio.npy', flat)
np.save('/tmp/codec_codes.npy', codes.numpy())
print('saved', flush=True)