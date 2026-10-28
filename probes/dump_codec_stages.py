import sys

import numpy as np
import torch
from safetensors.torch import load_file

CODEC = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/audio_tokenizer'
sys.path.insert(0, '/tmp/codecpkg')

from liquid.configuration_liquid_codec import LiquidCodecConfig
from liquid.modeling_liquid_codec import LiquidCodecModel

cfg = LiquidCodecConfig.from_pretrained(CODEC)
model = LiquidCodecModel(cfg)
model.load_state_dict(load_file(CODEC + '/model.safetensors'), strict=False)
model.eval()

codes = torch.from_numpy(np.load('/tmp/codec_codes.npy')).long()
dumps = {}

def hook(name):
    def inner(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(value, torch.Tensor):
            dumps[name] = value.detach().float().clone()
    return inner

model.from_quantizer.register_forward_hook(hook('from_quantizer'))
model.decoder.lfm.register_forward_hook(hook('lfm'))
model.decoder.lin.register_forward_hook(hook('lin'))
with torch.no_grad():
    out = model.decode(codes)
np.savez('/tmp/codec_stages.npz', **{k: v.numpy() for k, v in dumps.items()})
for name, value in dumps.items():
    print(name, list(value.shape), round(float(value.abs().mean()), 4), flush=True)