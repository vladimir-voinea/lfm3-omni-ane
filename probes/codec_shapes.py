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

inner = model.decoder.lfm
if not hasattr(inner, 'layers'):
    inner = inner.lfm
print({'decoder_lfm_type': type(model.decoder.lfm).__name__, 'n_layers': len(inner.layers),
       'layer_types': [('conv' if hasattr(l, 'conv') else 'attn') for l in inner.layers]},
      flush=True)

shapes = {}

def hook(name):
    def inner(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(value, torch.Tensor):
            shapes[name] = list(value.shape)
    return inner

model.from_quantizer.register_forward_hook(hook('from_quantizer'))
model.decoder.lfm.register_forward_hook(hook('lfm'))
for index, layer in enumerate(inner.layers):
    if index in (0, 1, 2):
        layer.register_forward_hook(hook('layer%d' % index))

torch.manual_seed(0)
codes = torch.randint(0, 2017, (1, 12, 16))
with torch.no_grad():
    out = model.decode(codes)
print(shapes, flush=True)
print({'quantizer_out': list(model.quantizer.decode(codes).shape)}, flush=True)