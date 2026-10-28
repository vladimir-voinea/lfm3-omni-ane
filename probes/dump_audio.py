import numpy as np
import torch
from transformers import AutoModel

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
data = np.load('/tmp/official_audio.npz')
feats = torch.from_numpy(data['input_features']).float()
lengths = torch.tensor([feats.shape[-1]])

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
model.eval()
enc = model.audio_encoder
dumps = {}

def hook(name):
    def inner(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(value, torch.Tensor):
            dumps[name] = value.detach().float().numpy()
    return inner

handles = [enc.downsample.register_forward_hook(hook('downsample'))]
for i, layer in enumerate(enc.lfm.layers):
    handles.append(layer.register_forward_hook(hook('layer%d' % i)))
with torch.no_grad():
    out, _ = enc(feats, lengths)
for handle in handles:
    handle.remove()
dumps['final'] = out.detach().float().numpy()
np.savez('/tmp/official_audio_layers.npz', **dumps)
for name in ('downsample', 'layer0', 'layer1', 'layer2', 'final'):
    print(name, list(dumps[name].shape), round(float(np.abs(dumps[name]).mean()), 4), flush=True)