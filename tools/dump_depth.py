# Dump the depth decoder's internals: trunk input, each layer's output, and the trunk's
# final hidden state, to find where the transcription diverges.
import numpy as np
import torch
from transformers import AutoModel

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
data = np.load('/tmp/depth_input.npz')
hidden = torch.from_numpy(data['hidden']).float()
codes = torch.from_numpy(data['codes']).long()
masked = torch.from_numpy(data['masked'])

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
model.eval()
dd = model.depth_decoder
dumps = {}

def hook(name):
    def inner(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(value, torch.Tensor):
            dumps[name] = value.detach().float().numpy()
    return inner

handles = [dd.model.layers[i].register_forward_hook(hook('layer%d' % i)) for i in range(3)]
handles.append(dd.model.layers[0].self_attn.register_forward_hook(hook('attn0')))
with torch.no_grad():
    logits = dd(hidden, codes, masked)
for handle in handles:
    handle.remove()

np.savez('/tmp/depth_layers.npz', **dumps)
for name, value in dumps.items():
    print(name, list(value.shape), round(float(np.abs(value).mean()), 4), flush=True)