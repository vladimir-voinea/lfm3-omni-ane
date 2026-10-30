import numpy as np
import torch
from transformers import AutoModel

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
data = np.load('/tmp/depth_input.npz')
hidden = torch.from_numpy(data['hidden']).float()
codes = torch.from_numpy(data['codes']).long()
masked = torch.from_numpy(data['masked'])
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True).eval()
dd = model.depth_decoder
dumps = {}

def hook(name):
    def inner(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(value, torch.Tensor):
            dumps[name] = value.detach().float().numpy()
    return inner

handles = [dd.depth_linear.register_forward_hook(hook('depth_linear')),
           dd.model.layers[0].register_forward_hook(hook('layer0_in'))]
handles += [emb.register_forward_hook(hook('emb%d' % i))
            for i, emb in enumerate(dd.depth_embeddings)]
with torch.no_grad():
    dd(hidden, codes, masked)
for handle in handles:
    handle.remove()
np.savez('/tmp/depth_parts.npz', **{k: v for k, v in dumps.items()})
for name in ('depth_linear', 'emb0', 'emb5'):
    print(name, list(dumps[name].shape), round(float(np.abs(dumps[name]).mean()), 5), flush=True)