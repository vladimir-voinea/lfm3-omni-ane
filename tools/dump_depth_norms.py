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
        dumps[name] = (inputs[0].detach().float().numpy(),
                       (output[0] if isinstance(output, (tuple, list)) else output).detach().float().numpy())
    return inner

handles = [dd.model.layers[0].operator_norm.register_forward_hook(hook('opnorm')),
           dd.model.layers[0].self_attn.q_norm.register_forward_hook(hook('qnorm'))]
with torch.no_grad():
    dd(hidden, codes, masked)
for handle in handles:
    handle.remove()
np.savez('/tmp/depth_norms.npz',
         opnorm_in=dumps['opnorm'][0], opnorm_out=dumps['opnorm'][1],
         qnorm_in=dumps['qnorm'][0], qnorm_out=dumps['qnorm'][1])
print({k: list(v[1].shape) for k, v in dumps.items()}, flush=True)
print('saved', flush=True)