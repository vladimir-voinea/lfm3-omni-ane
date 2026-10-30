# What does layer 0 actually do?  Capture its input, its conv output and its output.
import numpy as np
import torch
from transformers import AutoModel

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
data = np.load('/tmp/official_inputs.npz')
pixels = torch.from_numpy(data['pixel_values']).float()
grid = torch.from_numpy(data['grid'])

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
model.eval()
tower = model.vision_tower
layer0 = tower.layers[0]
print('layer0 children:', [n for n, _ in layer0.named_children()], flush=True)

stats = {}

def stat(name):
    def inner(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(value, torch.Tensor):
            stats[name] = (list(value.shape), round(float(value.abs().mean()), 5),
                           round(float(value.norm(dim=-1).mean()), 3))
        else:
            stats[name] = ('nontensor', type(value).__name__)
    return inner

handles = [layer0.register_forward_hook(stat('layer0_out')),
           layer0.conv.register_forward_hook(stat('conv_out')),
           layer0.feed_forward.register_forward_hook(stat('ffn_out')),
           layer0.operator_norm.register_forward_hook(stat('opnorm_out')),
           layer0.ffn_norm.register_forward_hook(stat('ffnnorm_out'))]
with torch.no_grad():
    out = tower(pixels, grid_thw=grid, return_dict=True)
for handle in handles:
    handle.remove()
for name, value in stats.items():
    print(name, value, flush=True)
print('patchifier out absmean', flush=True)