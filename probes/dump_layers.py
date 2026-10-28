# Dump the official hidden state after the patchifier and after every layer, so the
# first diverging layer can be found by bisection instead of guesswork.
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
print('tower children:', [type(m).__name__ for m in tower.children()], flush=True)
print('layer names:', [n for n, _ in tower.named_children()], flush=True)

dumps = {}

def hook(name):
    def inner(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(value, torch.Tensor):
            dumps[name] = value.detach().float().numpy()
    return inner

handles = []
for index, layer in enumerate(tower.layers):
    handles.append(layer.register_forward_hook(hook('layer%d' % index)))
handles.append(tower.patchifier.register_forward_hook(hook('patchifier')))

with torch.no_grad():
    out = tower(pixels, grid_thw=grid, return_dict=True)
for handle in handles:
    handle.remove()

dumps['final'] = out.last_hidden_state.detach().float().numpy()
np.savez('/tmp/official_layers.npz', **{k: v.astype(np.float32) for k, v in dumps.items()})
print({k: [list(v.shape), round(float(np.abs(v).mean()), 3)] for k, v in
       list(dumps.items())[:3] + [('final', dumps['final'])]}, flush=True)
print('saved', len(dumps), 'tensors', flush=True)