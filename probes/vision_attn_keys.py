from safetensors import safe_open
P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
with safe_open(P, framework='pt') as f:
    keys = [k for k in f.keys() if 'vision_tower.layers.2.' in k]
print('layer 2 (attention) keys:')
for k in keys: print('  ', k)