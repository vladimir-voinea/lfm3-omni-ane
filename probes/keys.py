from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
with safe_open(P, framework='pt') as f:
    keys = list(f.keys())
    shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}

print('--- layer 0:')
for k in keys:
    if 'language_model.layers.0.' in k:
        print('  ', k.split('layers.0.')[1], shapes[k])
print('--- layer 2 (full attention):')
for k in keys:
    if 'language_model.layers.2.' in k:
        print('  ', k.split('layers.2.')[1], shapes[k])
print('--- outside the stack:')
for k in keys:
    if 'language_model.layers.' not in k and k.startswith('model.language_model'):
        print('  ', k, shapes[k])