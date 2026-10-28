from safetensors import safe_open
import collections

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
with safe_open(P, framework='pt') as f:
    keys = [k for k in f.keys() if 'vision_tower' in k or 'multi_modal_projector' in k]
    shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}
print(len(keys), 'vision/projector tensors')
for k in keys[:8] + keys[-10:]:
    print('  ', k, shapes[k])
heads = collections.Counter('.'.join(k.split('.')[:3]) for k in keys)
for k, v in heads.most_common(6):
    print(k, v)