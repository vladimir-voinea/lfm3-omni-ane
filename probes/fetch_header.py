import json, struct, subprocess, os

# The safetensors header is at the front: 8-byte little-endian length, then JSON.
url = 'https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/main/model.safetensors'
head = subprocess.run(['curl', '-sL', '-r', '0-1000000', url], capture_output=True).stdout
length = struct.unpack('<Q', head[:8])[0]
print({'header_bytes': length})
header = json.loads(head[8:8 + length].decode('utf-8'))
tensors = {k: v for k, v in header.items() if k != '__metadata__'}
print({'tensors_in_file': len(tensors)})

# Everything the 6 full-attention layers need, and nothing else.
import re
needed = {}
for key, info in tensors.items():
    match = re.match(r'model\.language_model\.layers\.(\d+)\.', key)
    if match and ('self_attn' in key or 'feed_forward' in key or 'norm' in key):
        needed[key] = info
bytes_needed = sum(i['data_offsets'][1] - i['data_offsets'][0] for i in needed.values())
print({'layer_tensors': len(needed), 'GB_if_all_layers': round(bytes_needed / 1e9, 2)})
json.dump({k: v for k, v in needed.items()}, open('/tmp/needed_tensors.json', 'w'))