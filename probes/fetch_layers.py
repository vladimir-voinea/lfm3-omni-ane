import json, os, subprocess

URL = 'https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/main/model.safetensors'
HEADER = 84864
DATA_START = 8 + HEADER
TARGET = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/full_layer_weights.bin'
needed = json.load(open('/tmp/needed_tensors.json'))

# only the six full-attention layers
import sys
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from paths import MODEL
types = json.load(open(os.path.join(MODEL, 'layer_types.json')))
full_layers = [i for i, kind in enumerate(types) if kind == 'full_attention']
print({'full_layers': full_layers}, flush=True)

keep = {}
for key, info in needed.items():
    index = int(key.split('layers.')[1].split('.')[0])
    if index in full_layers:
        keep[key] = info
total = sum(i['data_offsets'][1] - i['data_offsets'][0] for i in keep.values())
print({'tensors': len(keep), 'GB': round(total / 1e9, 2)}, flush=True)

# one range request for the whole span, then carve the tensors out locally
spans = sorted((i['data_offsets'][0], i['data_offsets'][1]) for i in keep.values())
start, end = spans[0][0], spans[-1][1]
print({'span_GB': round((end - start) / 1e9, 2)}, flush=True)
with open(TARGET, 'wb') as out:
    done = 0
    chunk = 64 * 1024 * 1024
    while start + done < end:
        left = end - (start + done)
        size = min(chunk, left)
        request = subprocess.run(['curl', '-sL', '-r',
                                  '%d-%d' % (start + done, start + done + size - 1), URL],
                                 capture_output=True)
        out.write(request.stdout)
        done += size
        print({'MB_done': int(done / 1e6)}, flush=True)
print({'written_GB': round(os.path.getsize(TARGET) / 1e9, 2)}, flush=True)
json.dump({k: v for k, v in keep.items()}, open('/tmp/full_layer_index.json', 'w'))