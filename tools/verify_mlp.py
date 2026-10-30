import json, subprocess, sys
import torch
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_tiled as B
URL = 'https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/main/model.safetensors'
index = json.load(open('/tmp/full_layer_index.json'))
start = min(i['data_offsets'][0] for i in index.values())
blob = open(B.BIN, 'rb').read()
for key in ('model.language_model.layers.2.feed_forward.w1.weight',
            'model.language_model.layers财务2.feed_forward.w2.weight'):
    pass
targets = ['model.language_model.layers.2.feed_forward.w1.weight',
           'model.language_model.layers.2.self_attn.o_proj.weight',
           'model.language_model.layers.2.operator_norm.weight']
for key in targets:
    info = index[key]
    begin, end = info['data_offsets']
    fresh = subprocess.run(['curl', '-sL', '-r', '%d-%d' % (begin, end - 1), URL],
                           capture_output=True).stdout
    width = end - begin
    fresh_values = torch.frombuffer(bytearray(fresh), dtype=torch.bfloat16).float()
    mine = B.load_tensor(blob, start, info).reshape(-1)
    same_size = fresh_values.numel() == mine.numel()
    diff = float((fresh_values[:mine.numel()] - mine).abs().max()) if same_size else -1
    print({'tensor': key.split('layers.2.')[1], 'bytes': width, 'fresh_std': round(float(fresh_values.std()), 5),
           'mine_std': round(float(mine.std()), 5), 'max_abs_diff': round(diff, 8)}, flush=True)