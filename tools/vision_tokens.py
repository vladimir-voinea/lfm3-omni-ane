import json
from tokenizers import Tokenizer
cfg = json.load(open('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/config.json'))
print({k: cfg[k] for k in ('image_token_id', 'video_token_id', 'audio_token_id') if k in cfg})
tok = Tokenizer.from_file('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/tokenizer.json')
for marker in ('<|vision_start|>', '<|image|>', '<|vision_end|>', '<|audio|>', '<|im_start|>'):
    ids = tok.encode(marker).ids
    print(marker, '->', ids)