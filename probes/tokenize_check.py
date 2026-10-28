import numpy as np
from tokenizers import Tokenizer

tok = Tokenizer.from_file('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/tokenizer.json')
head = '<|im_start|>user\nSay hello in a friendly voice.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>'
mine = tok.encode(head).ids
official = np.load('/tmp/prompt_ids.npy')[0].tolist()
print({'mine_len': len(mine), 'official_len': len(official)}, flush=True)
print({'mine_first': mine[:8], 'official_first': official[:8]}, flush=True)
print({'mine_last': mine[-4:], 'official_last': official[-4:]}, flush=True)
print({'same': mine == official}, flush=True)