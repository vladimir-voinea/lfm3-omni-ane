import json, os, sys, numpy as np
sys.path.insert(0, os.path.expanduser('~/dev/lfm3-ane-public-src/tools'))
os.environ.setdefault('LFM3_PROJECT', os.path.expanduser('~/dev/lfm3-ane'))
from lfm3_runtime import Runtime
from tokenizers import Tokenizer
M = os.path.expanduser('~/dev/lfm3-ane/model-28k-lut6')
tok = Tokenizer.from_file(M + '/tokenizer.json')
text = ("<|startoftext|><|im_start|>user\nThe Danube flows through ten countries before reaching the Black Sea. "
        "Name three of them, and explain briefly why rivers matter for trade.<|im_end|>\n<|im_start|>assistant\n")
ids = tok.encode(text, add_special_tokens=False).ids
print(len(ids), ids[:6], flush=True)
json.dump(ids, open(sys.argv[1] + '/ids.json', 'w'))
if len(sys.argv) > 2 and sys.argv[2] == 'idsonly':
    sys.exit(0)
r = Runtime(M, mode='28k')
s = r.new_session()
L, T = [], []
for t in ids:
    r.trace = []
    L.append(r.step(s, [t]).astype(np.float32))
    T.append(np.stack([x.reshape(-1).astype(np.float32) for x in r.trace]))
np.savez(sys.argv[1] + '/ane.npz', logits=np.stack(L), trace=np.stack(T), emb=np.asarray(r.emb[np.array(ids)]).astype(np.float32))
print('done', np.stack(L).shape, np.stack(T).shape)
