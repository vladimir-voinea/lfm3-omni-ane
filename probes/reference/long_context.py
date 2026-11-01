import json, os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser('~/dev/lfm3-ane-public-src/tools'))
from lfm3_runtime import Runtime
from tokenizers import Tokenizer
M = os.path.expanduser('~/dev/lfm3-ane/model-28k-lut8s')
tok = Tokenizer.from_file(M + '/tokenizer.json')
enc = lambda s: tok.encode(s, add_special_tokens=False).ids
long = enc(open(os.path.expanduser('~/dev/dory-ane/dane6-artifacts/refs-24k/eval-long-text.txt')).read())
head = enc("<|startoftext|><|im_start|>user\nRemember this: the secret code is OMEGA-99. Here is a long document.\n\n")
tail = enc("\n\nEnd of the document. What was the secret code I told you at the very start? Answer with the code only.<|im_end|>\n<|im_start|>assistant\n")
TARGET = 28500
ids = head + (long * (TARGET // len(long) + 1))[:TARGET - len(head) - len(tail)] + tail
json.dump(ids[:4200], open(sys.argv[1] + '/ids_4200.json', 'w'))
print('tokens', len(ids), 'unique source tokens', len(long), flush=True)
r = Runtime(M, mode='28k'); s = r.new_session()
marks = {1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 28000}
window = 40; saved = {}; start = time.time(); recent = []
for i, t in enumerate(ids[:-1]):
    a = time.perf_counter()
    if any(m <= i < m + window for m in marks):
        logits = r.step(s, [t])
        if 4000 <= i < 4000 + window: saved[i] = logits.astype(np.float32)
        recent.append(time.perf_counter() - a)
        if i + 1 in {m + window for m in marks}:
            print({'position': i + 1, 'tok_s_with_head': round(1 / float(np.median(recent)), 1), 'elapsed_s': round(time.time() - start)}, flush=True); recent = []
    else:
        r.step_hidden(s, r.emb[int(t)])
np.savez(sys.argv[1] + '/logits_4000.npz', positions=np.array(sorted(saved)), logits=np.stack([saved[k] for k in sorted(saved)]))
out = []; logits = r.step(s, [ids[-1]]); times = []
for _ in range(16):
    n = int(logits.argmax()); out.append(n)
    if n in (tok.token_to_id('<|im_end|>'),): break
    a = time.perf_counter(); logits = r.step(s, [n]); times.append(time.perf_counter() - a)
print({'answer': tok.decode(out, skip_special_tokens=False), 'decode_tok_s': round(1 / float(np.median(times)), 1) if times else None, 'position': s.position, 'total_s': round(time.time() - start)}, flush=True)
