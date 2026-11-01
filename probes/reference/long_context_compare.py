import json, sys, numpy as np, torch
from transformers import AutoModel
ids = json.load(open('ids_4200.json'))
a = np.load('logits_4000.npz'); pos = a['positions']; L = a['logits']
m = AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True).eval().language_model
with torch.no_grad():
    h = m(input_ids=torch.tensor([ids[:int(pos[-1]) + 1]]), use_cache=False).last_hidden_state[0]
    off = (h[pos] @ m.embed_tokens.weight.T).float().numpy()
top1 = []; ov = []; kl = []; cos = []
for A, O in zip(L, off):
    pa = np.exp(A - A.max()); pa /= pa.sum(); po = np.exp(O - O.max()); po /= po.sum()
    kl.append(float((po * (np.log(po + 1e-12) - np.log(pa + 1e-12))).sum())); top1.append(A.argmax() == O.argmax())
    ov.append(len(set(np.argsort(-A)[:5]) & set(np.argsort(-O)[:5]))); cos.append(float((A * O).sum() / np.linalg.norm(A) / np.linalg.norm(O)))
print({'positions': [int(pos[0]), int(pos[-1])], 'top1': round(float(np.mean(top1)), 3), 'top5': round(float(np.mean(ov)), 2), 'kl_mean': round(float(np.mean(kl)), 4), 'kl_median': round(float(np.median(kl)), 4), 'cos': round(float(np.mean(cos)), 5)}, flush=True)
