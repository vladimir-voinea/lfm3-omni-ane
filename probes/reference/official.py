import json, sys, numpy as np, torch
from transformers import AutoModel
SRC = sys.argv[1]; OUT = sys.argv[2]
ids = json.load(open(sys.argv[3]))
m = AutoModel.from_pretrained(SRC, trust_remote_code=True, dtype=torch.float32, low_cpu_mem_usage=True).eval()
lm = m.language_model
print(type(lm).__name__, [n for n,_ in lm.named_children()], flush=True)
caught = []
hooks = [l.register_forward_hook(lambda mod, a, o: caught.append((o[0] if isinstance(o, tuple) else o)[0].detach().float().numpy().copy())) for l in lm.layers]
with torch.no_grad():
    o = lm(input_ids=torch.tensor([ids]), use_cache=False)
h = o.last_hidden_state[0]
emb = lm.embed_tokens.weight
logits = (h @ emb.T).detach(); emb = emb.detach(); h = h.detach()
print('logits', logits.shape, 'layers', len(caught), caught[0].shape, 'emb std', float(emb.std()), flush=True)
np.savez(OUT, logits=logits.float().numpy(), last=h.float().numpy(), trace=np.stack(caught).transpose(1,0,2), emb=emb[torch.tensor(ids)].float().numpy())
