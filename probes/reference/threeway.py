import os, sys, json, numpy as np, torch
T = os.path.expanduser('~/dev/lfm3-ane-public-src/tools'); sys.path.insert(0, T)
os.environ['LFM3_PROJECT'] = os.path.expanduser('~/dev/lfm3-ane')
import build_model as M
from lfm3_runtime import Runtime, H, K_HIST, D
from safetensors import safe_open
o = np.load('official.npz'); tr = o['trace'].astype(np.float32); N = tr.shape[0]
r = Runtime(os.path.expanduser('~/dev/lfm3-ane/model-28k-lut6'), mode='28k')
f = safe_open(M.P, 'pt')
def tensors(L):
    p = 'model.language_model.layers.%d.' % L
    return {k[len(p):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(p)}
def rel(a, b): return round(100 * float(np.linalg.norm(a - b) / np.linalg.norm(b)), 2)
def report(name, built, shipped, off):
    print(name)
    for label, a, b in (('built vs official', built, off), ('shipped vs official', shipped, off), ('built vs shipped', built, shipped)):
        print('  %-20s pos0 %6.2f  pos1 %6.2f  pos20 %6.2f  pos35 %6.2f' % (label, *[rel(a[p], b[p]) for p in (0, 1, 20, 35)]))
for L in (0, 1, 2, 3, 4, 6, 22):
    kind = r.layer_types[L]; t = tensors(L); off = tr[:, L]; inp = tr[:, L - 1] if L else o['emb'].astype(np.float32)
    s = r._plain_session(); bo = []; so = []
    if kind == 'conv':
        m = M.ConvBlock(t).eval(); h = torch.zeros(1, H, 1, K_HIST)
    else:
        rows = M.FULL_ROWS if kind == 'full_attention' else M.SLIDING_ROWS
        m = M.AttnBlock(t, rows).eval()
        mask = torch.full((1, 4, 1, rows), -1e4)
    for p in range(N):
        x16 = inp[p].astype(np.float16).reshape(1, H, 1, 1); x = torch.from_numpy(inp[p].reshape(1, H, 1, 1))
        with torch.no_grad():
            if kind == 'conv':
                y, h = m(x, h); 
                out = r.programs[L].predict({'in0': x16, 'in1': s.hists[L]})
                for v in out.values():
                    a = np.asarray(v)
                    if a.shape == (1, H, 1, 1): sy = a
                    else: s.hists[L] = a
            else:
                c, si = r._rotation(kind, p); c = np.asarray(c, np.float32).reshape(D); si = np.asarray(si, np.float32).reshape(D)
                mask[..., p] = 0
                y = m(x, p, torch.from_numpy(c), torch.from_numpy(si), mask)
                sy = np.asarray(list(r.programs[L].predict({'in0': x16, 'in1': np.array([p], np.int32), 'in2': c.astype(np.float16), 'in3': si.astype(np.float16), 'in4': r.mask(s, kind, p)}, state=s.states[L]).values())[0])
        bo.append(y.numpy().reshape(-1)); so.append(sy.reshape(-1).astype(np.float32))
    report('layer %d %s' % (L, kind), np.stack(bo), np.stack(so), off)
