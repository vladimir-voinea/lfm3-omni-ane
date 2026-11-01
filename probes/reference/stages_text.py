import os, sys, shutil, numpy as np, torch, coremltools as ct
sys.path.insert(0, os.path.expanduser('~/dev/lfm3-ane-public-src/tools'))
from safetensors import safe_open
import build_model as M
from lfm3_runtime import H, K_HIST
o = np.load(S + '/ref/run2/official.npz') if False else np.load('../ref/run2/official.npz'); emb = o['emb'].astype(np.float32)
f = safe_open('layers.safetensors', 'pt'); p = 'model.language_model.layers.0.'
t = {k[len(p):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(p)}
names = ['pre norm', 'in_b', 'in_x', 'gated', 'in_c', 'dw', 'c*dw', 'out_proj', 'post norm', 'h', 'ffn norm', 'w1', 'act', 'w3', 'product', 'w2', 'ffn post', 'final']
class St(torch.nn.Module):
    def __init__(s, b): super().__init__(); s.b = b
    def forward(s, x, hist):
        b = s.b; h = b.pre(x); ib = b.in_b(h); ix = b.in_x(h); g = ib * ix; full = torch.cat([hist, g], dim=3); ic = b.in_c(h); d = b.dw(full); cd = ic * d; y = b.out_proj(cd); po = b.post(y); h2 = po + x
        fn = b.ffn_norm(h2); w1 = b.mlp.w1(fn); a = torch.nn.functional.silu(w1); w3 = b.mlp.w3(fn); pr = a * w3; w2 = b.mlp.w2(pr); fp = b.ffn_post(w2)
        return h, ib, ix, g, ic, d, cd, y, po, h2, fn, w1, a, w3, pr, w2, fp, fp + h2
b = M.ConvBlock(t).eval(); m = St(b).eval()
x = torch.from_numpy(emb[8].reshape(1, H, 1, 1)); hist = torch.zeros(1, H, 1, K_HIST)
with torch.no_grad():
    for i in range(1, 8):
        g = b.in_b(b.pre(torch.from_numpy(emb[i].reshape(1, H, 1, 1)))) * b.in_x(b.pre(torch.from_numpy(emb[i].reshape(1, H, 1, 1))))
        hist = torch.cat([hist, g], 3)[..., 1:]
    want = [v.numpy() for v in m(x, hist)]
c = ct.convert(torch.jit.trace(m, (torch.zeros(*x.shape), torch.zeros(*hist.shape))), inputs=[ct.TensorType(name='x', shape=x.shape, dtype=np.float32), ct.TensorType(name='hist', shape=hist.shape, dtype=np.float32)], outputs=[ct.TensorType(name='s%d' % i) for i in range(len(names))], convert_to='mlprogram', compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.iOS18)
c.save('/tmp/ts.mlpackage'); r = {}
for cu, n in ((ct.ComputeUnit.CPU_AND_NE, 'NE'), (ct.ComputeUnit.CPU_ONLY, 'CPU')):
    r[n] = ct.models.MLModel('/tmp/ts.mlpackage', compute_units=cu).predict({'x': x.numpy(), 'hist': hist.numpy()})
for i, name in enumerate(names):
    print('%-10s NE %6.2f%%  CPU %6.2f%%  max|v| %.3g' % (name, *[100 * np.linalg.norm(np.asarray(r[k]['s%d' % i]).reshape(want[i].shape) - want[i]) / np.linalg.norm(want[i]) for k in ('NE', 'CPU')], np.abs(want[i]).max()), flush=True)
shutil.rmtree('/tmp/ts.mlpackage')
