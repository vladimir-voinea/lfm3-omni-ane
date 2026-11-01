import os, sys, time, numpy as np, torch, coremltools as ct, coremltools.optimize.coreml as cto
os.environ['TMPDIR'] = os.path.abspath('tmp')
sys.path.insert(0, os.path.expanduser('~/dev/lfm3-ane-public-src/tools'))
os.environ['LFM3_PROJECT'] = os.path.expanduser('~/dev/lfm3-ane')
import build_model as M
from lfm3_runtime import H, K_HIST
from safetensors import safe_open
o = np.load('official.npz'); tr = o['trace'].astype(np.float32); emb = o['emb'].astype(np.float32); N = tr.shape[0]
f = safe_open(M.P, 'pt'); p = 'model.language_model.layers.0.'
t = {k[len(p):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(p)}
base = 'var/l0_fp16.mlpackage'
if not os.path.exists(base):
    M.save(M.ConvBlock(t), [(1, H, 1, 1), (1, H, 1, M.K - 1)], [], base)
paths = {'fp16': base}
for bits in (8, 6):
    q = 'var/l0_lut%d.mlpackage' % bits
    if not os.path.exists(q):
        cto.palettize_weights(ct.models.MLModel(base, compute_units=ct.ComputeUnit.CPU_ONLY), cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode='kmeans', nbits=bits))).save(q)
    paths['lut%d' % bits] = q
paths['shipped lut6'] = os.path.expanduser('~/dev/lfm3-ane/model-28k-lut6/layer0_conv.mlpackage')
def run(path, cu, inp):
    m = ct.models.MLModel(path, compute_units=cu); hist = np.zeros((1, H, 1, K_HIST), np.float16); out = []; ts = []
    for p in range(len(inp)):
        s = time.perf_counter(); r = m.predict({'in0': inp[p].astype(np.float16).reshape(1, H, 1, 1), 'in1': hist}); ts.append(time.perf_counter() - s)
        for v in r.values():
            a = np.asarray(v)
            if a.shape == (1, H, 1, 1): out.append(a.reshape(-1).astype(np.float32))
            else: hist = a
    return np.stack(out), 1000 * float(np.median(ts))
for name, q in paths.items():
    for cu, l in ((ct.ComputeUnit.CPU_AND_NE, 'NE'), (ct.ComputeUnit.CPU_ONLY, 'CPU')):
        y, ms = run(q, cu, emb)
        e = 100 * np.linalg.norm(y - tr[:, 0], axis=1) / np.linalg.norm(tr[:, 0], axis=1)
        print('%-13s %-3s rel%% to official: median %.2f max %.2f  %.2f ms' % (name, l, np.median(e), e.max(), ms), flush=True)
# layer 2: does a Gaussian input reproduce the old 37 %?
from lfm3_runtime import Runtime, NKV, D
r = Runtime(os.path.expanduser('~/dev/lfm3-ane/model-28k-lut6'), mode='28k')
p2 = 'model.language_model.layers.2.'; t2 = {k[len(p2):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(p2)}
c, s_ = r._rotation('full_attention', 0); c = np.asarray(c, np.float32).reshape(D); s_ = np.asarray(s_, np.float32).reshape(D)
for std in (0.017, 0.1):
    x = np.random.RandomState(5).randn(1, H, 1, 1).astype(np.float32) * std
    mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0
    m = M.AttnBlock(t2, M.FULL_ROWS).eval()
    with torch.no_grad(): b = m(torch.from_numpy(x), 0, torch.from_numpy(c), torch.from_numpy(s_), torch.from_numpy(mask.astype(np.float32))).numpy().reshape(-1)
    st = r.programs[2].make_state()
    sh = np.asarray(list(r.programs[2].predict({'in0': x.astype(np.float16), 'in1': np.array([0], np.int32), 'in2': c.astype(np.float16), 'in3': s_.astype(np.float16), 'in4': mask}, state=st).values())[0]).reshape(-1).astype(np.float32)
    print('layer 2 Gaussian std %.3f: built vs shipped rel %.2f%%  output std built %.4f shipped %.4f' % (std, 100 * np.linalg.norm(b - sh) / np.linalg.norm(sh), b.std(), sh.std()), flush=True)
