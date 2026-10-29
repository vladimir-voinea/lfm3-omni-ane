import numpy as np, torch, sys, os, json, coremltools as ct
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
os.environ['LFM3_TILED'] = '1'
import build_tiled as B
import build_model as M
from lfm3_runtime import Runtime, H, NKV, D

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
index = json.load(open('/tmp/full_layer_index.json'))
start = min(i['data_offsets'][0] for i in index.values())
blob = open(B.BIN, 'rb').read()
tensors = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        tensors[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info)
x = (np.random.RandomState(3).randn(1, H, 1, 1) * 0.1).astype(np.float16)
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0
session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(
    {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask},
    state=session.states[2]).values())[0]).reshape(-1)
print({'shipped_scale': round(float(np.abs(shipped).max()), 3)}, flush=True)


def build(norm_one, rope, tag):
    t = dict(tensors)
    if norm_one:
        t['self_attn.q_norm.weight'] = torch.ones(D)
        t['self_attn.k_norm.weight'] = torch.ones(D)
    layer = M.AttnBlock(t, M.FULL_ROWS).eval().half()
    c = torch.from_numpy(cos) if rope else torch.ones(D, dtype=torch.float16)
    s = torch.from_numpy(sin) if rope else torch.zeros(D, dtype=torch.float16)
    pos = torch.zeros(1, dtype=torch.int32)
    traced = torch.jit.trace(layer, (torch.from_numpy(x), pos, c, s, torch.from_numpy(mask)))
    converted = ct.convert(traced,
        inputs=[ct.TensorType(name='in0', shape=(1, H, 1, 1), dtype=np.float16),
                ct.TensorType(name='in1', shape=(1,), dtype=np.int32),
                ct.TensorType(name='in2', shape=(D,), dtype=np.float16),
                ct.TensorType(name='in3', shape=(D,), dtype=np.float16),
                ct.TensorType(name='in4', shape=(1, NKV, 1, M.FULL_ROWS), dtype=np.float16)],
        states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
            shape=(2 * NKV, 1, M.FULL_ROWS, D), dtype=np.float16))],
        convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
    path = '/tmp/var_%s.mlpackage' % tag
    converted.save(path)
    loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    out = np.asarray(list(loaded.predict({'in0': x, 'in1': np.array([0], np.int32),
                                          'in2': cos, 'in3': sin, 'in4': mask},
                                         state=loaded.make_state()).values())[0]).reshape(-1)
    diff = float(np.abs(out - shipped).max() / (np.abs(shipped).max() + 1e-9))
    print({tag: round(diff, 4)}, flush=True)


build(True, True, 'qk_norm_to_one')
build(False, False, 'no_rope')
build(True, False, 'norm_one_no_rope')