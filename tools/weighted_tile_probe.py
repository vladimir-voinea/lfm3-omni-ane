# A tile that includes the layer's q/k/v projections (like the shipped program) but holds
# only an 8,192-row slice.  Random weights are enough: placement does not care about values.
import time
import numpy as np
import torch
import coremltools as ct

H, NH, NKV, D = 2048, 16, 4, 128
TILE = 8192


class Tile(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.k = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.v = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, TILE, D), persistent=False)

    def forward(self, x, cos, sin):
        q = self.q(x).view(1, NKV, 4, D)
        k = self.k(x).view(1, NKV, 1, D)
        v = self.v(x).view(1, NKV, 1, D)
        c, s = cos.view(1, 1, 1, D), sin.view(1, 1, 1, D)
        def rotate(z):
            half = z.shape[-1] // 2
            flipped = torch.cat([-z[..., half:], z[..., :half]], dim=-1)
            return z * c + flipped * s
        q, k = rotate(q), rotate(k)
        key = self.cache[0:NKV].view(1, NKV, TILE, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, TILE, D)
        scores = torch.matmul(q, key) * 0.0884
        probs = torch.softmax(scores, dim=-1)
        return probs.sum(dim=-1), torch.matmul(probs, val).reshape(1, NH * D)


module = Tile().eval().half()
example = (torch.zeros(1, H, 1, 1, dtype=torch.float16),
           torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16))
traced = torch.jit.trace(module, example)
model = ct.convert(traced,
    inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
            ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
            ct.TensorType(name='sin', shape=(D,), dtype=np.float16)],
    states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
        shape=(2 * NKV, 1, TILE, D), dtype=np.float16))],
    convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18, compute_units=ct.ComputeUnit.CPU_AND_NE)
model.save('/tmp/weighted_tile.mlpackage')


def timing(units, repeats=20):
    loaded = ct.models.MLModel('/tmp/weighted_tile.mlpackage', compute_units=units)
    state = loaded.make_state()
    feeds = {'x': np.zeros((1, H, 1, 1), np.float16), 'cos': np.ones((D,), np.float16),
             'sin': np.zeros((D,), np.float16)}
    loaded.predict(feeds, state=state)
    start = time.perf_counter()
    for _ in range(repeats):
        loaded.predict(feeds, state=state)
    return (time.perf_counter() - start) / repeats * 1000


ne = timing(ct.ComputeUnit.CPU_AND_NE)
cpu = timing(ct.ComputeUnit.CPU_ONLY)
print({'weighted_tile': TILE, 'ne_ms': round(ne, 3), 'cpu_ms': round(cpu, 3),
       'on_engine': ne < cpu * 0.6, 'est_tok_s': round(1000 / (ne * 60 + 20), 1)}, flush=True)