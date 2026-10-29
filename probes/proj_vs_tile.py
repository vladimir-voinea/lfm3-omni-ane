# Where does a tile's time actually go?  Three programs, same weights, measured separately:
#   proj   -- pre-norm + q/k/v only
#   tile_w -- weightless attention + an identity conv so the planner sees a weight
#   tile_p -- attention with the projections inside (what I measured last round)
import time
import numpy as np
import torch
import coremltools as ct

H, NH, NKV, D = 2048, 16, 4, 128


class Proj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = torch.nn.Conv2d(H, H, 1, bias=False)
        self.q = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.k = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.v = torch.nn.Conv2d(H, NKV * D, 1, bias=False)

    def forward(self, x):
        h = self.pre(x)
        return self.q(h), self.k(h), self.v(h)


class Tile(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        # identity 1x1 conv: the planner refused a graph with no weights at all
        self.eye = torch.nn.Conv2d(NKV * D, NKV * D, 1, bias=False)
        with torch.no_grad():
            self.eye.weight.copy_(torch.eye(NKV * D).view(NKV * D, NKV * D, 1, 1))
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, rows, D), persistent=False)

    def forward(self, q, k, v, pos, mask):
        k = self.eye(k).view(1, NKV, 1, D)
        v = self.eye(v).view(1, NKV, 1, D)
        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q.view(1, NKV, 4, D), key) * 0.0884 + mask
        probs = torch.softmax(scores, dim=-1)
        return probs.sum(dim=-1), torch.matmul(probs, val).reshape(1, NH * D)


def timeit(path, feeds, has_state, repeats=15):
    def run(units):
        model = ct.models.MLModel(path, compute_units=units)
        state = model.make_state() if has_state else None
        model.predict(feeds, state=state) if has_state else model.predict(feeds)
        start = time.perf_counter()
        for _ in range(repeats):
            model.predict(feeds, state=state) if has_state else model.predict(feeds)
        return (time.perf_counter() - start) / repeats * 1000
    return run(ct.ComputeUnit.CPU_AND_NE), run(ct.ComputeUnit.CPU_ONLY)


proj = Proj().eval().half()
traced = torch.jit.trace(proj, torch.zeros(1, H, 1, 1, dtype=torch.float16))
ct.convert(traced, inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16)],
           convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
           minimum_deployment_target=ct.target.iOS18,
           compute_units=ct.ComputeUnit.CPU_AND_NE).save('/tmp/proj.mlpackage')
ne, cpu = timeit('/tmp/proj.mlpackage', {'x': np.zeros((1, H, 1, 1), np.float16)}, False)
print({'proj': True, 'ne_ms': round(ne, 3), 'cpu_ms': round(cpu, 3),
       'on_engine': ne < cpu * 0.6}, flush=True)

for rows in (8192, 16384):
    tile = Tile(rows).eval().half()
    example = (torch.zeros(1, NKV, 4, D, dtype=torch.float16),
               torch.zeros(1, NKV * D, 1, 1, dtype=torch.float16),
               torch.zeros(1, NKV * D, 1, 1, dtype=torch.float16),
               torch.zeros(1, dtype=torch.int32),
               torch.full((1, NKV, 4, rows), -1e4, dtype=torch.float16))
    traced = torch.jit.trace(tile, example)
    path = '/tmp/tilek_%d.mlpackage' % rows
    ct.convert(traced,
        inputs=[ct.TensorType(name='q', shape=(1, NKV, 4, D), dtype=np.float16),
                ct.TensorType(name='k', shape=(1, NKV * D, 1, 1), dtype=np.float16),
                ct.TensorType(name='v', shape=(1, NKV * D, 1, 1), dtype=np.float16),
                ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                ct.TensorType(name='mask', shape=(1, NKV, 4, rows), dtype=np.float16)],
        states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
            shape=(2 * NKV, 1, rows, D), dtype=np.float16))],
        convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    feeds = {'q': np.zeros((1, NKV, 4, D), np.float16),
             'k': np.zeros((1, NKV * D, 1, 1), np.float16),
             'v': np.zeros((1, NKV * D, 1, 1), np.float16),
             'pos': np.array([0], np.int32),
             'mask': np.full((1, NKV, 4, rows), -1e4, np.float16)}
    ne, cpu = timeit(path, feeds, True)
    tiles = (78000 + rows - 1) // rows
    print({'rows': rows, 'ne_ms': round(ne, 3), 'cpu_ms': round(cpu, 3),
           'on_engine': ne < cpu * 0.6, 'tiles': tiles,
           'attn_ms_total': round(ne * tiles * 6, 1)}, flush=True)