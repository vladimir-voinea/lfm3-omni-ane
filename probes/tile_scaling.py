# Tile that returns ONLY the weighted sum, with the softmax denominator folded in as an
# extra V column -- no separate reduction output.  Measured at three tile sizes.
import time
import numpy as np
import torch
import coremltools as ct

H, NH, NKV, D = 2048, 16, 4, 128


class Tile(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.pre = torch.nn.Conv2d(H, H, 1, bias=False)
        self.q = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.k = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.v = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, rows, D), persistent=False)

    def forward(self, x, pos, cos, sin, mask):
        h = self.pre(x)
        q = self.q(h).view(1, NKV, 4, D)
        k = self.k(h).view(1, NKV, 1, D)
        v = self.v(h).view(1, NKV, 1, D)
        c, s = cos.view(1, 1, 1, D), sin.view(1, 1, 1, D)
        def rotate(z):
            half = z.shape[-1] // 2
            flipped = torch.cat([-z[..., half:], z[..., :half]], dim=-1)
            return z * c + flipped * s
        q, k = rotate(q), rotate(k)
        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q, key) * 0.0884 + mask
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, val).reshape(1, NH * D)


def run(rows):
    module = Tile(rows).eval().half()
    example = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
               torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16),
               torch.full((1, 4, 1, rows), -1e4, dtype=torch.float16))
    traced = torch.jit.trace(module, example)
    path = '/tmp/tile_acc_%d.mlpackage' % rows
    ct.convert(traced,
        inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
                ct.TensorType(name='sin', shape=(D,), dtype=np.float16),
                ct.TensorType(name='mask', shape=(1, 4, 1, rows), dtype=np.float16)],
        states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
            shape=(2 * NKV, 1, rows, D), dtype=np.float16))],
        convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    def timing(units, repeats=12):
        loaded = ct.models.MLModel(path, compute_units=units)
        state = loaded.make_state()
        feeds = {'x': np.zeros((1, H, 1, 1), np.float16), 'pos': np.array([0], np.int32),
                 'cos': np.ones((D,), np.float16), 'sin': np.zeros((D,), np.float16),
                 'mask': np.full((1, 4, 1, rows), -1e4, np.float16)}
        loaded.predict(feeds, state=state)
        start = time.perf_counter()
        for _ in range(repeats):
            loaded.predict(feeds, state=state)
        return (time.perf_counter() - start) / repeats * 1000
    ne, cpu = timing(ct.ComputeUnit.CPU_AND_NE), timing(ct.ComputeUnit.CPU_ONLY)
    tiles = (78000 + rows - 1) // rows
    print({'rows': rows, 'ne_ms': round(ne, 3), 'cpu_ms': round(cpu, 3),
           'on_engine': ne < cpu * 0.6, 'tiles_for_78k': tiles,
           'total_attn_ms': round(ne * tiles * 6, 1),
           'est_tok_s': round(1000 / (ne * tiles * 6 + 20), 1)}, flush=True)
    import shutil; shutil.rmtree(path, ignore_errors=True)


for rows in (8192, 16384, 32768):
    run(rows)