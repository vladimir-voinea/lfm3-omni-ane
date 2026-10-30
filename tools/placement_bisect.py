# Bisect from the structure that IS on the engine.  Start from a shipped-like block and
# remove one thing at a time until placement breaks.
import time
import numpy as np
import torch
import coremltools as ct

H, NH, NKV, D, ROWS = 2048, 16, 4, 128, 8192


class Block(torch.nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.pre = torch.nn.Conv2d(H, H, 1, bias=False)
        self.q = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.k = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.v = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.o = torch.nn.Conv2d(H, H, 1, bias=False)
        self.mlp1 = torch.nn.Conv2d(H, 7680, 1, bias=False)
        self.mlp2 = torch.nn.Conv2d(7680, H, 1, bias=False)
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, ROWS, D), persistent=False)

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
        key = self.cache[0:NKV].view(1, NKV, ROWS, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, ROWS, D)
        scores = torch.matmul(q, key) * 0.0884 + mask
        probs = torch.softmax(scores, dim=-1)
        if self.variant == 'partial':
            return probs.sum(dim=-1), torch.matmul(probs, val).reshape(1, NH * D)
        attn = torch.matmul(probs, val).reshape(1, NH * D, 1, 1)
        y = self.o(attn)
        if self.variant == 'no_mlp':
            return y
        h2 = y + x
        return self.mlp2(torch.nn.functional.silu(self.mlp1(h2))) + h2


def run(variant):
    module = Block(variant).eval().half()
    example = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
               torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16),
               torch.full((1, 4, 1, ROWS), -1e4, dtype=torch.float16))
    traced = torch.jit.trace(module, example)
    path = '/tmp/block_%s.mlpackage' % variant
    ct.convert(traced,
        inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
                ct.TensorType(name='sin', shape=(D,), dtype=np.float16),
                ct.TensorType(name='mask', shape=(1, 4, 1, ROWS), dtype=np.float16)],
        states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
            shape=(2 * NKV, 1, ROWS, D), dtype=np.float16))],
        convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    def timing(units, repeats=15):
        loaded = ct.models.MLModel(path, compute_units=units)
        state = loaded.make_state()
        feeds = {'x': np.zeros((1, H, 1, 1), np.float16), 'pos': np.array([0], np.int32),
                 'cos': np.ones((D,), np.float16), 'sin': np.zeros((D,), np.float16),
                 'mask': np.full((1, 4, 1, ROWS), -1e4, np.float16)}
        loaded.predict(feeds, state=state)
        start = time.perf_counter()
        for _ in range(repeats):
            loaded.predict(feeds, state=state)
        return (time.perf_counter() - start) / repeats * 1000
    ne, cpu = timing(ct.ComputeUnit.CPU_AND_NE), timing(ct.ComputeUnit.CPU_ONLY)
    print({variant: True, 'ne_ms': round(ne, 3), 'cpu_ms': round(cpu, 3),
           'on_engine': ne < cpu * 0.6}, flush=True)


for variant in ('full', 'no_mlp', 'partial'):
    run(variant)