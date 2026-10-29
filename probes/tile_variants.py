# Which formulation of tiled attention actually lands on the engine?
import time
import numpy as np
import torch
import coremltools as ct

TILE, NKV, NH, D = 8192, 4, 16, 128


def build(variant):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('cache', torch.zeros(2 * NKV, 1, TILE, D), persistent=False)

        def forward(self, q, k, v, slot):
            key = self.cache[0:NKV].view(1, NKV, TILE, D).transpose(-1, -2)
            val = self.cache[NKV:2 * NKV].view(1, NKV, TILE, D)
            qq = q.view(1, NKV, 4, D)
            if variant == 'exp_sum':
                scores = torch.matmul(qq, key) * 0.0884
                best = scores.max(dim=-1, keepdim=True).values
                weights = torch.exp(scores - best)
                return weights.sum(dim=-1), torch.matmul(weights, val).reshape(1, NH * D)
            if variant == 'softmax':
                scores = torch.matmul(qq, key) * 0.0884
                probs = torch.softmax(scores, dim=-1)
                return probs.sum(dim=-1), torch.matmul(probs, val).reshape(1, NH * D)
            if variant == 'no_softmax':
                scores = torch.matmul(qq, key) * 0.0884
                return scores.sum(dim=-1), torch.matmul(scores, val).reshape(1, NH * D)
            raise ValueError(variant)

    module = M().eval().half()
    example = (torch.zeros(1, NKV, 4, D, dtype=torch.float16),
               torch.zeros(1, NKV, 1, D, dtype=torch.float16),
               torch.zeros(1, NKV, 1, D, dtype=torch.float16),
               torch.zeros(1, dtype=torch.int32))
    traced = torch.jit.trace(module, example)
    return ct.convert(traced,
        inputs=[ct.TensorType(name='q', shape=(1, NKV, 4, D), dtype=np.float16),
                ct.TensorType(name='k', shape=(1, NKV, 1, D), dtype=np.float16),
                ct.TensorType(name='v', shape=(1, NKV, 1, D), dtype=np.float16),
                ct.TensorType(name='slot', shape=(1,), dtype=np.int32)],
        states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
            shape=(2 * NKV, 1, TILE, D), dtype=np.float16))],
        convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE)


def timing(model, repeats=20):
    state = model.make_state()
    feeds = {'q': np.zeros((1, NKV, 4, D), np.float16), 'k': np.zeros((1, NKV, 1, D), np.float16),
             'v': np.zeros((1, NKV, 1, D), np.float16), 'slot': np.array([0], np.int32)}
    model.predict(feeds, state=state)
    start = time.perf_counter()
    for _ in range(repeats):
        model.predict(feeds, state=state)
    return (time.perf_counter() - start) / repeats * 1000


for variant in ('no_softmax', 'softmax', 'exp_sum'):
    path = '/tmp/tile_%s.mlpackage' % variant
    build(variant).save(path)
    ne = timing(ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE))
    cpu = timing(ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY))
    print({variant: True, 'ne_ms': round(ne, 3), 'cpu_ms': round(cpu, 3),
           'on_engine': ne < cpu * 0.6}, flush=True)