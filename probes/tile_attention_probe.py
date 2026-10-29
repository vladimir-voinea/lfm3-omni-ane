# The load-bearing question for exact-78k on ANE: can a WEIGHTLESS tile program hold a slice
# of KV in its state, compute scores+softmax+weighted-V over it, and return the three values
# flash-attention needs to combine?  If it dispatches in well under a millisecond, a token
# costs (tiles x layers) dispatches and the design is alive.
import time
import numpy as np
import torch
import coremltools as ct

TILE = 8192
NKV, NH, D = 4, 16, 128


class Tile(torch.nn.Module):
    """No weights at all: q in, KV in state, partial softmax out."""

    def __init__(self):
        super().__init__()
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, TILE, D), persistent=False)

    def forward(self, q, k, v, slot, write):
        # write this token's k/v into this tile's buffer, at the slot we are told. Only the
        # tile that owns the current position maps to slot = position % TILE, the others
        # write to a slot they will never read back at that index -- so the write is a no-op
        # there in effect, and the buffer stays exact.
        key = self.cache[0:NKV].view(1, NKV, TILE, D).transpose(-1, -2)   # (1, NKV, D, TILE)
        val = self.cache[NKV:2 * NKV].view(1, NKV, TILE, D)               # (1, NKV, TILE, D)
        scores = torch.matmul(q.view(1, NKV, 4, D), key) * 0.0884   # (1, NKV, 4, TILE)
        best = scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(scores - best)
        total = weights.sum(dim=-1)                          # (1, NKV, 4)
        acc = torch.matmul(weights, val)                     # (1, NKV, 4, D)
        return best.reshape(1, NH), total.reshape(1, NH), acc.reshape(1, NH * D)


if __name__ == '__main__':
    module = Tile().eval().half()
    example = (torch.zeros(1, NKV, 4, D, dtype=torch.float16),
               torch.zeros(1, NKV, 1, D, dtype=torch.float16),
               torch.zeros(1, NKV, 1, D, dtype=torch.float16),
               torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.float16))
    traced = torch.jit.trace(module, example)
    model = ct.convert(traced,
        inputs=[ct.TensorType(name='q', shape=(1, NKV, 4, D), dtype=np.float16),
                ct.TensorType(name='k', shape=(1, NKV, 1, D), dtype=np.float16),
                ct.TensorType(name='v', shape=(1, NKV, 1, D), dtype=np.float16),
                ct.TensorType(name='slot', shape=(1,), dtype=np.int32),
                ct.TensorType(name='write', shape=(1,), dtype=np.float16)],
        states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
            shape=(2 * NKV, 1, TILE, D), dtype=np.float16))],
        convert_to='mlprogram', compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_NE)
    model.save('/tmp/tile.mlpackage')
    loaded = ct.models.MLModel('/tmp/tile.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = loaded.make_state()
    feeds = {'q': np.zeros((1, NKV, 4, D), np.float16), 'k': np.zeros((1, NKV, 1, D), np.float16),
             'v': np.zeros((1, NKV, 1, D), np.float16), 'slot': np.array([0], np.int32),
             'write': np.zeros((1,), np.float16)}
    loaded.predict(feeds, state=state)
    start = time.perf_counter()
    for _ in range(30):
        loaded.predict(feeds, state=state)
    ms = (time.perf_counter() - start) / 30 * 1000
    print({'tile_rows': TILE, 'state_MB': round(2 * NKV * TILE * D * 2 / 1e6, 1),
           'dispatch_ms': round(ms, 3),
           'dispatches_per_token': 6 * (78000 // TILE + 1),
           'est_ms_per_token': round(ms * 6 * (78000 // TILE + 1), 1)}, flush=True)