import os

from paths import PROJECT, SOURCE
# Check the tiled path against plain full attention over a 40-token sequence.
import sys

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, PROJECT)

from tiled_parts import Epilogue, Tile, rotate_half, rope

P = os.path.join(SOURCE, 'model.safetensors')
H, D, NH, NKV, FF = 2048, 128, 16, 4, 7680
EPS, SCALE = 1e-5, 0.0884
T, TILES, ROWS = 40, 3, 16


def reference(t, x):
    """The layer over the whole sequence at once, plain causal attention."""
    pre = t['operator_norm.weight']
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * pre
    q = (h @ t['self_attn.q_proj.weight'].T).view(1, T, NH, D)
    k = (h @ t['self_attn.k_proj.weight'].T).view(1, T, NKV, D)
    v = (h @ t['self_attn.v_proj.weight'].T).view(1, T, NKV, D)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * t['self_attn.q_norm.weight']
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * t['self_attn.k_norm.weight']
    cos, sin = rope(T, 1e6, 0.5)
    cos, sin = cos.view(1, T, 1, D), sin.view(1, T, 1, D)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    mask = torch.full((T, T), float('-inf')).triu(1)
    out = torch.zeros(T, NH, D)
    qh, kh, vh = q[0].permute(1, 0, 2), k[0].permute(1, 0, 2), v[0].permute(1, 0, 2)
    for hh in range(NH):
        kv = hh // (NH // NKV)
        s = qh[hh] @ kh[kv].T * SCALE + mask
        out[:, hh] = torch.softmax(s, dim=-1) @ vh[kv]
    gate = torch.sigmoid(h @ t['self_attn.gate_proj.weight'].T)
    out = out * gate.view(T, NH, 1)
    y = out.reshape(1, T, NH * D) @ t['self_attn.o_proj.weight'].T
    h1 = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS) * t['post_operator_norm.weight'] + x
    hn = h1 * torch.rsqrt(h1.pow(2).mean(-1, keepdim=True) + EPS) * t['ffn_norm.weight']
    f = (torch.nn.functional.silu(hn @ t['feed_forward.w1.weight'].T)
         * (hn @ t['feed_forward.w3.weight'].T)) @ t['feed_forward.w2.weight'].T
    return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS) * t['post_ffn_norm.weight'] + h1


if __name__ == '__main__':
    t = {}
    with safe_open(P, framework='pt') as fh:
        for key in fh.keys():
            if 'language_model.layers.2.' in key:
                t[key.split('layers.2.')[1]] = fh.get_tensor(key).float()
    torch.manual_seed(0)
    x = torch.randn(1, T, H) * 0.5
    ref = reference(t, x)

    cos, sin = rope(TILES * ROWS, 1e6, 0.5)
    tile_paths, epi_path = [], None
    tile_progs = []
    for j in range(TILES):
        tile = Tile(t, ROWS).eval()
        ex = (torch.zeros(1, H, 1, 1, dtype=torch.float16),
              torch.zeros(1, dtype=torch.int32),
              torch.zeros(D, dtype=torch.float16),
              torch.zeros(D, dtype=torch.float16),
              torch.zeros(1, NKV, 1, ROWS, dtype=torch.float16))
        traced = torch.jit.trace(tile.half(), ex)
        prog = ct.convert(traced,
                          inputs=[ct.TensorType(name='x', shape=(1, H, 1, 1), dtype=np.float16),
                                  ct.TensorType(name='pos', shape=(1,), dtype=np.int32),
                                  ct.TensorType(name='cos', shape=(D,), dtype=np.float16),
                                  ct.TensorType(name='sin', shape=(D,), dtype=np.float16),
                                  ct.TensorType(name='mask', shape=(1, NKV, 1, ROWS),
                                                dtype=np.float16)],
                          states=[ct.StateType(name='cache', wrapped_type=ct.TensorType(
                              shape=(2 * NKV, 1, ROWS, D), dtype=np.float16))],
                          compute_precision=ct.precision.FLOAT16,
                          minimum_deployment_target=ct.target.iOS18,
                          compute_units=ct.ComputeUnit.CPU_AND_NE)
        tile_progs.append((prog, prog.make_state()))

    epi = Epilogue(t).eval()
    ex2 = [torch.zeros(1, H, 1, 1, dtype=torch.float16),
           torch.zeros(1, H, 1, 1, dtype=torch.float16)]
    for _ in range(3):
        ex2 += [torch.zeros(1, NKV, 4, D, dtype=torch.float16),
                torch.zeros(1, NKV, 4, dtype=torch.float16),
                torch.zeros(1, NKV, 4, dtype=torch.float16)]
    ex2 = tuple(ex2)
    epi_prog = ct.convert(torch.jit.trace(epi.half(), ex2),
                          inputs=[ct.TensorType(name='in%d' % i, shape=tuple(a.shape),
                                                dtype=np.float16)
                                  for i, a in enumerate(ex2)],
                          compute_precision=ct.precision.FLOAT16,
                          minimum_deployment_target=ct.target.iOS18,
                          compute_units=ct.ComputeUnit.CPU_AND_NE)

    got = []
    for pos in range(T):
        hidden = x[0, pos].view(1, H, 1, 1).half().numpy()
        partials = []
        for j, (prog, state) in enumerate(tile_progs):
            start = j * ROWS
            local = pos - start
            mask = np.full((1, NKV, 1, ROWS), -1e4, np.float16)
            if local >= 0:
                mask[:, :, :, :local + 1] = 0
            out = prog.predict({'x': hidden,
                                'pos': np.array([local], np.int32),
                                'cos': cos[pos].numpy().astype(np.float16),
                                'sin': sin[pos].numpy().astype(np.float16),
                                'mask': mask}, state=state)
            values = list(out.values())
            wv = [v for v in values if np.asarray(v).shape == (1, NKV, 4, D)][0]
            stats = [v for v in values if np.asarray(v).shape == (1, NKV, 4)]
            partials.append((wv, stats[0], stats[1]))
        feed = {'in0': hidden, 'in1': hidden}
        for j, (wv, m, s) in enumerate(partials):
            feed['in%d' % (2 + 3 * j)] = np.asarray(wv, np.float16)
            feed['in%d' % (3 + 3 * j)] = np.asarray(m, np.float16)
            feed['in%d' % (4 + 3 * j)] = np.asarray(s, np.float16)
        out = epi_prog.predict(feed)
        got.append(np.asarray(list(out.values())[0], np.float32).reshape(H))
    got = np.stack(got)
    r = ref[0].detach().numpy().astype(np.float32)
    rel = float(np.linalg.norm(got - r) / max(np.linalg.norm(r), 1e-9))
    print({'tiled_vs_full_rel': round(rel, 5),
           'max_abs': round(float(np.abs(got - r).max()), 4),
           'ref_norm': round(float(np.linalg.norm(r)), 2)}, flush=True)