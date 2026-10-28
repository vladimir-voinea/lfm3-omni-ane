# Verify tiled attention with a host-side merge against plain full attention.
import sys

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

from tiled_parts2 import D, H, NKV, NH, SCALE, EPS, Epilogue, Tile, rotate_half, rms1, rope

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
T, TILES, ROWS = 40, 3, 16


def reference(t, x):
    h = rms1(x.view(1, H, 1, T), t['operator_norm.weight']).view(1, T, H)
    q = (h @ t['self_attn.q_proj.weight'].T).view(1, T, NH, D)
    k = (h @ t['self_attn.k_proj.weight'].T).view(1, T, NKV, D)
    v = (h @ t['self_attn.v_proj.weight'].T).view(1, T, NKV, D)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * t['self_attn.q_norm.weight']
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS) * t['self_attn.k_norm.weight']
    cos, sin = rope(T, 1e6, 0.5)
    q = q * cos.view(1, T, 1, D) + rotate_half(q) * sin.view(1, T, 1, D)
    k = k * cos.view(1, T, 1, D) + rotate_half(k) * sin.view(1, T, 1, D)
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
    h1 = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS) * t['post_operator_norm.weight'] + x.view(1, T, H)
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
    ref = reference(t, x)[0].detach().numpy()

    cos, sin = rope(TILES * ROWS, 1e6, 0.5)
    tiles = []
    for j in range(TILES):
        tile = Tile(t, ROWS).eval().half()
        ex = (torch.zeros(1, H, 1, 1, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
              torch.zeros(D, dtype=torch.float16), torch.zeros(D, dtype=torch.float16),
              torch.zeros(1, NKV, 1, ROWS, dtype=torch.float16))
        prog = ct.convert(torch.jit.trace(tile, ex),
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
        tiles.append((prog, prog.make_state()))

    epi = Epilogue(t).eval().half()
    epi_prog = ct.convert(torch.jit.trace(epi, (torch.zeros(1, H, 1, 1, dtype=torch.float16),
                                                    torch.zeros(1, H, 1, 1, dtype=torch.float16))),
                          inputs=[ct.TensorType(name='in0', shape=(1, H, 1, 1), dtype=np.float16),
                                  ct.TensorType(name='in1', shape=(1, H, 1, 1), dtype=np.float16)],
                          compute_precision=ct.precision.FLOAT16,
                          minimum_deployment_target=ct.target.iOS18,
                          compute_units=ct.ComputeUnit.CPU_AND_NE)

    out_rows = []
    for pos in range(T):
        hidden = x[0, pos].view(1, H, 1, 1).half().numpy()
        wvs, ms, ss = [], [], []
        for j, (prog, state) in enumerate(tiles):
            local = pos - j * ROWS
            if local < 0:
                # a tile that has not started yet contributes nothing
                wvs.append(np.zeros((1, NKV, 4, D), np.float32))
                ms.append(np.full((1, NKV, 4, 1), -1e4, np.float32))
                ss.append(np.zeros((1, NKV, 4), np.float32))
                continue
            mask = np.full((1, NKV, 1, ROWS), -1e4, np.float16)
            mask[:, :, :, :local + 1] = 0
            out = prog.predict({'x': hidden, 'pos': np.array([local], np.int32),
                                'cos': cos[pos].numpy().astype(np.float16),
                                'sin': sin[pos].numpy().astype(np.float16),
                                'mask': mask}, state=state)
            vals = {np.asarray(v).shape: np.asarray(v, np.float32) for v in out.values()}
            wvs.append(vals[(1, NKV, 4, D)])
            stats = [v for k, v in vals.items() if k == (1, NKV, 4, 1)]
            ss.append([v for k, v in vals.items() if k == (1, NKV, 4)][0])
            ms.append(stats[0])
        m = np.maximum(np.maximum(ms[0], ms[1]), ms[2])
        w0 = np.exp(ms[0] - m)
        w1 = np.exp(ms[1] - m)
        w2 = np.exp(ms[2] - m)
        num = wvs[0] * w0 + wvs[1] * w1 + wvs[2] * w2
        den = ss[0] * w0[..., 0] + ss[1] * w1[..., 0] + ss[2] * w2[..., 0]
        attn = (num / den[..., None]).reshape(1, H, 1, 1).astype(np.float16)
        out = epi_prog.predict({'in0': hidden, 'in1': attn})
        out_rows.append(np.asarray(list(out.values())[0], np.float32).reshape(H))
    got = np.stack(out_rows)
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))
    print({'tiled_vs_full_rel': round(rel, 5),
           'max_abs': round(float(np.abs(got - ref).max()), 4)}, flush=True)