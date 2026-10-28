# Is the tiled merge itself wrong, or is 2.5% just fp16?  Do the tiled math in
# torch fp32 with no Core ML at all and compare it against full attention.
import numpy as np
import torch

T, ROWS, TILES = 40, 16, 3
torch.manual_seed(0)
q = torch.randn(T, 4, 128)
k = torch.randn(T, 4, 128)
v = torch.randn(T, 4, 128)

mask = torch.full((T, T), float('-inf')).triu(1)
full = torch.zeros(T, 4, 128)
for hh in range(4):
    s = q[:, hh] @ k[:, hh].T * 0.0884 + mask
    full[:, hh] = torch.softmax(s, dim=-1) @ v[:, hh]

rows = []
for pos in range(T):
    wvs, ms, ss = [], [], []
    for j in range(TILES):
        lo, hi = j * ROWS, min((j + 1) * ROWS, T)
        if pos < lo:
            wvs.append(torch.zeros(4, 128))
            ms.append(torch.full((4, 1), -1e4))
            ss.append(torch.zeros(4))
            continue
        upto = min(pos + 1, hi)
        sc = torch.zeros(4, hi - lo)
        for hh in range(4):
            sc[hh] = q[pos, hh] @ k[lo:hi, hh].T * 0.0884
        sc[:, upto - lo:] = -1e4
        m = sc.max(dim=-1, keepdim=True).values
        e = torch.exp(sc - m)
        wv = torch.zeros(4, 128)
        for hh in range(4):
            wv[hh] = e[hh] @ v[lo:hi, hh]
        wvs.append(wv)
        ms.append(m)
        ss.append(e.sum(-1))
    m = torch.maximum(torch.maximum(ms[0], ms[1]), ms[2])
    w = [torch.exp(mi - m) for mi in ms]
    num = sum(a * b for a, b in zip(wvs, w))
    den = sum(s * b.squeeze(-1) for s, b in zip(ss, w))
    rows.append(num / den.unsqueeze(-1))
tiled = torch.stack(rows)
print({'fp32_tiled_vs_full': round(float((tiled - full).norm() / full.norm()), 8)}, flush=True)