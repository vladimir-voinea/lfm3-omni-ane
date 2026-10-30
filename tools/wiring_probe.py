# Same ops, same weights -- so the difference is in the wiring.  Probe the MLP activation and
# a few other wirings at position 0, where attention is trivially v.
import json, sys
import numpy as np
import torch
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_model as M
import build_tiled as B
from lfm3_runtime import Runtime, H, NKV, D, NH

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
index = json.load(open('/tmp/full_layer_index.json'))
start = min(i['data_offsets'][0] for i in index.values())
blob = open(B.BIN, 'rb').read()
t = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        t[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info).float()

x = (np.random.RandomState(5).randn(1, H, 1, 1) * 0.1).astype(np.float16)
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0
session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(
    {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask},
    state=session.states[2]).values())[0]).reshape(-1)


class Wiring(M.AttnBlock):
    def __init__(self, tensors, rows, activation='silu', gate='attn'):
        super().__init__(tensors, rows)
        self.activation = activation
        self.gate_mode = gate

    def forward(self, x, pos, cos, sin, mask):
        h = self.pre(x)
        q = self.q_proj(h).view(1, NH, 1, D)
        k = self.k_proj(h).view(1, NKV, 1, D)
        v = self.v_proj(h).view(1, NKV, 1, D)
        def rms(z, w):
            return z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        q = rms(q, self.q_norm)
        k = rms(k, self.k_norm)
        c, s = cos.view(1, 1, 1, D), sin.view(1, 1, 1, D)
        def rot(z):
            half = z.shape[-1] // 2
            return z * c + torch.cat([-z[..., half:], z[..., :half]], dim=-1) * s
        q, k = rot(q), rot(k)
        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q.view(1, NKV, 4, D), key) * 0.0884 + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), val).reshape(1, NH, D)
        g = torch.sigmoid(self.gate(h)).view(1, NH, 1)
        if self.gate_mode == 'attn':
            attn = attn * g
        y = self.o_proj(attn.reshape(1, NH * D, 1, 1))
        h2 = self.post(y) + x
        inner = self.ffn_norm(h2)
        if self.activation == 'silu':
            act = torch.nn.functional.silu(self.mlp.w1(inner)) if hasattr(self.mlp, 'w1') else None
        pre = self.mlp.w1(inner) if hasattr(self.mlp, 'w1') else self.mlp.gate(inner)
        up = self.mlp.w3(inner) if hasattr(self.mlp, 'w3') else self.mlp.up(inner)
        if self.activation == 'silu':
            mixed = torch.nn.functional.silu(pre) * up
        elif self.activation == 'gelu':
            mixed = torch.nn.functional.gelu(pre) * up
        else:
            mixed = torch.nn.functional.relu(pre) * up
        down = self.mlp.w2(mixed) if hasattr(self.mlp, 'w2') else self.mlp.down(mixed)
        return self.ffn_post(down) + h2


for activation in ('silu', 'gelu', 'relu'):
    model = Wiring(t, M.FULL_ROWS, activation).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                    torch.from_numpy(mask)).numpy().reshape(-1)
    rel = float(np.linalg.norm(out - shipped) / (np.linalg.norm(shipped) + 1e-9)) * 100
    cos_sim = float(((out - out.mean()) * (shipped - shipped.mean())).sum() /
                    (out.std() * shipped.std() * out.size + 1e-9))
    print({activation: round(rel, 2)}, flush=True)