# The shipped program is the oracle.  Variants are computed in torch on the CPU -- seconds
# each -- and compared with what the shipped program returns for the same input.
import json, sys, os
import numpy as np
import torch
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import build_model as M
import build_tiled as B
from lfm3_runtime import Runtime, H, NKV, D

runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
index = json.load(open('/tmp/full_layer_index.json'))
start = min(i['data_offsets'][0] for i in index.values())
blob = open(B.BIN, 'rb').read()
t = {}
for key, info in index.items():
    if int(key.split('layers.')[1].split('.')[0]) == 2:
        t[key.split('layers.2.')[1]] = B.load_tensor(blob, start, info).float()

x = (np.random.RandomState(3).randn(1, H, 1, 1) * 0.1).astype(np.float16)
rope = np.load('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz')
cos = rope['full_cos'][0].reshape(D).astype(np.float16)
sin = rope['full_sin'][0].reshape(D).astype(np.float16)
mask = np.full((1, NKV, 1, M.FULL_ROWS), -1e4, np.float16); mask[..., 0] = 0
feeds = {'in0': x, 'in1': np.array([0], np.int32), 'in2': cos, 'in3': sin, 'in4': mask}
session = runtime._plain_session()
shipped = np.asarray(list(runtime.programs[2].predict(feeds, state=session.states[2]).values())[0]).reshape(-1)
scale = float(np.abs(shipped).max()) + 1e-9


class Variant(M.AttnBlock):
    def __init__(self, tensors, rows, use_norm=True, use_rope=True, use_gate=True, attn_scale=0.0884):
        super().__init__(tensors, rows)
        self.use_norm = use_norm
        self.use_rope = use_rope
        self.use_gate = use_gate
        self.attn_scale = attn_scale

    def forward(self, x, pos, cos, sin, mask):
        h = self.pre(x)
        q = self.q_proj(h).view(1, NH_FIX, 1, D)
        k = self.k_proj(h).view(1, NKV, 1, D)
        v = self.v_proj(h).view(1, NKV, 1, D)
        def rms(z, w):
            return z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * w.view(1, 1, 1, D)
        if self.use_norm:
            q, k = rms(q, self.q_norm), rms(k, self.k_norm)
        if self.use_rope:
            c, s = cos.view(1, 1, 1, D), sin.view(1, 1, 1, D)
            def rot(z):
                half = z.shape[-1] // 2
                return z * c + torch.cat([-z[..., half:], z[..., :half]], dim=-1) * s
            q, k = rot(q), rot(k)
        self.cache[0:NKV, :, pos:pos + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, pos:pos + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q.view(1, NKV, 4, D), key) * self.attn_scale + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), val).reshape(1, NH_FIX, D)
        if self.use_gate:
            attn = attn * torch.sigmoid(self.gate(h)).view(1, NH_FIX, 1)
        y = self.o_proj(attn.reshape(1, NH_FIX * D, 1, 1))
        h2 = self.post(y) + x
        return self.ffn_post(self.mlp(self.ffn_norm(h2))) + h2


NH_FIX = 16
for label, kwargs in (('baseline', {}),
                      ('no_qk_norm', {'use_norm': False}),
                      ('no_rope', {'use_rope': False}),
                      ('no_gate', {'use_gate': False}),
                      ('scale_1', {'attn_scale': 1.0}),
                      ('no_norm_no_gate', {'use_norm': False, 'use_gate': False})):
    model = Variant(t, M.FULL_ROWS, **kwargs).eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x), 0, torch.from_numpy(cos), torch.from_numpy(sin),
                    torch.from_numpy(mask)).numpy().reshape(-1)
    print({label: round(float(np.abs(out - shipped).max() / scale), 4)}, flush=True)