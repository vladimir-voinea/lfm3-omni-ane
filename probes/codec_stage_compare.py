import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
import codec_ref as C

w = {}
with safe_open(C.CODEC, framework='pt') as f:
    for k in f.keys():
        w[k] = f.get_tensor(k).float()
codes = torch.from_numpy(np.load('/tmp/codec_codes.npy')).long()
stages = np.load('/tmp/codec_stages.npz')

lat = C.quantizer_decode(codes, w)
x = lat @ w['from_quantizer.weight'].T + w['from_quantizer.bias']
ref_fq = torch.from_numpy(stages['from_quantizer'])
print({'from_quantizer_cos': round(float(F.cosine_similarity(x, ref_fq, dim=-1).mean()), 5),
       'maxdiff': round(float((x - ref_fq).abs().max()), 5)}, flush=True)

# run the stack, then lin, and compare against the official lin output
layer_types = ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv']
x2 = x.repeat_interleave(C.FRAME_STACK, dim=1)
seq, head_dim = x2.shape[1], C.H // 8
cos_t, sin_t = C.rope_tables(seq, head_dim, 1e6)
for index, kind in enumerate(layer_types):
    p = 'decoder.lfm.layers.%d.' % index
    hn = C.rms(x2, w[p + 'operator_norm.weight'])
    if kind == 'conv':
        proj = hn @ w[p + 'conv.in_proj.weight'].T
        b, c, gate = proj.chunk(3, dim=-1)
        padded = F.pad((b * gate).transpose(1, 2), (6, 0))
        conv = F.conv1d(padded, w[p + 'conv.conv.weight'], groups=C.H).transpose(1, 2)
        y = (c * conv) @ w[p + 'conv.out_proj.weight'].T
    else:
        nh, nkv = w[p + 'self_attn.q_proj.weight'].shape[0] // head_dim, w[p + 'self_attn.k_proj.weight'].shape[0] // head_dim
        q = C.rms((hn @ w[p + 'self_attn.q_proj.weight'].T).view(1, seq, nh, head_dim), w[p + 'self_attn.q_norm.weight'])
        k = C.rms((hn @ w[p + 'self_attn.k_proj.weight'].T).view(1, seq, nkv, head_dim), w[p + 'self_attn.k_norm.weight'])
        v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(1, seq, nkv, head_dim)
        q = q * cos_t[None, :, None, :] + C.rotate_half(q) * sin_t[None, :, None, :]
        k = k * cos_t[None, :, None, :] + C.rotate_half(k) * sin_t[None, :, None, :]
        qp = q.transpose(1, 2)
        kp = k.transpose(1, 2).repeat_interleave(nh // nkv, dim=1)
        vp = v.transpose(1, 2).repeat_interleave(nh // nkv, dim=1)
        sc = torch.matmul(qp, kp.transpose(2, 3)) * (head_dim ** -0.5)
        sc = sc + torch.full((seq, seq), float('-inf')).triu(1)
        y = torch.matmul(torch.softmax(sc, dim=-1), vp).transpose(1, 2).reshape(1, seq, nh * head_dim)
        gate = torch.sigmoid(hn @ w[p + 'self_attn.gate_proj.weight'].T)
        y = (y * gate.repeat_interleave(head_dim, dim=-1)) @ w[p + 'self_attn.o_proj.weight'].T
    x2 = x2 + C.rms(y, w[p + 'post_operator_norm.weight'])
    f0 = C.rms(x2, w[p + 'ffn_norm.weight'])
    mlp = (F.silu(f0 @ w[p + 'feed_forward.w1.weight'].T) * (f0 @ w[p + 'feed_forward.w3.weight'].T)) @ w[p + 'feed_forward.w2.weight'].T
    x2 = x2 + C.rms(mlp, w[p + 'post_ffn_norm.weight'])
x2 = C.rms(x2, w['decoder.lfm.embedding_norm.weight'])
spec = x2 @ w['decoder.lin.weight'].T + w['decoder.lin.bias']
ref_lin = torch.from_numpy(stages['lin'])
print({'lin_cos': round(float(F.cosine_similarity(spec, ref_lin, dim=-1).mean()), 5),
       'mine_absmean': round(float(spec.abs().mean()), 4),
       'ref_absmean': round(float(ref_lin.abs().mean()), 4)}, flush=True)