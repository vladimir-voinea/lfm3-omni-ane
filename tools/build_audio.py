# The audio encoder and projector as one Core ML program.  Lengths vary and programs do not, so it
# takes a fixed mel window padded with zeros.  The stack is causal and the convolutions pad with
# zeros themselves, so rows inside the real audio do not change.  Transcribed from audio_ref.py,
# which matches the official model at cosine 1.0, with the engine fixes from notes/STATE.md.
from __future__ import annotations

import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

import audio_ref as A

FRAMES = 3000                    # 30 s of mel
ROWS = FRAMES // 8               # encoder rows
PROJECT_SCALE = 8.0


def silu_engine(x):
    return 0.5 * x * (1 + torch.tanh(0.5 * x))


def project(weight, x):
    return (x * PROJECT_SCALE) @ weight.T / PROJECT_SCALE


class Encoder(torch.nn.Module):
    def __init__(self, w, frames=FRAMES):
        super().__init__()
        self.w = w
        self.rows = frames // 8
        self.cos, self.sin = A.rope(self.rows)
        index = torch.arange(self.rows)
        distance = index[:, None] - index[None, :]
        self.mask = torch.where((distance < 0) | (distance >= A.WINDOW), -1e4, 0.0)

    def downsample(self, feats):
        w, d = self.w, 'model.audio_encoder.downsample.'
        x = feats.unsqueeze(1)
        x = silu_engine(F.conv2d(x, w[d + 'conv_in.weight'], w[d + 'conv_in.bias'], stride=2, padding=1))
        x = F.conv2d(x, w[d + 'conv_mid.weight'], w[d + 'conv_mid.bias'], stride=2, padding=1, groups=256)
        x = silu_engine(F.conv2d(x, w[d + 'lin_mid.weight'], w[d + 'lin_mid.bias']))
        x = F.conv2d(x, w[d + 'conv_end.weight'], w[d + 'conv_end.bias'], stride=2, padding=1, groups=256)
        x = silu_engine(F.conv2d(x, w[d + 'lin_end.weight'], w[d + 'lin_end.bias']))
        x = x.flatten(1, 2)
        return F.conv1d(x, w[d + 'lin_out.weight'], w[d + 'lin_out.bias']).transpose(1, 2)[0]

    def attention(self, hn, p):
        w, n, rep = self.w, self.rows, A.NH // A.NKV
        q = A.rms((hn @ w[p + 'q_proj.weight'].T).view(n, A.NH, A.HD), w[p + 'q_layernorm.weight'])
        k = A.rms((hn @ w[p + 'k_proj.weight'].T).view(n, A.NKV, A.HD), w[p + 'k_layernorm.weight'])
        v = (hn @ w[p + 'v_proj.weight'].T).view(n, A.NKV, A.HD)
        cos, sin = self.cos[:, None, :], self.sin[:, None, :]
        q = q * cos + A.rotate_half(q) * sin
        k = k * cos + A.rotate_half(k) * sin
        q = q.permute(1, 0, 2).view(A.NKV, rep, n, A.HD)
        k = k.permute(1, 0, 2).view(A.NKV, 1, n, A.HD)
        v = v.permute(1, 0, 2).view(A.NKV, 1, n, A.HD)
        scores = torch.matmul(q, k.transpose(2, 3)) * (A.HD ** -0.5) + self.mask
        y = torch.matmul(torch.softmax(scores, dim=-1), v).view(A.NH, n, A.HD).permute(1, 0, 2)
        return project(w[p + 'out_proj.weight'], y.reshape(n, A.NH * A.HD))

    def convolution(self, hn, p):
        w = self.w
        proj = hn @ w[p + 'conv.in_proj.weight'].T
        b, c, gate = proj[:, :A.H], proj[:, A.H:2 * A.H], proj[:, 2 * A.H:]
        padded = F.pad((b * gate).T[None], (A.CONV_L - 1, 0))
        conv = F.conv1d(padded, w[p + 'conv.conv.weight'], groups=A.H)[0].T
        return project(w[p + 'conv.out_proj.weight'], c * conv)

    def forward(self, feats):
        w = self.w
        x = self.downsample(feats)
        for i in range(17):
            p = '%slayers.%d.' % (A.PRE, i)
            hn = A.rms(x, w[p + 'operator_norm.weight'])
            if (p + 'conv.conv.weight') in w:
                x = x + self.convolution(hn, p)
            else:
                x = x + self.attention(hn, p + 'self_attn.')
            f = A.rms(x, w[p + 'ffn_norm.weight'])
            x = x + project(w[p + 'feed_forward.w2.weight'],
                            silu_engine(f @ w[p + 'feed_forward.w1.weight'].T)
                            * (f @ w[p + 'feed_forward.w3.weight'].T))
        x = A.rms(x, w['model.audio_encoder.lfm.embedding_norm.weight'])
        pre = 'model.audio_projector.'
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + 1e-5) * w[pre + 'layer_norm.weight'] + w[pre + 'layer_norm.bias']
        y = F.gelu(y @ w[pre + 'linear_1.weight'].T + w[pre + 'linear_1.bias'])
        return project(w[pre + 'linear_2.weight'], y) + w[pre + 'linear_2.bias']


def load():
    w = {}
    with safe_open(A.P, framework='pt') as f:
        for k in f.keys():
            if 'audio_encoder' in k or 'audio_projector' in k:
                w[k] = f.get_tensor(k).float()
    return w


if __name__ == '__main__':
    model = Encoder(load()).eval()
    example = torch.zeros(1, 128, FRAMES)
    with torch.no_grad():
        traced = torch.jit.trace(model, (example,))
    ct.convert(traced, inputs=[ct.TensorType(name='mel', shape=example.shape, dtype=np.float32)],
               outputs=[ct.TensorType(name='rows')], convert_to='mlprogram',
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(sys.argv[1])
    print({'saved': sys.argv[1]}, flush=True)
