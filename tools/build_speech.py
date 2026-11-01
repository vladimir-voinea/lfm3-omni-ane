# Speech output as Core ML: the depth decoder and the codec's network, each one program, plus the
# tables the host needs so that speech no longer reads the tower or codec checkpoints.  Sampling and
# the inverse STFT stay on the host.  Transcribed from depth_ref.py and codec_ref.py, which match
# the official model exactly, with the engine fixes from notes/STATE.md.
from __future__ import annotations

import os
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

import codec_ref as C
import depth_ref as D

CODEC_FRAMES = 128              # 10.2 s
PROJECT_SCALE = 8.0
CODEC_LAYERS = ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv']


def silu_engine(x):
    return 0.5 * x * (1 + torch.tanh(0.5 * x))


def sigmoid_engine(x):
    return 0.5 * (1 + torch.tanh(0.5 * x))


def project(weight, x):
    return (x * PROJECT_SCALE) @ weight.T / PROJECT_SCALE


class Depth(torch.nn.Module):
    """hidden (1, 2048) and slots (1, 12, 1024, the code or mask embeddings) -> logits (12, 2017)."""

    def __init__(self, w):
        super().__init__()
        self.w = w
        self.cos, self.sin = D.rope(D.CB)

    def forward(self, hidden, slots):
        w, pre = self.w, D.PRE
        x = (hidden @ w[pre + 'depth_linear.weight'].T + w[pre + 'depth_linear.bias']).view(D.CB, D.H)
        x = x + slots[0]
        cos, sin = self.cos[:, None, :], self.sin[:, None, :]
        rep = D.NH // D.NKV
        for layer in range(3):
            p = pre + 'model.layers.%d.' % layer
            hn = D.rms(x, w[p + 'operator_norm.weight'])
            q = D.rms((hn @ w[p + 'self_attn.q_proj.weight'].T).view(D.CB, D.NH, D.HD), w[p + 'self_attn.q_norm.weight'])
            k = D.rms((hn @ w[p + 'self_attn.k_proj.weight'].T).view(D.CB, D.NKV, D.HD), w[p + 'self_attn.k_norm.weight'])
            v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(D.CB, D.NKV, D.HD)
            q = q * cos + D.rotate_half(q) * sin
            k = k * cos + D.rotate_half(k) * sin
            q = q.permute(1, 0, 2).view(D.NKV, rep, D.CB, D.HD)
            k = k.permute(1, 0, 2).view(D.NKV, 1, D.CB, D.HD)
            v = v.permute(1, 0, 2).view(D.NKV, 1, D.CB, D.HD)
            scores = torch.matmul(q, k.transpose(2, 3)) * (D.HD ** -0.5)
            attn = torch.matmul(torch.softmax(scores, dim=-1), v).view(D.NH, D.CB, D.HD).permute(1, 0, 2)
            gate = sigmoid_engine(hn @ w[p + 'self_attn.gate_proj.weight'].T).view(D.CB, D.NH, 1)
            x = x + project(w[p + 'self_attn.o_proj.weight'], (attn * gate).reshape(D.CB, D.NH * D.HD))
            f = D.rms(x, w[p + 'ffn_norm.weight'])
            x = x + project(w[p + 'feed_forward.w2.weight'],
                            silu_engine(f @ w[p + 'feed_forward.w1.weight'].T)
                            * (f @ w[p + 'feed_forward.w3.weight'].T))
        x = D.rms(x, w[pre + 'model.embedding_norm.weight'])
        logits = []
        for i in range(D.CB):
            emb = w[pre + 'depth_embeddings.%d.embedding.weight' % i]
            logits.append(D.rms(x[i:i + 1], w[pre + 'depth_embeddings.%d.embedding_norm.weight' % i]) @ emb.T)
        return torch.cat(logits, dim=0)


class Codec(torch.nn.Module):
    """latents (1, frames, 48) -> spectrum (1, frames * 6, 1282).  Causal, so padding after the real
    frames does not change the rows before it."""

    def __init__(self, w, frames=CODEC_FRAMES):
        super().__init__()
        self.w = w
        self.seq = frames * C.FRAME_STACK
        self.head_dim = C.H // 8
        self.cos, self.sin = C.rope_tables(self.seq, self.head_dim, 1e6)
        index = torch.arange(self.seq)
        distance = index[:, None] - index[None, :]
        self.mask = torch.where((distance < 0) | (distance >= C.WINDOW), -1e4, 0.0)

    def forward(self, latents):
        w, n, hd = self.w, self.seq, self.head_dim
        lat = latents[0] @ w['quantizer.project_out.weight'].T + w['quantizer.project_out.bias']
        x = lat @ w['from_quantizer.weight'].T + w['from_quantizer.bias']
        x = x.repeat_interleave(C.FRAME_STACK, dim=0)
        cos, sin = self.cos[:, None, :], self.sin[:, None, :]
        for index, kind in enumerate(CODEC_LAYERS):
            p = 'decoder.lfm.layers.%d.' % index
            hn = C.rms(x, w[p + 'operator_norm.weight'])
            if kind == 'conv':
                proj = hn @ w[p + 'conv.in_proj.weight'].T
                b, c, gate = proj[:, :C.H], proj[:, C.H:2 * C.H], proj[:, 2 * C.H:]
                padded = F.pad((b * gate).T[None], (6, 0))
                conv = F.conv1d(padded, w[p + 'conv.conv.weight'], groups=C.H)[0].T
                y = project(w[p + 'conv.out_proj.weight'], c * conv)
            else:
                a = p + 'self_attn.'
                nh = w[a + 'q_proj.weight'].shape[0] // hd
                nkv = w[a + 'k_proj.weight'].shape[0] // hd
                rep = nh // nkv
                q = C.rms((hn @ w[a + 'q_proj.weight'].T).view(n, nh, hd), w[a + 'q_norm.weight'])
                k = C.rms((hn @ w[a + 'k_proj.weight'].T).view(n, nkv, hd), w[a + 'k_norm.weight'])
                v = (hn @ w[a + 'v_proj.weight'].T).view(n, nkv, hd)
                q = q * cos + C.rotate_half(q) * sin
                k = k * cos + C.rotate_half(k) * sin
                q = q.permute(1, 0, 2).view(nkv, rep, n, hd)
                k = k.permute(1, 0, 2).view(nkv, 1, n, hd)
                v = v.permute(1, 0, 2).view(nkv, 1, n, hd)
                scores = torch.matmul(q, k.transpose(2, 3)) * (hd ** -0.5) + self.mask
                attn = torch.matmul(torch.softmax(scores, dim=-1), v).view(nh, n, hd).permute(1, 0, 2)
                gate = sigmoid_engine(hn @ w[a + 'gate_proj.weight'].T).view(n, nh, 1)
                y = project(w[a + 'o_proj.weight'], (attn * gate).reshape(n, nh * hd))
            x = x + C.rms(y, w[p + 'post_operator_norm.weight'])
            f = C.rms(x, w[p + 'ffn_norm.weight'])
            mlp = project(w[p + 'feed_forward.w2.weight'],
                          silu_engine(f @ w[p + 'feed_forward.w1.weight'].T)
                          * (f @ w[p + 'feed_forward.w3.weight'].T))
            x = x + C.rms(mlp, w[p + 'post_ffn_norm.weight'])
        x = C.rms(x, w['decoder.lfm.embedding_norm.weight'])
        return (x @ w['decoder.lin.weight'].T + w['decoder.lin.bias'])[None]


def load_depth():
    with safe_open(D.P, framework='pt') as f:
        return {k: f.get_tensor(k).float() for k in f.keys() if k.startswith(D.PRE)}


def load_codec():
    with safe_open(C.CODEC, framework='pt') as f:
        return {k: f.get_tensor(k).float() for k in f.keys()}


def convert(model, inputs, output, path):
    with torch.no_grad():
        traced = torch.jit.trace(model.eval(), tuple(torch.zeros(*s) for _, s in inputs))
    ct.convert(traced, inputs=[ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in inputs],
               outputs=[ct.TensorType(name=output)], convert_to='mlprogram',
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    print({'saved': path}, flush=True)


if __name__ == '__main__':
    out = sys.argv[1]
    depth, codec = load_depth(), load_codec()
    convert(Depth(depth), [('hidden', (1, 2048)), ('slots', (1, D.CB, D.H))], 'logits',
            os.path.join(out, 'depth_decoder.mlpackage'))
    convert(Codec(codec), [('latents', (1, CODEC_FRAMES, C.CB * C.FSQ_DIM))], 'spectrum',
            os.path.join(out, 'codec.mlpackage'))
    with safe_open(D.P, framework='pt') as f:
        book = f.get_tensor('model.codebook_embedding.weight').float().numpy()
    tables = {'embedding_%d' % i: depth[D.PRE + 'depth_embeddings.%d.embedding.weight' % i].numpy()
              for i in range(D.CB)}
    np.savez(os.path.join(out, 'speech_tables.npz'), mask=depth[D.PRE + 'mask_embedding'].numpy(),
             book=book, window=codec['decoder.istft.window'].numpy(), **tables)
    print('done', flush=True)
