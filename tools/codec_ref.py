# Liquid codec decoder, transcribed: FSQ codes -> latents -> 512-wide sandwich-norm LFM
# stack -> magnitude/phase -> inverse STFT -> audio.
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

CODEC = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/audio_tokenizer/model.safetensors'
H, FF, CB, FSQ_DIM = 512, 2304, 12, 4
N_FFT, HOP, WIN, FRAME_STACK = 1280, 320, 1280, 6
WINDOW = 30
LEVELS = [8, 7, 6, 6]
BASIS = [1, 8, 56, 336]
TRIM = (WIN - HOP) // 2
EPS = 1e-5


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)


def quantizer_decode(codes, w):
    """codes (1, 12, T) -> latents (1, T, 256)."""
    idx = codes.transpose(1, 2).unsqueeze(-1).float()          # (1, T, 12, 1)
    levels = torch.tensor(LEVELS, dtype=torch.float32)
    basis = torch.tensor(BASIS, dtype=torch.float32)
    half = torch.div(levels, 2, rounding_mode='floor')
    lat = ((idx // basis) % levels - half) / half             # (1, T, 12, 4)
    lat = lat.reshape(codes.shape[0], codes.shape[2], CB * FSQ_DIM)
    lat = lat @ w['quantizer.project_out.weight'].T + w['quantizer.project_out.bias']
    return lat


def istft(spec, w):
    """spec (bins, frames) complex -> (audio, envelope)."""
    num_frames = spec.shape[-1]
    window = w['decoder.istft.window']
    frames = torch.fft.irfft(spec, N_FFT, dim=0, norm='backward') * window[:, None]
    output = (num_frames - 1) * HOP + WIN
    audio = F.fold(frames.unsqueeze(0), output_size=(1, output),
                   kernel_size=(1, WIN), stride=(1, HOP))[0, 0, 0]
    env = F.fold(window.square().unsqueeze(0).unsqueeze(0).expand(1, num_frames, -1).transpose(1, 2),
                 output_size=(1, output), kernel_size=(1, WIN), stride=(1, HOP))[0, 0, 0]
    return audio, env


def rope_tables(seq, head_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    freqs = torch.arange(seq, dtype=torch.float)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def decode(codes, w, layer_types, theta=1e6, norm_at='end'):
    lat = quantizer_decode(codes, w)
    x = lat @ w['from_quantizer.weight'].T + w['from_quantizer.bias']      # (1, T, 512)
    x = x.repeat_interleave(FRAME_STACK, dim=1)                            # nearest x6
    if norm_at == 'start':
        x = rms(x, w['decoder.lfm.embedding_norm.weight'])
    seq = x.shape[1]
    head_dim = H // 8
    layers = [name for name in layer_types]
    full_cos, full_sin = rope_tables(seq, head_dim, theta)
    for index, kind in enumerate(layer_types):
        p = 'decoder.lfm.layers.%d.' % index
        hn = rms(x, w[p + 'operator_norm.weight'])
        if kind == 'conv':
            proj = hn @ w[p + 'conv.in_proj.weight'].T
            b, c, gate = proj.chunk(3, dim=-1)
            mixed = b * gate
            padded = F.pad(mixed.transpose(1, 2), (6, 0))
            conv = F.conv1d(padded, w[p + 'conv.conv.weight'], groups=H).transpose(1, 2)
            y = (c * conv) @ w[p + 'conv.out_proj.weight'].T
        else:
            nh = w[p + 'self_attn.q_proj.weight'].shape[0] // head_dim
            nkv = w[p + 'self_attn.k_proj.weight'].shape[0] // head_dim
            q = (hn @ w[p + 'self_attn.q_proj.weight'].T).view(1, seq, nh, head_dim)
            k = (hn @ w[p + 'self_attn.k_proj.weight'].T).view(1, seq, nkv, head_dim)
            v = (hn @ w[p + 'self_attn.v_proj.weight'].T).view(1, seq, nkv, head_dim)
            q = rms(q, w[p + 'self_attn.q_norm.weight'])
            k = rms(k, w[p + 'self_attn.k_norm.weight'])
            q = q * full_cos[None, :, None, :] + rotate_half(q) * full_sin[None, :, None, :]
            k = k * full_cos[None, :, None, :] + rotate_half(k) * full_sin[None, :, None, :]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2).repeat_interleave(nh // nkv, dim=1)
            v = v.transpose(1, 2).repeat_interleave(nh // nkv, dim=1)
            scores = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
            # the decoder attends only within `decoder_window` frames (30): a sliding
            # window, not plain causal -- with a 96-frame sequence the difference is total
            idx = torch.arange(seq)
            future = idx[:, None] < idx[None, :]
            far = (idx[:, None] - idx[None, :]) >= WINDOW
            scores = scores.masked_fill((future | far).unsqueeze(0).unsqueeze(0), float('-inf'))
            y = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2)
            y = y.reshape(1, seq, nh * head_dim)
            gate = torch.sigmoid(hn @ w[p + 'self_attn.gate_proj.weight'].T)
            y = y * gate.repeat_interleave(head_dim, dim=-1)
            y = y @ w[p + 'self_attn.o_proj.weight'].T
        x = x + rms(y, w[p + 'post_operator_norm.weight'])
        f = rms(x, w[p + 'ffn_norm.weight'])
        mlp = (F.silu(f @ w[p + 'feed_forward.w1.weight'].T)
               * (f @ w[p + 'feed_forward.w3.weight'].T)) @ w[p + 'feed_forward.w2.weight'].T
        x = x + rms(mlp, w[p + 'post_ffn_norm.weight'])
    if norm_at == 'end':
        x = rms(x, w['decoder.lfm.embedding_norm.weight'])
    spec = (x @ w['decoder.lin.weight'].T + w['decoder.lin.bias'])[0].mT      # (1282, frames)
    log_mag, angle = spec.chunk(2, dim=0)
    audio, env = istft(torch.polar(log_mag.exp(), angle), w)
    audio, env = audio[TRIM:-TRIM], env[TRIM:-TRIM]
    return (audio / env).unsqueeze(0)


if __name__ == '__main__':
    w = {}
    with safe_open(CODEC, framework='pt') as f:
        for k in f.keys():
            w[k] = f.get_tensor(k).float()
    layer_types = ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv']
    codes = torch.from_numpy(np.load('/tmp/codec_codes.npy')).long()
    ref = torch.from_numpy(np.load('/tmp/official_codec_audio.npy')).float().unsqueeze(0)
    for theta, norm_at in ((1e6, 'end'),):
        mine = decode(codes, w, layer_types, theta, norm_at)
        k = min(mine.shape[-1], ref.shape[-1])
        a, b = mine[..., :k], ref[..., :k]
        print({'theta': theta, 'norm_at': norm_at, 'rel': round(float((a - b).norm() / b.norm()), 5),
               'corr': round(float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1]), 5)},
              flush=True)