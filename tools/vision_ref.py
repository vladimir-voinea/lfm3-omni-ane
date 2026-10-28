# The vision tower, transcribed from modeling_lfm3_vl.py and run in torch fp32.
# Checked by shape and by finiteness on a synthetic image; the behavioural check
# (a caption that matches the picture) comes after it is converted.
from __future__ import annotations

import numpy as np
import torch
from safetensors import safe_open

P = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/model.safetensors'
H, NH, NKV, HD, FF = 1024, 16, 8, 64, 2560
MERGE, KS, EPS, THETA = 2, 3, 1e-5, 1e6


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rotate_half(x):
    return torch.cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)


def axial_rope(h_idx, w_idx):
    spatial = HD // 2                        # 32
    inv = 1.0 / (THETA ** (torch.arange(0, spatial, 2, dtype=torch.float) / spatial))
    fh = h_idx.float()[:, None] * inv[None, :]
    fw = w_idx.float()[:, None] * inv[None, :]
    hw = torch.cat([fh, fw], dim=-1)          # 32
    ang = torch.cat([hw, hw], dim=-1)         # 64 = head_dim
    return ang.cos(), ang.sin()


def position_ids(h, w, frames=1):
    hi = torch.arange(h).view(h, 1).expand(h, w)
    wi = torch.arange(w).view(1, w).expand(h, w)
    hi = hi.view(h // MERGE, MERGE, w // MERGE, MERGE).permute(0, 2, 1, 3).reshape(-1)
    wi = wi.view(h // MERGE, MERGE, w // MERGE, MERGE).permute(0, 2, 1, 3).reshape(-1)
    return hi.repeat(frames), wi.repeat(frames)


class Vision:
    def __init__(self, w, layer_types):
        self.w = w
        self.layer_types = layer_types

    def conv_block(self, x, i, h, w, frames):
        p = 'model.vision_tower.layers.%d.conv.' % i
        proj = x @ self.w[p + 'in_proj.weight'].T
        b, c, t = proj.chunk(3, dim=-1)
        gated = b * t                                   # (N, H)
        fr = gated.view(frames, h // MERGE, w // MERGE, MERGE, MERGE, -1)
        fr = fr.permute(0, 5, 1, 3, 2, 4).reshape(frames, H, h, w)
        kernel = self.w[p + 'spatial_conv.weight']       # (H, 1, 3, 3)
        fr = torch.nn.functional.conv2d(fr, kernel, padding=KS // 2, groups=H)
        fr = fr.view(frames, H, h // MERGE, MERGE, w // MERGE, MERGE).permute(0, 2, 4, 3, 5, 1)
        conv_out = fr.reshape(-1, H)
        return (c * conv_out) @ self.w[p + 'out_proj.weight'].T

    def attn_block(self, x, i, h, w, frames, cos, sin):
        p = 'model.vision_tower.layers.%d.self_attn.' % i
        n = x.shape[0]
        q = (x @ self.w[p + 'q_proj.weight'].T).view(n, NH, HD)
        k = (x @ self.w[p + 'k_proj.weight'].T).view(n, NKV, HD)
        v = (x @ self.w[p + 'v_proj.weight'].T).view(n, NKV, HD)
        q = rms(q, self.w[p + 'q_layernorm.weight'])
        k = rms(k, self.w[p + 'k_layernorm.weight'])
        q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
        k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
        per = h * w
        out = torch.zeros(n, NH, HD)
        rep = NH // NKV
        for f in range(frames):
            sl = slice(f * per, (f + 1) * per)
            qf = q[sl].permute(1, 0, 2)
            kf = k[sl].permute(1, 0, 2)
            vf = v[sl].permute(1, 0, 2)
            kf = kf.repeat_interleave(rep, dim=0)
            vf = vf.repeat_interleave(rep, dim=0)
            s = torch.matmul(qf, kf.transpose(1, 2)) * (HD ** -0.5)
            out[sl] = torch.matmul(torch.softmax(s, dim=-1), vf).permute(1, 0, 2)
        return out.reshape(n, NH * HD) @ self.w[p + 'out_proj.weight'].T

    def forward(self, pixel_values, h, w, frames=1):
        x = pixel_values @ self.w['model.vision_tower.patchifier.weight'].T
        # the norm sits right after the patchifier, and the returned hidden state is the
        # raw residual stream -- there is no norm at the end
        x = rms(x, self.w['model.vision_tower.embedding_norm.weight'])
        hi, wi = position_ids(h, w, frames)
        cos, sin = axial_rope(hi, wi)
        for i, kind in enumerate(self.layer_types):
            p = 'model.vision_tower.layers.%d.' % i
            hn = rms(x, self.w[p + 'operator_norm.weight'])
            if kind == 'full_attention':
                y = self.attn_block(hn, i, h, w, frames, cos, sin)
            else:
                y = self.conv_block(hn, i, h, w, frames)
            x = x + y
            pre = 'model.vision_tower.layers.%d.feed_forward.' % i
            f = rms(x, self.w[p + 'ffn_norm.weight'])
            x = x + (torch.nn.functional.silu(f @ self.w[pre + 'w1.weight'].T)
                     * (f @ self.w[pre + 'w3.weight'].T)) @ self.w[pre + 'w2.weight'].T
        return x


def projector(features, w):
    pre = 'model.multi_modal_projector.'
    x = torch.nn.functional.layer_norm(features, (H,), w[pre + 'layer_norm.weight'],
                                       w[pre + 'layer_norm.bias'], 1e-5)
    x = x.view(-1, H * 4)                      # merge 2x2 blocks (downsample_factor 2)
    x = torch.nn.functional.gelu(x @ w[pre + 'linear_1.weight'].T + w[pre + 'linear_1.bias'])
    return x @ w[pre + 'linear_2.weight'].T + w[pre + 'linear_2.bias']


if __name__ == '__main__':
    import json
    w = {}
    with safe_open(P, framework='pt') as f:
        for k in f.keys():
            if 'vision_tower' in k or 'multi_modal_projector' in k:
                w[k] = f.get_tensor(k).float()
    types = json.load(open('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/config.json'))
    layer_types = types['vision_config']['layer_types']
    torch.manual_seed(0)
    h, wd = 16, 16                       # 256 patches, 64 after merge
    pixels = torch.randn(h * wd, 3 * 2 * 16 * 16) * 0.5
    tower = Vision(w, layer_types)
    feats = tower.forward(pixels, h, wd)
    proj = projector(feats, w)
    print({'tokens_in': h * wd, 'features': list(feats.shape), 'finite': bool(torch.isfinite(feats).all()),
           'feature_norm': round(float(feats.norm()), 2), 'projected': list(proj.shape),
           'proj_finite': bool(torch.isfinite(proj).all()),
           'proj_norm': round(float(proj.norm()), 2)}, flush=True)