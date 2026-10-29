"""Build exact-78k attention: per full-attention layer, a projection program, N tile
programs (no weights, KV in state) and an output program that applies o_proj + MLP.

Tiles return the three flash-attention numbers (max, sum, weighted V) and the runtime
combines them in float64, so the result is the exact softmax over the whole context --
no approximation, no dropped positions.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import MODEL, SOURCE

H, NH, NKV, D, K = 2048, 16, 4, 128, 7
TILE = int(os.environ.get("LFM3_TILE", "16384"))
OUT = os.path.join(os.path.dirname(SOURCE), 'model-tiled')
BIN = os.path.join(SOURCE, 'full_layer_weights.bin')
INDEX = '/tmp/full_layer_index.json'


def load_tensor(handle_bytes, start, info):
    """The checkpoint stores BF16, so decode that explicitly."""
    begin, end = info['data_offsets']
    raw = handle_bytes[begin - start:end - start]
    dtype = torch.bfloat16 if info['dtype'] == 'BF16' else torch.float32
    array = torch.frombuffer(bytearray(raw), dtype=dtype).float()
    return array.reshape(info['shape']).clone()


class Proj(torch.nn.Module):
    """pre-norm, q/k/v, and the attention gate, for one full-attention layer."""

    def __init__(self, t):
        super().__init__()
        self.pre = Rms(t['operator_norm.weight'])
        self.q = torch.nn.Conv2d(H, NH * D, 1, bias=False)
        self.k = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.v = torch.nn.Conv2d(H, NKV * D, 1, bias=False)
        self.gate = torch.nn.Conv2d(H, NH, 1, bias=False)
        self.q.weight.data = t['self_attn.q_proj.weight'].view(NH * D, H, 1, 1).clone()
        self.k.weight.data = t['self_attn.k_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.v.weight.data = t['self_attn.v_proj.weight'].view(NKV * D, H, 1, 1).clone()
        self.gate.weight.data = t['self_attn.gate_proj.weight'].view(NH, H, 1, 1).clone()
        self.q_norm = torch.nn.Parameter(t['self_attn.q_norm.weight'].clone())
        self.k_norm = torch.nn.Parameter(t['self_attn.k_norm.weight'].clone())

    def forward(self, x, cos, sin):
        h = self.pre(x)
        q = self.q(h).view(1, NH, D)
        k = self.k(h).view(1, NKV, D)
        v = self.v(h)
        q = head_norm(q, self.q_norm)
        k = head_norm(k, self.k_norm)
        q = rotate(q, cos, sin)
        k = rotate(k, cos, sin)
        return (q.view(1, NKV, 4, D), k.view(1, NKV * D, 1, 1), v,
                torch.sigmoid(self.gate(h)))


class Tile(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        # identity conv: the planner will not place a graph with no weights in it
        self.eye = torch.nn.Conv2d(NKV * D, NKV * D, 1, bias=False)
        with torch.no_grad():
            self.eye.weight.copy_(torch.eye(NKV * D).view(NKV * D, NKV * D, 1, 1))
        self.register_buffer('cache', torch.zeros(2 * NKV, 1, rows, D), persistent=False)

    def forward(self, q, k, v, slot, mask):
        k = self.eye(k).view(1, NKV, 1, D)
        v = self.eye(v).view(1, NKV, 1, D)
        self.cache[0:NKV, :, slot:slot + 1, :] = k.view(NKV, 1, 1, D)
        self.cache[NKV:2 * NKV, :, slot:slot + 1, :] = v.view(NKV, 1, 1, D)
        key = self.cache[0:NKV].view(1, NKV, self.rows, D).transpose(-1, -2)
        val = self.cache[NKV:2 * NKV].view(1, NKV, self.rows, D)
        scores = torch.matmul(q.view(1, NKV, 4, D), key) * 0.0884 + mask
        best = scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(scores - best)
        return (best.reshape(1, NH), weights.sum(dim=-1).reshape(1, NH),
                torch.matmul(weights, val).reshape(1, NH * D))


class Out(torch.nn.Module):
    """gate, o_proj, post-norm, MLP -- everything after attention is combined."""

    def __init__(self, t):
        super().__init__()
        self.post = Rms(t['post_operator_norm.weight'])
        self.ffn_norm = Rms(t['ffn_norm.weight'])
        self.ffn_post = Rms(t['post_ffn_norm.weight'])
        self.o = torch.nn.Conv2d(H, H, 1, bias=False)
        self.o.weight.data = t['self_attn.o_proj.weight'].view(H, H, 1, 1).clone()
        self.w1 = torch.nn.Conv2d(H, 7680, 1, bias=False)
        self.w3 = torch.nn.Conv2d(H, 7680, 1, bias=False)
        self.w2 = torch.nn.Conv2d(7680, H, 1, bias=False)
        self.w1.weight.data = t['feed_forward.w1.weight'].view(7680, H, 1, 1).clone()
        self.w3.weight.data = t['feed_forward.w3.weight'].view(7680, H, 1, 1).clone()
        self.w2.weight.data = t['feed_forward.w2.weight'].view(H, 7680, 1, 1).clone()

    def forward(self, x, attn, gate):
        scaled = attn.view(1, NH, D) * gate.view(1, NH, 1)
        y = self.o(scaled.reshape(1, NH * D, 1, 1))
        h = self.post(y) + x
        return self.ffn_post(self.w2(torch.nn.functional.silu(self.w1(self.ffn_norm(h)))
                                     * self.w3(self.ffn_norm(h)))) + h


def rms(z, w, eps=1e-5):
    return z * torch.rsqrt(z.pow(2).mean(dim=1, keepdim=True) + eps) * w.view(1, -1, 1, 1)


def head_norm(z, w, eps=1e-5):
    return z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + eps) * w.view(1, 1, -1)


def rotate(z, cos, sin):
    half = z.shape[-1] // 2
    flipped = torch.cat([-z[..., half:], z[..., :half]], dim=-1)
    return z * cos.view(1, 1, D) + flipped * sin.view(1, 1, D)


class Rms(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w.clone())

    def forward(self, x):
        return rms(x, self.w)


def save(model, inputs, states, path):
    example = [torch.zeros(*s, dtype=torch.int32 if s == (1,) else torch.float16) for s in inputs]
    traced = torch.jit.trace(model.half().eval(), tuple(example))
    ct.convert(traced,
               inputs=[ct.TensorType(name='in%d' % i, shape=tuple(s),
                                     dtype=np.int32 if s == (1,) else np.float16)
                       for i, s in enumerate(inputs)],
               states=states, convert_to='mlprogram',
               compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.iOS18,
               compute_units=ct.ComputeUnit.CPU_AND_NE).save(path)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    index = json.load(open(INDEX))
    start = min(i['data_offsets'][0] for i in index.values())
    blob = open(BIN, 'rb').read()
    types = json.load(open(os.path.join(MODEL, 'layer_types.json')))
    full = [i for i, kind in enumerate(types) if kind == 'full_attention']
    print({'tile': TILE, 'full_layers': full}, flush=True)
    spans = []
    for key, info in sorted(index.items()):
        index_number = int(key.split('layers.')[1].split('.')[0])
        spans.append((index_number, key.split('layers.%d.' % index_number)[1], load_tensor(blob, start, info)))
    by_layer = {}
    for index_number, name, tensor in spans:
        by_layer.setdefault(index_number, {})[name] = tensor
    for layer in full:
        tensors = by_layer[layer]
        proj = os.path.join(OUT, 'L%d_proj.mlpackage' % layer)
        if not os.path.isdir(proj):
            save(Proj(tensors), [(1, H, 1, 1), (D,), (D,)], [], proj)
        out = os.path.join(OUT, 'L%d_out.mlpackage' % layer)
        if not os.path.isdir(out):
            save(Out(tensors), [(1, H, 1, 1), (1, NH * D), (1, NH, 1, 1)], [], out)
        for tile_index in range((78000 + TILE - 1) // TILE):
            path = os.path.join(OUT, 'L%d_tile%d.mlpackage' % (layer, tile_index))
            if os.path.isdir(path):
                continue
            save(Tile(TILE),
                 [(1, NKV, 4, D), (1, NKV * D, 1, 1), (1, NKV * D, 1, 1), (1,), (1, NKV, 4, TILE)],
                 [ct.StateType(name='cache', wrapped_type=ct.TensorType(
                     shape=(2 * NKV, 1, TILE, D), dtype=np.float16))], path)
        print('built layer', layer, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()