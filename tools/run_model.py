# A minimal runtime: embed on the host, then 24 programs in order, then the LM head.
# Built to answer one question first -- does a token come out, and how fast.
from __future__ import annotations

import json
import os

from paths import BUILD, PROJECT
import time

import numpy as np
import coremltools as ct

D, H, NKV = 128, 2048, 4


class Lfm3Ane:
    def __init__(self, model_dir: str, full_rows: int = 28672, slide_rows: int = 256):
        self.dir = model_dir
        self.layer_types = json.load(open(os.path.join(model_dir, 'layer_types.json')))
        rope = np.load(os.path.join(model_dir, 'rope.npz'))
        self.full_cos, self.full_sin = rope['full_cos'], rope['full_sin']
        self.slide_cos, self.slide_sin = rope['slide_cos'], rope['slide_sin']
        self.emb = np.load(os.path.join(model_dir, 'embed_tokens.npy'), mmap_mode='r')
        self.full_rows, self.slide_rows = full_rows, slide_rows

        self.programs = []
        self.states = []
        self.hists = []
        for index, kind in enumerate(self.layer_types):
            if kind == 'conv':
                path = os.path.join(model_dir, 'layer%d_conv.mlpackage' % index)
                self.programs.append(ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE))
                self.states.append(None)
                self.hists.append(np.zeros((1, H, 1, 6), np.float16))
            else:
                rows = slide_rows if kind == 'sliding_attention' else full_rows
                tag = 'slide' if kind == 'sliding_attention' else 'full'
                path = os.path.join(model_dir, 'layer%d_%s.mlpackage' % (index, tag))
                program = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
                self.programs.append(program)
                self.states.append(program.make_state())
                self.hists.append(None)
        self.norm = ct.models.MLModel(os.path.join(model_dir, 'final_norm.mlpackage'),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.head = ct.models.MLModel(os.path.join(model_dir, 'lm_head.mlpackage'),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.position = 0

    def _mask(self, kind: str, position: int) -> np.ndarray:
        rows = self.slide_rows if kind == 'sliding_attention' else self.full_rows
        mask = np.full((1, NKV, 1, rows), -1e4, np.float16)
        visible = min(position + 1, rows)
        mask[:, :, :, :visible] = 0
        return mask

    def step(self, token_ids) -> np.ndarray:
        started = time.perf_counter()
        logits = None
        for token in token_ids:
            hidden = np.asarray(self.emb[int(token)], np.float16).reshape(1, H, 1, 1)
            for index, kind in enumerate(self.layer_types):
                program = self.programs[index]
                if kind == 'conv':
                    out = program.predict({'in0': hidden, 'in1': self.hists[index]})
                    hidden = None
                    for name, value in out.items():
                        arr = np.asarray(value)
                        if arr.shape == (1, H, 1, 1):
                            hidden = arr
                        else:
                            self.hists[index] = arr
                else:
                    slide = kind == 'sliding_attention'
                    cos = (self.slide_cos if slide else self.full_cos)[self.position]
                    sin = (self.slide_sin if slide else self.full_sin)[self.position]
                    out = program.predict(
                        {'in0': hidden,
                         'in1': np.array([self.position], np.int32),
                         'in2': cos.reshape(D),
                         'in3': sin.reshape(D),
                         'in4': self._mask(kind, self.position)},
                        state=self.states[index])
                    hidden = np.asarray(list(out.values())[0])
            self.position += 1
            logits = hidden
        normed = self.norm.predict({'in0': logits})
        out = self.head.predict({'in0': np.asarray(list(normed.values())[0], np.float16)})
        return np.asarray(list(out.values())[0]).reshape(-1), time.perf_counter() - started


if __name__ == '__main__':
    import sys
    runtime = Lfm3Ane(BUILD)
    print({'layers': len(runtime.layer_types), 'kinds': {k: runtime.layer_types.count(k)
          for k in set(runtime.layer_types)}}, flush=True)
    logits, seconds = runtime.step([1, 1000, 2000, 3000])
    np.save(os.path.join(PROJECT, 'ane_logits.npy'), logits.astype(np.float32))
    top = np.argsort(logits)[-5:][::-1]
    print({'first_token_seconds': round(seconds, 2),
           'logits_shape': list(logits.shape),
           'top5': top.tolist(),
           'top5_values': [round(float(logits[i]), 3) for i in top]}, flush=True)