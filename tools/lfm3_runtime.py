# Runtime with sessions: the 26 programs are loaded ONCE; a session is a set of
# MLStates plus conv histories and a position over those same models.
from __future__ import annotations

import json
import os

import numpy as np
import coremltools as ct

# Each load asks Core ML to compile the program, and it leaves an .mlmodelc behind in the
# user temp directory. A few hundred loads was tens of gigabytes, and TMPDIR does not
# redirect it, so the runtime deletes whatever it creates.
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'disk-tmp')
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ['TMPDIR'] = _CACHE_DIR

import atexit
import glob
import shutil
import tempfile as _tempfile


def _compiled_dirs():
    return set(glob.glob(os.path.join(_tempfile.gettempdir(), '*.mlmodelc')))


def _clean_compiled(before):
    for path in _compiled_dirs() - before:
        shutil.rmtree(path, ignore_errors=True)

D, H, NKV = 128, 2048, 4


class Session:
    def __init__(self, states, hists, full_rows, slide_rows):
        self.states = states
        self.hists = hists
        self.position = 0
        self.tokens = []
        # one mask per layer type, allocated once and revealed one position at a time:
        # rebuilding a 28,672-entry array for every attention layer of every token was
        # pure waste
        self.masks = {
            'full': np.full((1, NKV, 1, full_rows), -1e4, np.float16),
            'slide': np.full((1, NKV, 1, slide_rows), -1e4, np.float16),
        }


class Runtime:
    def __init__(self, model_dir: str, full_rows: int = 28672, slide_rows: int = 256):
        self.dir = model_dir
        _before = _compiled_dirs()
        atexit.register(_clean_compiled, _before)
        self.layer_types = json.load(open(os.path.join(model_dir, 'layer_types.json')))
        rope = np.load(os.path.join(model_dir, 'rope.npz'))
        self.rope = {k: rope[k] for k in ('full_cos', 'full_sin', 'slide_cos', 'slide_sin')}
        self.emb = np.load(os.path.join(model_dir, 'embed_tokens.npy'), mmap_mode='r')
        self.full_rows, self.slide_rows = full_rows, slide_rows

        self.programs = []
        self.stateful = []
        for index, kind in enumerate(self.layer_types):
            if kind == 'conv':
                path = os.path.join(model_dir, 'layer%d_conv.mlpackage' % index)
            else:
                tag = 'slide' if kind == 'sliding_attention' else 'full'
                path = os.path.join(model_dir, 'layer%d_%s.mlpackage' % (index, tag))
            program = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.programs.append(program)
            self.stateful.append(kind != 'conv')
        self.norm = ct.models.MLModel(os.path.join(model_dir, 'final_norm.mlpackage'),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE)
        # a single 128,000-way matmul is not placed on the engine; two chunks of 64,000 are
        parts = [n for n in ('lm_head_a.mlpackage', 'lm_head_b.mlpackage')
                 if os.path.exists(os.path.join(model_dir, n))]
        self.head_parts = [
            ct.models.MLModel(os.path.join(model_dir, n), compute_units=ct.ComputeUnit.CPU_AND_NE)
            for n in parts] if os.environ.get('LFM3_HEAD_PARTS', '1') == '1' and len(parts) == 2 else []
        head_name = os.environ.get('LFM3_HEAD', 'lm_head.mlpackage')
        self.head = ct.models.MLModel(os.path.join(model_dir, head_name),
                                      compute_units=ct.ComputeUnit.CPU_AND_NE)

    def new_session(self) -> Session:
        states, hists = [], []
        for index, stateful in enumerate(self.stateful):
            if stateful:
                states.append(self.programs[index].make_state())
                hists.append(None)
            else:
                states.append(None)
                hists.append(np.zeros((1, H, 1, 6), np.float16))
        return Session(states, hists, self.full_rows, self.slide_rows)

    def mask(self, session: Session, kind: str, position: int) -> np.ndarray:
        key = 'slide' if kind == 'sliding_attention' else 'full'
        mask = session.masks[key]
        rows = mask.shape[-1]
        if position < rows:
            mask[:, :, :, position] = 0
        return mask

    def _run_position(self, session: Session, hidden) -> None:
        """Run one position through the 24 layers."""
        for index, kind in enumerate(self.layer_types):
            program = self.programs[index]
            if kind == 'conv':
                out = program.predict({'in0': hidden, 'in1': session.hists[index]})
                hidden = None
                for value in out.values():
                    arr = np.asarray(value)
                    if arr.shape == (1, H, 1, 1):
                        hidden = arr
                    else:
                        session.hists[index] = arr
            else:
                slide = kind == 'sliding_attention'
                cos = self.rope['slide_cos' if slide else 'full_cos'][session.position]
                sin = self.rope['slide_sin' if slide else 'full_sin'][session.position]
                out = program.predict(
                    {'in0': hidden,
                     'in1': np.array([session.position], np.int32),
                     'in2': cos.reshape(D),
                     'in3': sin.reshape(D),
                     'in4': self.mask(session, kind, session.position)},
                    state=session.states[index])
                hidden = np.asarray(list(out.values())[0])
        session.position += 1
        session.hidden = hidden

    def step_hidden(self, session: Session, hidden) -> None:
        """Feed one position from a ready-made vector instead of an embedding row."""
        self._run_position(session, np.asarray(hidden, np.float16).reshape(1, H, 1, 1))

    def step(self, session: Session, token_ids) -> np.ndarray:
        for token in token_ids:
            self._run_position(session, np.asarray(self.emb[int(token)], np.float16).reshape(1, H, 1, 1))
        logits = session.hidden
        normed = self.norm.predict({'in0': logits})
        ready = np.asarray(list(normed.values())[0], np.float16)
        if self.head_parts:
            out = {'logits': np.concatenate([
                np.asarray(list(part.predict({'in0': ready}).values())[0]).reshape(-1)
                for part in self.head_parts])}
        else:
            out = self.head.predict({'in0': ready})
        return np.asarray(list(out.values())[0]).reshape(-1)