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

D, H, NKV, NH = 128, 2048, 4, 16
K_HIST = 6


def rope_frequencies(theta, partial):
    """The inverse frequencies build_model uses, so positions can be rotated on the fly.

    A table long enough for an unbounded window would be 38 MB of numpy for no reason: the
    rotation for any absolute position is four multiplications.
    """
    inv = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float64) / D))
    angles = int(partial * D) // 2
    if angles < D // 2:
        inv = np.concatenate([inv[:angles], np.zeros(D // 2 - angles)])
    return inv


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
        # lengths the tables cover; beyond them the runtime computes the rotation directly
        self.rope_rows = int(rope['full_cos'].shape[0])
        self.full_inv = rope_frequencies(1e6, 0.5)
        self.slide_inv = rope_frequencies(1e4, 1.0)
        # 0 means exact: no ring, and a position past the cache is an error rather than an
        # overwrite.  A window turns the full-attention caches into rings, which is what makes
        # contexts beyond the cache size usable at all (exact inside the window).
        self.window = int(os.environ.get('LFM3_WINDOW', '0'))
        if self.window:
            print('WARNING: LFM3_WINDOW=%d -- the full-attention caches are rings, so the model'
                  % self.window)
            print('         attends to the last %d tokens only. Past that, earlier context is'
                  % self.window)
            print('         overwritten and forgotten. For exact attention leave LFM3_WINDOW unset.')
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

    def _plain_session(self) -> Session:
        states, hists = [], []
        for index, stateful in enumerate(self.stateful):
            if stateful:
                states.append(self.programs[index].make_state())
                hists.append(None)
            else:
                states.append(None)
                hists.append(np.zeros((1, H, 1, K_HIST), np.float16))
        return Session(states, hists, self.full_rows, self.slide_rows)

    def new_session(self) -> Session:
        if os.environ.get('LFM3_TILED', '0') == '1':
            if not self._load_tiled():
                raise RuntimeError('LFM3_TILED=1 but the tiled build is not in %s'
                                   % getattr(self, 'tiled_dir', 'models/model-tiled'))
            session = self._plain_session()
            session.tile_states = self.new_tile_states()
            return session
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
        elif key == 'full' and self.window:
            # past the end of a ring every row holds a live position, so nothing is masked
            mask.fill(0)
        return mask

    def _rotation(self, kind, position):
        """cos/sin for one absolute position: from the table when it reaches, else computed."""
        sliding = kind == 'sliding_attention'
        if position < self.rope_rows:
            prefix = 'slide' if sliding else 'full'
            return self.rope[prefix + '_cos'][position], self.rope[prefix + '_sin'][position]
        inv = self.slide_inv if sliding else self.full_inv
        angles = position * inv
        turn = np.concatenate([angles, angles])
        return np.cos(turn).astype(np.float16), np.sin(turn).astype(np.float16)

# ---- batched prefill -------------------------------------------------------------
    # The shipped programs take one position per call, so a 4k prompt is 4k dispatches.
    # A second program set with a sequence axis of T moves T positions per dispatch; the
    # two sets keep their own KV caches, so the handoff copies the batched cache into the
    # single-position states once, after the prefill.

    def _load_batch(self):
        if getattr(self, 'batch_programs', None):
            return True
        directory = os.environ.get('LFM3_BATCH_DIR',
                                   os.path.join(os.path.dirname(self.dir), 'model-28k-b8'))
        self.batch_dir = directory
        if not os.path.isdir(directory):
            return False
        self.batch_T = int(os.environ.get('LFM3_BATCH_T', '8'))
        programs = []
        for index, kind in enumerate(self.layer_types):
            if kind == 'conv':
                name = 'layer%d_conv.mlpackage' % index
            else:
                name = 'layer%d_%s.mlpackage' % (
                    index, 'slide' if kind == 'sliding_attention' else 'full')
            path = os.path.join(directory, name)
            if not os.path.isdir(path):
                return False
            programs.append(ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE))
        self.batch_programs = programs
        self.batch_masks = {
            'full': np.full((1, NKV, self.batch_T * 4, self.full_rows), -1e4, np.float16),
            'slide': np.full((1, NKV, self.batch_T * 4, self.slide_rows), -1e4, np.float16),
        }
        return True

    def _chunk_mask(self, kind, position, count):
        """Rows ordered as q.view(1, NKV, T*4, D) flattens them: head-major, position-minor.

        Getting this backwards produced a plausible but wrong answer (relative difference 1.07
        against the serial path), which is exactly the kind of bug a crash would have saved me
        from.
        """
        key = 'slide' if kind == 'sliding_attention' else 'full'
        mask = self.batch_masks[key]
        rows = mask.shape[-1]
        mask.fill(-1e4)
        for head in range(4):
            for step in range(count):
                visible = min(position + step + 1, rows)
                mask[:, :, head * count + step, :visible] = 0
        return mask

    def prefill_batch(self, session, token_ids):
        """Run the leading whole multiple of T tokens through the batched programs."""
        if not self._load_batch():
            return 0
        T = self.batch_T
        count = (len(token_ids) // T) * T
        if count == 0:
            return 0
        if getattr(session, 'batch_states', None) is None:
            session.batch_states = [program.make_state() if stateful else None
                                    for program, stateful in zip(self.batch_programs, self.stateful)]
            session.batch_hists = [np.zeros((1, H, K_HIST, 1), np.float16) if not stateful else None
                                   for stateful in self.stateful]
        rows = np.stack([self.emb[int(t)] for t in token_ids[:count]]).astype(np.float16)
        hidden_all = rows.reshape(1, count, H).transpose(0, 2, 1)[..., None]
        start = session.position
        last = None
        for offset in range(0, count, T):
            hidden = np.ascontiguousarray(hidden_all[..., offset:offset + T, :])
            position = start + offset
            # one mask per layer type per chunk, not one per layer: rebuilding the 28,672-wide
            # array eleven times per chunk was most of the cost
            chunk_masks = {}
            for index, kind in enumerate(self.layer_types):
                program = self.batch_programs[index]
                if kind == 'conv':
                    out = program.predict({'in0': hidden, 'in1': session.batch_hists[index]})
                    values = [np.asarray(v) for v in out.values()]
                    hidden = next(v for v in values if v.shape[2] == T)
                    session.batch_hists[index] = next(v for v in values if v.shape[2] != T)
                else:
                    sliding = kind == 'sliding_attention'
                    cos = self.rope['slide_cos' if sliding else 'full_cos'][position:position + T]
                    sin = self.rope['slide_sin' if sliding else 'full_sin'][position:position + T]
                    if kind not in chunk_masks:
                        chunk_masks[kind] = self._chunk_mask(kind, position, T)
                    out = program.predict(
                        {'in0': hidden,
                         'in1': np.array([position], np.int32),
                         'in2': cos.reshape(T, D),
                         'in3': sin.reshape(T, D),
                         'in4': chunk_masks[kind]},
                        state=session.batch_states[index])
                    hidden = np.asarray(list(out.values())[0])
            last = hidden
        if os.environ.get('LFM3_NO_SYNC', '0') != '1':
            self._sync_batch_to_session(session)
        session.position = start + count
        # the caller wants the hidden state of the LAST position, not the whole chunk
        session.hidden = np.ascontiguousarray(last[..., -1:, :])
        return count

    def _sync_batch_to_session(self, session):
        """Copy the batched KV caches into the single-position states, once per prefill."""
        for index, kind in enumerate(self.layer_types):
            if kind == 'conv':
                hist = session.batch_hists[index]                  # (1, H, K-1, 1)
                session.hists[index] = np.ascontiguousarray(hist.transpose(0, 1, 3, 2))
            else:
                source = session.batch_states[index]
                target = session.states[index]
                target.cache[:] = source.cache[:]

    # ---- exact tiled attention -------------------------------------------------------
    # The shipped full-attention programs hold one 28,672-row cache and cap the context
    # there.  Here each full-attention layer is three programs: a projection, N tiles that
    # hold KV slices with no weights of their own, and an output program.  The tiles return
    # the three flash-attention numbers and the host combines them in float64, so the
    # softmax is exact over the whole context.

    def _load_tiled(self):
        if getattr(self, 'tiled_programs', None):
            return True
        candidates = [os.environ.get('LFM3_TILED_DIR'),
                      os.path.join(os.path.dirname(self.dir), 'model-tiled'),
                      os.path.join(os.path.dirname(self.dir), 'models', 'model-tiled')]
        directory = next((path for path in candidates if path and os.path.isdir(path)), '')
        if not os.path.isdir(directory):
            return False
        self.tiled_dir = directory
        self.tile_rows = int(os.environ.get('LFM3_TILE', '16384'))
        self.tiled_layers = [i for i, kind in enumerate(self.layer_types)
                             if kind == 'full_attention']
        self.tiled_programs = {}
        for layer in self.tiled_layers:
            proj = os.path.join(directory, 'L%d_proj.mlpackage' % layer)
            out = os.path.join(directory, 'L%d_out.mlpackage' % layer)
            if not (os.path.isdir(proj) and os.path.isdir(out)):
                return False
            proj_model = ct.models.MLModel(proj, compute_units=ct.ComputeUnit.CPU_AND_NE)
            entry = {'proj': proj_model,
                     'proj_outputs': [o.name for o in
                                      proj_model.get_spec().description.output],
                     'out': ct.models.MLModel(out, compute_units=ct.ComputeUnit.CPU_AND_NE),
                     'tiles': []}
            index = 0
            while True:
                path = os.path.join(directory, 'L%d_tile%d.mlpackage' % (layer, index))
                if not os.path.isdir(path):
                    break
                program = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
                # dict order out of predict() is not the declared order, so keep the names
                declared = [output.name for output in
                            program.get_spec().description.output]
                entry.setdefault('outputs', declared)
                entry['tiles'].append(program)
                index += 1
            self.tiled_programs[layer] = entry
        return True

    def new_tile_states(self):
        states = {}
        for layer, entry in self.tiled_programs.items():
            states[layer] = [tile.make_state() for tile in entry['tiles']]
        return states

    def _tild_mask(self, slot, is_diagonal):
        mask = np.zeros((1, NKV, 4, self.tile_rows), np.float16)
        if is_diagonal:
            mask[..., slot + 1:] = -1e4
        return mask

    def _run_tiled_attention(self, session, layer, x, cos, sin, position):
        entry = self.tiled_programs[layer]
        result = entry['proj'].predict({'in0': x,
                                        'in1': cos.reshape(D),
                                        'in2': sin.reshape(D)})
        # declared order: q, k, v, gate -- k and v share a shape, so names, not sizes
        names = entry['proj_outputs']
        q, k, v, gate = (np.asarray(result[name]) for name in names[:4])
        slot = position % self.tile_rows
        owners = position // self.tile_rows
        best = None
        weight = None
        acc = None
        for index, tile in enumerate(entry['tiles']):
            if index > owners:
                break
            out = tile.predict({'in0': q, 'in1': k, 'in2': v,
                                'in3': np.array([slot], np.int32),
                                'in4': self._tild_mask(slot, index == owners)},
                               state=session.tile_states[layer][index])
            # declared order is (tile max, sum of exp, weighted V)
            names = entry['outputs']
            tile_max = np.asarray(out[names[0]], np.float64).reshape(-1)
            tile_sum = np.asarray(out[names[1]], np.float64).reshape(-1)
            tile_acc = np.asarray(out[names[2]], np.float64).reshape(NH, D)
            if best is None:
                best, weight, acc = tile_max, tile_sum, tile_acc
            else:
                merged = np.maximum(best, tile_max)
                left = np.exp(best - merged)
                right = np.exp(tile_max - merged)
                weight = weight * left + tile_sum * right
                acc = acc * left[:, None] + tile_acc * right[:, None]
                best = merged
        attention = (acc / weight[:, None]).astype(np.float16).reshape(1, NH * D)
        out = entry['out'].predict({'in0': x, 'in1': attention,
                                    'in2': gate.reshape(1, NH, 1, 1).astype(np.float16)})
        return np.asarray(list(out.values())[0])

    def _run_position(self, session: Session, hidden) -> None:
        """Run one position through the 24 layers."""
        trace = getattr(self, 'trace', None)
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
            elif kind == 'full_attention' and getattr(self, 'tiled_programs', None):
                cos, sin = self._rotation(kind, session.position)
                hidden = self._run_tiled_attention(session, index, hidden, cos, sin,
                                                   session.position)
            else:
                slide = kind == 'sliding_attention'
                # the rotation is by ABSOLUTE position, so the tables must cover the whole
                # context; the sliding cache itself is a ring, so the write slot wraps
                cos, sin = self._rotation(kind, session.position)
                if slide:
                    slot = session.position % self.slide_rows
                elif self.window:
                    slot = session.position % self.full_rows
                else:
                    slot = session.position
                out = program.predict(
                    {'in0': hidden,
                     'in1': np.array([slot], np.int32),
                     'in2': cos.reshape(D),
                     'in3': sin.reshape(D),
                     'in4': self.mask(session, kind, session.position)},
                    state=session.states[index])
                hidden = np.asarray(list(out.values())[0])
            if trace is not None:
                trace.append(np.asarray(hidden).copy())
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