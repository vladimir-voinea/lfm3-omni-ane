"""Execution modes.

Context length is fixed when the model is built, because the full-attention caches hold a
fixed number of rows inside the compiled programs.  The shipped build holds 28,672.

There is one mode.  A longer context is not available by relabelling anything: the caches
cannot hold more rows, so a run past 28,672 tokens can only be a window over the most recent
tokens, which is not the same thing as a longer context and is not offered as one.

`LFM3_WINDOW=<rows>` still exists as a probe hook for experiments.  It turns the caches into
rings and forgets old tokens; it is not a supported mode.
"""
from __future__ import annotations

import os
from typing import NamedTuple, Optional


class Mode(NamedTuple):
    name: str
    context: int          # longest prompt + completion the mode accepts
    window_rows: int      # 0 keeps the caches exact
    exact: bool
    note: str
    ceiling_tps: float    # measured decode ceiling at short context


MODES = {
    '28k': Mode(
        name='28k',
        context=28672,
        window_rows=0,
        exact=True,
        note='Every token stays in the attention cache; nothing is forgotten.',
        ceiling_tps=30.6,
    ),
}

DEFAULT_MODE = '28k'

_module: dict = {}


def resolve(name: Optional[str] = None) -> Mode:
    """Look up a mode by name, falling back to LFM3_MODE and then the default."""
    key = (name or os.environ.get('LFM3_MODE') or DEFAULT_MODE).strip().lower()
    if key not in MODES:
        raise SystemExit('unknown mode %r; available: %s (a longer context is not a mode, it '
                         'needs a build with larger caches)'
                         % (key, ', '.join(sorted(MODES))))
    return MODES[key]


def current() -> Mode:
    if 'mode' not in _module:
        _module['mode'] = resolve()
    return _module['mode']


def summary() -> dict:
    mode = current()
    return {
        'mode': mode.name,
        'context_length': mode.context,
        'exact': mode.exact,
        'ceiling_tps': mode.ceiling_tps,
        'note': mode.note,
    }
