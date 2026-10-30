"""Execution modes: one context length each, with its own measured ceiling.

The full-attention caches inside the compiled programs have a fixed number of rows,
so how much context a run can hold is decided when the model is built, not when it
runs.  Two modes cover the useful cases.

28k  every token stays resident, so attention sees the whole conversation.  This is
     the exact mode: nothing is ever dropped.

78k  the same caches are used as rings, so a run can continue past 28,672 tokens.
     The model then attends to the most recent 28,672 tokens and earlier ones are
     overwritten.  It is not an approximation of exact attention -- it is ordinary
     sliding-window attention, and the window is the whole memory.

The ceilings below are decode speed measured on this machine with the shipped LUT6
build, at a 64-token context (28k: 30.6 tok/s, 78k: 30.9).  Decode slows as the full
cache fills, so treat them as the best case.  The two modes diverge as context grows:
28k pays more for every token once its cache is full, while 78k stops growing at its
window and stays flat from there.
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
    '78k': Mode(
        name='78k',
        context=78042,
        window_rows=28672,
        exact=False,
        note='Ring caches: attends to the most recent 28,672 tokens; older context is '
             'overwritten and forgotten. Cost per token stops growing once the window is full.',
        ceiling_tps=30.9,
    ),
}

DEFAULT_MODE = '28k'


_module: dict = {}


def resolve(name: Optional[str] = None) -> Mode:
    """Look up a mode by name, falling back to LFM3_MODE and then the default."""
    key = (name or os.environ.get('LFM3_MODE') or DEFAULT_MODE).strip().lower()
    if key not in MODES:
        raise SystemExit('unknown mode %r; choose one of: %s'
                         % (key, ', '.join(sorted(MODES))))
    return MODES[key]


def current() -> Mode:
    """The mode this process is running in (resolved once, so it cannot drift)."""
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