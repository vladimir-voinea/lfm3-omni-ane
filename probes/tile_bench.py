"""What does a token of full attention actually cost at 75k, all on the engine?

18 programs per token (6 layers x 3 tiles), each carrying 4 KV heads over 28,672
rows.  Time the 18 in series, a few passes, and report the median.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import coremltools as ct

sys.path.insert(0, "/Users/vladimir/dev/lfm3-ane/probes")

from tile_all_heads2 import build

ROWS, HIDDEN, KV, REP, DIM = 28672, 2048, 4, 4, 128


def main() -> None:
    pkg = build(ROWS, KV, "/tmp/tile_bench.mlpackage")
    model = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = model.make_state()
    feed = {
        "hidden": np.zeros((1, HIDDEN, 1, 1), np.float16),
        "mask": np.zeros((1, KV, 1, ROWS), np.float16),
        "pos": np.zeros((1,), np.int32),
    }
    model.predict(feed, state=state)

    passes = []
    for _ in range(6):
        t0 = time.perf_counter()
        for _ in range(18):
            model.predict(feed, state=state)
        passes.append((time.perf_counter() - t0) * 1000)

    per_token = statistics.median(passes)
    print({"program_ms": round(per_token / 18, 3),
           "attention_ms_per_token": round(per_token, 1),
           "passes_ms": [round(p, 1) for p in passes],
           "implied_tps_if_only_attention": round(1000 / per_token, 1)}, flush=True)


if __name__ == "__main__":
    main()