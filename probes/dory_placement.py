"""Does a program we KNOW ran on the ANE (Dory attn_full) still show an NE advantage?

If yes, then the LFM3 probes above are simply the wrong shape and the fix is to copy
Dory structure.  If no, my placement test itself is wrong and every verdict above is
meaningless.  Either answer is worth more than another guess.
"""
from __future__ import annotations

import time

import numpy as np
import coremltools as ct

PATH = "/Users/vladimir/dev/dory-ane/dane5-artifacts/set24k/dory_attn_full_0_h0_lut6_infer.mlpackage"


def run(units, label):
    model = ct.models.MLModel(PATH, compute_units=units)
    state = model.make_state()
    spec = model.get_spec()
    feed = {}
    for entry in spec.description.input:
        shape = [int(d) for d in entry.type.multiArrayType.shape]
        dtype = np.int32 if shape == [1] else np.float16
        feed[entry.name] = np.zeros(shape, dtype)
    model.predict(feed, state=state)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        model.predict(feed, state=state)
    ms = (time.perf_counter() - t0) / n * 1000
    return {"units": label, "ms": round(ms, 3), "inputs": list(feed)[:4]}


if __name__ == "__main__":
    for label, units in (("cpu_only", ct.ComputeUnit.CPU_ONLY),
                         ("cpu_and_ne", ct.ComputeUnit.CPU_AND_NE)):
        try:
            print(run(units, label), flush=True)
        except Exception as exc:  # noqa: BLE001
            print({"units": label, "error": str(exc)[:140]}, flush=True)