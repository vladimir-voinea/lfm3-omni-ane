import sys, os
import numpy as np
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_runtime import Runtime

TOKENS = [1, 9707, 500]


def logits_for(tiled):
    os.environ['LFM3_TILED'] = '1' if tiled else '0'
    runtime = Runtime('/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
    session = runtime.new_session()
    out = runtime.step(session, TOKENS)
    return np.asarray(out).reshape(-1)


shipped = logits_for(False)
tiled = logits_for(True)
a = shipped - shipped.mean()
b = tiled - tiled.mean()
cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
top_a = set(np.argsort(shipped)[-5:].tolist())
top_b = set(np.argsort(tiled)[-5:].tolist())
print({'cosine': round(cos, 4), 'top5_overlap': len(top_a & top_b),
       'shipped_top1': int(np.argmax(shipped)), 'tiled_top1': int(np.argmax(tiled)),
       'rel_l2_pct': round(float(np.linalg.norm(shipped - tiled) / np.linalg.norm(shipped) * 100), 1)}, flush=True)