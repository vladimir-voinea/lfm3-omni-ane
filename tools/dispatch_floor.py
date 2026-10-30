# How much of a token is dispatch overhead rather than arithmetic?
# final_norm is a tiny program: its dispatch time is the floor per call.
import statistics
import time

import numpy as np
import coremltools as ct

ANE = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6'
H = 2048


def timeit(label, path, feed, state=None, n=30):
    model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    if state:
        state = model.make_state()
        model.predict(feed, state=state)
    else:
        model.predict(feed)
    times = []
    for _ in range(n):
        start = time.perf_counter()
        if state:
            model.predict(feed, state=state)
        else:
            model.predict(feed)
        times.append((time.perf_counter() - start) * 1000)
    print({'program': label, 'median_ms': round(statistics.median(times), 3),
           'min_ms': round(min(times), 3)}, flush=True)


if __name__ == '__main__':
    hidden = np.zeros((1, H, 1, 1), np.float16)
    timeit('final_norm (tiny)', ANE + '/final_norm.mlpackage', {'in0': hidden})
    timeit('lm_head', ANE + '/lm_head.mlpackage', {'in0': hidden})
    hist = np.zeros((1, H, 1, 6), np.float16)
    timeit('layer0_conv', ANE + '/layer0_conv.mlpackage', {'in0': hidden, 'in1': hist})
    mask = np.full((1, 4, 1, 28672), -1e4, np.float16)
    mask[:, :, :, :8] = 0
    cos = np.zeros(128, np.float16)
    sin = np.zeros(128, np.float16)
    feed = {'in0': hidden, 'in1': np.array([8], np.int32), 'in2': cos, 'in3': sin, 'in4': mask}
    timeit('layer10_full (28k mask)', ANE + '/layer10_full.mlpackage', feed, state=True)
    slide_feed = {'in0': hidden, 'in1': np.array([8], np.int32), 'in2': cos, 'in3': sin,
                  'in4': np.full((1, 4, 1, 256), 0, np.float16)}
    timeit('layer12_slide (256)', ANE + '/layer12_slide.mlpackage', slide_feed, state=True)