# The Core ML encoder against audio_ref on real speech, and rows inside the audio only.
import sys
import time
import wave

import coremltools as ct
import numpy as np
import torch

import audio_ref as A
import build_audio as B
from mel_front import log_mel

with wave.open(sys.argv[2]) as h:
    rate = h.getframerate()
    samples = np.frombuffer(h.readframes(h.getnframes()), np.int16).astype(np.float32) / 32768.0
if rate != 16000:
    samples = np.interp(np.arange(0, len(samples), rate / 16000), np.arange(len(samples)), samples).astype(np.float32)
mel = torch.as_tensor(np.asarray(log_mel(samples))).float()
mel = mel.view(1, 128, -1)
w = B.load()
with torch.no_grad():
    want = A.forward(mel, w)[1][0].numpy()
n = want.shape[0]
padded = np.zeros((1, 128, B.FRAMES), np.float32)
padded[0, :, :mel.shape[2]] = mel.numpy()
for cu, label in ((ct.ComputeUnit.CPU_AND_NE, 'NE'), (ct.ComputeUnit.CPU_ONLY, 'CPU')):
    start = time.perf_counter()
    model = ct.models.MLModel(sys.argv[1], compute_units=cu)
    loaded = time.perf_counter() - start
    start = time.perf_counter()
    got = np.asarray(model.predict({'mel': padded})['rows'])[:n]
    took = time.perf_counter() - start
    cos = float((got * want).sum() / np.linalg.norm(got) / np.linalg.norm(want))
    print({'on': label, 'rows': n, 'cos': round(cos, 5), 'rel_pct': round(100 * float(np.linalg.norm(got - want) / np.linalg.norm(want)), 2),
           'ms': round(took * 1000), 'load_s': round(loaded, 1)}, flush=True)
