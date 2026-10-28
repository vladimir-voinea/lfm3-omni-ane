# Audio in, end to end: real speech -> my mel -> my encoder -> projector -> text model.
import json
import os
import sys
import wave

import numpy as np
import torch
from tokenizers import Tokenizer
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

import audio_ref
from lfm3_runtime import Runtime
from mel_front import log_mel

ANE = os.environ.get('LFM3_ANE', '/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
EOS = 124900


def read_wav(path):
    with wave.open(path) as handle:
        frames = handle.readframes(handle.getnframes())
        return np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0


w = {}
with safe_open(audio_ref.P, framework='pt') as f:
    for k in f.keys():
        if 'audio_encoder' in k or 'audio_projector' in k:
            w[k] = f.get_tensor(k).float()

wave_samples = read_wav('/tmp/speech.wav')
mel = log_mel(wave_samples)
feats = torch.from_numpy(mel).float().unsqueeze(0)
enc, proj = audio_ref.forward(feats, w)
tokens = proj[0].detach().numpy()
print({'seconds': round(len(wave_samples) / 16000, 2), 'mel': list(mel.shape),
       'audio_tokens': list(tokens.shape)}, flush=True)
del w
import gc
gc.collect()

tok = Tokenizer.from_file('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/tokenizer.json')
runtime = Runtime(ANE)

for question in ('Transcribe the audio.', 'What did the speaker say?'):
    head = ('<|im_start|>user\n<|audio_start|><|audio|><|audio_end|>%s<|im_end|>\n'
            '<|im_start|>assistant\n') % question
    ids = [1] + tok.encode(head).ids
    session = runtime.new_session()
    at = ids.index(125031)
    runtime.step(session, ids[:at])
    for row in tokens:
        runtime.step_hidden(session, row)
    logits = runtime.step(session, ids[at + 1:])
    out = []
    for _ in range(24):
        nxt = int(np.argmax(logits))
        if nxt == EOS:
            break
        out.append(nxt)
        logits = runtime.step(session, [nxt])
    print({'question': question, 'answer': tok.decode(out)}, flush=True)