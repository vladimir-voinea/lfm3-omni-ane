# Audio for the server: a base64 WAV becomes the vectors the text model reads.
import base64
import io
import wave

import numpy as np
import torch
from safetensors import safe_open

import audio_ref
from mel_front import log_mel

_AUDIO = {}


def _load():
    if 'weights' in _AUDIO:
        return _AUDIO['weights']
    weights = {}
    with safe_open(audio_ref.P, framework='pt') as handle:
        for key in handle.keys():
            if 'audio_encoder' in key or 'audio_projector' in key:
                weights[key] = handle.get_tensor(key).float()
    _AUDIO['weights'] = weights
    return weights


def read_wav_bytes(data):
    with wave.open(io.BytesIO(data)) as handle:
        rate = handle.getframerate()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    if width == 2:
        samples = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError('unsupported sample width %d' % width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(1)
    if rate != 16000:
        raise ValueError('need 16 kHz mono, got %d Hz x %d' % (rate, channels))
    return samples, rate


def rows_from_base64(payload):
    data = base64.b64decode(payload.split(',', 1)[1] if ',' in payload else payload)
    samples, _ = read_wav_bytes(data)
    mel = log_mel(samples)
    feats = torch.from_numpy(mel).float().unsqueeze(0)
    weights = _load()
    with torch.no_grad():
        _, projected = audio_ref.forward(feats, weights)
    rows = [row for row in projected[0].detach().numpy()]
    # the encoder weights are only needed for this projection; holding them beside the 26
    # resident programs is what made make_state() fail on the next request
    _AUDIO.clear()
    del weights
    import gc
    gc.collect()
    return rows