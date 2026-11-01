# Audio for the server: a base64 WAV becomes the vectors the text model reads.
import base64
import io
import os
import wave

import numpy as np
import torch
from safetensors import safe_open

import audio_ref
from mel_front import log_mel
from paths import MODEL

_AUDIO = {}


def _load():
    if 'weights' in _AUDIO:
        return _AUDIO['weights']
    from source_check import require_source
    require_source("audio input")
    weights = {}
    with safe_open(audio_ref.P, framework='pt') as handle:
        for key in handle.keys():
            if 'audio_encoder' in key or 'audio_projector' in key:
                weights[key] = handle.get_tensor(key).float()
    _AUDIO['weights'] = weights
    return weights


def _coreml_rows(feats):
    '''The Core ML encoder's rows when the build has it and the clip fits its window, or None.'''
    path = os.path.join(MODEL, 'audio_encoder.mlpackage')
    frames = feats.shape[-1]
    if not os.path.exists(path) or frames > 3000:
        return None
    if 'coreml' not in _AUDIO:
        import coremltools as ct
        _AUDIO['coreml'] = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    padded = np.zeros((1, 128, 3000), np.float32)
    padded[0, :, :frames] = feats.numpy()[0]
    rows = np.asarray(_AUDIO['coreml'].predict({'mel': padded})['rows'], np.float32)
    # the same row count the convolutions give the clip at its own length
    return list(rows[:(frames + 7) // 8])


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
    rows = _coreml_rows(feats)
    if rows is not None:
        return rows
    weights = _load()
    with torch.no_grad():
        _, projected = audio_ref.forward(feats, weights)
    rows = [row for row in projected[0].detach().numpy()]
    # the encoder weights are only needed for this projection; holding them beside the 26
    # resident programs is what made make_state() fail on the next request
    _AUDIO.pop('weights', None)
    del weights
    import gc
    gc.collect()
    return rows