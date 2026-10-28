# Mel front end, matching feature_extraction_lfm3_omni.py: torch.stft centre=False,
# power spectrum, slaney mel filters, log10 clamped at -12, then (x + 4) / 4.
from __future__ import annotations

import numpy as np

SR, N_FFT, HOP, WIN, N_MELS = 16000, 400, 160, 400, 128
LOG_FLOOR, OFFSET, DIVISOR = -12.0, 4.0, 4.0


def hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    linear = f / (200.0 / 3)
    log = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log, linear)


def mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    linear = m * (200.0 / 3)
    log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
    return np.where(m >= 15.0, log, linear)


def slaney_filters(n_mels=N_MELS, n_fft=N_FFT, sr=SR):
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    weights = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lower, centre, upper = hz[i], hz[i + 1], hz[i + 2]
        rising = (fft_freqs - lower) / max(centre - lower, 1e-10)
        falling = (upper - fft_freqs) / max(upper - centre, 1e-10)
        weights[i] = np.maximum(0.0, np.minimum(rising, falling))
        weights[i] *= 2.0 / (upper - lower)      # slaney area normalisation
    return weights


MEL = slaney_filters()
WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)   # periodic hann


def log_mel(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64).reshape(-1)
    n = len(wave)
    if n < N_FFT:
        padded = N_FFT
    else:
        padded = n + (-(n - N_FFT) % HOP)
    wave = np.pad(wave, (0, padded - n))
    frames = 1 + (padded - N_FFT) // HOP
    out = np.empty((N_MELS, frames), np.float64)
    for i in range(frames):
        chunk = wave[i * HOP:i * HOP + WIN] * WINDOW
        spec = np.fft.rfft(chunk, n=N_FFT)
        power = spec.real ** 2 + spec.imag ** 2
        out[:, i] = MEL @ power
    out = np.log10(np.maximum(out, 1e-30)).clip(min=LOG_FLOOR)
    return (out + OFFSET) / DIVISOR


if __name__ == '__main__':
    data = np.load('/tmp/official_audio.npz')
    official = data['input_features'][0]          # (128, frames)
    sr, t = 16000, np.arange(16000 * 2) / 16000
    wave = 0.4 * np.sin(2 * np.pi * (200 + 300 * t) * t)
    mine = log_mel(wave)
    print({'mine': list(mine.shape), 'official': list(official.shape)}, flush=True)
    k = min(mine.shape[1], official.shape[1])
    diff = np.abs(mine[:, :k] - official[:, :k])
    print({'max_abs_diff': round(float(diff.max()), 5),
           'mean_abs_diff': round(float(diff.mean()), 6),
           'corr': round(float(np.corrcoef(mine[:, :k].ravel(), official[:, :k].ravel())[0, 1]), 6)},
          flush=True)