# Is the generated audio speech-like, or just loud?  Compare its spectrum to the official
# decode and to silence.
import wave

import numpy as np


def load(path):
    with wave.open(path, 'rb') as handle:
        raw = handle.readframes(handle.getnframes())
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0, handle.getframerate()


def profile(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1 / rate)
    total = spectrum.sum() + 1e-12
    low = spectrum[(freqs >= 80) & (freqs < 1000)].sum() / total
    mid = spectrum[(freqs >= 1000) & (freqs < 4000)].sum() / total
    high = spectrum[(freqs >= 4000)].sum() / total
    centroid = float((spectrum * freqs).sum() / total)
    return {'rms': round(float(np.sqrt((samples ** 2).mean())), 4),
            'low_80_1k': round(float(low), 3), 'mid_1k_4k': round(float(mid), 3),
            'high_4k_plus': round(float(high), 3), 'centroid_hz': round(centroid)}


if __name__ == '__main__':
    mine, rate = load('/tmp/speech_out.wav')
    print({'mine': profile(mine, rate), 'seconds': round(len(mine) / rate, 2)}, flush=True)
    official = np.load('/tmp/official_codes_audio.npy')
    print({'official': profile(official.astype(np.float32), 24000)}, flush=True)
    print({'silence': profile(np.zeros(24000, np.float32) + 1e-4, 24000)}, flush=True)