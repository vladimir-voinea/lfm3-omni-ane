# The speech programs for the host: depth logits and codes -> waveform, without the checkpoints.
import os

import coremltools as ct
import numpy as np
import torch

import codec_ref as C
import depth_ref as D

CODEC_FRAMES = 128


def available(model_dir):
    return all(os.path.exists(os.path.join(model_dir, n))
               for n in ('depth_decoder.mlpackage', 'codec.mlpackage', 'speech_tables.npz'))


class Speech:
    def __init__(self, model_dir):
        units = ct.ComputeUnit.CPU_AND_NE
        self.depth = ct.models.MLModel(os.path.join(model_dir, 'depth_decoder.mlpackage'), compute_units=units)
        self.codec = ct.models.MLModel(os.path.join(model_dir, 'codec.mlpackage'), compute_units=units)
        tables = np.load(os.path.join(model_dir, 'speech_tables.npz'))
        self.tables = {k: tables[k] for k in tables.files}
        self.book = torch.from_numpy(self.tables['book'])

    def depth_logits(self, hidden, codes, masked):
        slots = np.stack([self.tables['mask'] if masked[0, i] else
                          self.tables['embedding_%d' % i][int(codes[0, i])] for i in range(D.CB)])
        logits = self.depth.predict({'hidden': hidden.reshape(1, -1).numpy().astype(np.float32),
                                     'slots': slots[None].astype(np.float32)})['logits']
        return torch.from_numpy(np.asarray(logits, np.float32))[None]

    def waveform(self, frames):
        """codes (frames, 12) -> samples at 24 kHz; None when too long for the program."""
        count = frames.shape[0]
        if count > CODEC_FRAMES:
            return None
        levels, basis = np.array(C.LEVELS, np.float32), np.array(C.BASIS, np.float32)
        half = np.floor(levels / 2)
        index = frames.numpy().astype(np.float32)[..., None]
        latents = np.zeros((1, CODEC_FRAMES, C.CB * C.FSQ_DIM), np.float32)
        latents[0, :count] = ((np.floor(index / basis) % levels - half) / half).reshape(count, -1)
        spectrum = np.asarray(self.codec.predict({'latents': latents})['spectrum'], np.float32)[0]
        spec = torch.from_numpy(spectrum[:count * C.FRAME_STACK]).T
        log_mag, angle = spec.chunk(2, dim=0)
        audio, env = C.istft(torch.polar(log_mag.exp(), angle), {'decoder.istft.window': torch.from_numpy(self.tables['window'])})
        return (audio[C.TRIM:-C.TRIM] / env[C.TRIM:-C.TRIM]).clamp(-1, 1).numpy()
