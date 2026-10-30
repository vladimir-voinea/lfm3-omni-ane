import numpy as np

official = np.load('/tmp/official_audio_codes.npy')[0]      # (12, frames)
mine = np.load('/tmp/my_audio_codes.npy').T                 # (12, frames)
synth = np.load('/tmp/synth_codes.npy').T
print({'shapes': [official.shape, mine.shape, synth.shape]}, flush=True)
for name, codes in (('official', official), ('mine', mine), ('synth', synth)):
    print({name: {'mean': round(float(codes.mean()), 1), 'std': round(float(codes.std()), 1),
                  'min': int(codes.min()), 'max': int(codes.max()),
                  'codebook0_mean': round(float(codes[0].mean()), 1)}}, flush=True)