import numpy as np

official = np.load('/tmp/official_audio_codes.npy')[0].astype(float)
mine = np.load('/tmp/my_audio_codes.npy').T.astype(float)
synth = np.load('/tmp/synth_codes.npy').T.astype(float)


def smoothness(codes):
    steps = np.abs(np.diff(codes, axis=1))
    return round(float(steps.mean()), 1), round(float(steps.std()), 1)


for name, codes in (('official', official), ('mine', mine), ('synth', synth)):
    mean_step, std_step = smoothness(codes)
    # a speech trajectory moves in small steps; scatter means the frames are unrelated
    print({name: {'mean_frame_to_frame_jump': mean_step, 'std': std_step}}, flush=True)