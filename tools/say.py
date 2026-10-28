# Text to speech through the engine: prompt -> ANE text model -> depth decoder -> codec.
#   python tools/say.py 'Good morning, how are you?' out.wav
import os
import sys
import wave

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import codec_ref
import depth_ref
from audio_out import sample_codes
from lfm3_runtime import Runtime

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
ANE = os.environ.get('LFM3_ANE', '/Users/vladimir/dev/lfm3-ane/model-28k-lut6')
AUDIO_START = 125012
EOS_AUDIO = 2016
LAYERS = ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv']


def load_depth():
    weights = {}
    with safe_open(depth_ref.P, framework='pt') as handle:
        for key in handle.keys():
            if key.startswith(depth_ref.PRE):
                weights[key] = handle.get_tensor(key).float()
    return weights


def load_codec():
    weights = {}
    with safe_open(codec_ref.CODEC, framework='pt') as handle:
        for key in handle.keys():
            weights[key] = handle.get_tensor(key).float()
    return weights


def say(text, consonance_frames=48):
    tokenizer = Tokenizer.from_file(os.path.join(MODEL, 'tokenizer.json'))
    prompt = ('<|im_start|>user\n' + text + '<|im_end|>\n<|im_start|>assistant\n<|audio_start|>')
    ids = tokenizer.encode(prompt).ids          # no BOS: the processor does not add one
    runtime = Runtime(ANE)
    depth = load_depth()
    codec = load_codec()
    session = runtime.new_session()
    runtime.step(session, ids)
    # the audio feedback table is the codebook embedding, not the text embedding
    with safe_open(depth_ref.P, framework='pt') as handle:
        book = handle.get_tensor('model.codebook_embedding.weight').float()
    torch.manual_seed(0)
    frames = []
    for _ in range(consonance_frames):
        normed = runtime.norm.predict({'in0': session.hidden})
        hidden = torch.from_numpy(np.asarray(list(normed.values())[0], np.float16)
                                  .reshape(-1).astype(np.float32))
        codes = sample_codes(hidden, depth, history=frames[-64:])
        frames.append(codes[0].tolist())
        if int(codes[0, 0]) == EOS_AUDIO:
            break
        offsets = torch.arange(12) * 2017
        embed = book[codes[0] + offsets].sum(0).numpy()
        runtime.step_hidden(session, embed)
    codes = torch.tensor(frames)
    audio = codec_ref.decode(codes.unsqueeze(0).transpose(1, 2), codec, LAYERS)[0]
    return audio.clamp(-1, 1).numpy(), len(frames)


if __name__ == '__main__':
    text = sys.argv[1] if len(sys.argv) > 1 else 'Say hello in a friendly voice.'
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/speech_out.wav'
    samples, frames = say(text)
    data = (samples * 32767).astype(np.int16)
    with wave.open(out, 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(data.tobytes())
    print({'text': text, 'frames': frames, 'seconds': round(len(samples) / 24000, 2),
           'rms': round(float(np.sqrt((samples ** 2).mean())), 4),
           'peak': round(float(np.abs(samples).max()), 3), 'path': out}, flush=True)