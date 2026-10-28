# Speech out, end to end: the ANE text model supplies a hidden state per frame, the depth
# decoder turns it into 12 codes, the codec turns the codes into 24 kHz audio.
import sys
import wave

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')

import codec_ref
import depth_ref
from lfm3_runtime import Runtime

P = depth_ref.P
ANE = '/Users/vladimir/dev/lfm3-ane/model-28k'
AUDIO_START = 125012
EOS_AUDIO = 2016
FRAMES = 24


def load(prefixes):
    out = {}
    with safe_open(P, framework='pt') as f:
        for key in f.keys():
            if any(key.startswith(p) for p in prefixes):
                out[key] = f.get_tensor(key).float()
    return out


def sample_codes(hidden, w, history=None, schedule=(4, 4, 4), temperature=0.4, penalty=1.2):
    """Maskgit with the model's own audio sampler: temperature 0.4 and a repetition penalty
    of 1.2 over the previous frames. Greedy here produces silence -- the most likely code
    of a frame is not the code that carries sound."""
    codes = torch.zeros(1, 12, dtype=torch.long)
    revealed = torch.zeros(1, 12, dtype=torch.bool)
    start = 0
    for width in schedule:
        pick = list(range(start, start + width))
        logits = depth_ref.forward(hidden.view(1, -1), codes, ~revealed, w)
        for index in pick:
            scores = logits[0, index].float().clone()
            if history is not None and len(history):
                seen = torch.tensor(history, dtype=torch.long)[:, index]
                for code in set(seen.tolist()):
                    scores[code] = scores[code] / penalty if scores[code] > 0 else scores[code] * penalty
            probabilities = (scores / temperature).softmax(-1)
            codes[0, index] = int(torch.multinomial(probabilities, 1)[0])
        revealed[0, pick] = True
        start += width
    return codes


def main():
    tok = Tokenizer.from_file('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/tokenizer.json')
    head = '<|im_start|>user\nSay hello in a friendly voice.<|im_end|>\n<|im_start|>assistant\n<|audio_start|>'
    ids = [1] + tok.encode(head).ids

    text_w = load(['model.codebook_embedding.'])
    book = text_w['model.codebook_embedding.weight']
    del text_w
    depth_w = load([depth_ref.PRE])
    codec_w = {}
    with safe_open(codec_ref.CODEC, framework='pt') as f:
        for key in f.keys():
            codec_w[key] = f.get_tensor(key).float()

    runtime = Runtime(ANE)
    session = runtime.new_session()
    logits = runtime.step(session, ids)
    frames = []
    torch.manual_seed(0)
    for _ in range(FRAMES):
        # the depth decoder is fed the model's last_hidden_state, which is the final-norm
        # output -- the raw pre-norm residual gives near-silent frames
        normed = runtime.norm.predict({'in0': session.hidden})
        hidden = torch.from_numpy(np.asarray(list(normed.values())[0], np.float16).reshape(-1).astype(np.float32))
        codes = sample_codes(hidden, depth_w, history=frames[-64:])
        frames.append(codes[0].tolist())
        if int(codes[0, 0]) == EOS_AUDIO:
            break
        offsets = torch.arange(12) * 2017
        embed = book[codes[0] + offsets].sum(0).numpy()
        logits = runtime.step_hidden(session, embed) or logits
    codes = torch.tensor(frames)
    print({'frames': len(frames), 'first_codes': frames[0] if frames else None}, flush=True)

    audio = codec_ref.decode(codes.unsqueeze(0).transpose(1, 2), codec_w,
                             ['conv', 'conv', 'attn', 'conv', 'attn', 'conv', 'attn', 'conv'])
    samples = (audio[0].clamp(-1, 1).numpy() * 32767).astype(np.int16)
    with wave.open('/tmp/speech_out.wav', 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(samples.tobytes())
    print({'samples': len(samples), 'seconds': round(len(samples) / 24000, 2),
           'rms': round(float(np.sqrt((samples.astype(np.float32) ** 2).mean())), 1),
           'peak': int(np.abs(samples).max())}, flush=True)


if __name__ == '__main__':
    main()