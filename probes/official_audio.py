# Official audio path reference: processor features -> encoder -> projector, dumped so a
# transcription can be checked number against number (the method that cracked vision).
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel

MODEL = '/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B'
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
print('processor ok', flush=True)

# a spoken sentence is not needed to check the encoder: a swept tone exercises the
# mel front end and every layer, and its features must match my transcription exactly
sr = 16000
t = np.arange(sr * 2) / sr
wave = 0.4 * np.sin(2 * np.pi * (200 + 300 * t) * t)
inputs = processor(audio=wave, sampling_rate=sr, text='<|audio_start|><|audio|><|audio_end|>hi',
                   return_tensors='pt')
print({k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}, flush=True)
np.savez('/tmp/official_audio.npz',
         input_features=inputs['input_features'].float().numpy(),
         input_ids=inputs['input_ids'].numpy())

model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32,
                                  low_cpu_mem_usage=True)
model.eval()
features = inputs['input_features'].float()
lengths = torch.tensor([features.shape[-1]])
with torch.no_grad():
    encoded, out_lengths = model.audio_encoder(features, lengths)
    projected = model.audio_projector(encoded)
print({'encoded': list(encoded.shape), 'projected': list(projected.shape),
       'enc_norm': round(float(encoded.norm(dim=-1).mean()), 3)}, flush=True)
np.save('/tmp/official_audio_enc.npy', encoded.float().numpy())
np.save('/tmp/official_audio_proj.npy', projected.float().numpy())
print('saved', flush=True)