"""Did the shipped build use this checkpoint's weights?

Compare the first rows of the embedding table dumped by the build (embed_tokens.npy) with the
same rows fetched fresh from the checkpoint.  The embedding is shared with the checkpoint's
own tensor, so a mismatch means the published model was built from different weights.
"""
import json
import struct
import subprocess
import sys

import numpy as np

URL = 'https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/main/model.safetensors'
ANE = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6'

head = subprocess.run(['curl', '-sL', '-r', '0-7', URL], capture_output=True).stdout
header_length = struct.unpack('<Q', head)[0]
header = subprocess.run(['curl', '-sL', '-r', '8-%d' % (7 + header_length), URL],
                        capture_output=True).stdout
index = json.loads(header.decode('utf-8').rstrip(' '))
entry = index['model.language_model.embed_tokens.weight']
begin, end = entry['data_offsets']
print({'tensor': entry['dtype'], 'begin': begin, 'rows_bytes': end - begin}, flush=True)

rows = 64
payload = subprocess.run(['curl', '-sL', '-r', '%d-%d' % (begin, begin + 2048 * 2 * rows - 1), URL],
                         capture_output=True).stdout
fresh = np.frombuffer(payload, dtype=np.float16).reshape(rows, 2048).astype(np.float32)

local = np.load(ANE + '/embed_tokens.npy', mmap_mode='r')[:rows].astype(np.float32)
diff = float(np.abs(fresh - local).max())
print({'max_abs_diff': round(diff, 6),
       'fresh_row0_first4': [round(float(v), 4) for v in fresh[0][:4]],
       'local_row0_first4': [round(float(v), 4) for v in local[0][:4]],
       'fresh_std': round(float(fresh.std()), 5), 'local_std': round(float(local.std()), 5)}, flush=True)
