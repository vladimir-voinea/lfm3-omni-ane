import base64
import sys

import numpy as np

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
sys.path.insert(0, '/Users/vladimir/dev/dory-ane-worktrees/DANE-5')

from anemll.runtime.dory_sampling import SamplingParams
from lfm3_engine import Engine, render_chat
from lfm3_vision_api import rows_from_data_uri
from png_codec import png_encode

pixels = np.zeros((224, 224, 3), np.uint8)
pixels[:, :] = (255, 0, 0)
uri = 'data:image/png;base64,' + base64.b64encode(png_encode(pixels)).decode()
rows = rows_from_data_uri(uri)

messages = [{'role': 'user',
             'content': '<|vision_start|><|image|><|vision_end|>What colour fills this image? One word.'}]
print({'rendered': render_chat(messages)[:80]}, flush=True)
engine = Engine()
params = SamplingParams()
greedy = SamplingParams(temperature=0.0)
for name, p in (('default', params), ('greedy', greedy)):
    text = ''.join(engine.tokenizer.decode([t]) for t, _ in
                   engine.stream(messages, p, max_tokens=8, image_rows=rows))
    print({name: text}, flush=True)
    break