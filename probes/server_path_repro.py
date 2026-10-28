# Reproduce the server's image path step by step, printing what each stage sees.
import base64
import json
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

# exactly what the server does with content parts
messages = [{'role': 'user', 'content': [
    {'type': 'text', 'text': 'What colour fills this image? One word.'},
    {'type': 'image_url', 'image_url': {'url': uri}}]}]
image_rows = None
for message in messages:
    content = message.get('content')
    if isinstance(content, list):
        text_parts, images = [], []
        for part in content:
            if part.get('type') == 'text':
                text_parts.append(part.get('text') or '')
            elif part.get('type') == 'image_url':
                url = (part.get('image_url') or {}).get('url') or ''
                if url.startswith('data:'):
                    images.append(url)
        if images:
            image_rows = rows_from_data_uri(images[0])
            message['content'] = ('<|vision_start|><|image|><|vision_end|>' + ' '.join(text_parts))

print({'rows': len(image_rows), 'content': messages[0]['content'][:60]}, flush=True)
print({'rendered': repr(render_chat(messages)[:90])}, flush=True)
engine = Engine()
for label, params in (('default', SamplingParams()), ('greedy', SamplingParams(temperature=0.0))):
    text = ''.join(engine.tokenizer.decode([t]) for t, _ in
                   engine.stream(messages, params, max_tokens=8, image_rows=image_rows))
    print({label: text}, flush=True)