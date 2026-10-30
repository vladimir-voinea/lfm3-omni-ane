# End-to-end API test: post a solid-colour PNG as a data URI and check that the answer
# follows the colour -- proof that the pixels reach the text model through HTTP.
import base64
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from png_codec import png_encode


def uri(colour):
    pixels = np.zeros((224, 224, 3), np.uint8)
    pixels[:, :] = colour
    return 'data:image/png;base64,' + base64.b64encode(png_encode(pixels)).decode()


for name, colour in (('red', (255, 0, 0)), ('green', (0, 128, 0))):
    payload = {
        'model': 'lfm3',
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'What colour fills this image? One word.'},
            {'type': 'image_url', 'image_url': {'url': uri(colour)}}]}],
        'max_tokens': 12}
    request = urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1200) as response:
        body = json.loads(response.read())
    print({'colour': name, 'answer': body['choices'][0]['message']['content']}, flush=True)