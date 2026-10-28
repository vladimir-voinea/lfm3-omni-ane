import base64
import io
import json

import numpy as np
import torch
from safetensors import safe_open

from png_codec import png_decode
from vision_chat import IDS, patchify
from vision_ref import P, Vision, projector

_VISION = {}


def _load():
    # check for the tower, not for 'something cached': a half-built cache from a failed
    # first attempt would otherwise be returned forever
    if 'tower' in _VISION:
        return _VISION
    weights = {}
    with safe_open(P, framework='pt') as handle:
        for key in handle.keys():
            if 'vision_tower' in key or 'multi_modal_projector' in key:
                weights[key] = handle.get_tensor(key).float()
    config = json.load(open('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/config.json'))
    tower = Vision(weights, config['vision_config']['layer_types'])
    _VISION['weights'] = weights
    _VISION['tower'] = tower
    return _VISION


def rows_from_data_uri(uri, size=224):
    '''data URI -> the vectors the text model sees, one per merged patch.'''
    payload = uri.split(',', 1)[1] if ',' in uri else uri
    pixels = png_decode(base64.b64decode(payload))
    height, width = pixels.shape[:2]
    if (height, width) != (size, size):
        ys = (np.arange(size) * height // size).clip(0, height - 1)
        xs = (np.arange(size) * width // size).clip(0, width - 1)
        pixels = pixels[ys][:, xs]
    array = pixels.astype(np.float32).transpose(2, 0, 1) / 255.0
    state = _load()
    patches, gh, gw = patchify(torch.from_numpy(array))
    patches = ((patches - 0.5) / 0.5).float()
    with torch.no_grad():
        tokens = projector(state['tower'].forward(patches, gh, gw), state['weights'])
    return [row.numpy() for row in tokens]