import sys

import numpy as np

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from png_codec import png_decode, png_encode

for colour in ((255, 0, 0), (0, 128, 0), (0, 0, 255)):
    pixels = np.zeros((224, 224, 3), np.uint8)
    pixels[:, :] = colour
    decoded = png_decode(png_encode(pixels))
    print({'sent': colour, 'decoded_mean': decoded.reshape(-1, 3).mean(0).round(1).tolist(),
           'shape': list(decoded.shape), 'match': bool((decoded == pixels).all())}, flush=True)