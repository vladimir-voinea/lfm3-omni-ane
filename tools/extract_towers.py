"""Cut the 4.3 GB checkpoint down to the 0.88 GB the towers actually read.

The text backbone is already converted to Core ML, so vision, audio input and speech out are
the only run-time consumers of the original safetensors. This keeps those tensors and drops
the rest; rerun it after a fresh download if you deleted towers.safetensors.
"""
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SOURCE, TOWERS

PREFIXES = ('model.vision_tower.', 'model.multi_modal_projector.', 'model.audio_encoder.',
            'model.audio_projector.', 'model.depth_decoder.', 'model.codebook_embedding.')


def main():
    source = os.path.join(SOURCE, 'model.safetensors')
    if not os.path.exists(source):
        print('nothing to extract: %s is not there' % source)
        return 1
    kept = {}
    with safe_open(source, framework='pt') as handle:
        for key in handle.keys():
            if key.startswith(PREFIXES):
                kept[key] = handle.get_tensor(key)
    save_file(kept, TOWERS)
    print({'tensors': len(kept), 'GB': round(os.path.getsize(TOWERS) / 1e9, 2)})
    return 0


if __name__ == '__main__':
    sys.exit(main())