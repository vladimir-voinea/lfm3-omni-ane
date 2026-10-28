# One place for the directories, so a fresh clone does not depend on anyone's home folder.
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.environ.get('LFM3_PROJECT', os.path.dirname(HERE))

# the converted Core ML build (download from HuggingFace)
MODEL = os.environ.get('LFM3_MODEL', os.path.join(PROJECT, 'model-28k-lut6'))
# the runtime's own programs
ANE = os.environ.get('LFM3_ANE', MODEL)
# the original weights: vision, audio in, speech out, and rebuilding
SOURCE = os.environ.get('LFM3_SOURCE', os.path.join(PROJECT, 'models', 'LFM3-Omni-2B'))
# where build_model.py writes before palettisation
BUILD = os.environ.get('LFM3_BUILD', os.path.join(PROJECT, 'model-28k'))

TOKENIZER = os.path.join(MODEL, 'tokenizer.json')
CONFIG = os.path.join(MODEL, 'config.json')
