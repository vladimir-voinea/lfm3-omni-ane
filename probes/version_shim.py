import os
import sys

os.makedirs('/tmp/torchfake/torch-2.6.0.dist-info', exist_ok=True)
with open('/tmp/torchfake/torch-2.6.0.dist-info/METADATA', 'w') as handle:
    handle.write('Metadata-Version: 2.1\nName: torch\nVersion: 2.6.0\n')
sys.path.insert(0, '/tmp/torchfake')
sys.path.insert(1, '/Users/vladimir/dev/lfm3-ane/pylibs')

import torch
print('real torch', torch.__version__)
import transformers
from transformers.utils import is_torch_available
print('reported torch available:', is_torch_available())
from transformers import AutoModel
print('AutoModel ok')