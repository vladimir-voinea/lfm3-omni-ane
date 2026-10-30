import sys
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/pylibs')
import torch  # noqa: F401  brought in first so transformers sees it
print('torch', torch.__version__)
import transformers
print('transformers', transformers.__version__)
from transformers.utils import import_utils
print('torch available:', import_utils.is_torch_available())
print('torch info:', import_utils._torch_available, import_utils._torch_version)
try:
    from transformers import AutoModel
    print('AutoModel import ok')
except Exception as exc:
    print('AutoModel failed:', str(exc)[:200])