from huggingface_hub import hf_hub_download
import shutil

for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json', 'generation_config.json'):
    path = hf_hub_download('LiquidAI/LFM3-Omni-2B', name)
    shutil.copy(path, '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/' + name)
    print(name, 'ok', flush=True)