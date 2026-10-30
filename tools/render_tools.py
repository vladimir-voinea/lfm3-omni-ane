from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download('LiquidAI/LFM3-Omni-2B', 'chat_template.jinja')
shutil.copy(path, '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/chat_template.jinja')
print('template saved', flush=True)

import sys
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_chat import render

tools = [{'type': 'function', 'function': {
    'name': 'get_weather',
    'description': 'Current weather for a city',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                   'required': ['city']}}}]
messages = [{'role': 'user', 'content': 'What is the weather in Paris?'}]
out = render(messages, tools=tools)
print({'chars': len(out)}, flush=True)
print(repr(out[-600:]), flush=True)