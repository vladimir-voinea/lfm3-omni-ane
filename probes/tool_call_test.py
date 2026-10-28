import sys

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
sys.path.insert(0, '/Users/vladimir/dev/dory-ane-worktrees/DANE-5')

from anemll.runtime.dory_sampling import SamplingParams
from lfm3_engine import Engine

tools = [{'type': 'function', 'function': {
    'name': 'get_weather',
    'description': 'Current weather for a city',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                   'required': ['city']}}}]
messages = [{'role': 'user', 'content': 'What is the weather in Paris? Use the get_weather tool.'}]
engine = Engine()
text = ''.join(engine.tokenizer.decode([t]) for t, _ in
               engine.stream(messages, SamplingParams(), max_tokens=96, tools=tools))
print({'with_tools': text[:400]}, flush=True)