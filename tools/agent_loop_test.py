import sys

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
sys.path.insert(0, '/Users/vladimir/dev/dory-ane-worktrees/DANE-5')

from anemll.runtime.dory_sampling import SamplingParams
from lfm3_engine import Engine, render_chat
from lfm3_tools import parse_tool_calls, strip_tool_calls

tools = [{'type': 'function', 'function': {
    'name': 'get_weather',
    'description': 'Current weather for a city',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                   'required': ['city']}}}]
engine = Engine()
params = SamplingParams()

first = [{'role': 'user', 'content': 'What is the weather in Paris? Use the get_weather tool.'}]
raw = ''.join(engine.tokenizer.decode([t]) for t, _ in
              engine.stream(first, params, max_tokens=96, tools=tools))
calls = parse_tool_calls(raw)
print({'step1_call': calls[0]['function'] if calls else None}, flush=True)

# the harness feeds the result back, then asks again
second = first + [
    {'role': 'assistant', 'content': raw.strip()},
    {'role': 'tool', 'content': '{"temperature_c": 18, "conditions": "cloudy"}'}]
print({'rendered_tail': repr(render_chat(second, tools=tools)[-260:])}, flush=True)
answer = ''.join(engine.tokenizer.decode([t]) for t, _ in
                 engine.stream(second, params, max_tokens=64, tools=tools))
print({'step2_answer': strip_tool_calls(answer)[:220]}, flush=True)