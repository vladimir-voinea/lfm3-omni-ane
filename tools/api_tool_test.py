import json
import urllib.request

payload = {
    'model': 'lfm3',
    'messages': [{'role': 'user',
                  'content': 'What is the weather in Paris? Use the get_weather tool.'}],
    'tools': [{'type': 'function', 'function': {
        'name': 'get_weather',
        'description': 'Current weather for a city',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}],
    'max_tokens': 96}
request = urllib.request.Request('http://127.0.0.1:8010/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=900) as response:
    body = json.loads(response.read())
print({'finish': body['choices'][0]['finish_reason'],
       'tool_calls': body['choices'][0].get('message', {}).get('tool_calls')}, flush=True)