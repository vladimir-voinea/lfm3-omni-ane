"""A complete agent loop against an OpenAI-compatible endpoint, no SDK required.

    python tools/agent_demo.py                       # starts its own server
    python tools/agent_demo.py http://127.0.0.1:8010  # or use a running one

The point is the wire format: the model answers with tool_calls, the harness runs the tool,
sends the result back as role=tool, and the model writes the final answer. Anything that
speaks this protocol -- a chat UI, an agent framework, a script -- can drive this engine.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = int(os.environ.get('LFM3_PORT', '8096'))

TOOLS = [{'type': 'function', 'function': {
    'name': 'get_weather',
    'description': 'Current weather for a city',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                   'required': ['city']}}},
    {'type': 'function', 'function': {
    'name': 'convert_currency',
    'description': 'Convert an amount between currencies',
    'parameters': {'type': 'object',
                   'properties': {'amount': {'type': 'number'},
                                  'from_currency': {'type': 'string'},
                                  'to_currency': {'type': 'string'}},
                   'required': ['amount', 'from_currency', 'to_currency']}}}]


def run_tool(name, arguments):
    """The harness's side of the contract: it is just a function call."""
    if name == 'get_weather':
        return json.dumps({'city': arguments.get('city'), 'temperature_c': 18,
                           'conditions': 'cloudy'})
    if name == 'convert_currency':
        rates = {('EUR', 'USD'): 1.08, ('USD', 'EUR'): 0.93}
        rate = rates.get((arguments.get('from_currency'), arguments.get('to_currency')), 1.0)
        return json.dumps({'converted': round(float(arguments.get('amount', 0)) * rate, 2),
                           'rate': rate})
    return json.dumps({'error': 'unknown tool %s' % name})


def post(base, payload):
    request = urllib.request.Request(base + '/v1/chat/completions',
                                     data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def agent_turn(base, question, max_steps=4):
    messages = [{'role': 'user', 'content': question}]
    for step in range(max_steps):
        reply = post(base, {'messages': messages, 'tools': TOOLS, 'max_tokens': 256})
        message = reply['choices'][0]['message']
        calls = message.get('tool_calls') or []
        if not calls:
            return message.get('content', '').strip(), step
        messages.append({'role': 'assistant', 'content': message.get('content'),
                         'tool_calls': calls})
        for call in calls:
            name = call['function']['name']
            arguments = json.loads(call['function']['arguments'] or '{}')
            print('  tool -> %s(%s)' % (name, arguments), flush=True)
            messages.append({'role': 'tool', 'tool_call_id': call.get('id', 'call_%d' % step),
                             'content': run_tool(name, arguments)})
    return '(gave up after %d steps)' % max_steps, max_steps


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:%d' % PORT
    server = None
    if len(sys.argv) == 1:
        environment = dict(os.environ, LFM3_PORT=str(PORT))
        server = subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'lfm3_serve.py')],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment)
        for _ in range(180):
            try:
                urllib.request.urlopen(base + '/health', timeout=2).read()
                break
            except Exception:
                time.sleep(2)
    try:
        for question in ('What is the weather in Paris?',
                         'How much is 100 EUR in USD?'):
            print(question, flush=True)
            answer, steps = agent_turn(base, question)
            print('  answer (%d tool round(s)): %s' % (steps, answer), flush=True)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == '__main__':
    main()