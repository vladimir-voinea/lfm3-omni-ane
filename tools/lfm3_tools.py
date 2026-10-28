# The model emits tool calls as text in its own format:
#   <tool_call>
#   <function=NAME>
#   <parameter=ARG>
#   value
#   </parameter>
#   </function>
#   </tool_call>
# This turns that into the OpenAI shape clients expect.
import json
import re
import uuid

BLOCK = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
FUNCTION = re.compile(r'<function=([^>]+)>\s*(.*?)\s*</function>', re.S)
PARAMETER = re.compile(r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', re.S)


def _coerce(value):
    text = value.strip()
    try:
        return json.loads(text)
    except Exception:
        return text


def parse_tool_calls(text):
    calls = []
    for block in BLOCK.findall(text or ''):
        for name, body in FUNCTION.findall(block):
            arguments = {key.strip(): _coerce(value) for key, value in PARAMETER.findall(body)}
            calls.append({
                'id': 'call_' + uuid.uuid4().hex[:24],
                'type': 'function',
                'function': {'name': name.strip(),
                             'arguments': json.dumps(arguments, ensure_ascii=False)}})
    return calls


def strip_tool_calls(text):
    return BLOCK.sub('', text or '').strip()