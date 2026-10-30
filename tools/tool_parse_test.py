import sys
sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from lfm3_tools import parse_tool_calls, strip_tool_calls

raw = '<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>'
print(parse_tool_calls(raw), flush=True)
print(repr(strip_tool_calls('thinking ' + raw)), flush=True)