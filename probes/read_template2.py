from huggingface_hub import hf_hub_download

path = hf_hub_download('LiquidAI/LFM3-Omni-2B', 'chat_template.jinja')
text = open(path).read()
for marker in ('tool_list', 'render_tools', 'tool_call_start', 'tool_call_end'):
    index = text.find(marker)
    print('=== ' + marker + ' @ ' + str(index), flush=True)
    if index >= 0:
        print(text[max(0, index - 350):index + 700], flush=True)