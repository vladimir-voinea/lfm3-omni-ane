from huggingface_hub import hf_hub_download

path = hf_hub_download('LiquidAI/LFM3-Omni-2B', 'chat_template.jinja')
text = open(path).read()
print({'chars': len(text)}, flush=True)
start = text.find('tool')
print(text[max(0, start - 200):start + 1800], flush=True)