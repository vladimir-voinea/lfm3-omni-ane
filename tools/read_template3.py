from huggingface_hub import hf_hub_download

path = hf_hub_download('LiquidAI/LFM3-Omni-2B', 'chat_template.jinja')
text = open(path).read()
print(text[:2000], flush=True)