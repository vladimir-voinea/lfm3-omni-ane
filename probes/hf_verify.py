from huggingface_hub import HfApi

token = open('/tmp/.hf_token').read().strip()
api = HfApi(token=token)
REPO = 'vladimir94/lfm3-omni-2b-ane-coreml'
info = api.model_info(REPO, files_metadata=True)
names = [s.rfilename for s in info.siblings]
total = sum(s.size or 0 for s in info.siblings)
key = [n for n in names if not n.startswith('layer')]
print({'files': len(names), 'gb': round(total / 1e9, 2)}, flush=True)
print({'non_layer': key}, flush=True)