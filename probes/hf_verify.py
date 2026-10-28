from huggingface_hub import HfApi

token = open('/tmp/.hf_token').read().strip()
api = HfApi(token=token)
info = api.model_info('vladimir94/lfm3-omni-2b-ane-coreml', files_metadata=True)
files = [s.rfilename for s in info.siblings]
total = sum(s.size or 0 for s in info.siblings)
print({'files': len(files), 'gb': round(total / 1e9, 2)}, flush=True)
print({'examples': files[:4]}, flush=True)