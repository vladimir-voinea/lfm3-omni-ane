from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
files = [s.rfilename for s in api.model_info('LiquidAI/LFM3-Omni-2B', files_metadata=False).siblings]
print({'files': files}, flush=True)