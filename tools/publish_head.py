from huggingface_hub import HfApi

token = open('/tmp/.hf_token').read().strip()
api = HfApi(token=token)
REPO = 'vladimir94/lfm3-omni-2b-ane-coreml'
DIR = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6'
api.upload_folder(folder_path=DIR, repo_id=REPO, repo_type='model',
                  allow_patterns=['lm_head_a.mlpackage/*', 'lm_head_b.mlpackage/*'],
                  commit_message='chunked LM head: 2 x 64,000 rows, placed on the engine')
print('uploaded', flush=True)
info = api.model_info(REPO, files_metadata=True)
names = [s.rfilename for s in info.siblings]
print({'head_files': [n for n in names if n.startswith('lm_head')][:6]}, flush=True)
print({'total_files': len(names), 'gb': round(sum(s.size or 0 for s in info.siblings) / 1e9, 2)},
      flush=True)