from huggingface_hub import HfApi
import subprocess

token = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dgx3', 'cat ~/.cache/huggingface/token'],
                       capture_output=True, text=True).stdout.strip()
api = HfApi(token=token)
repo = 'vladimir94/lfm3-omni-2b-ane-coreml'
files = [f for f in api.list_repo_tree(repo, repo_type='model', recursive=True) if hasattr(f, 'size')]
print({'files': len(files), 'GB': round(sum(f.size for f in files) / 1e9, 2),
       'has_towers': any(f.rfilename == 'towers.safetensors' for f in files)})