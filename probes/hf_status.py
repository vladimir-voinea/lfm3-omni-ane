from huggingface_hub import HfApi
import subprocess

token = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dgx3', 'cat ~/.cache/huggingface/token'],
                       capture_output=True, text=True).stdout.strip()
api = HfApi(token=token)
remote = {f.rfilename: f.size for f in api.list_repo_tree(
    'vladimir94/lfm3-omni-2b-ane-coreml', repo_type='model', recursive=True) if hasattr(f, 'size')}
print({'remote_files': len(remote), 'remote_bytes': sum(remote.values())})
print({'rope.npz_remote_bytes': remote.get('rope.npz')})