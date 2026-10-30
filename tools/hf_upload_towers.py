from huggingface_hub import HfApi
import subprocess, os

token = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dgx3', 'cat ~/.cache/huggingface/token'],
                       capture_output=True, text=True).stdout.strip()
api = HfApi(token=token)
api.upload_file(
    path_or_fileobj='/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/towers.safetensors',
    path_in_repo='towers.safetensors', repo_id='vladimir94/lfm3-omni-2b-ane-coreml',
    repo_type='model',
    commit_message='towers.safetensors: the 0.88 GB the vision/audio/depth paths read')
print({'uploaded_GB': round(os.path.getsize('/Users/vladimir/dev/lfm3-ane/models/LFM3-Omni-2B/towers.safetensors') / 1e9, 2)}, flush=True)