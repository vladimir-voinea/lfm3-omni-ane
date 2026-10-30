from huggingface_hub import HfApi
import subprocess

token = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dgx3', 'cat ~/.cache/huggingface/token'],
                       capture_output=True, text=True).stdout.strip()
api = HfApi(token=token)
api.upload_file(path_or_fileobj='/Users/vladimir/dev/lfm3-ane/model-28k-lut6/rope.npz',
                path_in_repo='rope.npz', repo_id='vladimir94/lfm3-omni-2b-ane-coreml',
                repo_type='model',
                commit_message='rope.npz: sliding tables cover the full context (256 rows broke >255 tokens)')
print('uploaded', flush=True)