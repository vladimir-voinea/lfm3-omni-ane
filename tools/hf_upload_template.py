from huggingface_hub import HfApi
import subprocess

token = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'dgx3', 'cat ~/.cache/huggingface/token'],
                       capture_output=True, text=True).stdout.strip()
api = HfApi(token=token)
api.upload_file(path_or_fileobj='/Users/vladimir/dev/lfm3-ane/model-28k-lut6/chat_template.jinja',
                path_in_repo='chat_template.jinja',
                repo_id='vladimir94/lfm3-omni-2b-ane-coreml', repo_type='model',
                commit_message='add chat_template.jinja (needed for tool calling)')
info = api.model_info('vladimir94/lfm3-omni-2b-ane-coreml')
print({'files': len(info.siblings), 'has_template': any(s.rfilename == 'chat_template.jinja' for s in info.siblings)}, flush=True)