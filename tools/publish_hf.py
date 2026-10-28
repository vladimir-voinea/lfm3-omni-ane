# Publish the converted programs. The token is read from a file and never printed.
import os
import sys

from huggingface_hub import HfApi

REPO = 'vladimir94/lfm3-omni-2b-ane-coreml'
MODEL_DIR = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6'
CARD = '/Users/vladimir/dev/lfm3-ane-public-src/hf/README.md'

token = open('/tmp/.hf_token').read().strip()
api = HfApi(token=token)
who = api.whoami()
print({'account': who.get('name')}, flush=True)
api.create_repo(repo_id=REPO, repo_type='model', exist_ok=True, private=False)
print({'created': REPO}, flush=True)
api.upload_file(path_or_fileobj=CARD, path_in_repo='README.md', repo_id=REPO,
                repo_type='model', commit_message='model card')
print({'card': 'uploaded'}, flush=True)
api.upload_folder(folder_path=MODEL_DIR, repo_id=REPO, repo_type='model',
                  commit_message='LUT6 Core ML programs for LFM3-Omni-2B',
                  allow_patterns=['*.mlpackage/*', '*.npy', '*.npz', '*.json'])
print({'folder': 'uploaded'}, flush=True)