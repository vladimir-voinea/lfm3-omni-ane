from huggingface_hub import HfApi

api = HfApi(token=open('/tmp/.hf_token').read().strip())
REPO = 'vladimir94/lfm3-omni-2b-ane-coreml'
DIR = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/'
for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json',
             'generation_config.json', 'layer_types.json'):
    api.upload_file(path_or_fileobj=DIR + name, path_in_repo=name, repo_id=REPO,
                    repo_type='model', commit_message='add ' + name)
    print('uploaded', name, flush=True)
info = api.model_info(REPO, files_metadata=True)
print({'files': len(info.siblings),
       'gb': round(sum(s.size or 0 for s in info.siblings) / 1e9, 2)}, flush=True)