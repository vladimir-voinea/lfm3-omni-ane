# Vision tower -> Core ML. Unlike the text model this is one program: the tower is
# non-autoregressive, so it needs no state and no tiling.
import os
import sys

import numpy as np
import torch
import coremltools as ct
from safetensors import safe_open

sys.path.insert(0, '/Users/vladimir/dev/lfm3-ane/tools')
from vision_ref import P, Vision, projector

OUT = '/Users/vladimir/dev/lfm3-ane/vision.mlpackage'
SIZE = 224


class Tower(torch.nn.Module):
    def __init__(self, weights, types):
        super().__init__()
        self.tower = Vision(weights, types)
        self.weights = weights

    def forward(self, patches, gh, gw):
        hidden = self.tower.forward(patches, gh, gw)
        return projector(hidden, self.weights)


if __name__ == '__main__':
    import json
    weights = {}
    with safe_open(P, framework='pt') as handle:
        for key in handle.keys():
            if 'vision_tower' in key or 'multi_modal_projector' in key:
                weights[key] = handle.get_tensor(key).float()
    config = json.load(open('/Users/vladimir/dev/lfm3-ane/model-28k-lut6/config.json'))
    types = config['vision_config']['layer_types']
    print({'tensors': len(weights), 'layer_types': types[:4]}, flush=True)

    from vision_chat import patchify
    image = torch.zeros(3, SIZE, SIZE)
    image[0] = 1.0
    patches, gh, gw = patchify(image)
    print({'patches': list(patches.shape), 'grid': [gh, gw]}, flush=True)

    module = Tower(weights, types).eval()
    with torch.no_grad():
        reference = module(patches, gh, gw)
    print({'reference_rows': list(reference.shape)}, flush=True)

    traced = torch.jit.trace(module, (patches, torch.tensor(gh), torch.tensor(gw)))
    model = ct.convert(traced,
                       inputs=[ct.TensorType(name='patches', shape=tuple(patches.shape), dtype=np.float32),
                               ct.TensorType(name='gh', shape=(), dtype=np.int32),
                               ct.TensorType(name='gw', shape=(), dtype=np.int32)],
                       outputs=[ct.TensorType(name='rows', dtype=np.float32)],
                       compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.iOS18,
                       compute_units=ct.ComputeUnit.CPU_AND_NE)
    model.save(OUT)
    print({'saved': OUT}, flush=True)

    loaded = ct.models.MLModel(OUT, compute_units=ct.ComputeUnit.CPU_AND_NE)
    got = loaded.predict({'patches': patches.numpy(), 'gh': np.array(gh, np.int32),
                          'gw': np.array(gw, np.int32)})
    rows = np.asarray(list(got.values())[0], np.float32).reshape(reference.shape)
    ref = reference.numpy()
    import numpy as np2
    cos = float((rows * ref).sum() / (np.linalg.norm(rows) * np.linalg.norm(ref) + 1e-9))
    print({'cos_vs_torch': round(cos, 5), 'rel': round(float(np.linalg.norm(rows - ref) / np.linalg.norm(ref)), 5)},
          flush=True)