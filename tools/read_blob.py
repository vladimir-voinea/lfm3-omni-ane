import numpy as np, os
import coremltools as ct
PKG = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/layer2_full.mlpackage'
spec = ct.models.MLModel(PKG).get_spec()
function = list(spec.mlProgram.functions.values())[0]
block = function.block_specializations[list(function.block_specializations)[0]]
blob = open(os.path.join(PKG, 'Data/com.apple.CoreML/weights/weight.bin'), 'rb').read()
print('blob_bytes:', len(blob))
for op in block.operations:
    if op.type != 'const' or 'val' not in list(op.attributes):
        continue
    weight = op.attributes['val']
    if weight.WhichOneof('value') != 'blobFileValue':
        continue
    dims = [d.constant.size for d in op.outputs[0].type.tensorType.dimensions] if op.outputs else []
    count = 1
    for size in dims:
        count *= size
    if count > 400000 or count == 0:
        continue
    offset = weight.blobFileValue.offset
    values = np.frombuffer(blob[offset:offset + 2 * count], dtype=np.float16).astype(np.float64)
    print({'op': op.outputs[0].name, 'dims': dims, 'offset': offset,
           'min': round(float(values.min()), 5), 'max': round(float(values.max()), 5),
           'std': round(float(values.std()), 5)})