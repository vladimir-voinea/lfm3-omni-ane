import numpy as np, json
import coremltools as ct
sys_path = '/Users/vladimir/dev/lfm3-ane/tools'
import sys; sys.path.insert(0, sys_path)
import build_tiled as B

MODEL = '/Users/vladimir/dev/lfm3-ane/model-28k-lut6/layer2_full.mlpackage'
spec = ct.models.MLModel(MODEL).get_spec()
function = list(spec.mlProgram.functions.values())[0]
block = function.block_specializations[list(function.block_specializations)[0]]
by_output = {op.outputs[0].name: op for op in block.operations if op.outputs}


def decode(weight):
    field = weight.WhichOneof('value')
    if field == 'floatValue':
        return field, np.array(list(weight.floatValue), dtype=np.float64)
    if field == 'float16Value':
        return field, np.frombuffer(bytes(weight.float16Value), dtype=np.float16).astype(np.float64)
    if field == 'blobFileValue':
        return field, None
    return field, None


rows = []
for op in block.operations:
    if op.type != 'constexpr_lut_to_dense':
        continue
    dims = [dimension.constant.size for dimension in op.outputs[0].type.tensorType.dimensions]
    names = list(op.inputs)
    lut_op = by_output.get(names[0])
    if lut_op is None or not lut_op.attributes:
        continue
    field, values = decode(lut_op.attributes[0].val)
    if values is None:
        rows.append((dims, field, None))
    else:
        rows.append((dims, field, (len(values), round(float(values.min()), 5),
                                  round(float(values.max()), 5), round(float(values.std()), 5))))
for dims, field, stats in rows:
    print({'dims': dims, 'field': field, 'lut': stats})