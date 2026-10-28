import collections, os, subprocess
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
pkg = "/tmp/v_state3d.mlpackage"
out = "/tmp/plan_detail"
subprocess.run(["rm", "-rf", out], check=False)
os.makedirs(out, exist_ok=True)
subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out], check=True, capture_output=True)
name = [f for f in os.listdir(out) if f.endswith(".mlmodelc")][0]
plan = MLComputePlan.load_from_path(os.path.join(out, name),
                                    compute_units=ct.ComputeUnit.CPU_AND_NE)
rows = []


def walk(block):
    for op in block.operations:
        try:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = type(usage.preferred_compute_device).__name__.replace("ComputeDevice", "")
        except Exception:
            dev = "unsupported"
        outs = ",".join(getattr(getattr(o, "value", o), "name", str(o))
                       for o in (getattr(op, "outputs", None) or []))
        rows.append((op.operator_name, dev, outs[:40]))
    for nested in getattr(block, "blocks", []) or []:
        walk(nested)


for fn in plan.model_structure.program.functions.values():
    walk(fn.block)
for op_name, dev, outs in rows:
    print("  %-16s %-22s %s" % (dev, op_name, outs))