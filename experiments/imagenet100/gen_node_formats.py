#!/usr/bin/env python3
"""
Generate POSIT_NODE_FORMATS strings for selective per-layer posit experiments.

Usage examples:
  # All Conv/Gemm in resnet18 → p8e1
  python gen_node_formats.py resnet18 all:8:1

  # First 10 Conv layers → p8e1, rest → p16e2
  python gen_node_formats.py resnet18 0-9:8:1 10-20:16:2

  # Only Gemm layer → p16e2 (by type index)
  python gen_node_formats.py resnet18 Gemm0:16:2

  # mobilenetv2: first half Conv → p8e1, second half → p8e0
  python gen_node_formats.py mobilenetv2 0-25:8:1 26-51:8:0

Output: POSIT_NODE_FORMATS=node_Conv_291:8:1,...
"""

import json
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")

OP_FILES = {
    "resnet18":    "resnet18_posit_ops.json",
    "mobilenetv2": "mobilenetv2_posit_ops.json",
}


def load_ops(model_name):
    path = os.path.join(MODEL_DIR, OP_FILES[model_name])
    with open(path) as f:
        return json.load(f)


def parse_spec(spec, ops):
    """
    Parse one range spec and return list of (name, nbits, es).
    Formats:
      all:N:E            - all ops
      A-B:N:E            - global_idx range [A, B] inclusive
      ConvA-B:N:E        - type Conv, type_idx range [A, B]
      TypeK:N:E          - specific type_idx K (e.g. Gemm0, Conv3)
    """
    import re

    # all:N:E
    m = re.fullmatch(r'all:(\d+):(\d+)', spec)
    if m:
        nb, es = int(m.group(1)), int(m.group(2))
        return [(op['name'], nb, es) for op in ops]

    # A-B:N:E  (global index range)
    m = re.fullmatch(r'(\d+)-(\d+):(\d+):(\d+)', spec)
    if m:
        a, b, nb, es = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return [(op['name'], nb, es) for op in ops if a <= op['global_idx'] <= b]

    # Conv0-9:N:E  (type + type_idx range)
    m = re.fullmatch(r'([A-Za-z]+)(\d+)-(\d+):(\d+):(\d+)', spec)
    if m:
        typ, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        nb, es = int(m.group(4)), int(m.group(5))
        return [(op['name'], nb, es) for op in ops
                if op['type'] == typ and a <= op['type_idx'] <= b]

    # Gemm0:N:E  (type + single type_idx)
    m = re.fullmatch(r'([A-Za-z]+)(\d+):(\d+):(\d+)', spec)
    if m:
        typ, k, nb, es = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return [(op['name'], nb, es) for op in ops
                if op['type'] == typ and op['type_idx'] == k]

    raise ValueError(f"Cannot parse spec: {spec!r}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in OP_FILES:
        print(f"ERROR: unknown model '{model_name}'. Choose: {list(OP_FILES)}")
        sys.exit(1)

    ops = load_ops(model_name)
    specs = sys.argv[2:]

    # Accumulate assignments; later specs override earlier ones for the same node.
    assignments = {}
    for spec in specs:
        for name, nb, es in parse_spec(spec, ops):
            assignments[name] = (nb, es)

    if not assignments:
        print("ERROR: no ops matched the given specs", file=sys.stderr)
        sys.exit(1)

    parts = [f"{name}:{nb}:{es}" for name, (nb, es) in assignments.items()]
    print(f"POSIT_NODE_FORMATS={','.join(parts)}")

    # also show human summary
    print(f"\n# {len(assignments)} ops assigned:", file=sys.stderr)
    for name, (nb, es) in assignments.items():
        print(f"#   {name:30s} → p{nb}e{es}", file=sys.stderr)


if __name__ == "__main__":
    main()
