#!/usr/bin/env python3
"""
Convert a precision_plan.json's per-node "chosen" formats into a POSIT_NODE_FORMATS
env-var string, generalizing experiments/imagenet100/gen_node_formats.py's
range-spec DSL to read a plan file's `entities` dict directly instead.

Usage:
  eval "$(python plan_to_node_formats.py --plan precision_plan.json --export)"
  POSIT_NODE_FORMATS=$(python plan_to_node_formats.py --plan precision_plan.json --value-only)
"""

import argparse
import json


def plan_to_node_formats(plan):
    parts = []
    for node, entry in plan["entities"].items():
        chosen = entry["chosen"]
        if chosen == "FP32":
            continue
        nbits, es = chosen.replace("posit_", "").split("_")
        parts.append(f"{node}:{nbits}:{es}")
    return ",".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--export", action="store_true",
                     help="print as 'export POSIT_NODE_FORMATS=...' for eval $(...)")
    ap.add_argument("--value-only", action="store_true",
                     help="print only the value, no POSIT_NODE_FORMATS= prefix")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    value = plan_to_node_formats(plan)

    if args.value_only:
        print(value)
    elif args.export:
        print(f"export POSIT_NODE_FORMATS={value}")
    else:
        print(f"POSIT_NODE_FORMATS={value}")


if __name__ == "__main__":
    main()
