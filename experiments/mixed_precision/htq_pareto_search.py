#!/usr/bin/env python3
"""Generalization of HTQ/pareto_frontier.py's Node/tree-search: a layer-by-layer
branch-and-bound over per-node format choices, pruned at each layer to the
top-1/20 "profit" (cumulative sensitivity-weighted perturbation) band -- first
sorted by cumulative cost (weight bytes), then by cumulative BOPs proxy -- kept
as the union of both, so the search stays tractable without exhaustively
enumerating every format combination.

Differences from HTQ's original, both driven by this pass's own design rather
than a redesign of the algorithm:
- HTQ branches each layer into a fixed 6-way [weight_bits x activation_bits]
  grid (2/4/8 x 4/8, or a 3-way weight-only grid for "downsample" layers with no
  independent activation choice). Here every node -- weight-bearing or not --
  always gets its own activation-format choice (this pass's MIXEDP_NODE_FORMATS
  gives every supported op, including Relu/Add/Pool, an `a=` format), and only
  the weight-format axis is optional (fixed to "fp32" for non-weight-bearing
  ops); the branch grid size is len(weight_formats) x len(act_formats) (or just
  len(act_formats) with no weight choice), read from the caller's candidate
  lists instead of hardcoded.
- HTQ's own pareto_frontier() never reconstructs a concrete per-layer config
  from a leaf -- it only returns the three aggregate numbers (cost, BOPs,
  profit) for plotting. This repo needs the actual per-node format assignment
  to drive MIXEDP_NODE_FORMATS, so each Node keeps a `parent` pointer and its
  own (node_name, w_fmt, a_fmt) choice, and leaf_config() walks that chain back
  to the root to recover the full assignment.
"""
from math import floor

FORMAT_BITS = {
    "fp32": 32, "bf16": 16, "f16": 16,
    "int4": 4, "int5": 5, "int6": 6, "int7": 7, "int8": 8,
    "p4e1": 4, "p6e1": 6, "p8e1": 8,
}


class _Node:
    __slots__ = ("parent", "cost", "bops", "profit", "node_name", "w_fmt", "a_fmt")

    def __init__(self, parent, cost, bops, profit, node_name, w_fmt, a_fmt):
        self.parent, self.cost, self.bops, self.profit = parent, cost, bops, profit
        self.node_name, self.w_fmt, self.a_fmt = node_name, w_fmt, a_fmt


def leaf_config(leaf):
    """Walk a leaf's parent chain back to the root, collecting each layer's own
    choice -- the same per-node shape random_config()/uniform_config() produce."""
    formats = {}
    node = leaf
    while node.parent is not None:
        formats[node.node_name] = {"w": node.w_fmt, "a": node.a_fmt}
        node = node.parent
    return formats


def _prune_by(next_list, order_key, prune_fraction):
    """Faithful port of HTQ/pareto_frontier.py's pruning loop: walk `next_list`
    sorted by `order_key` (cost, then separately BOPs), keeping a node only if
    its profit isn't much worse than the profit floor(prune_fraction*len) back
    in the list built so far. The returned list stays in ascending `order_key`
    order (a strict subsequence of `ordered`)."""
    ordered = sorted(next_list, key=order_key)
    pruned = []
    for node in ordered:
        idx = floor(prune_fraction * len(pruned))
        if len(pruned) == 0 or node.profit <= pruned[-idx].profit:
            pruned.append(node)
    return pruned


def _cap_evenly(ordered_list, cap):
    """Evenly-strided subsample across `ordered_list` (which must already be
    sorted by the axis whose spread we want to preserve -- cost or BOPs),
    keeping representatives across the full low-to-high range. Capping by
    profit instead (sort by profit, keep the lowest) would systematically
    collapse the frontier toward whichever format is cheapest in profit (fp32
    weight quantizes to exactly 0 perturbation, so it never loses a profit
    comparison regardless of its cost) -- confirmed empirically: doing that
    made every single surviving leaf keep every weight at fp32."""
    if len(ordered_list) <= cap:
        return ordered_list
    if cap <= 1:
        return ordered_list[:1]
    idxs = sorted({round(i * (len(ordered_list) - 1) / (cap - 1)) for i in range(cap)})
    return [ordered_list[i] for i in idxs]


def pareto_search(nodes, weight_formats, act_formats, sensitivity_w, sensitivity_a, weight_table,
                   activation_table, weight_elems, flops_proxy, prune_fraction=1 / 20,
                   beam_width=2000, log=lambda msg: None):
    """nodes: supported_nodes() output, in ONNX graph order. sensitivity_w:
    {node_name: score}, weight sensitivity (0.0 for nodes with no weight, e.g.
    Relu/Add/Pool). sensitivity_a: {node_name: score}, activation sensitivity
    (every supported node has one). htq_snip_ort.py's --mode shared makes
    sensitivity_a == sensitivity_w (HTQ's original weight-only sensitivity
    reused for both terms); --mode split_connection / split_direct compute
    each independently (see that file's docstring for the exact definitions).
    weight_table/activation_table:
    htq_perturbation.build_perturbation_tables() output. weight_elems:
    {node_name: int} for weight-bearing nodes. flops_proxy: {node_name: float}
    for every node (0.0 where not meaningful, e.g. Relu/Add/Pool).

    beam_width: hard per-axis cap on by_cost/by_bops after each layer's pruning
    (so the union is at most 2*beam_width), kept as an evenly-strided
    subsample across each list's already-sorted cost/BOPs range (see
    _cap_evenly) -- NOT by lowest profit, which would collapse the whole
    frontier toward whichever format is cheapest in profit regardless of cost
    (fp32 weight always costs exactly 0 perturbation, so it never loses a
    profit comparison). HTQ's own pruning threshold is a FRACTION of the
    current list's length (floor(prune_fraction*len(pruned))), which only
    stays small in HTQ's own paper setting (21 tunable nodes, a 6-way branch
    grid) -- on a real 60+-node network with this pass's larger per-node
    format grid, that fraction-of-a-growing-list threshold gets looser every
    layer and the frontier compounds into millions of nodes within ~20 layers
    (confirmed: unbounded, it does not converge on its own). beam_width is an
    additive cap on top of HTQ's original two-pass cost/BOPs pruning, not a
    replacement for it -- it only fires when that pruning still leaves more
    than beam_width survivors on a given axis.

    Returns the final list of leaf _Node objects (call leaf_config() on each to
    get its concrete per-node format assignment).
    """
    root = _Node(parent=None, cost=0.0, bops=0.0, profit=0.0, node_name=None, w_fmt=None, a_fmt=None)
    current = [root]

    for layer_id, n in enumerate(nodes):
        name = n["name"]
        has_weight = bool(n.get("weight_const"))
        w_choices = weight_formats if has_weight else ["fp32"]
        elems = weight_elems.get(name, 0)
        fp = flops_proxy.get(name, 0.0)
        sens_w = sensitivity_w.get(name, 0.0)
        sens_a = sensitivity_a.get(name, 0.0)

        next_list = []
        for parent in current:
            for w_fmt in w_choices:
                w_bits = FORMAT_BITS[w_fmt]
                w_err = weight_table.get(name, {}).get(w_fmt, 0.0) if has_weight else 0.0
                w_bytes = elems * w_bits / 8 if has_weight else 0.0
                for a_fmt in act_formats:
                    a_bits = FORMAT_BITS[a_fmt]
                    a_err = activation_table[name][a_fmt]
                    next_list.append(_Node(
                        parent=parent,
                        cost=parent.cost + w_bytes,
                        bops=parent.bops + fp * w_bits * a_bits,
                        profit=parent.profit + sens_w * w_err + sens_a * a_err,
                        node_name=name, w_fmt=w_fmt if has_weight else None, a_fmt=a_fmt,
                    ))

        by_cost = _cap_evenly(_prune_by(next_list, lambda x: x.cost, prune_fraction), beam_width)
        by_bops = _cap_evenly(_prune_by(next_list, lambda x: x.bops, prune_fraction), beam_width)
        current = list(set(by_cost) | set(by_bops))
        log(f"  layer {layer_id} ({name}): {len(next_list)} candidates -> "
            f"{len(by_cost)} by cost, {len(by_bops)} by bops, {len(current)} kept")

    return current
