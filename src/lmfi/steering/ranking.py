"""Rank LM components for steering (RASteer ordering).

Unifies get_heads (steer_attn) and get_neurons (steer_neuron): group components by
generalizability (4-general first ... 1-general last) and, within each group, sort
by the chosen reliability metric (descending). Returns a list of (layer, idx)
tuples in that order. Behaviour matches the originals exactly.
"""
from lmfi.io import read_json

_METRIC_KEY = {
    "f1-score": "macro_f1",
    "precision": "average_precision",
    "recall": "average_recall",
}


def parse_attn_name(name):
    """'L{layer}_H{head}' -> (layer, head)."""
    name = name.split("L")
    layer = int(name[1].split("_")[0])
    head = int(name[1].split("_")[1].split("H")[1])
    return (layer, head)


def parse_mlp_name(name):
    """'L{layer}N{neuron}' -> (layer, neuron)."""
    layer = int(name.split("N")[0].split("L")[1])
    neuron = int(name.split("N")[1])
    return (layer, neuron)


def rank_components(generalization_path, macro_path, noun, metric="f1-score"):
    """`noun` in {"heads","neurons"}. Returns ranked [(layer, idx), ...].

    4-general group first, then 3-, 2-, 1-general; within each, sorted by `metric`
    descending. Mirrors steer_attn.get_heads / steer_neuron.get_neurons.
    """
    if metric not in _METRIC_KEY:
        raise ValueError("Invalid metric!")
    key = _METRIC_KEY[metric]
    parse = parse_attn_name if noun == "heads" else parse_mlp_name

    groups = read_json(generalization_path)["generalization_groups"]
    macro = read_json(macro_path)

    def group_members(n):
        g = f"{n}-general-{noun}"
        return groups.get(g, {}).get(noun, []) if g in groups else []

    def ranked(members):
        scored = [(name, macro[name][key][0]) for name in members]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored]

    ordered = ranked(group_members(4)) + ranked(group_members(3)) \
        + ranked(group_members(2)) + ranked(group_members(1))
    return [parse(name) for name in ordered]
