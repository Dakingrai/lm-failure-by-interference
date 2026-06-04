#!/usr/bin/env python
"""Projection experiment entry point (the hub).

Computes per-component logit-lens projections that feed generalizability
(Table 1 / Fig 1) and steering. Migrated from proj_attn.py + proj_neuron.py.
All knobs come from config/CLI (CLAUDE.md hard-rule 1). Output paths and file
contents match the originals so results are directly comparable.

Outputs (per model folder):
  attn   -> results/projections/<folder>/attn/proj/L{l}_H{h}_proj.json
  neuron -> results/projections/<folder>/mlp/<model_name>_paren_neurons.json
            results/projections/<folder>/mlp/proj/L{l}N{n}_proj.json

Usage:
    python experiments/02_projection/run.py --component attn   --models gpt2
    python experiments/02_projection/run.py --component neuron --models gpt2
    # fast subset (for validation): cap examples and components
    python experiments/02_projection/run.py --component attn --models gpt2 \
        --limit 10 --max-heads 8
"""
import argparse
import os

from lmfi.analysis import find_paren_neurons, run_attn_projection, run_neuron_projection
from lmfi.data import SubTask, labeled_path, model_folder, paren_token_ids
from lmfi.io import clear_cache, read_json, save_file
from lmfi.models import load_model


def load_registry(path):
    return {m["name"]: m.get("cache") for m in read_json(path)["models"]}


def load_train(data_root, model_name, n_subtasks, limit_per_subtask):
    """Concatenate train data across sub-tasks, optionally capping each
    sub-task to the first `limit_per_subtask` examples (0 = all)."""
    data = []
    for i in range(n_subtasks):
        rows = read_json(labeled_path(data_root, model_name, "train", SubTask(i)))
        if limit_per_subtask and limit_per_subtask > 0:
            rows = rows[:limit_per_subtask]
        data += rows
    return data


def run_attn(model, model_name, cfg, args, results_root):
    folder = model_folder(model_name)
    limit = args.limit if args.limit is not None else cfg["attn"].get("train_limit_per_subtask", 0)
    data = load_train(cfg["data_root"], model_name, cfg["n_subtasks"], limit)
    tids = paren_token_ids(cfg["data_root"], model_name, cfg["n_subtasks"])

    heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    if args.max_heads:
        heads = heads[: args.max_heads]
    print(f"  attn: {len(heads)} heads x {len(data)} examples", flush=True)

    all_results = run_attn_projection(model, data, heads, tids)

    proj_dir = os.path.join(results_root, folder, "attn", "proj")
    for layer, head in heads:
        save_file(all_results[(layer, head)], os.path.join(proj_dir, f"L{layer}_H{head}_proj.json"))
    print(f"  attn: wrote {len(heads)} files -> {proj_dir}", flush=True)


def run_neuron(model, model_name, cfg, args, results_root):
    folder = model_folder(model_name)
    ncfg = cfg["neuron"]
    limit = args.limit if args.limit is not None else ncfg.get("train_limit_per_subtask", 50)
    data = load_train(cfg["data_root"], model_name, cfg["n_subtasks"], limit)
    tids = paren_token_ids(cfg["data_root"], model_name, cfg["n_subtasks"])

    mlp_dir = os.path.join(results_root, folder, "mlp")
    neurons = find_paren_neurons(model, tids, top_k=ncfg.get("filter_top_k", 50))
    if args.max_neurons:
        neurons = neurons[: args.max_neurons]
    save_file(neurons, os.path.join(mlp_dir, f"{model.cfg.model_name}_paren_neurons.json"))
    print(f"  neuron: {len(neurons)} paren-neurons x {len(data)} examples", flush=True)

    all_results = run_neuron_projection(model, data, neurons, tids,
                                        settle_seconds=ncfg.get("settle_seconds", 0.0))
    proj_dir = os.path.join(mlp_dir, "proj")
    for neuron in neurons:
        key = f"L{neuron['layer']}N{neuron['neuron_idx']}"
        save_file(all_results[key], os.path.join(proj_dir, f"{key}_proj.json"))
    print(f"  neuron: wrote {len(neurons)} files -> {proj_dir}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Per-component logit projection experiment.")
    p.add_argument("--config", default="configs/projection.json")
    p.add_argument("--registry", default="configs/models.json")
    p.add_argument("--component", choices=["attn", "neuron", "both"], default="both")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--results-root", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="cap train examples per sub-task (overrides config; subset/smoke)")
    p.add_argument("--max-heads", type=int, default=0, help="subset of heads (validation)")
    p.add_argument("--max-neurons", type=int, default=0, help="subset of neurons (validation)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)
    registry = load_registry(args.registry)
    if args.data_root:
        cfg["data_root"] = args.data_root
    results_root = args.results_root or cfg["results_root"]
    models = args.models if args.models is not None else cfg["models"]

    for model_name in models:
        print(f"\n=== {model_name} (component={args.component}) ===", flush=True)
        model = load_model(model_name, registry.get(model_name))
        if args.component in ("attn", "both"):
            run_attn(model, model_name, cfg, args, results_root)
        if args.component in ("neuron", "both"):
            run_neuron(model, model_name, cfg, args, results_root)
        del model
        clear_cache()
    print("\n=== DONE projection ===")


if __name__ == "__main__":
    main()
