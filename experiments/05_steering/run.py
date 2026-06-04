#!/usr/bin/env python
"""Unified steering experiment entry point (RASteer -> Fig 4, Appendix D).

One driver for component {attn,neuron,both} x mode {sweep,analyze,apply}.
Config/CLI-driven (CLAUDE.md hard-rule 1). Replaces steer_{attn,neuron,both},
steer_{attn,both}_test and analyze_dev_steer.

Pipeline:
    # 1. dev sweep (full grid) -- needs a GPU
    run.py --component attn --mode sweep --split dev
    # 2. analyze the dev sweep -> coeffs_attn.json (CPU)
    run.py --component attn --mode analyze
    # 3. apply the best dev coeff on the test set (GPU)
    run.py --component attn --mode apply

`--models/--metric/--subtasks/--n-grid/--coeffs` override the per-component config.
"""
import argparse
import os

from lmfi.data import model_folder
from lmfi.io import clear_cache, read_json, save_file
from lmfi.models import load_model
from lmfi.steering import driver


def load_registry(path):
    return {m["name"]: m.get("cache") for m in read_json(path)["models"]}


def parse_args():
    p = argparse.ArgumentParser(description="Unified steering driver.")
    p.add_argument("--config", default="configs/steering.json")
    p.add_argument("--registry", default="configs/models.json")
    p.add_argument("--component", choices=["attn", "neuron", "both"], required=True)
    p.add_argument("--mode", choices=["sweep", "analyze", "apply"], required=True)
    p.add_argument("--split", choices=["dev", "test"], default="dev", help="sweep split")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--metric", default=None, help="single metric override")
    p.add_argument("--subtasks", nargs="+", type=int, default=None)
    p.add_argument("--n-grid", nargs="+", type=int, default=None)
    p.add_argument("--coeffs", nargs="+", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)
    registry = load_registry(args.registry)
    ccfg = cfg["components"][args.component]

    models = args.models if args.models is not None else ccfg["models"]
    metrics = [args.metric] if args.metric else ccfg["metrics"]
    subtasks = args.subtasks if args.subtasks is not None else ccfg["subtasks"]
    n_grid = args.n_grid if args.n_grid is not None else cfg["n_grid"]
    coeffs = args.coeffs if args.coeffs is not None else cfg["coeffs"]

    if args.mode == "analyze":
        # coeffs are derived from the f1-score dev sweep (as in the original)
        metric = args.metric or "f1-score"
        all_acc, all_coeffs = {}, {}
        for m in models:
            folder = model_folder(m)
            acc, co = driver.analyze(args.component, folder, metric, cfg, subtasks, n_grid, coeffs)
            all_acc[folder] = acc
            all_coeffs[folder] = co
        save_file(all_acc, os.path.join(cfg["meta_root"], f"{metric}_{args.component}_accuracy.json"))
        save_file(all_coeffs, os.path.join(cfg["meta_root"], f"coeffs_{args.component}.json"))
        print(f"=== analyze {args.component}: wrote coeffs_{args.component}.json ===")
        return

    # sweep / apply both need the model on GPU
    for m in models:
        folder = model_folder(m)
        print(f"\n=== {m} ({args.component}/{args.mode}) ===", flush=True)
        model = load_model(m, registry.get(m))
        if args.mode == "sweep":
            for metric in metrics:
                driver.sweep(model, args.component, folder, args.split, metric, cfg,
                             subtasks, n_grid, coeffs)
        elif args.mode == "apply":
            metric = args.metric or "f1-score"
            coeffs_data = read_json(os.path.join(cfg["meta_root"], f"coeffs_{args.component}.json"))
            driver.apply_best(model, args.component, folder, metric, cfg, coeffs_data[folder])
        del model
        clear_cache()
    print(f"=== DONE steering {args.component}/{args.mode} ===")


if __name__ == "__main__":
    main()
