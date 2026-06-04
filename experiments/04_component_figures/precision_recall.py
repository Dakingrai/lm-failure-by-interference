#!/usr/bin/env python
"""Fig 3 (+ appendix Fig 8): precision-recall scatter for heads and FF neurons.

Reads the generalizability results (no model load). Two presets select the model
set (CLAUDE.md rule 9): fig3 = {gpt2, CodeLlama-7b-hf}, fig8 = the other four.

Usage:
    python experiments/04_component_figures/precision_recall.py --preset fig3
    python experiments/04_component_figures/precision_recall.py --models gpt2
"""
import argparse
import os

from lmfi.data import model_folder
from lmfi.io import read_json
from lmfi.viz.precision_recall import plot_precision_recall, process_attn_results, process_mlp_results


def parse_args():
    p = argparse.ArgumentParser(description="Precision-recall scatter (Fig 3 / Fig 8).")
    p.add_argument("--config", default="configs/component_figures.json")
    p.add_argument("--preset", default=None, help="fig3 | fig8 | all")
    p.add_argument("--models", nargs="+", default=None, help="override the model list")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)["precision_recall"]
    if args.models is not None:
        models = args.models
    else:
        preset = args.preset or cfg["default_preset"]
        if preset == "all":
            models = [m for ms in cfg["presets"].values() for m in ms]
        else:
            models = cfg["presets"][preset]

    out_root = cfg["out_root"]
    for m in models:
        folder = model_folder(m)
        attn_path = os.path.join(cfg["attn_results_root"], folder)
        _, precisions, recalls = process_attn_results(attn_path, folder, out_root)
        plot_precision_recall(precisions, recalls, folder, out_root, neuron=False)

        mlp_path = os.path.join(cfg["mlp_results_root"], folder, "proj")
        mlp_p, mlp_r = process_mlp_results(mlp_path)
        plot_precision_recall(mlp_p, mlp_r, folder, out_root, neuron=True)
        print(f"  {folder}: attn heads={len(precisions)}, neurons={len(mlp_p)}", flush=True)
    print("=== DONE precision_recall ===")


if __name__ == "__main__":
    main()
