#!/usr/bin/env python
"""Fig 2: dual-sign FF neuron (CodeLlama L19N11) means + bar chart.

Reads the neuron's projection file (no model load). Needs the projection output
for the configured model (default CodeLlama-7b-hf).

Usage:
    python experiments/04_component_figures/plot_dual_neuron.py
    python experiments/04_component_figures/plot_dual_neuron.py --model gpt2 --neurons 19 11
"""
import argparse

from lmfi.data import model_folder
from lmfi.io import read_json
from lmfi.viz.dual_neuron import dual_neuron_means, plot_neuron_results


def parse_args():
    p = argparse.ArgumentParser(description="Dual-sign neuron figure (Fig 2).")
    p.add_argument("--config", default="configs/component_figures.json")
    p.add_argument("--model", default=None, help="override model (HF id or folder)")
    p.add_argument("--neurons", nargs="+", type=int, default=None,
                   help="flat (layer neuron) pairs, e.g. --neurons 19 11 20 3998")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)["dual_neuron"]
    model = args.model or cfg["model"]
    folder = model_folder(model)
    if args.neurons is not None:
        flat = args.neurons
        neurons = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    else:
        neurons = [tuple(n) for n in cfg["neurons"]]

    first_results = None
    for neuron in neurons:
        results = dual_neuron_means(neuron, folder, cfg["proj_root"], cfg["out_root"])
        print(f"  L{neuron[0]}N{neuron[1]} means computed", flush=True)
        if first_results is None:
            first_results = results
    if first_results is not None:
        out = plot_neuron_results(first_results, cfg["out_root"])
        print(f"  plotted -> {out}", flush=True)
    print("=== DONE dual_neuron ===")


if __name__ == "__main__":
    main()
