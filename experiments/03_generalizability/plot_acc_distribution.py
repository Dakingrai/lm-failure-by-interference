#!/usr/bin/env python
"""Fig 1 (and appendix Fig 7): per-head accuracy distribution histograms.

Reads the with-rank head results produced by run.py and writes one PNG per
(model, n_paren). Default model set = all 7 (the [:2] in the original was a debug
leftover; Fig 1 = CodeLlama, the rest = appendix). Run after run.py --component attn.

Usage:
    python experiments/03_generalizability/plot_acc_distribution.py --models CodeLlama-7b-hf
"""
import argparse
import os

from lmfi.data import model_folder
from lmfi.io import read_json
from lmfi.viz.attn_acc_distribution import plot_acc_distribution


def parse_args():
    p = argparse.ArgumentParser(description="Fig 1 head-accuracy distributions.")
    p.add_argument("--config", default="configs/generalizability.json")
    p.add_argument("--models", nargs="+", default=None,
                   help="model folder names (e.g. CodeLlama-7b-hf) or HF ids")
    p.add_argument("--attn-results-root", default=None)
    p.add_argument("--out-root", default="results/plot_results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)
    attn_root = args.attn_results_root or cfg["attn_results_root"]
    models = args.models if args.models is not None else cfg["models"]
    for m in models:
        folder = model_folder(m)
        for n_paren in range(1, 5):
            out = plot_acc_distribution(attn_root, folder, n_paren, args.out_root)
            print(f"  {out}", flush=True)


if __name__ == "__main__":
    main()
