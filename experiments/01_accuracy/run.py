#!/usr/bin/env python
"""Accuracy experiment entry point (paper Table 3 / Appendix A).

Thin driver: load config -> for each model/split/sub-task, run the model and
record token-prediction accuracy. All knobs (models, splits, sub-tasks, paths)
come from config/CLI — nothing is hard-coded in a main() (CLAUDE.md hard-rule 1).

Migrated from get_accuracy.py. Output paths and per-file contents match the
original (results/accuracy/<folder>/<split>_last_paren_<n>.json), so results are
directly comparable to the regression anchor.

Usage:
    python experiments/01_accuracy/run.py                      # all 7 models (config default)
    python experiments/01_accuracy/run.py --models gpt2        # canary
    python experiments/01_accuracy/run.py --limit 50           # quick smoke (first N examples/file)
"""
import argparse
import json
import os
import sys

from lmfi.analysis import compute_accuracy
from lmfi.data import SubTask, labeled_path, model_folder
from lmfi.io import clear_cache, read_json, save_file
from lmfi.models import load_model


def load_registry(path):
    reg = read_json(path)
    return {m["name"]: m.get("cache") for m in reg["models"]}


def parse_args():
    p = argparse.ArgumentParser(description="Balanced-parentheses accuracy experiment.")
    p.add_argument("--config", default="configs/accuracy.json")
    p.add_argument("--registry", default="configs/models.json")
    p.add_argument("--models", nargs="+", default=None, help="override model list")
    p.add_argument("--splits", nargs="+", default=None, help="override splits")
    p.add_argument("--n-subtasks", type=int, default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--results-root", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, only the first N examples per file (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)
    registry = load_registry(args.registry)

    models = args.models if args.models is not None else cfg["models"]
    splits = args.splits if args.splits is not None else cfg["splits"]
    n_subtasks = args.n_subtasks if args.n_subtasks is not None else cfg["n_subtasks"]
    data_root = args.data_root if args.data_root is not None else cfg["data_root"]
    results_root = args.results_root if args.results_root is not None else cfg["results_root"]
    label_pos = cfg.get("label_pos", "last_paren")

    summary = {}
    for model_name in models:
        cache_dir = registry.get(model_name)
        print(f"\n=== {model_name} (cache={cache_dir}) ===", flush=True)
        model = load_model(model_name, cache_dir)
        folder = model_folder(model_name)
        summary[folder] = {}

        for split in splits:
            for st in (SubTask(i) for i in range(n_subtasks)):
                data_path = labeled_path(data_root, model_name, split, st, label_pos)
                if not os.path.exists(data_path):
                    print(f"  [skip] missing {data_path}", flush=True)
                    continue
                data = read_json(data_path)
                if args.limit > 0:
                    data = data[: args.limit]

                results = compute_accuracy(model, data)

                out_path = os.path.join(results_root, folder, f"{split}_{label_pos}_{st.index}.json")
                save_file(results, out_path)
                acc = results["accuracy"]
                summary[folder][f"{split}/{st.name}"] = round(acc, 4)
                print(f"  {split:5s} {st.name}: acc={acc:.4f}  (n={len(data)}) -> {out_path}", flush=True)

        del model
        clear_cache()

    print("\n=== SUMMARY (accuracy) ===")
    print(json.dumps(summary, indent=2))
    # persist a compact summary next to the per-file results
    save_file(summary, os.path.join(results_root, "summary.json"))
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
