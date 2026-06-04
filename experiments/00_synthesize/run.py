#!/usr/bin/env python
"""Dataset synthesis entry point. Migrated from synthesize_data.py.

Regenerates the balanced-parentheses dataset (reproducibly, seeded). The committed
data/ is the canonical artifact; only run this to regenerate. No model forward --
just a tokenizer per model.

Usage:
    python experiments/00_synthesize/run.py                 # all 7 models
    python experiments/00_synthesize/run.py --models gpt2   # one model
"""
import argparse
import os
import random

from transformers import AutoTokenizer

from lmfi.data import model_folder
from lmfi.data.synthesis import base_prompts, label_prompt, train_dev_test_split
from lmfi.io import create_results_dir, read_json


def load_registry(path):
    return {m["name"]: m.get("cache") for m in read_json(path)["models"]}


def parse_args():
    p = argparse.ArgumentParser(description="Synthesize the balanced-parentheses dataset.")
    p.add_argument("--config", default="configs/synthesis.json")
    p.add_argument("--registry", default="configs/models.json")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)
    registry = load_registry(args.registry)

    models = args.models if args.models is not None else cfg["models"]
    data_root = args.data_root or cfg["data_root"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    n_paren = cfg["n_paren"]
    n_samples = cfg["n_samples"]

    for model_name in models:
        folder = model_folder(model_name)
        data_dir = os.path.join(data_root, folder)
        create_results_dir(data_dir)
        print(f"\n=== {model_name} -> {data_dir} ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=registry.get(model_name))

        random.seed(seed)
        all_prompts = base_prompts(n_paren, n_samples, seed)
        train, dev, test = train_dev_test_split(all_prompts, n_paren)
        label_prompt(train, tokenizer, os.path.join(data_dir, "train"), n_paren)
        label_prompt(dev, tokenizer, os.path.join(data_dir, "dev"), n_paren)
        label_prompt(test, tokenizer, os.path.join(data_dir, "test"), n_paren)
    print("\n=== DONE synthesize ===")


if __name__ == "__main__":
    main()
