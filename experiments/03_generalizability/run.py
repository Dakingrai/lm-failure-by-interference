#!/usr/bin/env python
"""Generalizability experiment (Section 3.2 -> Table 1 + head/neuron generalization).

Pure-CPU post-processing of the projection output. Migrated from
analyze_attn_proj.py + analyze_neuron.py. No model forward; a tokenizer is loaded
only to decode the closing-paren token ids (as the originals did via the model's
tokenizer). Config/CLI-driven (CLAUDE.md hard-rule 1).

Determinism: the `random` stream is reseeded to 42 at the start of each component
(attn, neuron) -- the two originals were separate scripts that each seeded 42 at
import. Within a component, models/sub-tasks/heads are processed in the original
order so the random draws (precision-set sampling) match.

Outputs (per model folder):
  attn   -> results/attn_results/<folder>/{1..4}_results.json, macro_metrics.json,
            head_generalization_0.json   (+ the same under .../with_rank/)
  neuron -> results/mlp_results/<folder>/proj/{1..4}_results.json, macro_metrics.json,
            neuron_generalization_0.json, neuron_coeffs.json
  Table 1 -> results/meta_results/generalization_heads.json

The broken analyze_neuron main() stage-2 (NameError; wrote a redundant flat
neuron_coeffs.json that feeds no figure) is intentionally dropped -- the rich
neuron_coeffs.json from analyze_generalization is kept.
"""
import argparse
import glob
import os
import random
import re

from transformers import AutoTokenizer

from lmfi.analysis import generalizability as G
from lmfi.data import model_folder, paren_token_ids
from lmfi.io import read_json, save_file


def load_registry(path):
    return {m["name"]: m.get("cache") for m in read_json(path)["models"]}


def decode_subtask_parens(model_name, cache_dir, tids):
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    return [tok.decode(t) for t in tids]


def derive_heads(proj_dir):
    """Reconstruct the (layer, head) list in the original numeric order from the
    per-head projection files present."""
    n_layers = n_heads = 0
    for f in glob.glob(os.path.join(proj_dir, "L*_H*_proj.json")):
        m = re.search(r"L(\d+)_H(\d+)_proj\.json$", os.path.basename(f))
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)
            n_heads = max(n_heads, int(m.group(2)) + 1)
    return [(l, h) for l in range(n_layers) for h in range(n_heads)]


def run_attn(models, registry, cfg):
    random.seed(42)
    thresholds = cfg["thresholds"]
    tv = cfg["generalization_threshold"]
    table1 = {}
    for model_name in models:
        folder = model_folder(model_name)
        tids = paren_token_ids(cfg["data_root"], model_name)
        subtask_parens = decode_subtask_parens(model_name, registry.get(model_name), tids)
        proj_dir = os.path.join(cfg["proj_root"], folder, "attn", "proj")
        heads = derive_heads(proj_dir)
        print(f"  attn {folder}: {len(heads)} heads", flush=True)

        # non-rank then with-rank (preserves original random-consumption order)
        for with_rank in (False, True):
            result_path = os.path.join(cfg["attn_results_root"], folder, "with_rank" if with_rank else "")
            for sp in subtask_parens:
                G.process_attention_data(proj_dir, result_path, thresholds, heads, sp, with_rank)
            G.analyze_macro_f1_results(result_path, subtask_parens, thresholds)
            G.accuracy_generalization(result_path, subtask_parens, "heads",
                                      "head_generalization_0.json", threshold_value=tv)

        gen_path = os.path.join(cfg["attn_results_root"], folder, "with_rank", "head_generalization_0.json")
        table1[folder] = G.count_generalization(gen_path, "heads")
    save_file(table1, os.path.join(cfg["meta_root"], "generalization_heads.json"))
    print(f"  Table 1 (heads): {table1}", flush=True)


def run_neuron(models, registry, cfg):
    random.seed(42)
    thresholds = cfg["thresholds"]
    tv = cfg["generalization_threshold"]
    table1 = {}
    for model_name in models:
        folder = model_folder(model_name)
        tids = paren_token_ids(cfg["data_root"], model_name)
        subtask_parens = decode_subtask_parens(model_name, registry.get(model_name), tids)
        # The paren-neurons metadata lives in .../mlp/ (where projection writes it);
        # the per-neuron projection files live in .../mlp/proj/. The original
        # analyze_neuron read both from .../mlp/proj/, a path mismatch -- fixed here.
        mlp_dir = os.path.join(cfg["proj_root"], folder, "mlp")
        proj_dir = os.path.join(mlp_dir, "proj")
        result_path = os.path.join(cfg["mlp_results_root"], folder, "proj")
        neurons = G.get_mlp_neurons(mlp_dir, folder)
        print(f"  neuron {folder}: {len(neurons)} neurons", flush=True)

        for sp in subtask_parens:
            G.process_mlp_data(proj_dir, result_path, thresholds, neurons, sp)
        G.analyze_macro_f1_results(result_path, subtask_parens, thresholds)
        G.accuracy_generalization(result_path, subtask_parens, "neurons",
                                  "neuron_generalization_0.json", threshold_value=tv)
        G.analyze_generalization(result_path, proj_dir, subtask_parens, subtask_parens)

        gen_path = os.path.join(result_path, "neuron_generalization_0.json")
        table1[folder] = G.count_generalization(gen_path, "neurons")
    save_file(table1, os.path.join(cfg["meta_root"], "generalization_neurons.json"))
    print(f"  Table 1 (neurons): {table1}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Generalizability analysis (Table 1).")
    p.add_argument("--config", default="configs/generalizability.json")
    p.add_argument("--registry", default="configs/models.json")
    p.add_argument("--component", choices=["attn", "neuron", "both"], default="both")
    p.add_argument("--models", nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = read_json(args.config)
    registry = load_registry(args.registry)
    models = args.models if args.models is not None else cfg["models"]
    if args.component in ("attn", "both"):
        run_attn(models, registry, cfg)
    if args.component in ("neuron", "both"):
        run_neuron(models, registry, cfg)
    print("=== DONE generalizability ===")


if __name__ == "__main__":
    main()
