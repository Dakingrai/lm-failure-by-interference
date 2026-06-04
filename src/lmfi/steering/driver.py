"""Unified steering driver (RASteer): component {attn,neuron,both} x split {dev,test}.

Collapses steer_attn / steer_neuron / steer_both (+ the *_test variants) and
analyze_dev_steer into one place, on a single canonical path scheme so the sweep
writer and the analysis reader always agree (this dissolves the §3 path mismatches
in the originals, where analyze_dev_steer read paths the sweeps never wrote).

Modes:
  * sweep  : full grid (n_components x coeff x sub-task) on a split; one accuracy
             file per config. Reproduces steer_attn (dev), steer_both (dev/test),
             steer_neuron (the neuron sweep).
  * analyze: from the dev sweep, pick the best coeff + best accuracy per
             (sub-task, n_components) -> coeffs_<component>.json (+ accuracy meta).
             Reproduces analyze_dev_steer.attn_accuracy_analysis.
  * apply  : apply the best dev coeff per config on the test split. Reproduces
             steer_attn_test; the neuron/both test paths are *built* here on the
             same scheme (the original steer_neuron_test was broken and deleted).

The per-example predict rule is identical to the accuracy experiment, wrapped in
the intervention context (the originals' per-example gc/empty_cache is dropped --
timing only, no effect on the saved accuracy/detail).
"""
import os

import torch

from lmfi.data import SubTask, labeled_path, model_folder
from lmfi.io import read_json, save_file
from lmfi.steering.interventions import InterveneMLPNeuron, InterveneOV, InterveneOV_Neuron
from lmfi.steering.ranking import rank_components

_UNIT = {"attn": "heads", "neuron": "neurons", "both": "both"}


# --------------------------------------------------------------------------- #
# Paths (single source of truth for writer + reader)
# --------------------------------------------------------------------------- #
def sweep_dir(steer_root, folder, component, split, metric):
    return os.path.join(steer_root, folder, component, split, metric)


def config_filename(component, n, coeff, k):
    return f"subtask-{n}-coeff-{coeff}-{_UNIT[component]}-{k}.json"


# --------------------------------------------------------------------------- #
# Component ranking + intervention context
# --------------------------------------------------------------------------- #
def ranked_for(component, folder, metric, cfg):
    heads, neurons = [], []
    if component in ("attn", "both"):
        gp = os.path.join(cfg["attn_results_root"], folder, "head_generalization_0.json")
        mp = os.path.join(cfg["attn_results_root"], folder, "macro_metrics.json")
        heads = rank_components(gp, mp, "heads", metric)
    if component in ("neuron", "both"):
        gp = os.path.join(cfg["mlp_results_root"], folder, "proj", "neuron_generalization_0.json")
        mp = os.path.join(cfg["mlp_results_root"], folder, "proj", "macro_metrics.json")
        neurons = rank_components(gp, mp, "neurons", metric)
    return heads, neurons


def make_ctx(component, model, heads_k, neurons_k, coeff):
    if component == "attn":
        return lambda: InterveneOV(model, heads_k, coeff)
    if component == "neuron":
        return lambda: InterveneMLPNeuron(model, neurons_k, coeff)
    if component == "both":
        return lambda: InterveneOV_Neuron(model, heads_k, neurons_k, coeff)
    raise ValueError(component)


def evaluate(model, data, ctx_factory):
    """Predict the last token under the intervention; return accuracy + details.
    Same prediction rule as lmfi.analysis.compute_accuracy."""
    total = 0
    correct = 0
    detail = []
    for each in data:
        total += 1
        with ctx_factory():
            logits = model(each["prompt"], return_type="logits")
        l = logits.argmax(dim=-1).squeeze()[-1]
        pred = model.to_string(l)
        tmp = {"prompt": each["prompt"], "label": each["label"], "pred": pred}
        tmp["correct"] = (pred == each["label"])
        if tmp["correct"]:
            correct += 1
        detail.append(tmp)
    return {"accuracy": correct / total, "detail_results": detail}


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def sweep(model, component, folder, split, metric, cfg, subtasks=None, n_grid=None, coeffs=None):
    subtasks = cfg["subtasks"] if subtasks is None else subtasks
    n_grid = cfg["n_grid"] if n_grid is None else n_grid
    coeffs = cfg["coeffs"] if coeffs is None else coeffs
    heads, neurons = ranked_for(component, folder, metric, cfg)
    out_dir = sweep_dir(cfg["steer_root"], folder, component, split, metric)

    for k in n_grid:
        heads_k, neurons_k = heads[:k], neurons[:k]
        ks_coeffs = [1] if k == 0 else coeffs
        for c in ks_coeffs:
            for n in subtasks:
                data = read_json(labeled_path(cfg["data_root"], folder, split, SubTask(n)))
                res = evaluate(model, data, make_ctx(component, model, heads_k, neurons_k, c))
                save_file(res, os.path.join(out_dir, config_filename(component, n, c, k)))
        print(f"  sweep {component}/{split}/{metric} k={k} done", flush=True)


def analyze(component, folder, metric, cfg, subtasks=None, n_grid=None, coeffs=None):
    """From the dev sweep, best accuracy + best coeff per (sub-task, k)."""
    subtasks = cfg["subtasks"] if subtasks is None else subtasks
    n_grid = cfg["n_grid"] if n_grid is None else n_grid
    coeffs = cfg["coeffs"] if coeffs is None else coeffs
    in_dir = sweep_dir(cfg["steer_root"], folder, component, "dev", metric)

    acc_out, coeff_out = {}, {}
    for n in subtasks:
        acc_out[f"{n}-paren"], coeff_out[f"{n}-paren"] = {}, {}
        for k in n_grid:
            ks_coeffs = [1] if k == 0 else coeffs
            best_acc, best_c = 0, 1
            for c in ks_coeffs:
                acc = read_json(os.path.join(in_dir, config_filename(component, n, c, k)))["accuracy"]
                if acc > best_acc:
                    best_acc, best_c = round(acc, 3), c
            acc_out[f"{n}-paren"][k] = best_acc
            coeff_out[f"{n}-paren"][k] = best_c
    return acc_out, coeff_out


def apply_best(model, component, folder, metric, cfg, coeffs_for_folder):
    """Apply the best dev coeff per (sub-task, k) on the test split.
    `coeffs_for_folder` is {"{n}-paren": {k: coeff}} (one model's entry)."""
    heads, neurons = ranked_for(component, folder, metric, cfg)
    out_dir = sweep_dir(cfg["steer_root"], folder, component, "test", metric)
    count = 0
    for task, kmap in coeffs_for_folder.items():
        data = read_json(labeled_path(cfg["data_root"], folder, "test", SubTask(count)))
        for k_str, coeff in kmap.items():
            k = int(k_str)
            res = evaluate(model, data, make_ctx(component, model, heads[:k], neurons[:k], coeff))
            save_file(res, os.path.join(out_dir, config_filename(component, count, coeff, k)))
        print(f"  apply {component}/{metric} subtask={count} done", flush=True)
        count += 1
