"""Generalizability analysis of LM components (Section 3.2 -> Table 1, Fig 1).

Consolidates analyze_attn_proj.py + analyze_neuron.py. Pure CPU post-processing
over the projection JSON (no model forward). Numeric logic is preserved exactly
(CLAUDE.md prime directive). Helpers that were byte-identical across the two
originals are de-duplicated here; the genuinely different accuracy variants are
kept distinct (attn non-rank, attn with-rank, neuron gated).

Determinism note: get_data() draws `random.sample` for the precision set, so it
consumes the module `random` stream in call order (matching the originals, which
seed 42 at import). The *generalization counts* (Table 1) depend only on accuracy
(computed from the deterministic `new_data`), so they are reproducible regardless
of the random stream; only precision/recall/F1 depend on the sampling.

Per-call clear_cache()/time.sleep() from the originals are dropped here (timing
only, no GPU work in this pure-CPU analysis; no effect on outputs).
"""
import os
import random

import numpy as np

from lmfi.io import read_json, save_file


# --------------------------------------------------------------------------- #
# Shared predicate + data helpers (identical across both originals)
# --------------------------------------------------------------------------- #
def is_promoted(max_logit, logit, threshold):
    logit = round(logit, 4)
    threshold = round(threshold, 4)
    max_logit = round(max_logit, 4)
    return logit >= threshold * max_logit


def is_promoted_rank(ranks, threshold_rank=100):
    return any(r < threshold_rank for r in ranks)


def get_data(data_path, subtask_paren):
    """Returns (new_data, precision_data). `new_data` is the deterministic slice
    for this sub-task; `precision_data` adds random samples (n_data//3) from the
    other three sub-tasks. Random draws match the originals exactly."""
    data = read_json(data_path)
    n_data = len(data) / 4
    if len(data) % 4 != 0:
        raise ValueError("Length of data must be divisible by 4")
    n_data = int(n_data)
    subtask_n_paren = subtask_paren.count(")")

    if subtask_n_paren == 0:
        raise ValueError("subtask_n_paren is 0")
    elif subtask_n_paren == 1:
        new_data = data[:n_data]
        sample_from_two = random.sample(data[n_data:2 * n_data], n_data // 3)
        sample_from_three = random.sample(data[2 * n_data:3 * n_data], n_data // 3)
        sample_from_four = random.sample(data[3 * n_data:4 * n_data], n_data // 3)
        precision_data = new_data + sample_from_two + sample_from_three + sample_from_four
    elif subtask_n_paren == 2:
        new_data = data[n_data:2 * n_data]
        sample_from_one = random.sample(data[:n_data], n_data // 3)
        sample_from_three = random.sample(data[2 * n_data:3 * n_data], n_data // 3)
        sample_from_four = random.sample(data[3 * n_data:4 * n_data], n_data // 3)
        precision_data = new_data + sample_from_one + sample_from_three + sample_from_four
    elif subtask_n_paren == 3:
        new_data = data[2 * n_data:3 * n_data]
        sample_from_one = random.sample(data[:n_data], n_data // 3)
        sample_from_two = random.sample(data[n_data:2 * n_data], n_data // 3)
        sample_from_four = random.sample(data[3 * n_data:4 * n_data], n_data // 3)
        precision_data = new_data + sample_from_one + sample_from_two + sample_from_four
    elif subtask_n_paren == 4:
        new_data = data[3 * n_data:4 * n_data]
        sample_from_one = random.sample(data[:n_data], n_data // 3)
        sample_from_two = random.sample(data[n_data:2 * n_data], n_data // 3)
        sample_from_three = random.sample(data[2 * n_data:3 * n_data], n_data // 3)
        precision_data = new_data + sample_from_one + sample_from_two + sample_from_three

    return new_data, precision_data


def _pr_f1_from_counts(thresholds, tp_l, fp_l, fn_l, tn_l):
    precision, recall, f1_scores, fpr_l = [], [], [], []
    for i in range(len(thresholds)):
        tp, fp, fn, tn = tp_l[i], fp_l[i], fn_l[i], tn_l[i]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision.append(prec); recall.append(rec); f1_scores.append(f1); fpr_l.append(fpr)
    return precision, recall, f1_scores, fpr_l


def calculate_precision_recall_f1(thresholds, precision_data, subtask_n_paren):
    """Activation by logit threshold (is_promoted). Identical in both originals."""
    tp = [0] * len(thresholds); fp = [0] * len(thresholds)
    fn = [0] * len(thresholds); tn = [0] * len(thresholds)
    for entry in precision_data:
        max_logit = entry["paren_logits"]["max-logit"]
        label_n = entry["label"].count(")")
        label_logit = entry["paren_logits"].get(f"{label_n}-paren-logit", 0)
        for i, threshold in enumerate(thresholds):
            activated = is_promoted(max_logit, label_logit, threshold)
            if activated:
                if label_n == subtask_n_paren:
                    tp[i] += 1
                else:
                    fp[i] += 1
            elif label_n == subtask_n_paren:
                fn[i] += 1
            else:
                tn[i] += 1
    return _pr_f1_from_counts(thresholds, tp, fp, fn, tn)


def calculate_precision_recall_f1_with_rank(thresholds, precision_data, subtask_n_paren):
    """Activation by rank<100 (is_promoted_rank). attn '_with_rank' variant."""
    tp = [0] * len(thresholds); fp = [0] * len(thresholds)
    fn = [0] * len(thresholds); tn = [0] * len(thresholds)
    for entry in precision_data:
        label_n = entry["label"].count(")")
        for i, threshold in enumerate(thresholds):
            activated = is_promoted_rank(list(entry["paren_ranks"].values()), threshold_rank=100)
            if activated:
                if label_n == subtask_n_paren:
                    tp[i] += 1
                else:
                    fp[i] += 1
            elif label_n == subtask_n_paren:
                fn[i] += 1
            else:
                tn[i] += 1
    return _pr_f1_from_counts(thresholds, tp, fp, fn, tn)


# --------------------------------------------------------------------------- #
# Accuracy + confidence (three distinct variants, preserved exactly)
# --------------------------------------------------------------------------- #
def _confidence_inputs(entry, subtask_n, use_second_highest):
    paren_logits = entry["paren_logits"]
    paren_ranks = entry.get("paren_ranks", {})
    subtask_logit = paren_logits.get(f"{subtask_n}-paren-logit", float("-inf"))
    subtask_rank = paren_ranks.get(f"{subtask_n}-paren-rank", float("inf"))
    all_ranks = [paren_ranks.get(f"{n}-paren-rank", float("inf")) for n in range(1, 5)]
    min_rank = min(all_ranks)
    other_logits = [paren_logits.get(f"{n}-paren-logit", float("-inf"))
                    for n in range(1, 5) if n != subtask_n]
    other_logits = sorted(other_logits, reverse=True)
    ref_logit = (other_logits[1] if (use_second_highest and len(other_logits) > 1)
                 else other_logits[0] if other_logits else float("-inf"))
    confidence = subtask_logit - ref_logit
    return subtask_rank, min_rank, all_ranks, confidence


def _avg(num, den, conf_lists):
    avg_accuracy = [num[i] / den[i] if den[i] > 0 else 0 for i in range(len(num))]
    avg_conf = [sum(c) / len(c) if len(c) > 0 else 0 for c in conf_lists]
    return avg_accuracy, avg_conf


def attn_accuracy_and_confidence(thresholds, data, subtask_paren, paren_token_ids=None,
                                 use_second_highest=False):
    """attn non-rank variant: counts every entry; correct iff subtask_rank == min_rank."""
    subtask_n = subtask_paren.count(")")
    accuracy = [0] * len(thresholds); n_total = [0] * len(thresholds)
    correct_conf = [[] for _ in thresholds]; incorrect_conf = [[] for _ in thresholds]
    for entry in data:
        subtask_rank, min_rank, _, confidence = _confidence_inputs(entry, subtask_n, use_second_highest)
        for i in range(len(thresholds)):
            n_total[i] += 1
            if subtask_rank == min_rank:
                accuracy[i] += 1
                correct_conf[i].append(confidence)
            else:
                incorrect_conf[i].append(confidence)
    avg_acc, avg_c = _avg(accuracy, n_total, correct_conf)
    _, avg_ic = _avg(accuracy, n_total, incorrect_conf)
    return avg_acc, avg_c, avg_ic


def attn_accuracy_and_confidence_with_rank(thresholds, data, subtask_paren, paren_token_ids=None,
                                           use_second_highest=False):
    """attn with-rank variant: correct iff subtask_rank == min_rank and min_rank <= 100."""
    subtask_n = subtask_paren.count(")")
    accuracy = [0] * len(thresholds); n_total = [0] * len(thresholds)
    correct_conf = [[] for _ in thresholds]; incorrect_conf = [[] for _ in thresholds]
    for entry in data:
        subtask_rank, min_rank, _, confidence = _confidence_inputs(entry, subtask_n, use_second_highest)
        for i in range(len(thresholds)):
            n_total[i] += 1
            if subtask_rank == min_rank and min_rank <= 100:
                accuracy[i] += 1
                correct_conf[i].append(confidence)
            else:
                incorrect_conf[i].append(confidence)
    avg_acc, avg_c = _avg(accuracy, n_total, correct_conf)
    _, avg_ic = _avg(accuracy, n_total, incorrect_conf)
    return avg_acc, avg_c, avg_ic


def neuron_accuracy_and_confidence(thresholds, data, subtask_paren, paren_token_ids=None,
                                   use_second_highest=False):
    """neuron variant: only entries with some rank<100 are counted; correct iff
    subtask_rank == min_rank."""
    subtask_n = subtask_paren.count(")")
    accuracy = [0] * len(thresholds); n_total = [0] * len(thresholds)
    correct_conf = [[] for _ in thresholds]; incorrect_conf = [[] for _ in thresholds]
    for entry in data:
        subtask_rank, min_rank, all_ranks, confidence = _confidence_inputs(entry, subtask_n, use_second_highest)
        for i in range(len(thresholds)):
            if is_promoted_rank(all_ranks, threshold_rank=100):
                n_total[i] += 1
                if subtask_rank == min_rank:
                    accuracy[i] += 1
                    correct_conf[i].append(confidence)
                else:
                    incorrect_conf[i].append(confidence)
    avg_acc, avg_c = _avg(accuracy, n_total, correct_conf)
    _, avg_ic = _avg(accuracy, n_total, incorrect_conf)
    return avg_acc, avg_c, avg_ic


# --------------------------------------------------------------------------- #
# Per-subtask processing (attn heads / mlp neurons)
# --------------------------------------------------------------------------- #
def process_attention_data(proj_results_path, result_path, thresholds, heads, subtask_paren,
                           with_rank):
    subtask_n_paren = subtask_paren.count(")")
    pr_f1 = calculate_precision_recall_f1_with_rank if with_rank else calculate_precision_recall_f1
    acc_fn = attn_accuracy_and_confidence_with_rank if with_rank else attn_accuracy_and_confidence

    activated = {}
    for layer, head in heads:
        head_name = f"L{layer}_H{head}"
        data, precision_data = get_data(os.path.join(proj_results_path, f"{head_name}_proj.json"), subtask_paren)
        precision, recall, f1_scores, fpr = pr_f1(thresholds, precision_data, subtask_n_paren)
        accuracy, acc_conf, inc_conf = acc_fn(thresholds, data, subtask_paren, use_second_highest=True)
        activated[head_name] = {
            "precision": precision, "recall": recall, "f1": f1_scores,
            "false_positive_rate": fpr, "accuracy": accuracy,
            "avg_correct_conf": acc_conf, "avg_incorrect_conf": inc_conf,
        }
    save_file(activated, os.path.join(result_path, f"{subtask_n_paren}_results.json"))


def process_mlp_data(proj_results_path, result_path, thresholds, neurons, subtask_paren):
    subtask_n_paren = subtask_paren.count(")")
    activated = {}
    for layer, neuron_idx in neurons:
        neuron_name = f"L{layer}N{neuron_idx}"
        data, precision_data = get_data(os.path.join(proj_results_path, f"{neuron_name}_proj.json"), subtask_paren)
        precision, recall, f1_scores, fpr = calculate_precision_recall_f1(thresholds, precision_data, subtask_n_paren)
        accuracy, acc_conf, inc_conf = neuron_accuracy_and_confidence(thresholds, data, subtask_paren, use_second_highest=True)
        activated[neuron_name] = {
            "precision": precision, "recall": recall, "f1": f1_scores,
            "false_positive_rate": fpr, "accuracy": accuracy,
            "avg_correct_conf": acc_conf, "avg_incorrect_conf": inc_conf,
        }
    save_file(activated, os.path.join(result_path, f"{subtask_n_paren}_results.json"))


# --------------------------------------------------------------------------- #
# Aggregation: macro metrics + generalization grouping (parameterized by noun)
# --------------------------------------------------------------------------- #
def analyze_macro_f1_results(result_path, subtask_parens, thresholds):
    accum = {}
    for subtask_paren in subtask_parens:
        subtask_n = subtask_paren.count(")")
        result_file = os.path.join(result_path, f"{subtask_n}_results.json")
        if not os.path.exists(result_file):
            print(f"Warning: Result file {result_file} not found.")
            continue
        for name, metrics in read_json(result_file).items():
            if name not in accum:
                accum[name] = {"precision": [[] for _ in thresholds],
                               "recall": [[] for _ in thresholds],
                               "f1": [[] for _ in thresholds]}
            for i in range(len(thresholds)):
                accum[name]["precision"][i].append(metrics["precision"][i])
                accum[name]["recall"][i].append(metrics["recall"][i])
                accum[name]["f1"][i].append(metrics["f1"][i])

    avg_metrics = {}
    for name, lists in accum.items():
        avg_metrics[name] = {
            "average_precision": [float(np.mean(p)) for p in lists["precision"]],
            "average_recall": [float(np.mean(r)) for r in lists["recall"]],
            "macro_f1": [float(np.mean(f)) for f in lists["f1"]],
        }
    save_file(avg_metrics, os.path.join(result_path, "macro_metrics.json"))


def accuracy_generalization(result_path, subtask_parens, noun, out_name,
                            threshold_value=0.9, threshold_index=0):
    """`noun` in {"heads","neurons"}. Writes {out_name} with generalization groups
    (how many sub-tasks each component reaches >= threshold accuracy) and per-subtask
    component sets."""
    from collections import defaultdict
    passing = defaultdict(set)
    for subtask_paren in subtask_parens:
        subtask_n = subtask_paren.count(")")
        result_file = os.path.join(result_path, f"{subtask_n}_results.json")
        if not os.path.exists(result_file):
            print(f"Warning: Result file {result_file} not found.")
            continue
        for name, metrics in read_json(result_file).items():
            acc = metrics.get("accuracy", [])
            if acc[threshold_index] >= threshold_value:
                passing[subtask_n].add(name)

    per_subtask = {f"{k}-paren": sorted(list(v)) for k, v in passing.items()}
    comp_to_subtasks = defaultdict(set)
    for subtask_id, comp_set in passing.items():
        for comp in comp_set:
            comp_to_subtasks[comp].add(subtask_id)
    groups = defaultdict(list)
    for comp, subtasks in comp_to_subtasks.items():
        groups[f"{len(subtasks)}-general-{noun}"].append(comp)
    group_out = {g: {noun: sorted(c), "n_len": len(c)} for g, c in groups.items()}

    final = {"generalization_groups": group_out, f"per_subtask_{noun}": per_subtask}
    save_file(final, os.path.join(result_path, out_name))


def count_generalization(generalization_path, noun):
    """Table 1 column: counts of components generalizing to 1/2/3/4 sub-tasks."""
    data = read_json(generalization_path)["generalization_groups"]
    out = {}
    for n in range(1, 5):
        key = f"{n}-general-{noun}"
        out[key] = len(data[key][noun]) if key in data else 0
    return out


def parse_mlp_name(name):
    layer = int(name.split("N")[0].split("L")[1])
    neuron = int(name.split("N")[1])
    return (layer, neuron)


def get_mlp_neurons(proj_results_path, folder_name):
    """Read the (layer, neuron_idx) list from <model>_paren_neurons.json."""
    model_name = folder_name
    if model_name == "Llama-2-7b":
        model_name = "Llama-2-7b-hf"
    if model_name == "gpt2-small":
        model_name = "gpt2"
    neurons_data = read_json(os.path.join(proj_results_path, f"{model_name}_paren_neurons.json"))
    return [(each["layer"], each["neuron_idx"]) for each in neurons_data]


def analyze_generalization(result_path, proj_results_path, subtask_parens, paren_tokens):
    """Dual-sign neuron summary (avg logits/ranks per sub-task + generalization
    flags) -> neuron_coeffs.json. Migrated from analyze_neuron.analyze_generalization."""
    gen = read_json(os.path.join(result_path, "neuron_generalization_0.json"))
    groups = gen["generalization_groups"]
    general_neurons = []
    for n in range(1, 5):
        key = f"{n}-general-neurons"
        general_neurons += groups.get(key, {}).get("neurons", []) if key in groups else []

    all_neurons = {}
    for neuron in general_neurons:
        all_neurons[neuron] = {"generalization": {}, "avg_logits": {}, "ranks": {}}
        neuron_data = read_json(os.path.join(proj_results_path, f"{neuron}_proj.json"))
        n_count = {}
        for paren in paren_tokens:
            all_neurons[neuron]["avg_logits"][paren] = 0.0
            all_neurons[neuron]["ranks"][paren] = 0.0
            all_neurons[neuron]["generalization"][paren] = False
            n_count[paren] = 0
        for entry in neuron_data:
            entry_label = entry["label"]
            entry_label_count = entry_label.count(")")
            n_count[entry_label] += 1
            all_neurons[neuron]["avg_logits"][entry_label] += entry["paren_logits"][f"{entry_label_count}-paren-logit"]
            all_neurons[neuron]["ranks"][entry_label] += entry["paren_ranks"][f"{entry_label_count}-paren-rank"]
        for paren in paren_tokens:
            all_neurons[neuron]["avg_logits"][paren] /= n_count[paren]
            all_neurons[neuron]["ranks"][paren] /= n_count[paren]
        for paren in paren_tokens:
            n_paren = paren.count(")")
            if neuron in gen["per_subtask_neurons"][f"{n_paren}-paren"]:
                all_neurons[neuron]["generalization"][paren] = True

    save_file(all_neurons, os.path.join(result_path, "neuron_coeffs.json"))
