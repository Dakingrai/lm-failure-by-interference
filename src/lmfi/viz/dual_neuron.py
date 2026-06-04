"""Fig 2: dual-sign FF neuron (CodeLlama L19N11) — mean coefficient / logits per
input sub-task. Migrated from plot_dual_neuron.py.

Pure JSON aggregation over the neuron's projection file + a grouped bar chart.
The model/neuron are parameters (default: CodeLlama-7b-hf, (19, 11)).
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from lmfi.io import create_results_dir, read_json, save_file


def compute_mean_neuron_coefficients(data):
    return {
        "avg_neuron_coefs": np.mean([e["neuron_activation_score"] for e in data]),
        "avg_max_logits": np.mean([e["paren_logits"]["max-logit"] for e in data]),
        "avg_one_paren_paren_logits": np.mean([e["paren_logits"]["1-paren-logit"] for e in data]),
        "avg_two_paren_paren_logits": np.mean([e["paren_logits"]["2-paren-logit"] for e in data]),
        "avg_three_paren_paren_logits": np.mean([e["paren_logits"]["3-paren-logit"] for e in data]),
        "avg_four_paren_paren_logits": np.mean([e["paren_logits"]["4-paren-logit"] for e in data]),
    }


def dual_neuron_means(neuron, folder, proj_root, out_root):
    """Group a neuron's projection records by closing-paren count (1..4), compute
    mean coefficients/logits per group, save the means JSON, and return them."""
    layer, neuron_idx = neuron
    neuron_name = f"L{layer}N{neuron_idx}"
    proj_path = os.path.join(proj_root, folder, "mlp", "proj", f"{neuron_name}_proj.json")
    neuron_proj = read_json(proj_path)

    groups = {1: [], 2: [], 3: [], 4: []}
    for each in neuron_proj:
        if each["neuron_idx"] == neuron_idx and each["layer"] == layer:
            n = each["label"].count(")")
            if n in groups:
                groups[n].append(each)

    results = {
        "one_paren": compute_mean_neuron_coefficients(groups[1]),
        "two_paren": compute_mean_neuron_coefficients(groups[2]),
        "three_paren": compute_mean_neuron_coefficients(groups[3]),
        "four_paren": compute_mean_neuron_coefficients(groups[4]),
    }
    out_dir = os.path.join(out_root, "neuron_analysis")
    save_file(results, os.path.join(out_dir, f"{neuron_name}_mean_coefs.json"))
    return results


def plot_neuron_results(results, out_root, out_name="dual_neuron_plot.png"):
    x = np.arange(4)
    x_labels = ['One Paren Input', 'Two Paren Input', 'Three Paren Input', 'Four Paren Input']
    coefs = [results[k]['avg_neuron_coefs'] for k in ('one_paren', 'two_paren', 'three_paren', 'four_paren')]
    x_labels_with_coef = [f"{label}\n(Avg coef={coef:.2f})" for label, coef in zip(x_labels, coefs)]
    avg_max_logits = [results[k]['avg_max_logits'] for k in ('one_paren', 'two_paren', 'three_paren', 'four_paren')]

    tokens = ['1 Paren Logit', '2 Paren Logit', '3 Paren Logit', '4 Paren Logit']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    keys = ['avg_one_paren_paren_logits', 'avg_two_paren_paren_logits',
            'avg_three_paren_paren_logits', 'avg_four_paren_paren_logits']
    logits_matrix = [[results[g][k] for k in keys]
                     for g in ('one_paren', 'two_paren', 'three_paren', 'four_paren')]

    offsets = [-1.5, -0.5, 0.5, 1.5, 2.5]
    bar_width = 0.16
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (token, color) in enumerate(zip(tokens, colors)):
        logits = [row[i] for row in logits_matrix]
        ax.bar(x + offsets[i] * bar_width, logits, bar_width, label=token, color=color)
    ax.bar(x + offsets[4] * bar_width, avg_max_logits, bar_width, label='Max Logit', color='gray')
    ax.set_xticks(x + bar_width * 0.5)
    ax.set_xticklabels(x_labels_with_coef, fontsize=12)
    ax.set_ylabel("Value", fontsize=14)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(title="Metric")
    plt.grid(axis='y', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    out_dir = os.path.join(out_root, "neuron_analysis")
    create_results_dir(out_dir)
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path
