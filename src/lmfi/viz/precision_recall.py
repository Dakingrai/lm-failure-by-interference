"""Fig 3 (+ appendix Fig 8): precision-recall scatter for heads and FF neurons.

Migrated from precision_recall_plot.py. Pure plotting over the macro/per-subtask
results from the generalizability step (no model load).

Path fix (CLAUDE.md broken item): the original read attention results from
``results/attn_results/<folder>/final_v/`` which nothing writes; it now reads
the non-rank results at ``results/attn_results/<folder>/`` (where the
generalizability step writes them).
"""
import os

import matplotlib.pyplot as plt

from lmfi.io import create_results_dir, read_json


def plot_precision_recall(precisions, recalls, model_name, out_root, neuron=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(recalls, precisions, alpha=0.7, edgecolors='k',
                c=[[0.12156863, 0.46666667, 0.70588235, 0.7]])
    plt.xlabel("Recall", fontsize=32)
    if model_name == "CodeLlama-7b-hf":
        plt.ylabel("Precision", fontsize=32)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.grid(True)

    out_dir = os.path.join(out_root, "precision_recall")
    create_results_dir(out_dir)
    suffix = "neuron" if neuron else "attn"
    plot_path = os.path.join(out_dir, f"{model_name}_{suffix}_precision_recall_plot.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def process_attn_results(attn_results_path, folder_name, out_root):
    """Per-subtask scatter plots + return (heads, macro precisions, macro recalls)."""
    for n in range(4):
        data = read_json(os.path.join(attn_results_path, f"{n + 1}_results.json"))
        precisions = [v["precision"][0] for v in data.values()]
        recalls = [v["recall"][0] for v in data.values()]
        plot_precision_recall(precisions, recalls, f"{folder_name}_paren-{n}", out_root, neuron=True)

    macro = read_json(os.path.join(attn_results_path, "macro_metrics.json"))
    heads, precisions, recalls = [], [], []
    for head, metrics in macro.items():
        if metrics["average_precision"] and metrics["average_recall"]:
            heads.append(head)
            precisions.append(metrics["average_precision"][0])
            recalls.append(metrics["average_recall"][0])
    return heads, precisions, recalls


def process_mlp_results(mlp_results_path):
    """Return (macro precisions, macro recalls) for FF neurons."""
    macro = read_json(os.path.join(mlp_results_path, "macro_metrics.json"))
    precisions, recalls = [], []
    for neuron, metrics in macro.items():
        if metrics["average_precision"] and metrics["average_recall"]:
            precisions.append(metrics["average_precision"][0])
            recalls.append(metrics["average_recall"][0])
    return precisions, recalls
