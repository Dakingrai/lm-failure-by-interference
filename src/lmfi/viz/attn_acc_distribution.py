"""Fig 1: per-head accuracy distribution histograms.

Migrated from attention_acc_distribution.py. Reads the with-rank head results and
histograms head accuracy (threshold index 0) per sub-task. The paper's Fig 1 is
CodeLlama; the other models are the appendix version (Fig 7).
"""
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lmfi.io import create_results_dir, read_json


def plot_acc_distribution(attn_results_root, folder, n_paren, out_root):
    """One histogram for a (model folder, n_paren) pair -> PNG."""
    results = read_json(os.path.join(attn_results_root, folder, "with_rank", f"{n_paren}_results.json"))
    acc_list = [v["accuracy"][0] for v in results.values()]
    acc_list = [acc for acc in acc_list if acc > 0.01]

    plt.hist(acc_list, bins=20, color=[[0.12156863, 0.46666667, 0.70588235, 0.7]])
    if n_paren == 3 and folder == "CodeLlama-7b-hf":
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Accuracy", fontsize=32)
    if n_paren == 1:
        plt.ylabel("Frequency", fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    out_dir = os.path.join(out_root, "attn_results")
    create_results_dir(os.path.join(out_dir, folder))
    out_path = os.path.join(out_dir, f"{folder}_acc_distribution_{n_paren}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path
