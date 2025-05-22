

import json
import matplotlib.pyplot as plt
import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import numpy as np
import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load necessary utilities
from utils.general_utils import MyDataset, load_model
from activation_patching import InterveneOV


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

def save_file(data, file_name):
    """Save data to a JSON file."""
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)


def plot_precision_recall(precisions, recalls, model_name, neuron=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(recalls, precisions, alpha=0.7, edgecolors='k', c=[[0.12156863, 0.46666667, 0.70588235, 0.7]])
    # breakpoint()
    # plt.title("Precision-Recall Scatter Plot for Attention Heads")
    
    plt.xlabel("Recall", fontsize=32)
    if model_name == "CodeLlama-7b-hf":
        plt.ylabel("Precision", fontsize=32)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.grid(True)

    # Save the plot
    if neuron:
        plot_path = f"results/plot_results/precision_recall/{model_name}_neuron_precision_recall_plot.png"
    else:
        plot_path = f"results/plot_results/precision_recall/{model_name}_attn_precision_recall_plot.png"
    plt.savefig(plot_path)
    plt.close()


def process_attn_results(attn_results_path, folder_name):
    for n in range(4):
        data = f"{attn_results_path}/{n+1}_results.json"
        data = read_json(data)
        precisions = []
        recalls = []
        for k, v in data.items():
            precisions.append(v["precision"][0])
            recalls.append(v["recall"][0])
        
        plot_precision_recall(precisions, recalls, f"{folder_name}_paren-{n}", neuron=True)


    general_heads = f"{attn_results_path}/macro_metrics.json"
    general_heads = read_json(general_heads)

    attention_heads = []
    precisions = []
    recalls = []

    for head, metrics in general_heads.items():
        if metrics["average_precision"] and metrics["average_recall"]:
            precision = metrics["average_precision"][0]
            recall = metrics["average_recall"][0]
            attention_heads.append(head)
            precisions.append(precision)
            recalls.append(recall)

    return attention_heads, precisions, recalls

def process_mlp_results(mlp_results_path):
    neuron_generalization_heads = f"{mlp_results_path}/macro_metrics.json"
    neuron_generalization_heads = read_json(neuron_generalization_heads)

    precisions = []
    recalls = []

    for neuron, metrics in neuron_generalization_heads.items():
        if metrics["average_precision"] and metrics["average_recall"]:
            precision = metrics["average_precision"][0]
            recall = metrics["average_recall"][0]
            precisions.append(precision)
            recalls.append(recall)

    return precisions, recalls


def main():
    models = read_json("utils/models.json")[:2]
    model_generalization_heads = {}
    for m in models:
        model_name = m["name"]
        cache_dir = m["cache"]
        # model = load_model(model_name, cache_dir)
        folder_name = m["name"].split("/")[-1]

        attn_results_path = f"results/attn_results/{folder_name}/final_v"
        mlp_results_path = f"results/mlp_results/{folder_name}/proj"

        attn_heads, precisions, recalls = process_attn_results(attn_results_path, folder_name)
        plot_precision_recall(precisions, recalls, folder_name)
        # pdb.set_trace()
        mlp_precisions, mlp_recalls = process_mlp_results(mlp_results_path)
        plot_precision_recall(mlp_precisions, mlp_recalls, folder_name, neuron=True)


if __name__ == "__main__":
    main()