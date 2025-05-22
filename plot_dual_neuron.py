import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
import os
import gc
import time
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

# Load necessary utilities
from utils.general_utils import MyDataset, load_model, MyDatasetV2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_file(data, file_name):
    """Save data to a JSON file."""
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)


def compute_mean_neuron_coefficients(data):
    neuron_coefs = [ 
        each["neuron_activation_score"] for each in data
    ]
    # Paren_logits
    one_paren_paren_logits = [
        each["paren_logits"]["1-paren-logit"] for each in data
    ]
    two_paren_paren_logits = [
        each["paren_logits"]["2-paren-logit"] for each in data
    ]
    three_paren_paren_logits = [
        each["paren_logits"]["3-paren-logit"] for each in data
    ]
    four_paren_paren_logits = [
        each["paren_logits"]["4-paren-logit"] for each in data
    ]

    # max logits
    max_logits = [
        each["paren_logits"]["max-logit"] for each in data
    ]

    # get the mean of all the coefs
    results = {
        "avg_neuron_coefs": np.mean(neuron_coefs),
        "avg_max_logits": np.mean(max_logits),
        "avg_one_paren_paren_logits": np.mean(one_paren_paren_logits),
        "avg_two_paren_paren_logits": np.mean(two_paren_paren_logits),
        "avg_three_paren_paren_logits": np.mean(three_paren_paren_logits),
        "avg_four_paren_paren_logits": np.mean(four_paren_paren_logits)
    }

    return results

    

def plot_dual_neuron(neuron):
    layer, neuron_idx = neuron
    neuron_name = f"L{layer}N{neuron_idx}"
    neuron_proj_path = f"results/projections/CodeLlama-7b-hf/mlp/proj"
    neuron_proj = read_json(os.path.join(neuron_proj_path, f"{neuron_name}_proj.json"))

    one_paren_data = []
    two_paren_data = []
    three_paren_data = []
    four_paren_data = []
    for each in neuron_proj:
        if each["neuron_idx"] == neuron_idx and each["layer"] == layer:
            # count no of ")" in each["label"]
            no_of_right_paren = each["label"].count(")")
            if no_of_right_paren == 1:
                one_paren_data.append(each)
            elif no_of_right_paren == 2:
                two_paren_data.append(each)
            elif no_of_right_paren == 3:
                three_paren_data.append(each)
            elif no_of_right_paren == 4:
                four_paren_data.append(each)

    mean_coefs_one_paren = compute_mean_neuron_coefficients(one_paren_data)
    mean_coefs_two_paren = compute_mean_neuron_coefficients(two_paren_data)
    mean_coefs_three_paren = compute_mean_neuron_coefficients(three_paren_data)
    mean_coefs_four_paren = compute_mean_neuron_coefficients    (four_paren_data)

    # print(f"\nNeuron L{layer}N{neuron_idx} mean coefficients:")
    # print(f"One parenthesis cases:", mean_coefs_one_paren)
    # print(f"Two parentheses cases:", mean_coefs_two_paren) 
    # print(f"Three parentheses cases:", mean_coefs_three_paren)
    # print(f"Four parentheses cases:", mean_coefs_four_paren)

    results = {
        "one_paren": mean_coefs_one_paren,
        "two_paren": mean_coefs_two_paren,
        "three_paren": mean_coefs_three_paren,
        "four_paren": mean_coefs_four_paren
    }

    save_file(results, f"results/plot_results/neuron_analysis/L{layer}N{neuron_idx}_mean_coefs.json")
    return results



def plot_neuron_results(results):
    # Extract data
    x = np.arange(4)  # 4 categories
    x_labels = ['One Paren Input', 'Two Paren Input', 'Three Paren Input', 'Four Paren Input']
    
    # Get neuron coefficients for labels
    coefs = [
        results['one_paren']['avg_neuron_coefs'],
        results['two_paren']['avg_neuron_coefs'], 
        results['three_paren']['avg_neuron_coefs'],
        results['four_paren']['avg_neuron_coefs']
    ]
    
    # Create labels with coefficients
    x_labels_with_coef = [f"{label}\n(Avg coef={coef:.2f})" for label, coef in zip(x_labels, coefs)]
    
    # Extract logits data
    avg_max_logits = [
        results['one_paren']['avg_max_logits'],
        results['two_paren']['avg_max_logits'],
        results['three_paren']['avg_max_logits'], 
        results['four_paren']['avg_max_logits']
    ]
    
    # Token logits for each case
    tokens = ['1 Paren Logit', '2 Paren Logit', '3 Paren Logit', '4 Paren Logit']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    logits_matrix = [
        [results['one_paren']['avg_one_paren_paren_logits'], 
         results['one_paren']['avg_two_paren_paren_logits'],
         results['one_paren']['avg_three_paren_paren_logits'],
         results['one_paren']['avg_four_paren_paren_logits']],
        [results['two_paren']['avg_one_paren_paren_logits'],
         results['two_paren']['avg_two_paren_paren_logits'],
         results['two_paren']['avg_three_paren_paren_logits'],
         results['two_paren']['avg_four_paren_paren_logits']],
        [results['three_paren']['avg_one_paren_paren_logits'],
         results['three_paren']['avg_two_paren_paren_logits'], 
         results['three_paren']['avg_three_paren_paren_logits'],
         results['three_paren']['avg_four_paren_paren_logits']],
        [results['four_paren']['avg_one_paren_paren_logits'],
         results['four_paren']['avg_two_paren_paren_logits'],
         results['four_paren']['avg_three_paren_paren_logits'],
         results['four_paren']['avg_four_paren_paren_logits']]
    ]

    # Plot settings
    offsets = [-1.5, -0.5, 0.5, 1.5, 2.5]  # Four tokens + one for max logit
    bar_width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot token logits
    for i, (token, color) in enumerate(zip(tokens, colors)):
        logits = [row[i] for row in logits_matrix]
        ax.bar(x + offsets[i]*bar_width, logits, bar_width, label=token, color=color)

    # Add max logits as bars right after token bars
    ax.bar(x + offsets[4]*bar_width, avg_max_logits, bar_width, label='Max Logit', color='gray')

    # Formatting
    ax.set_xticks(x + bar_width * 0.5)
    ax.set_xticklabels(x_labels_with_coef, fontsize=12)
    ax.set_ylabel("Value", fontsize=14)
    # ax.set_title("Neuron Logits and Max Logits Across Input Types\n(Neuron Coefficients Shown in X-axis Labels)")
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.legend(title="Metric")
    plt.grid(axis='y', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    create_results_dir("results/plot_results/neuron_analysis")
    plt.savefig('results/plot_results/neuron_analysis/dual_neuron_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    results = plot_dual_neuron((19, 11))
    plot_neuron_results(results)
    # plot_dual_neuron((20, 3998))
    # plot_dual_neuron((22, 8326))
    # plot_dual_neuron((27, 9695))
    # plot_dual_neuron((29, 8515))

if __name__ == "__main__":
    main()