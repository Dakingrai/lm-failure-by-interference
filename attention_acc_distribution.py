import torch
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gc
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load necessary utilities
from utils.general_utils import MyDataset, load_model
from activation_patching import InterveneOV, InterveneNeurons


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def attention_acc(attn_results_path, n_paren, folder_name):
    attn_results = read_json(attn_results_path+f"/{n_paren}_results.json")
    acc_list = []
    for key, value in attn_results.items():
        acc_list.append(value["accuracy"][0])
    
    # plot the distribution of acc_list
    # first remove the accuracy below 0.01
    acc_list = [acc for acc in acc_list if acc > 0.01]
    plt.hist(acc_list, bins=20, color=[[0.12156863, 0.46666667, 0.70588235, 0.7]])
    
    if n_paren == 3 and folder_name == "CodeLlama-7b-hf":
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.xlabel("Accuracy", fontsize=32)

    if n_paren == 1:
        plt.ylabel("Frequency", fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    # plt.title(f"Accuracy Distribution for {folder_name} with {n_paren} parentheses")
    create_results_dir(f"results/plot_results/attn_results/{folder_name}")
    plt.savefig(f"results/plot_results/attn_results/{folder_name}_acc_distribution_{n_paren}.png")
    plt.close()


def main():
    models = read_json("utils/models.json")[:2]
    n_paren = 4 # Number of parentheses to consider
    for model in models:
        model_name = model["name"]
        print(f"Model name: {model_name}")
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        attn_results_path = f"results/attn_results/{folder_name}/with_rank"
        for n_paren in range(1, 5):
            attention_acc(attn_results_path, n_paren, folder_name)

if __name__ == "__main__":
    main()