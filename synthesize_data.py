import random
import pdb
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import csv
import torch
import os

from utils.general_utils import load_model, read_json, save_file, create_results_dir
random.seed(20)

def base_prompts(n_paren, results_dir, n_samples=750, save_file=False):
    sample_int = [100, 1000]
    all_prompts = {}
    for n in range(n_paren): 
        print(f"Preparing data for {n} parenthesis")
        random.seed(20)
        random_numbers = random.sample(range(sample_int[0], sample_int[1]), n_samples)
        prompts = []
        for num in random_numbers:
            prompt = f"#print the string {num}\nprint(" + "str("*(n) + f"{num})" + ")"*(n)
            prompts.append(prompt)
        
        if save_file:
            save_file(prompts, f"{results_dir}/prompts_{n}.json")
        all_prompts[f"paren_{n}"] = prompts
    return all_prompts

def get_corrupt_prompts(all_prompts, n_paren, results_dir, save_results=False):
    corrupt_prompts = {}
    for n in range(n_paren):
        paren_corrupt_prompts = []
        prompts = all_prompts[f"paren_{n}"]
        for prompt in prompts:
            tmp_corrupt_prompts = []
            
            for c in range(n+1):
                if c == 0:
                    to_be_replace_str = "print("
                    replace_str = "print((" 
                else:
                    to_be_replace_str = f"print(" + "str("*c
                    replace_str = f"print(" + "str("*c +"("
                corrupt_prompt = prompt.replace(to_be_replace_str, replace_str) + ")"
                tmp_corrupt_prompts.append(corrupt_prompt)
            paren_corrupt_prompts.append(tmp_corrupt_prompts)
        corrupt_prompts[f"paren_{n}"] = paren_corrupt_prompts
        if save_results:
            save_file(paren_corrupt_prompts, f"{results_dir}/corrupt_prompts_{n}.json")
    return corrupt_prompts



def label_prompt(all_prompts, tokenizer, results_dir, n_paren=10):
    
    for n in range(n_paren):
        prompts = all_prompts[f"paren_{n}"]
        label_prompts = {}
        label_prompts["last_paren"] = []
        label_prompts["second_last_paren"] = []
        label_prompts["third_last_paren"] = []
        for prompt in prompts:
            tmp_data = {}
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            n_closing_paren = len(tokenizer.tokenize(")"*(n+1), add_special_tokens=False))

            tmp_data["prompt"] = tokenizer.decode(tokenized_prompt["input_ids"][0][:-1], skip_special_tokens=True)
            tmp_data["label"] = tokenizer.decode(tokenized_prompt["input_ids"][0][-1], skip_special_tokens=True)
            tmp_data["label_idx"] = int(tokenized_prompt["input_ids"][0][-1])

            label_prompts["last_paren"].append(tmp_data)
            
            if n_closing_paren == 3: # because codellama tokenizer adds whitespace in front of token
                tmp_data = {}
                tmp_data["prompt"] = tokenizer.decode(tokenized_prompt["input_ids"][0][:-2], skip_special_tokens=True)
                tmp_data["label"] = tokenizer.decode(tokenized_prompt["input_ids"][0][-2], skip_special_tokens=True)
                tmp_data["label_idx"] = int(tokenized_prompt["input_ids"][0][-2])

                label_prompts["second_last_paren"].append(tmp_data)
            
            elif n_closing_paren == 4:
                tmp_data = {}
                tmp_data["prompt"] = tokenizer.decode(tokenized_prompt["input_ids"][0][:-3], skip_special_tokens=True)
                tmp_data["label"] = tokenizer.decode(tokenized_prompt["input_ids"][0][-3], skip_special_tokens=True)
                tmp_data["label_idx"] = int(tokenized_prompt["input_ids"][0][-3])

                label_prompts["third_last_paren"].append(tmp_data)

                tmp_data = {}
                tmp_data["prompt"] = tokenizer.decode(tokenized_prompt["input_ids"][0][:-2], skip_special_tokens=True)
                tmp_data["label"] = tokenizer.decode(tokenized_prompt["input_ids"][0][-2], skip_special_tokens=True)
                tmp_data["label_idx"] = int(tokenized_prompt["input_ids"][0][-2])
                label_prompts["second_last_paren"].append(tmp_data)

        save_file(label_prompts["last_paren"], f"{results_dir}_labeled_last_paren_{n}.json")

        if len(label_prompts["second_last_paren"]) > 1:
            save_file(label_prompts["second_last_paren"], f"{results_dir}_labeled_second_last_paren_{n}.json")
        if len(label_prompts["third_last_paren"]) > 1:
            save_file(label_prompts["third_last_paren"], f"{results_dir}_labeled_third_last_paren_{n}.json")



def train_dev_test_split(all_prompts, n_paren, data_dir, save_file_flag=False):
    train_prompts = {}
    dev_prompts = {}
    test_prompts = {}

    for n in range(n_paren):
        prompts = all_prompts[f"paren_{n}"]
        random.shuffle(prompts)

        # Assumes at least 650 prompts per category
        train_prompts[f"paren_{n}"] = prompts[:350]
        dev_prompts[f"paren_{n}"] = prompts[350:500]
        test_prompts[f"paren_{n}"] = prompts[500:650]

        if save_file_flag:
            save_file(train_prompts[f"paren_{n}"], f"{data_dir}/train_prompts_{n}.json")
            save_file(dev_prompts[f"paren_{n}"], f"{data_dir}/dev_prompts_{n}.json")
            save_file(test_prompts[f"paren_{n}"], f"{data_dir}/test_prompts_{n}.json")

    return train_prompts, dev_prompts, test_prompts
        
def main():
    # Variables Change
    models = read_json("utils/models.json")
    n_paren = 4
    for model in models:
        model_name = model["name"]
        cache_dir = model["cache"]
        folder_name = model["name"].split("/")[-1]
        data_dir = f"data/{folder_name}"
        create_results_dir(data_dir)
        # model = load_model(model_name, cache_dir)
        # tokenizer = model.tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        all_prompts = base_prompts(n_paren, data_dir, save_file=False)
        train_prompts, dev_prompts, test_prompts = train_dev_test_split(all_prompts, n_paren, data_dir)

        # Labeled Train Prompts
        train_labeled_prompts = label_prompt(train_prompts, tokenizer, data_dir+"/train", n_paren)
        # Labeled Dev Prompts
        dev_labeled_prompts = label_prompt(dev_prompts, tokenizer, data_dir+"/dev", n_paren)
        # Labeled Test Prompts
        test_labeled_prompts = label_prompt(test_prompts, tokenizer, data_dir+"/test", n_paren)


if __name__ == "__main__":
    main()
