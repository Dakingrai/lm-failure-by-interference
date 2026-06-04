"""Synthesize the balanced-parentheses dataset.

Migrated from synthesize_data.py. The committed ``data/`` is the canonical
artifact (the README notes the original was synthesized without a fixed seed);
this module regenerates it reproducibly given ``seed`` (default 20, matching the
original's module-level seed). The prompt template, train/dev/test split, and
last-token labeling are preserved exactly; the unused get_corrupt_prompts helper
is dropped.
"""
import random

from lmfi.io import save_file

SAMPLE_INT = (100, 1000)


def base_prompts(n_paren, n_samples=750, seed=20):
    """`#print the string {num}\\nprint(str(str(...{num}...)))` for each nesting n."""
    all_prompts = {}
    for n in range(n_paren):
        random.seed(seed)
        random_numbers = random.sample(range(SAMPLE_INT[0], SAMPLE_INT[1]), n_samples)
        prompts = []
        for num in random_numbers:
            prompt = f"#print the string {num}\nprint(" + "str(" * (n) + f"{num})" + ")" * (n)
            prompts.append(prompt)
        all_prompts[f"paren_{n}"] = prompts
    return all_prompts


def train_dev_test_split(all_prompts, n_paren, train_end=350, dev_end=500, test_end=650):
    """Shuffle and slice each nesting level into train/dev/test (350/150/150)."""
    train_prompts, dev_prompts, test_prompts = {}, {}, {}
    for n in range(n_paren):
        prompts = all_prompts[f"paren_{n}"]
        random.shuffle(prompts)
        train_prompts[f"paren_{n}"] = prompts[:train_end]
        dev_prompts[f"paren_{n}"] = prompts[train_end:dev_end]
        test_prompts[f"paren_{n}"] = prompts[dev_end:test_end]
    return train_prompts, dev_prompts, test_prompts


def label_prompt(all_prompts, tokenizer, out_prefix, n_paren=4):
    """Tokenize each prompt and record the final closing-paren token as the label.
    Writes ``{out_prefix}_labeled_last_paren_{n}.json`` (+ second/third-last for
    tokenizers that split the closing parens into 3-4 tokens, as in the original)."""
    for n in range(n_paren):
        prompts = all_prompts[f"paren_{n}"]
        label_prompts = {"last_paren": [], "second_last_paren": [], "third_last_paren": []}
        for prompt in prompts:
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            n_closing_paren = len(tokenizer.tokenize(")" * (n + 1), add_special_tokens=False))
            ids = tokenized_prompt["input_ids"][0]

            tmp_data = {
                "prompt": tokenizer.decode(ids[:-1], skip_special_tokens=True),
                "label": tokenizer.decode(ids[-1], skip_special_tokens=True),
                "label_idx": int(ids[-1]),
            }
            label_prompts["last_paren"].append(tmp_data)

            if n_closing_paren == 3:  # codellama tokenizer adds whitespace in front of token
                label_prompts["second_last_paren"].append({
                    "prompt": tokenizer.decode(ids[:-2], skip_special_tokens=True),
                    "label": tokenizer.decode(ids[-2], skip_special_tokens=True),
                    "label_idx": int(ids[-2]),
                })
            elif n_closing_paren == 4:
                label_prompts["third_last_paren"].append({
                    "prompt": tokenizer.decode(ids[:-3], skip_special_tokens=True),
                    "label": tokenizer.decode(ids[-3], skip_special_tokens=True),
                    "label_idx": int(ids[-3]),
                })
                label_prompts["second_last_paren"].append({
                    "prompt": tokenizer.decode(ids[:-2], skip_special_tokens=True),
                    "label": tokenizer.decode(ids[-2], skip_special_tokens=True),
                    "label_idx": int(ids[-2]),
                })

        save_file(label_prompts["last_paren"], f"{out_prefix}_labeled_last_paren_{n}.json")
        if len(label_prompts["second_last_paren"]) > 1:
            save_file(label_prompts["second_last_paren"], f"{out_prefix}_labeled_second_last_paren_{n}.json")
        if len(label_prompts["third_last_paren"]) > 1:
            save_file(label_prompts["third_last_paren"], f"{out_prefix}_labeled_third_last_paren_{n}.json")
