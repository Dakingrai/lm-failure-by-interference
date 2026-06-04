"""Shared IO + housekeeping helpers — the SINGLE source for these.

These existed as copy-pasted duplicates across get_accuracy.py and the steer_*
scripts. As each experiment migrates, the local copies are dropped in favour of
these. Behaviour matches the originals (json dump indent=4, mkdir-if-missing).
"""
import gc
import json
import os
import time

import torch


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_file(data, file_name, indent=4):
    create_results_dir(os.path.dirname(file_name) or ".")
    with open(file_name, "w") as f:
        json.dump(data, f, indent=indent)


def create_results_dir(results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)


def clear_cache(sleep_seconds=0.0):
    """Free Python + CUDA memory.

    The original `clear_cache` hard-coded `time.sleep(10)` to let GPU memory
    settle between large models. That delay is preserved but parameterised
    (default 0); callers that load multiple big models back-to-back can pass a
    small positive value. See CLAUDE.md hard-rule 8 (don't blindly delete it).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if sleep_seconds:
        time.sleep(sleep_seconds)
