"""The ONE place that encodes the sub-task indexing (CLAUDE.md hard-rule 4).

A balanced-parentheses sub-task is identified two ways in this codebase:
  * a 0-indexed file suffix `n` in {0,1,2,3}  (e.g. ``..._last_paren_2.json``)
  * a 1-indexed paren count `N` in {1,2,3,4}  (the "three-paren" sub-task etc.)

with ``N = n + 1``. No other module should do inline ``n - 1`` / ``f"_{n}"``
arithmetic; build paths and labels through here.
"""
import json
import os

N_SUBTASKS = 4
SPLITS = ("train", "dev", "test")


class SubTask:
    """A single sub-task: file suffix ``index`` (0..3) <-> ``paren_count`` (1..4)."""

    def __init__(self, index: int):
        if not 0 <= index < N_SUBTASKS:
            raise ValueError(f"sub-task index out of range: {index}")
        self.index = index

    @property
    def paren_count(self) -> int:
        return self.index + 1

    @property
    def name(self) -> str:
        return f"{self.paren_count}-paren"

    def __repr__(self) -> str:
        return f"SubTask(index={self.index}, {self.name})"


def all_subtasks(n_subtasks: int = N_SUBTASKS):
    return [SubTask(i) for i in range(n_subtasks)]


def model_folder(model_name: str) -> str:
    """``codellama/CodeLlama-7b-hf`` -> ``CodeLlama-7b-hf`` (matches originals)."""
    return model_name.split("/")[-1]


def labeled_path(data_root: str, model_name: str, split: str, subtask: SubTask,
                 label_pos: str = "last_paren") -> str:
    """Path of a labeled data file, e.g.
    ``data/gpt2/test_labeled_last_paren_2.json``."""
    folder = model_folder(model_name)
    return os.path.join(data_root, folder, f"{split}_labeled_{label_pos}_{subtask.index}.json")


def paren_token_ids(data_root: str, model_name: str, n_subtasks: int = N_SUBTASKS,
                    label_pos: str = "last_paren"):
    """Token ids of the 1..N closing-paren targets, read from the first example
    of each sub-task's train file (all examples in a sub-task share the label).

    Migrated from get_paren_logit_idx() in proj_attn.py / proj_neuron.py.
    """
    ids = []
    for st in all_subtasks(n_subtasks):
        path = labeled_path(data_root, model_name, "train", st, label_pos)
        with open(path, "r") as f:
            data = json.load(f)
        ids.append(data[0]["label_idx"])
    return ids
