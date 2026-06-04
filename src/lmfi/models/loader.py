"""Model loading — lifted verbatim from utils/general_utils.load_model.

The per-architecture HookedTransformer configuration branches are preserved
EXACTLY (fold_ln, center_writing_weights, center_unembed, use_split_qkv_input,
use_hook_mlp_in, use_attn_result, fp16 for the 7B coder models, the
refactor_factored_attn_matrices path for GPT-2). Refactors here are structural
only — what is computed must not change (CLAUDE.md hard-rule 8).

The only change vs the original: the default `cache_dir` is None instead of a
hard-coded personal path (`../models/llama3-8b-cache/`). With cache_dir=None,
HuggingFace/TransformerLens use the default HF hub cache, which is the
environment-specific location where weights actually live (hard-rule 6).
"""
import os

import torch
import transformer_lens as lens
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, cache_dir: str = None):
    """Load a model as a TransformerLens HookedTransformer.

    Arguments:
        model_name (str): HF / TransformerLens model id.
        cache_dir (str): weight cache dir. If None, the default HF hub cache is
            used. If given and missing, it is created (matches original).
    """
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if "codellama" in model_name:
        inner_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = lens.HookedTransformer.from_pretrained(
            model_name=model_name,
            hf_model=inner_model,
            tokenizer=tokenizer,
            fold_ln=True,
            center_unembed=True,
            center_writing_weights=False,
            device="cuda",
            dtype="float16",
        )
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
        # NOTE - if too much memory required, we may not need this
        model.cfg.use_attn_result = True

    elif 'Meta-Llama-3-8B' in model_name:
        inner_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = lens.HookedTransformer.from_pretrained(model_name=model_name, hf_model=inner_model, tokenizer=tokenizer, fold_ln=True, center_unembed=True, center_writing_weights=True, device="cuda")

    elif "EleutherAI/gpt-j-6b" in model_name:
        model = lens.HookedTransformer.from_pretrained(model_name, cache_dir=cache_dir, fold_ln=True, center_unembed=True, center_writing_weights=True, device="cuda")

    elif "pythia" in model_name or "Llama-2-7b" in model_name or "Llama-3-8b" in model_name:
        if "Llama-2-7b" in model_name:
            model_name = "Llama-2-7b"
        elif "pythia" in model_name:
            model_name = "pythia-6.9b"

        model = lens.HookedTransformer.from_pretrained(
            model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            cache_dir=cache_dir
        )
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.use_attn_result = True

    else:
        model = lens.HookedTransformer.from_pretrained(
            model_name,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            refactor_factored_attn_matrices=True,
            cache_dir=cache_dir
        )
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
        # NOTE - if too much memory required, we may not need this
        model.cfg.use_attn_result = True

    model.eval()
    return model
