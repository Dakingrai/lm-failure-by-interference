"""Logit-lens projection of LM components (attention heads & FF neurons).

Migrated from proj_attn.py and proj_neuron.py. The numeric computation is
preserved exactly (CLAUDE.md prime directive) — these per-component logit
projections are the hub feeding generalizability (Table 1 / Fig 1) and steering.

Shared helpers de-duplicated from the two originals:
  * softmax_entropy   (== proj_neuron.logit_entropy == proj_attn.softmax_entropy)
  * get_logit_rank    (proj_neuron's shape-robust version; a superset of proj_attn's)

Differences from the originals, both timing-only (no effect on outputs):
  * the per-neuron `torch.cuda.empty_cache()` + `time.sleep(0.5)` inside
    process_mlp_neuron is parameterized via `settle_seconds` (default 0).
  * file IO lives in the experiment runner, not here (these return data).

The dead proj_neuron.filter_neurons (v1, never called) is intentionally dropped;
only filter_neurons_v2 (the one main() used) is migrated.
"""
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lmfi.data import MyDatasetV2


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits); supports [vocab] or [batch, vocab]."""
    probs = F.softmax(logits, dim=-1)
    if len(logits.shape) == 1:
        return -torch.sum(probs * torch.log(probs + 1e-10))
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


def get_logit_rank(logits, index):
    """1-based rank of `index` in the last-token logit ordering.

    Accepts either a 1-D vocab vector or an [batch, seq, vocab] tensor (then the
    [0, -1, :] slice is used) — matching proj_neuron.get_logit_rank.
    """
    if len(logits.shape) > 1:
        last_token_logits = logits[0, -1, :]
    else:
        last_token_logits = logits
    sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
    return (sorted_indices == index).nonzero(as_tuple=True)[0].item() + 1


# --------------------------------------------------------------------------- #
# Attention-head projection
# --------------------------------------------------------------------------- #
def process_comp(model, comp_activations, results, correct_idx, paren_token_ids):
    with torch.no_grad():
        logit_projection = comp_activations @ model.W_U  # (batch, seq, vocab)

    results["pred"] = logit_projection[0, -1, :].argmax().item()
    paren_logits = [logit_projection[0, -1, i].item() for i in paren_token_ids]

    results["full_logit_entropy"] = softmax_entropy(logit_projection[0, -1, :]).item()
    results["paren_entropy"] = softmax_entropy(torch.tensor(paren_logits)).item()

    results["paren_ranks"] = {
        "1-paren-rank": get_logit_rank(logit_projection, paren_token_ids[0]),
        "2-paren-rank": get_logit_rank(logit_projection, paren_token_ids[1]),
        "3-paren-rank": get_logit_rank(logit_projection, paren_token_ids[2]),
        "4-paren-rank": get_logit_rank(logit_projection, paren_token_ids[3]),
    }
    results["paren_logits"] = {
        "max-logit": logit_projection[0, -1, :].max().item(),
        "1-paren-logit": logit_projection[0, -1, paren_token_ids[0]].item(),
        "2-paren-logit": logit_projection[0, -1, paren_token_ids[1]].item(),
        "3-paren-logit": logit_projection[0, -1, paren_token_ids[2]].item(),
        "4-paren-logit": logit_projection[0, -1, paren_token_ids[3]].item(),
    }
    results["head_l2_norm"] = torch.linalg.norm(comp_activations[0, -1, :]).item()
    return results


def process_attention_head(model, cache, clean_input, correct_idx, layer, head, paren_token_ids):
    results = {
        "prompt": clean_input,
        "label": model.tokenizer.decode(correct_idx),
        "label_idx": correct_idx,
        "layer": layer,
        "head": head,
        "paren_token_ids": paren_token_ids,
    }
    attn_activations = cache[f"blocks.{layer}.attn.hook_result"][:, :, head, :]  # (batch, seq, d_model)
    return process_comp(model, attn_activations, results, correct_idx, paren_token_ids)


def run_attn_projection(model, data, heads, paren_token_ids, device=None):
    """Return {(layer, head): [per-example result dict, ...]} for every head."""
    device = device or default_device()
    dataset = MyDatasetV2(data)
    dataloader = dataset.to_dataloader(batch_size=1)
    all_results = {head: [] for head in heads}
    for clean_input, correct_idx in tqdm(dataloader):
        correct_idx = int(correct_idx[0])
        tokens = model.to_tokens(clean_input, prepend_bos=True).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for layer, head in heads:
            results = process_attention_head(model, cache, clean_input, correct_idx, layer, head, paren_token_ids)
            all_results[(layer, head)].append(results)
    return all_results


# --------------------------------------------------------------------------- #
# FF-neuron projection
# --------------------------------------------------------------------------- #
def filter_neurons_v2(W2, W_U, paren_token_ids, layer, top_k: int = 50):
    """Static pre-filter: keep neurons whose parametric projection (W2 @ W_U)
    ranks a closing-paren token in its top-k or bottom-k logits."""
    with torch.no_grad():
        neuron_logits = W2 @ W_U  # [d_mlp, vocab]

    selected = []
    for neuron_idx in range(W2.shape[0]):
        logit_vec = neuron_logits[neuron_idx]
        sorted_indices = torch.argsort(logit_vec, descending=True)
        top_tokens = set(sorted_indices[:top_k].tolist())
        bottom_tokens = set(sorted_indices[-top_k:].tolist())
        promoted_tokens = [tok for tok in paren_token_ids if tok in top_tokens]
        suppressed_tokens = [tok for tok in paren_token_ids if tok in bottom_tokens]
        if promoted_tokens or suppressed_tokens:
            selected.append({
                "neuron_idx": neuron_idx,
                "layer": layer,
                "paren_token_ranks": {
                    tok: (logit_vec >= logit_vec[tok]).sum().item() for tok in paren_token_ids
                },
                "paren_logits": {tok: logit_vec[tok].item() for tok in paren_token_ids},
                "status": (
                    "promoter" if promoted_tokens else "suppressor"
                    if suppressed_tokens else "both"
                ),
                "promoted_tokens": promoted_tokens,
                "suppressed_tokens": suppressed_tokens,
            })
    return selected


def find_paren_neurons(model, paren_token_ids, top_k: int = 50):
    """All neurons (across layers) that project a paren token within top/bottom-k.
    Migrated from save_paren_neurons (minus the file write, now in the runner)."""
    num_layers, num_neurons = model.W_out.shape[:2]
    paren_neurons = []
    for layer in tqdm(range(num_layers)):
        paren_neurons.extend(filter_neurons_v2(model.W_out[layer], model.W_U, paren_token_ids, layer, top_k))
    return paren_neurons


def process_mlp_neuron(model, cache, clean_input, correct_idx, layer, neuron_idx,
                       paren_token_ids, settle_seconds: float = 0.0):
    results = {
        "prompt": clean_input,
        "label": model.tokenizer.decode(correct_idx),
        "label_idx": correct_idx,
        "layer": layer,
        "neuron_idx": neuron_idx,
        "paren_token_ids": paren_token_ids,
    }
    neuron_activation_score = cache[f"blocks.{layer}.mlp.hook_post"][:, -1, neuron_idx].item()
    results["neuron_activation_score"] = neuron_activation_score
    neuron_parameter = model.W_out[layer][neuron_idx]
    with torch.no_grad():
        neuron_contribution = neuron_activation_score * neuron_parameter
        logit_projection = neuron_contribution @ model.W_U  # (vocab,)
    paren_logits = [logit_projection[i].item() for i in paren_token_ids]

    results["full_logit_entropy"] = softmax_entropy(logit_projection).item()
    results["paren_entropy"] = softmax_entropy(torch.tensor(paren_logits)).item()
    results["paren_ranks"] = {
        "1-paren-rank": get_logit_rank(logit_projection, paren_token_ids[0]),
        "2-paren-rank": get_logit_rank(logit_projection, paren_token_ids[1]),
        "3-paren-rank": get_logit_rank(logit_projection, paren_token_ids[2]),
        "4-paren-rank": get_logit_rank(logit_projection, paren_token_ids[3]),
    }
    results["paren_logits"] = {
        "max-logit": logit_projection.max().item(),
        "1-paren-logit": logit_projection[paren_token_ids[0]].item(),
        "2-paren-logit": logit_projection[paren_token_ids[1]].item(),
        "3-paren-logit": logit_projection[paren_token_ids[2]].item(),
        "4-paren-logit": logit_projection[paren_token_ids[3]].item(),
    }
    results["head_l2_norm"] = torch.linalg.norm(neuron_contribution).item()

    # Original did empty_cache()+sleep(0.5) here per neuron (timing only). Off by default.
    if settle_seconds:
        torch.cuda.empty_cache()
        time.sleep(settle_seconds)
    return results


def run_neuron_projection(model, data, neurons, paren_token_ids, device=None,
                          settle_seconds: float = 0.0):
    """Return {"L{layer}N{neuron}": [per-example result dict, ...]}."""
    device = device or default_device()
    dataset = MyDatasetV2(data)
    dataloader = dataset.to_dataloader(batch_size=1)
    all_results = {f"L{n['layer']}N{n['neuron_idx']}": [] for n in neurons}
    for clean_input, correct_idx in tqdm(dataloader):
        correct_idx = int(correct_idx[0])
        tokens = model.to_tokens(clean_input, prepend_bos=True).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for neuron in neurons:
            layer = neuron["layer"]
            neuron_idx = neuron["neuron_idx"]
            results = process_mlp_neuron(model, cache, clean_input, correct_idx,
                                         layer, neuron_idx, paren_token_ids, settle_seconds)
            all_results[f"L{layer}N{neuron_idx}"].append(results)
    return all_results
