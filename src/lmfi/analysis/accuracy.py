"""Token-prediction accuracy on the balanced-parentheses task.

Pure compute — no IO. Migrated from get_accuracy.get_accuracy(); the prediction
rule is preserved EXACTLY so numbers do not move (CLAUDE.md prime directive):

    logits = model(prompt, return_type="logits")
    pred_id = logits.argmax(dim=-1).squeeze()[-1]
    pred    = model.to_string(pred_id)
    correct = (pred == label)   # exact string match

The original wrapped the forward in ``torch.no_grad()``; kept here.
"""
import torch


def compute_accuracy(model, data):
    """Run `model` over `data` (list of {"prompt","label",...}); return a dict
    ``{"accuracy": float, "prompt_results": [{prompt,label,pred,correct}, ...]}``
    identical in shape to the original get_accuracy output."""
    total = 0
    correct = 0
    prompt_results = []
    for each in data:
        total += 1
        with torch.no_grad():
            logits = model(each["prompt"], return_type="logits")
        l = logits.argmax(dim=-1).squeeze()[-1]
        pred = model.to_string(l)

        tmp = {"prompt": each["prompt"], "label": each["label"], "pred": pred}
        if pred == each["label"]:
            tmp["correct"] = True
            correct += 1
        else:
            tmp["correct"] = False
        prompt_results.append(tmp)

    return {"accuracy": correct / total, "prompt_results": prompt_results}
