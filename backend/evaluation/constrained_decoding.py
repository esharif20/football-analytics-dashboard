"""Findings-informed prompt + constrained decoding study.

Tests the chapter's central prescription claim (§4.8.20 / §4.8.22) on the same
test set the probing work used. Compares two input formats × two decoding modes:

                                 │ Free decode (argmax over all tokens)  │ Constrained (argmax over class tokens)
─────────────────────────────────┼───────────────────────────────────────┼──────────────────────────────────────
 Baseline: `d` digit prompt      │ current pipeline                      │ d + constrained
 Findings-informed: `nec+v`      │ updated prompt free generation        │ PRESCRIPTION (what the chapter says)

For each task × condition it reports macro-F1 and per-class F1 over a stratified
10-seed test split drawn the same way probing_rigor.multiseed does. Results land
in `constrained_decoding.json` for the chapter's "prescription realised" figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _class_token_sequences(tokenizer, classes: list[str]) -> dict[str, list[int]]:
    """Tokenize each class name. Use leading space — decoder outputs words with
    BPE space prefix."""
    out: dict[str, list[int]] = {}
    for c in classes:
        # Try " <class>" first; fall back to <class> if tokenizer strips it.
        ids = tokenizer.encode(f" {c}", add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(c, add_special_tokens=False)
        out[c] = ids
    return out


def _score_class_sequences(model, tokenizer, prompt: str, class_ids: dict[str, list[int]],
                           device) -> dict[str, float]:
    """Return summed log-prob of each class name sequence as a continuation of
    the prompt. Uses a single forward pass per class (cheap — classes are 1-4
    tokens long)."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)
    prompt_ids = inputs["input_ids"]
    scores: dict[str, float] = {}
    with torch.no_grad():
        for cls, ids in class_ids.items():
            cont = torch.tensor([ids], dtype=prompt_ids.dtype, device=device)
            full = torch.cat([prompt_ids, cont], dim=1)
            logits = model(full).logits  # (1, T, V)
            # Sum log-probs of each continuation token given its prefix
            logp = 0.0
            for i, tok_id in enumerate(ids):
                # position of the token in `full` is len(prompt_ids) + i
                # logits[pos-1] predicts full[pos]
                pos = prompt_ids.shape[1] + i
                lp = torch.log_softmax(logits[0, pos - 1], dim=-1)[tok_id].item()
                logp += lp
            scores[cls] = float(logp) / max(len(ids), 1)  # length-normalised
    return scores


def _free_decode(model, tokenizer, prompt: str, device, max_new: int = 12) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    return text.strip()


def _parse_pred(text: str, classes: list[str]) -> str:
    low = text.lower()
    for cls in sorted(classes, key=len, reverse=True):
        if cls.lower() in low:
            return cls
    return ""


def run(model_path: str, ground_truth_path: str, output_path: str,
        modalities: list[str], n_seeds: int = 5, test_frac: float = 0.2,
        load_in_8bit: bool = True) -> None:
    sys.path.insert(0, "/root")
    os.environ.setdefault("PIPELINE_SUBPROCESS", "1")
    import torch
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    from backend.evaluation.linear_probing import (
        CLASSIFICATION_TASKS, prepare_classification_data,
        _build_classification_prompt,
    )

    logger.info("loading Qwen2-VL-7B from %s (8-bit=%s)", model_path, load_in_8bit)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.eval()
    device = next(model.parameters()).device

    gt = json.loads(Path(ground_truth_path).read_text())
    task_data = prepare_classification_data([gt])

    out: dict[str, Any] = {}
    for task_name, (series_list, labels) in task_data.items():
        classes = CLASSIFICATION_TASKS[task_name]["classes"]
        class_ids = _class_token_sequences(tokenizer, classes)
        logger.info("task %s classes %s", task_name, classes)
        for cls, ids in class_ids.items():
            logger.info("  class '%s' -> %d tokens %s", cls, len(ids), ids)
        out[task_name] = {}

        y_all = np.array(labels)
        # Reuse probing convention: first n_train samples train, rest test.
        # For a fair comparison with multiseed we take stratified 10-seed splits.
        min_class = min((y_all == c).sum() for c in set(y_all))
        cv = n_seeds
        n_splits = min(cv, max(2, min_class))
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_frac,
                                          random_state=0)

        for mod in modalities:
            logger.info("─── task %s modality %s ───", task_name, mod)
            free_f1s: list[float] = []
            cons_f1s: list[float] = []
            free_preds: list[str] = []
            cons_preds: list[str] = []
            trues: list[str] = []
            for seed, (_tr, te) in enumerate(splitter.split(np.zeros(len(y_all)), y_all)):
                te_series = [series_list[i] for i in te]
                te_labels = [labels[i] for i in te]
                f_preds, c_preds = [], []
                for s, true_label in zip(te_series, te_labels):
                    prompt = _build_classification_prompt(s, task_name, mod)
                    # Free decode
                    text = _free_decode(model, tokenizer, prompt, device)
                    f_preds.append(_parse_pred(text, classes))
                    # Constrained decode
                    scores = _score_class_sequences(model, tokenizer, prompt, class_ids, device)
                    c_preds.append(max(scores, key=scores.get))
                # Normalise empty free-decode predictions to the first class
                # (same convention probing uses), counted as wrong if ≠ true.
                f_preds_norm = [p if p in classes else classes[0] for p in f_preds]
                free_f1 = float(f1_score(te_labels, f_preds_norm,
                                          labels=classes, average="macro", zero_division=0))
                cons_f1 = float(f1_score(te_labels, c_preds,
                                          labels=classes, average="macro", zero_division=0))
                free_f1s.append(free_f1)
                cons_f1s.append(cons_f1)
                free_preds.extend(f_preds)
                cons_preds.extend(c_preds)
                trues.extend(te_labels)
                logger.info("  seed %d  free F1 %.3f  constrained F1 %.3f  (n_test=%d)",
                            seed, free_f1, cons_f1, len(te_labels))

            out[task_name][mod] = {
                "free_decode": {
                    "mean_f1": round(float(np.mean(free_f1s)), 4),
                    "std_f1":  round(float(np.std(free_f1s)),  4),
                    "per_seed": [round(x, 4) for x in free_f1s],
                },
                "constrained_decode": {
                    "mean_f1": round(float(np.mean(cons_f1s)), 4),
                    "std_f1":  round(float(np.std(cons_f1s)),  4),
                    "per_seed": [round(x, 4) for x in cons_f1s],
                },
                "delta_constrained_minus_free": round(
                    float(np.mean(cons_f1s)) - float(np.mean(free_f1s)), 4
                ),
                "n_test_per_seed": len(trues) // n_splits,
                "n_seeds": n_splits,
            }
            Path(output_path).write_text(json.dumps(out, indent=2))
            logger.info("saved partial %s (after %s / %s)", output_path, task_name, mod)

    Path(output_path).write_text(json.dumps(out, indent=2))
    logger.info("wrote final %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--modalities", nargs="+",
                    default=["d", "nec+v", "nc+va", "d+v"],
                    help="Conditions to test. Default covers the chapter's four "
                         "task-specific prescriptions + baseline.")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--no-8bit", action="store_true")
    args = ap.parse_args()
    run(args.model_path, args.ground_truth, args.output, args.modalities,
        args.n_seeds, args.test_frac, not args.no_8bit)
