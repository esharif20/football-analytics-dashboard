"""Thin pod-side orchestrator for Gaps 3, 8, 9.

- class_token_rank: for each (task, modality, layer) compute the rank of each
  class-name token among all 152k vocab logits projected from the class-mean
  hidden state. Cross-modality extension of the existing d-only JSON.
- shuffled_extract: re-extract hidden states for modality d with frame order
  permuted before tokenisation (Gap 8 temporal control).
- random_digit_prompt: run prompting baseline where each value's digits are
  randomly reshuffled (Gap 9 input-readability control).

All three share the loaded model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _load(model_path: str, load_in_8bit: bool = True):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, processor


def class_token_rank(cache_dir: Path, model_path: str, output_path: Path,
                     modalities: list[str]) -> None:
    import torch
    from backend.evaluation.probing_rigor import _parse_task_modality
    from backend.evaluation.linear_probing import CLASSIFICATION_TASKS
    model, processor = _load(model_path)
    tokenizer = processor.tokenizer
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.model.lm_head
    device = next(model.parameters()).device

    want = set(modalities)
    out: dict[str, Any] = {}
    for npz in sorted(cache_dir.glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz)
        if not parsed:
            continue
        task, mod = parsed
        if mod not in want:
            continue
        data = np.load(npz, allow_pickle=True)
        hidden = data["hidden"]
        labels = np.asarray(data["labels"])
        probed = sorted([int(x) for x in data["layers_probed"]])
        classes = CLASSIFICATION_TASKS[task]["classes"]
        # Candidate class-name tokens with leading space (BPE)
        class_token_ids = {}
        for cls in classes:
            words = [cls.replace("_", " "), cls] + cls.split("_")
            ids = []
            for w in words:
                tid = tokenizer.encode(f" {w}", add_special_tokens=False)
                if tid:
                    ids.append(tid[0])
            class_token_ids[cls] = list(dict.fromkeys(ids))

        logger.info("=== %s / %s: classes=%s, layers=%s ===", task, mod,
                    classes, probed)
        task_out = out.setdefault(task, {}).setdefault(mod, {})
        for L in probed:
            layer_out: dict[str, Any] = {}
            for cls in classes:
                mask = np.array([str(l) == cls for l in labels])
                if mask.sum() == 0:
                    continue
                mean_h = hidden[mask, L, :].mean(axis=0)
                with torch.no_grad():
                    h = torch.tensor(mean_h, dtype=torch.float16, device=device)
                    logits = lm_head(h.unsqueeze(0))[0].float().cpu().numpy()
                # Sort descending; rank is 1-indexed position
                order = np.argsort(-logits)
                rank_of = {int(i): int(pos) for pos, i in enumerate(order)}
                ranks = {}
                for tid in class_token_ids[cls]:
                    tok = tokenizer.decode([tid]).strip()
                    ranks[tok] = rank_of.get(tid, -1)
                layer_out[cls] = ranks
            task_out[str(L)] = layer_out
            logger.info("  L%d done", L)
        output_path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", output_path)


def extract_shuffled(model, processor, ground_truth_path: str,
                     output_cache: Path) -> None:
    """Gap 8: extract hidden states for modality d with frames permuted."""
    sys.path.insert(0, "/workspace")
    os.environ.setdefault("PIPELINE_SUBPROCESS", "1")
    from backend.evaluation.linear_probing import (
        CLASSIFICATION_TASKS, prepare_classification_data,
        _build_classification_prompt,
    )
    import torch
    gt = json.loads(Path(ground_truth_path).read_text())
    task_data = prepare_classification_data([gt])
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    rng = random.Random(42)

    output_cache.mkdir(parents=True, exist_ok=True)
    for task in CLASSIFICATION_TASKS:
        if task not in task_data:
            continue
        series_list, labels = task_data[task]
        # Deterministic per-sample permutation
        shuffled = []
        for s in series_list:
            s2 = list(s)
            rng.shuffle(s2)
            shuffled.append(s2)
        hidden_all = []
        for s in shuffled:
            prompt = _build_classification_prompt(s, task, "d")
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            # (layers+1, 1, T, D) → take last token per layer
            h = torch.stack([hs[0, -1, :].float().cpu()
                             for hs in out.hidden_states], dim=0).numpy()
            hidden_all.append(h)
        hidden = np.stack(hidden_all, axis=0)  # (N, L, D)
        n = len(labels)
        probed = list(range(0, hidden.shape[1], 4))
        np.savez(output_cache / f"hidden_{task}_dshuf.npz",
                 hidden=hidden, labels=np.asarray(labels),
                 layers_probed=np.asarray(probed),
                 n_train=int(n * 0.8), n_test=int(n * 0.2))
        logger.info("shuffled: %s hidden %s", task, hidden.shape)


def run_random_digit_prompt(model, processor, ground_truth_path: str,
                             output_path: Path) -> None:
    """Gap 9: greedy prompting where each value's digits are reshuffled."""
    sys.path.insert(0, "/workspace")
    from backend.evaluation.linear_probing import (
        CLASSIFICATION_TASKS, prepare_classification_data,
        _build_classification_prompt,
    )
    from sklearn.metrics import f1_score
    import torch
    gt = json.loads(Path(ground_truth_path).read_text())
    task_data = prepare_classification_data([gt])
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    rng = random.Random(42)

    out: dict[str, Any] = {}
    for task in CLASSIFICATION_TASKS:
        if task not in task_data:
            continue
        series_list, labels = task_data[task]
        classes = CLASSIFICATION_TASKS[task]["classes"]

        # Pull first-token digit-randomised series
        def _jumble(series):
            out_s = []
            for v in series:
                s = f"{v:.1f}"
                digits = list(s.replace(".", ""))
                rng.shuffle(digits)
                # Re-insert decimal in original position
                idx = s.index(".")
                new = "".join(digits[:idx]) + "." + "".join(digits[idx:])
                try:
                    out_s.append(float(new))
                except Exception:
                    out_s.append(float(v))
            return out_s

        jumbled = [_jumble(s) for s in series_list]
        preds: list[str] = []
        for s, true_lbl in zip(jumbled, labels):
            prompt = _build_classification_prompt(s, task, "d")
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=12,
                                      do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().lower()
            pred = ""
            for cls in sorted(classes, key=len, reverse=True):
                if cls.lower() in text:
                    pred = cls
                    break
            preds.append(pred or classes[0])
        f1 = float(f1_score(labels, preds, labels=classes,
                            average="macro", zero_division=0))
        out[task] = {"f1_macro": round(f1, 4), "n": len(labels)}
        logger.info("random-digit %s F1=%.3f", task, f1)
        output_path.write_text(json.dumps(out, indent=2))


def _probe_cache(cache_file: Path, output: Path, label: str) -> None:
    """Gap 8 follow-up: stratified 10-seed probe on the shuffled cache."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    data = np.load(cache_file, allow_pickle=True)
    hidden = data["hidden"]
    labels = np.asarray(data["labels"])
    L = int(data["layers_probed"][-1])
    X = hidden[:, L, :]
    y = LabelEncoder().fit_transform(labels)
    classes = sorted(set(labels.tolist()))
    min_c = min((y == c).sum() for c in range(len(classes)))
    n_splits = min(10, max(2, min_c))
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2,
                                      random_state=0)
    f1s = []
    for _, (tr, te) in enumerate(splitter.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegressionCV(Cs=[0.001,0.01,0.1,1.0,10.0], cv=5,
                                    max_iter=1000, class_weight="balanced",
                                    scoring="f1_macro")
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        f1s.append(float(f1_score(y[te], pred,
                                   average="macro", zero_division=0)))
    return {"mean_f1": round(float(np.mean(f1s)), 4),
            "std_f1": round(float(np.std(f1s)), 4),
            "per_seed": [round(x, 4) for x in f1s]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rank", "shuffle", "randigit"], required=True)
    ap.add_argument("--cache-dir")
    ap.add_argument("--ground-truth")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--modalities", nargs="+", default=["d", "v", "d+v", "nec+v"])
    args = ap.parse_args()

    if args.mode == "rank":
        class_token_rank(Path(args.cache_dir), args.model_path,
                         Path(args.output), args.modalities)
    elif args.mode == "shuffle":
        m, p = _load(args.model_path)
        extract_shuffled(m, p, args.ground_truth, Path(args.output))
    elif args.mode == "randigit":
        m, p = _load(args.model_path)
        run_random_digit_prompt(m, p, args.ground_truth, Path(args.output))
