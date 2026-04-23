"""Heuristic baselines for the tactical-probing study.

Replicates the Majority / Prior / Uniform rows of Schumacher et al. (2026)
Table 1 using the same 80/20 split the probe uses. Pure pandas/numpy —
no GPU.

Majority:  always predict the most-frequent training-set class.
Prior:     sample from the training-set class prior, independently per test sample.
Uniform:   sample uniformly across classes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

# Re-use the probing script's data prep so the splits match exactly.
from backend.evaluation.linear_probing import (
    CLASSIFICATION_TASKS,
    prepare_classification_data,
)


def _load_task_data(ground_truth_path: Path) -> dict:
    """Thin wrapper: load the db_extractor JSON then run the probe's data prep."""
    gt = json.loads(Path(ground_truth_path).read_text())
    # prepare_classification_data expects a LIST of ground-truth dicts
    return prepare_classification_data([gt])

logger = logging.getLogger(__name__)


def _f1_macro(y_true, y_pred, classes):
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(list(classes))
    return float(
        f1_score(
            le.transform(y_true),
            le.transform(y_pred),
            average="macro",
            zero_division=0,
        )
    )


def compute_heuristics(
    ground_truth_path: Path,
    test_fraction: float = 0.2,
    n_seeds: int = 20,
) -> dict:
    """Return {task: {baseline: {f1_macro, n_train, n_test, classes}}}."""
    task_data = _load_task_data(ground_truth_path)
    results: dict = {}
    for task_name in CLASSIFICATION_TASKS:
        if task_name not in task_data:
            continue
        series_list, labels = task_data[task_name]
        n_total = len(labels)
        if n_total < 5:
            continue
        n_test = max(1, int(n_total * test_fraction))
        n_train = n_total - n_test
        y_train = list(labels[:n_train])
        y_test = list(labels[n_train:])
        classes = sorted(set(labels))

        # Majority — deterministic
        from collections import Counter
        top = Counter(y_train).most_common(1)[0][0]
        f1_maj = _f1_macro(y_test, [top] * len(y_test), classes)

        # Prior / Uniform — averaged over seeds
        rng = np.random.default_rng(42)
        prior_counts = np.array([y_train.count(c) for c in classes], dtype=float)
        prior_probs = prior_counts / prior_counts.sum()
        f1_priors, f1_unifs = [], []
        for _ in range(n_seeds):
            pred_prior = rng.choice(classes, size=len(y_test), p=prior_probs)
            pred_unif = rng.choice(classes, size=len(y_test))
            f1_priors.append(_f1_macro(y_test, list(pred_prior), classes))
            f1_unifs.append(_f1_macro(y_test, list(pred_unif), classes))
        results[task_name] = {
            "n_train": n_train,
            "n_test": n_test,
            "classes": classes,
            "class_counts_train": {c: int(y_train.count(c)) for c in classes},
            "majority": {
                "f1_macro": round(f1_maj, 4),
                "predicted_class": top,
            },
            "prior": {
                "f1_macro": round(float(np.mean(f1_priors)), 4),
                "f1_std": round(float(np.std(f1_priors)), 4),
                "n_seeds": n_seeds,
            },
            "uniform": {
                "f1_macro": round(float(np.mean(f1_unifs)), 4),
                "f1_std": round(float(np.std(f1_unifs)), 4),
                "n_seeds": n_seeds,
            },
        }
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-seeds", type=int, default=20)
    args = ap.parse_args()

    results = compute_heuristics(
        Path(args.ground_truth), n_seeds=args.n_seeds
    )
    Path(args.output).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.output)

    # Print compact table
    print("\ntask                        | majority | prior    | uniform")
    print("-" * 60)
    for task, r in results.items():
        print(
            f"{task:27} | {r['majority']['f1_macro']:.4f}   "
            f"| {r['prior']['f1_macro']:.4f}   "
            f"| {r['uniform']['f1_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
