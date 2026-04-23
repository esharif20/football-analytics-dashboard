"""Audit `ne` (class-leaking) prompt variants for actual signal.

The class-name enumeration (`[1] high_press — ...`) always appears in every
prompt, so surface-level "class-name-present" is trivially 100%. The real
question is whether the `ne` threshold-gated hint (e.g. "indicating high
pressing intent") mentions the *correct* class for each sample. That's the
actual leakage signal.

For each (task, variant_id):
    - build the `ne` prompt with AND without `extended=True`
    - diff the two to isolate the hint-only text
    - check whether the hint contains the TRUE class name
Report per-variant hint-correctness rate.

Usage:
    python -m backend.evaluation.audit_ne_leakage \
        --gt eval_output/dissertation/db_grounded/18_db_ground_truth.json \
        --out eval_output/dissertation/probing_new_run/ne_leakage_audit.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from backend.evaluation.linear_probing import (
    CLASSIFICATION_TASKS, prepare_classification_data,
    _build_classification_prompt, _PROMPT_VARIANTS,
)


# Hardcoded mapping of the English adjective phrases emitted by
# _format_series_string `extended` branches → which class they hint at.
# Keys are substrings to search for (case-insensitive); values are class names.
_HINT_PHRASES: dict[str, dict[str, str]] = {
    "pressing_type": {
        "high pressing intent": "high_press",
        "mid block": "mid_block",
        "lower defensive line": "low_block",
    },
    "compactness_trend": {
        "compact defensive block": "compact",
        "stretched attacking shape": "expansive",
        "moderate inter-player spacing": "moderate",
    },
    "possession_phase": {
        "chaotic with rapid possession": "chaotic",
        "transitional": "transitional",
        "sustained possession": "sustained",
    },
    "territorial_dominance": {
        "pressing high up the pitch": "pressing_high",
        "retreating deep": "retreating",
        "holding a balanced line": "balanced",
    },
}


def _hinted_classes(prompt: str, task_name: str) -> list[str]:
    """Return the class names whose phrase appears in the prompt."""
    low = prompt.lower()
    hits = []
    for phrase, cls in _HINT_PHRASES.get(task_name, {}).items():
        if phrase.lower() in low:
            hits.append(cls)
    return hits


def run(gt_path: Path, out_path: Path) -> dict:
    gt = json.loads(gt_path.read_text())
    task_data = prepare_classification_data([gt])
    results: dict = {}

    for task_name, (series_list, labels_list) in task_data.items():
        task_cfg = CLASSIFICATION_TASKS[task_name]
        classes = task_cfg["classes"]
        results[task_name] = {"classes": classes, "variants": {}}

        for variant_id in range(len(_PROMPT_VARIANTS)):
            n_total = len(series_list)
            correct_hints = 0
            per_class_correct = defaultdict(int)
            per_class_total = defaultdict(int)
            wrong_class_hinted = defaultdict(int)  # hinted class -> count of wrong hints
            empty_hint = 0

            for s, true_label in zip(series_list, labels_list):
                ne_prompt = _build_classification_prompt(
                    s, task_name, "ne", variant_id=variant_id
                )
                hinted_classes = _hinted_classes(ne_prompt, task_name)
                per_class_total[true_label] += 1
                if not hinted_classes:
                    empty_hint += 1
                    continue
                if true_label in hinted_classes and len(hinted_classes) == 1:
                    correct_hints += 1
                    per_class_correct[true_label] += 1
                else:
                    for c in hinted_classes:
                        if c != true_label:
                            wrong_class_hinted[c] += 1

            results[task_name]["variants"][variant_id] = {
                "n_samples": n_total,
                "hint_correct_rate": round(correct_hints / n_total, 3),
                "empty_hints": empty_hint,
                "per_class_recall": {
                    c: round(per_class_correct[c] / max(1, per_class_total[c]), 3)
                    for c in classes
                },
                "wrong_class_hinted_counts": dict(wrong_class_hinted),
            }

        per_var = results[task_name]["variants"]
        rates = [v["hint_correct_rate"] for v in per_var.values()]
        results[task_name]["summary"] = {
            "hint_correct_min": round(min(rates), 3),
            "hint_correct_max": round(max(rates), 3),
            "hint_correct_mean": round(sum(rates) / len(rates), 3),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


def _print_report(results: dict) -> None:
    for task, payload in results.items():
        s = payload["summary"]
        print(f"\n=== {task} ({', '.join(payload['classes'])}) ===")
        print(f"  hint_correct_rate  min={s['hint_correct_min']:.2f} "
              f"mean={s['hint_correct_mean']:.2f} max={s['hint_correct_max']:.2f}")
        classes = payload["classes"]
        # Per-class recall averaged across variants
        avg_recall = {c: 0.0 for c in classes}
        for v in payload["variants"].values():
            for c in classes:
                avg_recall[c] += v["per_class_recall"][c]
        for c in classes:
            avg_recall[c] /= len(payload["variants"])
        print("  mean per-class hint-recall across variants:")
        for c in classes:
            marker = " <-- hint never names this class" if avg_recall[c] < 0.05 else ""
            print(f"    {c:20s} {avg_recall[c]:.2f}{marker}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    res = run(args.gt, args.out)
    _print_report(res)
    print(f"\nWrote {args.out}")
