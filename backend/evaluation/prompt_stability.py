"""Prompt stability study — mirrors Schumacher et al. (2026) Table 3.

Measures:
  Δ = max(F1_macro) − min(F1_macro) across 10 meaning-preserving prompt variants
  P@K = fraction of test samples where ≥2/3 of variants agree on the predicted class

Both prompting (zero-shot text generation) and probing (linear probe) are evaluated.
The probing side uses 10 different (seed) combinations to measure probe stability.
Prompting is expected to show high Δ (brittle); probing should show low Δ (stable).

Usage (requires Qwen2-VL model on pod or local):
    python3 -m backend.evaluation.prompt_stability \\
        --ground-truth eval_output/dissertation/db_grounded/ \\
        --model-path /workspace/models/Qwen2-VL-7B-Instruct \\
        --output eval_output/dissertation/figures/ \\
        --tasks pressing_type,territorial_dominance \\
        --modalities d,v
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import load_db_ground_truth, ensure_output_dir, save_figure
from .linear_probing import (
    CLASSIFICATION_TASKS,
    _PROMPT_VARIANTS,
    _build_classification_prompt,
    prepare_classification_data,
    run_prompting_baseline,
    train_linear_probe,
    extract_hidden_states,
    format_time_series_for_llm,
)

logger = logging.getLogger(__name__)

N_VARIANTS = len(_PROMPT_VARIANTS)       # 10
P_AT_K_THRESHOLD = 2 / 3                # per-sample consistency threshold


def _compute_delta_pak(f1_list: list[float], predictions_list: list[list[str]]) -> tuple:
    """Compute Δ = max − min F1 and P@K consistency over variants.

    Args:
        f1_list: F1_macro for each variant.
        predictions_list: Per-variant list of per-sample predicted classes.

    Returns:
        (delta, pak): Δ float and P@K float.
    """
    delta = max(f1_list) - min(f1_list)
    # P@K: fraction of samples where ≥2/3 of variants agree
    n_variants = len(predictions_list)
    if n_variants == 0 or not predictions_list[0]:
        return delta, 0.0
    n_samples = len(predictions_list[0])
    consistent = 0
    for i in range(n_samples):
        sample_preds = [predictions_list[v][i] for v in range(n_variants)
                        if i < len(predictions_list[v])]
        if not sample_preds:
            continue
        most_common_count = Counter(sample_preds).most_common(1)[0][1]
        if most_common_count / len(sample_preds) >= P_AT_K_THRESHOLD:
            consistent += 1
    return delta, consistent / n_samples


def run_prompting_stability(
    model, tokenizer, series_list, labels, task_name, modality, fps=25.0
) -> dict:
    """Run prompting over all N_VARIANTS variants, return Δ and P@K."""
    f1_list = []
    predictions_list = []
    for vid in range(N_VARIANTS):
        import asyncio
        result = asyncio.run(run_prompting_baseline(
            model, tokenizer, series_list, labels, task_name, modality,
            fps=fps, variant_id=vid,
        ))
        f1_list.append(result["f1_macro"])
        predictions_list.append(result.get("predictions", []))
        logger.info("  variant %d/%d: F1=%.3f", vid + 1, N_VARIANTS, result["f1_macro"])
    delta, pak = _compute_delta_pak(f1_list, predictions_list)
    return {
        "f1_list": f1_list,
        "f1_mean": float(np.mean(f1_list)),
        "delta": round(delta, 4),
        "pak": round(pak, 4),
        "method": "prompting",
    }


def run_probing_stability(
    model, tokenizer, series_list, labels, task_name, modality,
    n_seeds=N_VARIANTS, images=None
) -> dict:
    """Run probing with N_SEEDS different random seeds, return Δ and P@K."""
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch required") from e

    prompts = [
        _build_classification_prompt(s, task_name, modality, fps=25.0, variant_id=0)
        for s in series_list
    ]
    hidden = extract_hidden_states(model, tokenizer, prompts, images=images, layer=-1)

    from sklearn.model_selection import train_test_split
    n_test = max(1, int(len(labels) * 0.2))
    n_train = len(labels) - n_test

    f1_list = []
    predictions_list = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        idx = np.random.permutation(len(labels))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        try:
            result = train_linear_probe(
                hidden[train_idx], [labels[i] for i in train_idx],
                hidden[test_idx], [labels[i] for i in test_idx],
            )
        except ValueError:
            logger.warning("  seed %d/%d: skipped (single-class split)", seed + 1, n_seeds)
            continue
        f1_list.append(result["f1_macro"])
        # Reconstruct per-sample predictions for P@K
        predictions_list.append(result.get("predictions", []))
        logger.info("  seed %d/%d: F1=%.3f", seed + 1, n_seeds, result["f1_macro"])
    if not f1_list:
        return {"f1_list": [], "f1_mean": 0.0, "delta": 0.0, "pak": 0.0, "method": "probing"}
    delta, pak = _compute_delta_pak(f1_list, predictions_list)
    return {
        "f1_list": f1_list,
        "f1_mean": float(np.mean(f1_list)),
        "delta": round(delta, 4),
        "pak": round(pak, 4),
        "method": "probing",
    }


def save_stability_table(results: dict, output_dir: Path) -> None:
    """Write markdown table and save results JSON."""
    lines = [
        "# Prompt Stability Study",
        "",
        "Mirrors Schumacher et al. (2026) Table 3.",
        "Δ = max(F1) − min(F1) across 10 variants. "
        "P@K = per-sample consistency (≥2/3 variants agree).",
        "",
        "| Task | Modality | Method | F1 mean | Δ | P@K |",
        "|---|---|---|---|---|---|",
    ]
    for key, r in sorted(results.items()):
        task, mod, method = key.split("|")
        lines.append(
            f"| {task} | {mod} | {method} | {r['f1_mean']:.3f} | {r['delta']:.3f} | {r['pak']:.3f} |"
        )
    (output_dir / "prompt_stability.md").write_text("\n".join(lines))
    (output_dir / "prompt_stability.json").write_text(json.dumps(results, indent=2))
    logger.info("Stability table written.")


def save_stability_figure(results: dict, output_dir: Path, tasks: list) -> None:
    """Bar chart: Δ per task × modality, grouped by method {prompting, probing}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    modalities = ["d", "v"]
    methods = ["prompting", "probing"]
    colours = {"prompting": "#d62728", "probing": "#1f77b4"}

    n_combos = len(tasks) * len(modalities)
    x = np.arange(n_combos)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_combos * 1.2), 4))
    for i, method in enumerate(methods):
        deltas = []
        for task in tasks:
            for mod in modalities:
                key = f"{task}|{mod}|{method}"
                deltas.append(results.get(key, {}).get("delta", 0.0))
        ax.bar(x + i * width, deltas, width, label=method.title(),
               color=colours[method], alpha=0.85)

    xlabels = [f"{t[:8]}\n{m}" for t in tasks for m in modalities]
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Δ (max−min macro F1)")
    ax.set_title("Prompt Stability: Δ F1 across 10 variants\n"
                 "Higher Δ = more brittle (prompting expected to dominate)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "prompt_stability", output_dir)
    logger.info("Stability bar chart saved.")


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
    except ImportError:
        pass

    output_dir = ensure_output_dir(args.output)

    # Load ground truth
    gt_path = Path(args.ground_truth)
    if gt_path.is_dir():
        gt_files = sorted(gt_path.glob("*_db_ground_truth.json"))
        gts = [load_db_ground_truth(str(f)) for f in gt_files]
    else:
        gts = [load_db_ground_truth(str(gt_path))]
    task_data = prepare_classification_data(gts)

    requested_tasks = (
        [t.strip() for t in args.tasks.split(",")] if args.tasks
        else list(CLASSIFICATION_TASKS.keys())
    )
    requested_mods = (
        [m.strip() for m in args.modalities.split(",")] if args.modalities
        else ["d", "v"]
    )

    # Subsample to 20 test samples for cost control
    test_n = 20

    # Load model
    if not args.model_path:
        logger.error("--model-path is required")
        sys.exit(1)

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as e:
        logger.error("Missing dep: %s", e)
        sys.exit(1)

    logger.info("Loading model from %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            output_hidden_states=True,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            output_hidden_states=True,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        processor = None

    results: dict = {}

    for task_name in requested_tasks:
        series_list, labels = task_data.get(task_name, ([], []))
        if not series_list:
            logger.warning("No data for %s", task_name)
            continue
        # Subsample to test_n
        n = min(test_n, len(series_list))
        idx = np.random.choice(len(series_list), n, replace=False)
        sl = [series_list[i] for i in idx]
        lb = [labels[i] for i in idx]

        for mod in requested_mods:
            logger.info("=== %s × %s ===", task_name, mod)

            logger.info("Prompting stability (%d samples × %d variants)...", n, N_VARIANTS)
            p_result = run_prompting_stability(model, tokenizer, sl, lb, task_name, mod)
            results[f"{task_name}|{mod}|prompting"] = p_result
            logger.info("  Δ=%.3f  P@K=%.3f", p_result["delta"], p_result["pak"])

            logger.info("Probing stability (%d samples × %d seeds)...", n, N_VARIANTS)
            imgs = None
            if mod == "v":
                imgs = []
                for s in sl:
                    fmt = format_time_series_for_llm(s, "v", task_name=task_name)
                    imgs.append(fmt[1] if isinstance(fmt, tuple) else None)
            pr_result = run_probing_stability(
                model, tokenizer, sl, lb, task_name, mod, images=imgs
            )
            results[f"{task_name}|{mod}|probing"] = pr_result
            logger.info("  Δ=%.3f  P@K=%.3f", pr_result["delta"], pr_result["pak"])

    save_stability_table(results, output_dir)
    save_stability_figure(results, output_dir, requested_tasks)
    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt stability study")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="eval_output/dissertation/figures")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names")
    parser.add_argument("--modalities", default="d,v")
    main(parser.parse_args())
