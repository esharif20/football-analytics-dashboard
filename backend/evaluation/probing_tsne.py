"""t-SNE visualisation of Qwen2-VL-7B hidden states per probing task × modality.

Reproduces Schumacher et al. (2026) Figure 4/5 style: 4 tasks × 2 modalities (d, v)
in a grid, coloured by ground-truth class. Visually demonstrates that visual modality
hidden states cluster by class (high probe F1) while text modality collapses to a blob
(low probe F1) — the core representation-vs-extraction gap.

Usage (model already loaded, e.g. on RunPod):
    python3 -m backend.evaluation.probing_tsne \\
        --ground-truth eval_output/dissertation/db_grounded/ \\
        --probing-results eval_output/dissertation/probing_vl/probing_results.json \\
        --output eval_output/dissertation/figures/ \\
        --model-path /workspace/models/Qwen2-VL-7B-Instruct

Usage (plot only from cached .npy embeddings):
    python3 -m backend.evaluation.probing_tsne \\
        --ground-truth eval_output/dissertation/db_grounded/ \\
        --probing-results eval_output/dissertation/probing_vl/probing_results.json \\
        --output eval_output/dissertation/figures/ \\
        --cache-dir eval_output/dissertation/probing_vl/embeddings \\
        --plot-only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import load_db_ground_truth, ensure_output_dir, save_figure
from .linear_probing import (
    CLASSIFICATION_TASKS,
    prepare_classification_data,
    extract_hidden_states,
    extract_hidden_states_vision,
    format_time_series_for_llm,
)

logger = logging.getLogger(__name__)

# Probe F1 scores from probing_results.json — used as panel annotations.
# These are the Qwen2-VL-7B v3 results from the dissertation probing study.
_PROBE_F1: dict[str, dict[str, float]] = {
    "pressing_type":        {"d": 0.664, "v": 0.496},
    "compactness_trend":    {"d": 0.740, "v": 0.879},
    "possession_phase":     {"d": 0.466, "v": 0.674},
    "territorial_dominance": {"d": 0.182, "v": 0.444},
}

_CLASS_COLOURS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
]

MODALITIES = ["d", "v"]


def _load_best_layers(results_path: Path | None) -> dict:
    """Load best-F1 layer per task × modality from probing_results.json.

    Returns {task: {"d": layer_idx, "v": layer_idx}}.
    """
    defaults = {
        "pressing_type":         {"d": 4,  "v": 4},
        "compactness_trend":     {"d": 24, "v": 4},
        "possession_phase":      {"d": 24, "v": 16},
        "territorial_dominance": {"d": 8,  "v": 4},
    }
    if not results_path or not results_path.exists():
        return defaults
    try:
        data = json.loads(results_path.read_text())
        out: dict = {}
        for task, tdata in data.items():
            if not isinstance(tdata, dict):
                continue
            out[task] = {}
            for mod in ("d", "v"):
                lw = tdata.get(mod, {}).get("layer_wise", {})
                if lw:
                    best = max(
                        lw,
                        key=lambda k: lw[k].get("f1_macro", 0) if isinstance(lw[k], dict) else lw[k],
                    )
                    out[task][mod] = int(best)
        for task, mods in defaults.items():
            out.setdefault(task, {})
            for mod, layer in mods.items():
                out[task].setdefault(mod, layer)
        return out
    except Exception:
        return defaults


def _load_probe_f1(results_path: Path) -> dict:
    """Load probe F1 from JSON; fall back to hard-coded values if missing."""
    if results_path and results_path.exists():
        try:
            data = json.loads(results_path.read_text())
            out: dict = {}
            for task, tdata in data.items():
                if not isinstance(tdata, dict):
                    continue
                out[task] = {}
                for mod in ("d", "v"):
                    probe = tdata.get(mod, {}).get("probe", {})
                    if isinstance(probe, dict) and "f1_macro" in probe:
                        out[task][mod] = probe["f1_macro"]
            # Fill gaps from hard-coded values
            for task, mods in _PROBE_F1.items():
                out.setdefault(task, {})
                for mod, f1 in mods.items():
                    out[task].setdefault(mod, f1)
            return out
        except Exception:
            pass
    return {t: dict(m) for t, m in _PROBE_F1.items()}


def extract_embeddings(
    model, tokenizer, processor, task_data: dict, cache_dir: Path,
    layer_map: "dict | None" = None,
) -> dict:
    """Extract hidden states for each task × modality and cache as .npy files.

    For modality='v', uses extract_hidden_states_vision (proper Qwen2-VL vision
    pathway). For modality='d', uses extract_hidden_states (text-only path).
    Extracts from the best-F1 layer per task rather than always the last layer.

    Returns:
        {task: {modality: {"hidden": np.array(n, d), "labels": list[str]}}}
    """
    from ._common import ensure_output_dir as _ensure
    _ensure(str(cache_dir))
    if layer_map is None:
        layer_map = {}
    results: dict = {}

    for task_name, (series_list, labels) in task_data.items():
        if not series_list:
            logger.warning("Skipping %s — no samples", task_name)
            continue
        results[task_name] = {}
        task_layers = layer_map.get(task_name, {})

        for modality in MODALITIES:
            layer_idx = task_layers.get(modality, -1)
            npy_path = cache_dir / f"{task_name}_{modality}.npy"
            if npy_path.exists():
                logger.info("Loading cached embeddings: %s", npy_path)
                hidden = np.load(str(npy_path))
                # Sanity: v and d should differ when both cached
                if modality == "v":
                    d_path = cache_dir / f"{task_name}_d.npy"
                    if d_path.exists():
                        d_hidden = np.load(str(d_path))
                        if np.array_equal(hidden, d_hidden):
                            logger.warning(
                                "CACHE BUG: %s v==d (identical). Delete %s and re-run.",
                                task_name, npy_path,
                            )
                results[task_name][modality] = {"hidden": hidden, "labels": labels}
                continue

            logger.info(
                "Extracting hidden states: %s × %s (%d samples, layer=%d)",
                task_name, modality, len(series_list), layer_idx,
            )

            if modality == "v" and processor is not None:
                hidden = extract_hidden_states_vision(
                    model, processor, series_list, task_name,
                    modality="v", layer=layer_idx,
                )
            else:
                # text-only path (modality='d' or no processor)
                prompts = [
                    format_time_series_for_llm(s, "d", task_name=task_name, fps=25.0)
                    for s in series_list
                ]
                hidden = extract_hidden_states(
                    model, tokenizer, prompts, layer=layer_idx
                )

            np.save(str(npy_path), hidden)
            logger.info("Saved %s shape=%s", npy_path, hidden.shape)
            results[task_name][modality] = {"hidden": hidden, "labels": labels}

    return results


def run_tsne(hidden: np.ndarray, n_samples: int = 500) -> np.ndarray:
    """PCA→50 then t-SNE reduction to 2D. Subsample to n_samples if larger."""
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("scikit-learn required: pip install scikit-learn") from e

    hidden = hidden.astype(np.float32)
    if len(hidden) > n_samples:
        idx = np.random.choice(len(hidden), n_samples, replace=False)
        hidden = hidden[idx]

    # PCA to 50 before t-SNE — standard practice for high-D embeddings
    n_pca = min(50, hidden.shape[0] - 1, hidden.shape[1])
    if n_pca > 2:
        hidden = PCA(n_components=n_pca, random_state=42).fit_transform(hidden)

    perplexity = min(10, max(5, len(hidden) // 5))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    return tsne.fit_transform(hidden)


def run_probe_lda(hidden: np.ndarray, labels: list) -> np.ndarray:
    """Project hidden states onto the LDA discriminant axes (2-D).

    This is the supervised analogue of t-SNE — it visualises the same linear
    subspace that a linear probe exploits. When probe F1 is high, clusters
    separate clearly. Fits on all data jointly (qualitative figure only).
    """
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except ImportError as e:
        raise ImportError("scikit-learn required: pip install scikit-learn") from e

    n_classes = len(set(labels))
    n_components = min(2, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    xy = lda.fit_transform(hidden.astype(np.float32), labels)
    if xy.shape[1] == 1:
        xy = np.hstack([xy, np.zeros((len(xy), 1), dtype=np.float32)])
    return xy


def _scatter_panel(
    ax, xy: np.ndarray, labels_plot: list, classes: list,
    colour_map: dict, f1: float, task_name: str, modality: str,
    proj_label: str = "t-SNE",
) -> None:
    """Shared scatter plot helper for t-SNE and LDA panels."""
    import matplotlib.patches as mpatches

    counts = {cls: labels_plot.count(cls) for cls in classes}
    for cls in classes:
        mask = np.array([lb == cls for lb in labels_plot])
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=18, color=colour_map[cls], alpha=0.65,
            label=cls.replace("_", " "), linewidths=0,
        )

    count_str = "  ".join(f"{cls.replace('_',' ')}:{counts.get(cls,0)}" for cls in classes)
    ax.set_title(
        f"{task_name.replace('_', ' ').title()}\n"
        f"mod={modality}  probe F1={f1:.3f}\n"
        f"{count_str}",
        fontsize=8, pad=3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        mpatches.Patch(color=colour_map[cls], label=cls.replace("_", " "))
        for cls in classes
    ]
    ax.legend(handles=handles, fontsize=6, loc="lower left", framealpha=0.75)


def make_tsne_figure(
    embeddings: dict, probe_f1: dict, output_dir: Path
) -> None:
    """Generate 4-row × 2-col t-SNE grid and save as PDF + PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib required") from e

    tasks = list(CLASSIFICATION_TASKS.keys())
    fig, axes = plt.subplots(
        len(tasks), len(MODALITIES),
        figsize=(5 * len(MODALITIES), 4 * len(tasks)),
        squeeze=False,
    )
    fig.suptitle(
        "Qwen2-VL-7B Hidden States (best layer) — t-SNE (PCA→50 preprocessing)\n"
        "Colour = ground-truth class  ·  d = text modality  ·  v = chart image modality",
        fontsize=11, y=1.01,
    )

    for row, task_name in enumerate(tasks):
        classes = CLASSIFICATION_TASKS[task_name]["classes"]
        colour_map = {cls: _CLASS_COLOURS[i] for i, cls in enumerate(classes)}

        for col, modality in enumerate(MODALITIES):
            ax = axes[row][col]
            emb_data = embeddings.get(task_name, {}).get(modality)

            if emb_data is None:
                ax.text(0.5, 0.5, "No embeddings", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_axis_off()
                continue

            hidden = emb_data["hidden"]
            labels = emb_data["labels"]

            logger.info("Running t-SNE: %s × %s (%d samples)", task_name, modality, len(hidden))
            xy = run_tsne(hidden)
            if len(xy) < len(labels):
                np.random.seed(42)
                idx = np.random.choice(len(labels), len(xy), replace=False)
                labels_plot = [labels[i] for i in idx]
            else:
                labels_plot = list(labels)

            f1 = probe_f1.get(task_name, {}).get(modality, 0.0)
            _scatter_panel(ax, xy, labels_plot, classes, colour_map, f1, task_name, modality)

    plt.tight_layout()
    save_figure(fig, "probe_tsne", output_dir)
    logger.info("t-SNE figure saved to %s/probe_tsne.{pdf,png}", output_dir)


def make_lda_figure(
    embeddings: dict, probe_f1: dict, output_dir: Path
) -> None:
    """Generate 4-row × 2-col LDA (probe-discriminant) projection grid.

    Unlike t-SNE, LDA directly visualises the linear subspace a logistic probe
    uses. When probe F1 is high, clusters separate clearly here.
    Fits on all data jointly — qualitative figure only, not a generalisation claim.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib required") from e

    tasks = list(CLASSIFICATION_TASKS.keys())
    fig, axes = plt.subplots(
        len(tasks), len(MODALITIES),
        figsize=(5 * len(MODALITIES), 4 * len(tasks)),
        squeeze=False,
    )
    fig.suptitle(
        "Qwen2-VL-7B Hidden States (best layer) — LDA projection\n"
        "Supervised: visualises the linear subspace the probe reads. "
        "Fitted on all data (qualitative only).\n"
        "d = text modality  ·  v = chart image modality",
        fontsize=10, y=1.01,
    )

    for row, task_name in enumerate(tasks):
        classes = CLASSIFICATION_TASKS[task_name]["classes"]
        colour_map = {cls: _CLASS_COLOURS[i] for i, cls in enumerate(classes)}

        for col, modality in enumerate(MODALITIES):
            ax = axes[row][col]
            emb_data = embeddings.get(task_name, {}).get(modality)

            if emb_data is None:
                ax.text(0.5, 0.5, "No embeddings", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_axis_off()
                continue

            hidden = emb_data["hidden"]
            labels = list(emb_data["labels"])

            logger.info("Running LDA: %s × %s (%d samples)", task_name, modality, len(hidden))
            try:
                xy = run_probe_lda(hidden, labels)
            except Exception as exc:
                logger.warning("LDA failed for %s × %s: %s", task_name, modality, exc)
                ax.text(0.5, 0.5, f"LDA failed:\n{exc}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="red")
                ax.set_axis_off()
                continue

            f1 = probe_f1.get(task_name, {}).get(modality, 0.0)
            _scatter_panel(ax, xy, labels, classes, colour_map, f1, task_name, modality,
                           proj_label="LDA")

    plt.tight_layout()
    save_figure(fig, "probe_lda", output_dir)
    logger.info("LDA figure saved to %s/probe_lda.{pdf,png}", output_dir)


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
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load probe F1 scores and per-task best layers
    results_path = Path(args.probing_results) if args.probing_results else None
    probe_f1 = _load_probe_f1(results_path)
    layer_map = _load_best_layers(results_path)
    logger.info("Probe F1 loaded: %s", probe_f1)
    logger.info("Best layers per task: %s", layer_map)

    # Load ground truth data
    gt_path = Path(args.ground_truth)
    if gt_path.is_dir():
        gt_files = sorted(gt_path.glob("*_db_ground_truth.json"))
        if not gt_files:
            logger.error("No *_db_ground_truth.json in %s", gt_path)
            sys.exit(1)
        gts = [load_db_ground_truth(str(f)) for f in gt_files]
        logger.info("Loaded %d ground truth files", len(gts))
    else:
        gts = [load_db_ground_truth(str(gt_path))]

    task_data = prepare_classification_data(gts)
    logger.info("Task data: %s",
                {t: len(s) for t, (s, _) in task_data.items()})

    if args.plot_only:
        # Load cached embeddings only
        embeddings: dict = {}
        for task_name in CLASSIFICATION_TASKS:
            embeddings[task_name] = {}
            _, labels = task_data[task_name]
            for modality in MODALITIES:
                npy_path = cache_dir / f"{task_name}_{modality}.npy"
                if npy_path.exists():
                    hidden = np.load(str(npy_path))
                    embeddings[task_name][modality] = {"hidden": hidden, "labels": labels}
                    logger.info("Loaded %s %s: shape %s", task_name, modality, hidden.shape)
                    # Warn if v == d (stale corrupt cache)
                    if modality == "v":
                        d_path = cache_dir / f"{task_name}_d.npy"
                        if d_path.exists() and np.array_equal(hidden, np.load(str(d_path))):
                            logger.warning(
                                "CACHE BUG DETECTED: %s v == d. "
                                "Delete %s and re-run without --plot-only to fix.",
                                task_name, npy_path,
                            )
                else:
                    logger.warning("Missing cache: %s — panel will be blank", npy_path)
    else:
        # Load model and extract
        if not args.model_path:
            logger.error("--model-path required unless --plot-only is set")
            sys.exit(1)
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as e:
            logger.error("Missing dep: %s", e)
            sys.exit(1)

        logger.info("Loading model from %s", args.model_path)
        model_path = args.model_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = None

        # Try Qwen2-VL processor for vision modality
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                output_hidden_states=True,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            logger.info("Loaded Qwen2VLForConditionalGeneration")
        except Exception:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                output_hidden_states=True,
                device_map="auto" if torch.cuda.is_available() else "cpu",
            )
            logger.info("Loaded AutoModelForCausalLM (text only)")

        embeddings = extract_embeddings(
            model, tokenizer, processor, task_data, cache_dir, layer_map=layer_map
        )

    make_tsne_figure(embeddings, probe_f1, output_dir)
    make_lda_figure(embeddings, probe_f1, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE probing visualisation")
    parser.add_argument("--ground-truth", required=True,
                        help="Path to db_ground_truth JSON or directory of them")
    parser.add_argument("--probing-results", default=None,
                        help="Path to probing_results.json for F1 annotations")
    parser.add_argument("--output", default="eval_output/dissertation/figures",
                        help="Output directory for figures")
    parser.add_argument("--model-path", default=None,
                        help="HuggingFace model path (required unless --plot-only)")
    parser.add_argument("--cache-dir",
                        default="eval_output/dissertation/probing_vl/embeddings",
                        help="Directory for cached .npy embeddings")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip extraction, load from cache and plot only")
    main(parser.parse_args())
