"""Generate dissertation supplementary figures from cached probing outputs.

Three figures (all CPU-only, no GPU):
  1. Layer-wise F1 curves — already generated; this re-generates with publication styling
  2. 1-D probe direction histogram — projects hidden states onto probe weight vector
  3. PCA scatter — replaces t-SNE with PCA to show linear separability

Usage:
    python3 -m backend.evaluation.dissertation_figures \
        --probing-results eval_output/dissertation/probing_vl/probing_results.json \
        --embeddings-dir eval_output/dissertation/probing_vl/embeddings/ \
        --ground-truth eval_output/dissertation/db_grounded/ \
        --output eval_output/dissertation/figures/
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
    train_linear_probe,
)

logger = logging.getLogger(__name__)

TASKS = list(CLASSIFICATION_TASKS.keys())
MODALITIES = ["d", "v"]
COLOURS = {"d": "#1f77b4", "v": "#d62728", "d+v": "#2ca02c"}
CLASS_COLOURS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Layer-wise F1 curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_layerwise(results: dict, output_dir: Path) -> None:
    """4-panel grid: one panel per task, lines per modality."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        task_data = results.get(task, {})
        has_data = False
        for mod in ["d", "v", "d+v"]:
            lw = task_data.get(mod, {}).get("layer_wise")
            if not lw:
                continue
            layers = sorted(int(k) for k in lw.keys())
            f1s = [lw[str(l)]["f1_macro"] if isinstance(lw[str(l)], dict) else lw[str(l)] for l in layers]
            ax.plot(layers, f1s, marker="o", markersize=4, linewidth=1.8,
                    color=COLOURS[mod], label=f"mod={mod}")
            has_data = True

        # Random baseline
        if has_data:
            ax.axhline(y=1 / len(CLASSIFICATION_TASKS[task]["classes"]),
                       color="gray", linestyle=":", linewidth=1, label="chance")

        ax.set_title(task.replace("_", " ").title(), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probe F1 macro", fontsize=8)
        ax.set_xlabel("Layer index", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Layer-wise linear probe F1 — Qwen2-VL-7B\n"
                 "Shows which layers encode each tactical concept",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save_figure(fig, "layerwise_f1", str(output_dir))
    logger.info("Layer-wise figure saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 1-D probe direction histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_probe_direction(embeddings_dir: Path, task_data: dict,
                         output_dir: Path) -> None:
    """For each task × modality: project hidden states onto probe weight vector,
    plot per-class histogram. Shows why probe F1 is high despite t-SNE blobs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_tasks = len(TASKS)
    n_mods = len(MODALITIES)
    fig, axes = plt.subplots(n_tasks, n_mods, figsize=(5 * n_mods, 3.5 * n_tasks),
                             squeeze=False)

    for row, task in enumerate(TASKS):
        series_list, labels = task_data.get(task, ([], []))
        classes = CLASSIFICATION_TASKS[task]["classes"]

        for col, mod in enumerate(MODALITIES):
            ax = axes[row][col]
            npy = embeddings_dir / f"{task}_{mod}.npy"
            if not npy.exists():
                ax.text(0.5, 0.5, "No embeddings", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_axis_off()
                continue

            hidden = np.load(str(npy)).astype(np.float32)
            if len(hidden) != len(labels):
                min_n = min(len(hidden), len(labels))
                hidden = hidden[:min_n]
                labels_use = labels[:min_n]
            else:
                labels_use = labels

            # Train probe, get weight vector
            n_test = max(1, int(len(labels_use) * 0.2))
            n_train = len(labels_use) - n_test
            result = train_linear_probe(
                hidden[:n_train], labels_use[:n_train],
                hidden[n_train:], labels_use[n_train:],
            )
            clf = result.get("_clf")
            if clf is None or not hasattr(clf, "coef_"):
                ax.text(0.5, 0.5, "Probe failed", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue

            # Project all samples onto first discriminative direction
            w = clf.coef_[0]  # (hidden_dim,) — first class vs rest
            w = w / (np.linalg.norm(w) + 1e-9)
            projections = hidden @ w

            for i, cls in enumerate(classes):
                mask = np.array([lb == cls for lb in labels_use])
                if mask.sum() == 0:
                    continue
                ax.hist(projections[mask], bins=20, alpha=0.6,
                        color=CLASS_COLOURS[i % len(CLASS_COLOURS)],
                        label=cls.replace("_", " "), density=True)

            f1 = result["f1_macro"]
            ax.set_title(f"{task.replace('_',' ').title()}\nmod={mod}  probe F1={f1:.3f}",
                         fontsize=9)
            ax.set_xlabel("Projection onto probe axis w", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Hidden states projected onto probe weight vector\n"
                 "Separated distributions → high F1 even when t-SNE shows blobs",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save_figure(fig, "probe_direction_hist", str(output_dir))
    logger.info("Probe direction histogram saved.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA scatter (linear alternative to t-SNE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca(embeddings_dir: Path, task_data: dict, results: dict,
             output_dir: Path) -> None:
    """Same 4×2 grid as t-SNE but using PCA — preserves linear variance,
    so linear separability is visible when it exists."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from sklearn.decomposition import PCA
    except ImportError:
        return

    n_tasks = len(TASKS)
    n_mods = len(MODALITIES)
    fig, axes = plt.subplots(n_tasks, n_mods, figsize=(5 * n_mods, 4 * n_tasks),
                             squeeze=False)

    for row, task in enumerate(TASKS):
        _, labels = task_data.get(task, ([], []))
        classes = CLASSIFICATION_TASKS[task]["classes"]
        colour_map = {cls: CLASS_COLOURS[i] for i, cls in enumerate(classes)}

        for col, mod in enumerate(MODALITIES):
            ax = axes[row][col]
            npy = embeddings_dir / f"{task}_{mod}.npy"
            if not npy.exists():
                ax.set_axis_off()
                continue

            hidden = np.load(str(npy)).astype(np.float32)
            if len(hidden) != len(labels):
                min_n = min(len(hidden), len(labels))
                hidden = hidden[:min_n]
                labels_plot = labels[:min_n]
            else:
                labels_plot = labels

            pca = PCA(n_components=2, random_state=42)
            xy = pca.fit_transform(hidden)
            var = pca.explained_variance_ratio_

            for cls in classes:
                mask = np.array([lb == cls for lb in labels_plot])
                if mask.sum() == 0:
                    continue
                ax.scatter(xy[mask, 0], xy[mask, 1], s=20,
                           color=colour_map[cls], alpha=0.7, linewidths=0)

            f1 = results.get(task, {}).get(mod, {}).get("probe", {}).get("f1_macro", 0.0)
            ax.set_title(
                f"{task.replace('_',' ').title()}\n"
                f"mod={mod}  probe F1={f1:.3f}  "
                f"PCA var={var[0]:.1%}+{var[1]:.1%}",
                fontsize=8, pad=3,
            )
            ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=7)
            ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

            if col == 0:
                handles = [mpatches.Patch(color=colour_map[cls],
                                          label=cls.replace("_", " "))
                           for cls in classes]
                ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.7)

    fig.suptitle("Qwen2-VL-7B Hidden States — PCA (linear projection)\n"
                 "Unlike t-SNE, PCA preserves linear structure — "
                 "separation here ↔ high probe F1",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    save_figure(fig, "probe_pca", str(output_dir))
    logger.info("PCA scatter saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Results table (LaTeX + markdown)
# ─────────────────────────────────────────────────────────────────────────────

def save_results_table(results: dict, random_baseline: dict, output_dir: Path) -> None:
    """Generate a Schumacher-style comparison table."""
    short = {
        "pressing_type": "PTY",
        "compactness_trend": "CTR",
        "possession_phase": "PPS",
        "territorial_dominance": "TDM",
    }

    md_rows = []
    tex_rows = []
    md_rows.append("| Model | Method | Mod | " + " | ".join(short.values()) + " | Avg |")
    md_rows.append("|---|---|---|" + "---|" * (len(short) + 1))

    for method_key, method_label in [("probe", "Probe"), ("prompting", "Prompt")]:
        for mod in ["d", "v", "d+v"]:
            f1s = []
            for task in TASKS:
                v = results.get(task, {}).get(mod, {}).get(method_key, {}).get("f1_macro")
                f1s.append(v if v is not None else 0.0)
            avg = np.mean(f1s)
            cells = " | ".join(f"{v:.3f}" for v in f1s)
            md_rows.append(f"| Qwen2-VL-7B | {method_label} | {mod} | {cells} | **{avg:.3f}** |")
            tex_rows.append(f"Qwen2-VL-7B & {method_label} & ${mod}$ & "
                            + " & ".join(f"{v:.3f}" for v in f1s)
                            + f" & \\textbf{{{avg:.3f}}} \\\\")

    # Random baseline
    for mod in ["d", "v", "d+v"]:
        f1s = []
        for task in TASKS:
            # Random baseline is the same regardless of modality
            v = random_baseline.get(task, {}).get("random_probe_f1", 0.0)
            f1s.append(v)
        avg = np.mean(f1s)
        cells = " | ".join(f"{v:.3f}" for v in f1s)
        md_rows.append(f"| Random | Probe | {mod} | {cells} | {avg:.3f} |")

    table_path = output_dir / "probing_results_table.md"
    table_path.write_text("\n".join(md_rows))

    tex_path = output_dir / "probing_results_table.tex"
    tex_content = (
        "\\begin{tabular}{llrccccr}\n"
        "\\toprule\n"
        "Model & Method & Mod & " + " & ".join(short.values()) + " & Avg \\\\\n"
        "\\midrule\n"
        + "\n".join(tex_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    tex_path.write_text(tex_content)
    logger.info("Results table written: %s and %s", table_path, tex_path)


# ─────────────────────────────────────────────────────────────────────────────
# Patch train_linear_probe to return clf
# ─────────────────────────────────────────────────────────────────────────────

def _patched_train_linear_probe(X_train, y_train, X_test, y_test):
    """Wraps train_linear_probe and also returns the fitted clf."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import f1_score, accuracy_score

    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform([y for y in y_test if y in le.classes_])
    if len(set(y_tr)) < 2:
        return {"f1_macro": 0.0, "_clf": None, "predictions": []}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegressionCV(cv=3, max_iter=1000, multi_class="multinomial",
                               solver="lbfgs")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    preds = list(le.inverse_transform(y_pred))
    return {"f1_macro": float(f1), "_clf": clf, "predictions": preds,
            "accuracy": float(accuracy_score(y_te, y_pred))}


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    output_dir = Path(ensure_output_dir(args.output))
    embeddings_dir = Path(args.embeddings_dir)

    with open(args.probing_results) as f:
        results = json.load(f)

    rb_path = Path(args.probing_results).parent / "random_baseline.json"
    random_baseline = json.loads(rb_path.read_text()) if rb_path.exists() else {}

    # Load ground truth for label access
    gt_path = Path(args.ground_truth)
    if gt_path.is_dir():
        gt_files = sorted(gt_path.glob("*_db_ground_truth.json"))
        gts = [load_db_ground_truth(str(f)) for f in gt_files]
    else:
        gts = [load_db_ground_truth(str(gt_path))]
    task_data = prepare_classification_data(gts)

    logger.info("Generating results table...")
    save_results_table(results, random_baseline, output_dir)

    logger.info("Generating layer-wise F1 curves...")
    plot_layerwise(results, output_dir)

    logger.info("Generating probe direction histograms...")
    # Monkey-patch to get clf back
    import backend.evaluation.dissertation_figures as _self
    _self.train_linear_probe = _patched_train_linear_probe
    plot_probe_direction(embeddings_dir, task_data, output_dir)

    logger.info("Generating PCA scatter...")
    plot_pca(embeddings_dir, task_data, results, output_dir)

    logger.info("All done. Outputs in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probing-results",
                        default="eval_output/dissertation/probing_vl/probing_results.json")
    parser.add_argument("--embeddings-dir",
                        default="eval_output/dissertation/probing_vl/embeddings/")
    parser.add_argument("--ground-truth",
                        default="eval_output/dissertation/db_grounded/")
    parser.add_argument("--output",
                        default="eval_output/dissertation/figures/")
    main(parser.parse_args())
