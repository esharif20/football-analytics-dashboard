"""Dissertation-grade "cool" visuals built from cached hidden states.

Four figures designed for the §4.8 addendum as hero plates:

  tsne-evolution   — Per task, 4-layer t-SNE-style grid showing class
                     separation evolving through transformer depth. Uses
                     PCA-50 → UMAP (preferred) or LDA-2 fallback because
                     sklearn TSNE segfaults on some Mac/MKL combos.
  modality-grid    — At layer 28, one 2D projection per modality for a
                     chosen task. Compares how clean the class clusters
                     are under each modality.
  multitask-embed  — All four tasks projected to 2D in a shared LDA
                     space. Visualises the cross-task orthogonality
                     finding: each task sits on its own axis.
  decision-boundary— Hidden states projected onto the linear probe's
                     weight direction. 1D / 2D histogram per class,
                     shows the hyperplane the probe learned.

All figures use a slate/navy academic palette matching the existing
CHAPTER_EVALUATION.md §4.8 figure set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)

PALETTE = {
    "ink":   "#0f172a",
    "slate": "#475569",
    "grid":  "#e2e8f0",
}
CLASS_CMAP = ["#1e3a8a", "#b45309", "#047857", "#be123c", "#7c3aed"]
TASK_CMAP = {
    "pressing_type":        "#1e3a8a",
    "compactness_trend":    "#b45309",
    "possession_phase":     "#047857",
    "territorial_dominance":"#be123c",
}
TASK_LABEL = {
    "pressing_type": "Pressing Type",
    "compactness_trend": "Compactness Trend",
    "possession_phase": "Possession Phase",
    "territorial_dominance": "Territorial Dominance",
}

_MOD_STEMS = [
    "ne_va", "ne_sp", "ne_v", "p_va", "p_sp", "p_v2", "p_vh",
    "n_va", "n_sp", "n_v",
    "d_v", "pi_v", "p_v", "b_v",
    "pi", "ne", "va", "sp", "v2", "vh",
    "d", "v", "p", "m", "o", "b", "n",
]


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titleweight": "semibold",
        "axes.edgecolor": PALETTE["slate"],
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
    })


def _parse_task_modality(path: Path):
    stem = path.stem.replace("hidden_", "")
    for m in _MOD_STEMS:
        if stem.endswith("_" + m):
            return stem[: -(len(m) + 1)], m.replace("_", "+") if m in ("d_v","p_v","pi_v","n_v","b_v","ne_v","p_va","p_sp","p_v2","p_vh","n_va","n_sp","ne_va","ne_sp") else m
    return None


def _reduce_2d(X, y=None, method="umap"):
    """Project (n, d) hidden states to 2D. UMAP preferred, LDA fallback,
    PCA final fallback. PCA-50 pre-reduction for efficiency."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    xs = StandardScaler().fit_transform(X.astype(np.float64))
    n_pca = min(50, xs.shape[0] - 1, xs.shape[1])
    xp = PCA(n_components=n_pca, random_state=42).fit_transform(xs)
    if method == "umap":
        try:
            import umap
            n_neighbors = min(15, max(3, xp.shape[0] // 3))
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                min_dist=0.3, random_state=42, metric="cosine")
            return reducer.fit_transform(xp), "UMAP"
        except Exception as e:
            logger.warning("UMAP unavailable (%s), falling back to LDA", e)
    if method in ("umap", "lda") and y is not None:
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            n_comp = min(2, len(np.unique(y)) - 1) or 1
            lda = LinearDiscriminantAnalysis(
                n_components=n_comp, solver="lsqr", shrinkage="auto",
            )
            emb = lda.fit(xp, y).transform(xp)
            if emb.shape[1] == 1:
                emb = np.hstack([emb, np.zeros_like(emb)])
            return emb, "LDA"
        except Exception as e:
            logger.warning("LDA failed (%s), using PCA-2", e)
    return PCA(n_components=2, random_state=42).fit_transform(xp), "PCA"


# ── 1. t-SNE-style evolution strip (4 layers × 4 tasks) ──────────────────────


def fig_tsne_evolution(cache_dir: Path, out_dir: Path,
                      modality: str = "p",
                      layers: list[int] | None = None) -> Path | None:
    """4×4 grid: rows = tasks, cols = layers {0, 8, 16, 28}. Shows class
    clusters emerging through transformer depth."""
    from sklearn.preprocessing import LabelEncoder
    layers = layers or [0, 8, 16, 28]
    safe = modality.replace("+", "_")
    tasks = ["pressing_type", "compactness_trend", "possession_phase", "territorial_dominance"]
    # Find caches
    cache_dir = Path(cache_dir)
    cached = {}
    for t in tasks:
        p = cache_dir / f"hidden_{t}_{safe}.npz"
        if p.exists():
            cached[t] = p
    if not cached:
        logger.warning("no cached npz for modality=%s", modality)
        return None

    fig, axes = plt.subplots(len(cached), len(layers),
                             figsize=(3.0 * len(layers), 2.8 * len(cached)),
                             squeeze=False)
    last_method = "PCA"
    for row_i, task in enumerate(tasks):
        if task not in cached:
            continue
        data = np.load(cached[task], allow_pickle=True)
        hidden = data["hidden"]
        labels = np.array([str(l) for l in data["labels"]])
        classes = list(dict.fromkeys(labels))
        le = LabelEncoder(); y = le.fit_transform(labels)
        for col_i, L in enumerate(layers):
            ax = axes[row_i, col_i]
            if L >= hidden.shape[1]:
                ax.set_visible(False); continue
            X = hidden[:, L, :]
            emb, method = _reduce_2d(X, y, method="umap")
            last_method = method
            for ci, cls in enumerate(classes):
                mask = labels == cls
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           color=CLASS_CMAP[ci % len(CLASS_CMAP)],
                           s=24, alpha=0.85, edgecolors="white",
                           linewidths=0.5, label=cls)
            if col_i == 0:
                ax.set_ylabel(TASK_LABEL.get(task, task),
                              fontsize=10, fontweight="semibold")
            if row_i == 0:
                ax.set_title(f"Layer {L}", fontsize=11, color=PALETTE["ink"])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(PALETTE["slate"])
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7, loc="best", frameon=False)
    fig.suptitle(f"Class-separation evolution across transformer depth — "
                 f"modality {modality}  ·  {last_method} projection",
                 fontsize=12, color=PALETTE["ink"], fontweight="semibold",
                 y=1.01)
    fig.tight_layout()
    p = out_dir / f"tsne_evolution_{safe}.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


# ── 2. Modality grid at last layer ───────────────────────────────────────────


def fig_modality_projection_grid(cache_dir: Path, out_dir: Path,
                                 task: str = "compactness_trend") -> Path | None:
    """At the last probed layer, show 2D projection per modality. Side-by-side
    comparison of how clean each modality's class clusters are."""
    from sklearn.preprocessing import LabelEncoder
    cache_dir = Path(cache_dir)
    # Find all modality caches for this task
    cached: list[tuple[str, Path]] = []
    for npz in sorted(cache_dir.glob("hidden_*.npz")):
        parsed = _parse_task_modality(npz)
        if parsed and parsed[0] == task:
            cached.append((parsed[1], npz))
    # Order modalities
    order = ["d", "p", "pi", "n", "ne", "v", "d+v", "pi+v", "ne+v",
             "va", "sp", "v2", "vh", "m", "o", "b",
             "p+va", "p+sp", "n+va", "n+sp", "ne+va", "ne+sp"]
    cached.sort(key=lambda kv: order.index(kv[0]) if kv[0] in order else 999)
    n = len(cached)
    if n == 0:
        return None
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.6 * rows),
                             squeeze=False)
    method_seen = "PCA"
    for idx, (mod, path) in enumerate(cached):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        data = np.load(path, allow_pickle=True)
        labels = np.array([str(l) for l in data["labels"]])
        classes = list(dict.fromkeys(labels))
        le = LabelEncoder(); y = le.fit_transform(labels)
        L = int(data["layers_probed"][-1])
        X = data["hidden"][:, L, :]
        emb, method = _reduce_2d(X, y, method="umap")
        method_seen = method
        for ci, cls in enumerate(classes):
            mask = labels == cls
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       color=CLASS_CMAP[ci % len(CLASS_CMAP)],
                       s=22, alpha=0.85, edgecolors="white", linewidths=0.5,
                       label=cls if idx == 0 else None)
        ax.set_title(mod, fontsize=10, color=PALETTE["ink"])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(PALETTE["slate"])
    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)
    # legend at the top-right cell that has data
    if n > 0:
        axes[0, 0].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(f"Modality class-separation comparison — {TASK_LABEL.get(task, task)}"
                 f"  ·  last transformer layer  ·  {method_seen} projection",
                 fontsize=12, color=PALETTE["ink"], fontweight="semibold",
                 y=1.01)
    fig.tight_layout()
    p = out_dir / f"modality_grid_{task}.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


# ── 3. Multi-task shared embedding ──────────────────────────────────────────


def fig_multitask_superimposed(cache_dir: Path, out_dir: Path,
                               modality: str = "p") -> Path | None:
    """All 4 tasks' last-layer hidden states projected into a shared 2D space.
    Each task coloured separately. Visualises the cross-task orthogonality
    finding — if tasks use disjoint subspaces they cluster into separate
    regions."""
    cache_dir = Path(cache_dir)
    safe = modality.replace("+", "_")
    tasks = ["pressing_type", "compactness_trend", "possession_phase", "territorial_dominance"]
    Xs, task_labels, sample_labels = [], [], []
    for t in tasks:
        p = cache_dir / f"hidden_{t}_{safe}.npz"
        if not p.exists():
            continue
        data = np.load(p, allow_pickle=True)
        L = int(data["layers_probed"][-1])
        X = data["hidden"][:, L, :]
        Xs.append(X)
        task_labels.extend([t] * X.shape[0])
        sample_labels.extend([str(l) for l in data["labels"]])
    if not Xs:
        return None
    X_all = np.vstack(Xs)
    emb, method = _reduce_2d(X_all, np.array(task_labels), method="umap")

    fig, ax = plt.subplots(figsize=(9, 7))
    for t in tasks:
        mask = np.array(task_labels) == t
        if mask.sum() == 0:
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   color=TASK_CMAP[t], s=28, alpha=0.75,
                   edgecolors="white", linewidths=0.5,
                   label=TASK_LABEL.get(t, t))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(PALETTE["slate"])
    ax.legend(fontsize=10, loc="best", frameon=False)
    ax.set_title(f"Multi-task superimposed embedding  ·  modality {modality}  ·  {method} projection"
                 "\ndisjoint task regions visualise the cross-task orthogonality finding",
                 fontsize=11, color=PALETTE["ink"])
    fig.tight_layout()
    p = out_dir / f"multitask_embed_{safe}.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


# ── 4. Probe-decision boundary ───────────────────────────────────────────────


def fig_decision_boundary(cache_dir: Path, out_dir: Path,
                          task: str = "compactness_trend",
                          modality: str = "pi") -> Path | None:
    """1-D histogram per class of hidden-state projections onto the linear
    probe's weight direction. Shows the hyperplane the probe learned."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    safe = modality.replace("+", "_")
    p = Path(cache_dir) / f"hidden_{task}_{safe}.npz"
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    L = int(data["layers_probed"][-1])
    X = data["hidden"][:, L, :]
    labels = np.array([str(l) for l in data["labels"]])
    classes = list(dict.fromkeys(labels))
    le = LabelEncoder(); y = le.fit_transform(labels)

    scaler = StandardScaler().fit(X)
    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5,
        max_iter=1000, class_weight="balanced", scoring="f1_macro",
    )
    clf.fit(scaler.transform(X), y)
    coef = clf.coef_ / (scaler.scale_ + 1e-9)

    if coef.shape[0] == 1:
        # Binary task
        direction = coef[0] / (np.linalg.norm(coef[0]) + 1e-9)
        proj = X @ direction
        fig, ax = plt.subplots(figsize=(8.5, 4))
        for ci, cls in enumerate(classes):
            mask = labels == cls
            ax.hist(proj[mask], bins=18, alpha=0.65,
                    color=CLASS_CMAP[ci % len(CLASS_CMAP)],
                    edgecolor="white", linewidth=0.8, label=cls)
        ax.axvline(0, color=PALETTE["ink"], linestyle="--", linewidth=1,
                   label="probe boundary")
        ax.set_xlabel("Projection onto probe weight direction")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=9)
    else:
        # Multiclass — project onto first 2 probe directions
        dirs = coef[:2] / (np.linalg.norm(coef[:2], axis=1, keepdims=True) + 1e-9)
        proj = X @ dirs.T
        fig, ax = plt.subplots(figsize=(7, 6))
        for ci, cls in enumerate(classes):
            mask = labels == cls
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       color=CLASS_CMAP[ci % len(CLASS_CMAP)],
                       s=38, alpha=0.85, edgecolors="white", linewidths=0.5,
                       label=cls)
        ax.set_xlabel(f"Projection onto class-0 direction")
        ax.set_ylabel(f"Projection onto class-1 direction")
        ax.legend(frameon=False, fontsize=9)
    for s in ax.spines.values():
        s.set_color(PALETTE["slate"])
    ax.set_title(f"Probe decision geometry  ·  {TASK_LABEL.get(task, task)}"
                 f"  ·  modality {modality}  ·  layer {L}",
                 fontsize=11, color=PALETTE["ink"])
    fig.tight_layout()
    p = out_dir / f"decision_boundary_{task}_{safe}.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--evolution-modality", default="p")
    ap.add_argument("--modality-grid-task", default="compactness_trend")
    ap.add_argument("--multitask-modality", default="p")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    _apply_style()

    cache = Path(args.cache_dir)
    # 1. t-SNE evolution for a few modalities
    for mod in ["p", "pi", "d", "d+v"]:
        r = fig_tsne_evolution(cache, out, modality=mod)
        if r: logger.info("wrote %s", r)

    # 2. Modality grid per task
    for task in ["pressing_type", "compactness_trend", "possession_phase", "territorial_dominance"]:
        r = fig_modality_projection_grid(cache, out, task=task)
        if r: logger.info("wrote %s", r)

    # 3. Multi-task superimposed
    for mod in ["p", "pi", "d"]:
        r = fig_multitask_superimposed(cache, out, modality=mod)
        if r: logger.info("wrote %s", r)

    # 4. Decision boundary for strongest cells
    for task, mod in [("compactness_trend", "pi"),
                      ("territorial_dominance", "pi"),
                      ("possession_phase", "d+v"),
                      ("pressing_type", "m")]:
        r = fig_decision_boundary(cache, out, task=task, modality=mod)
        if r: logger.info("wrote %s", r)


if __name__ == "__main__":
    main()
