"""Dissertation-grade aesthetic figures for LLM probing study.

Loads the JSON produced by `linear_probing.py` and renders a slate/navy
academic visual set:

  1. Layer × Task F1 heatmap (per modality) — single-glance story of where
     each tactical concept emerges through model depth.
  2. MLP − Linear gap heatmap — shows where non-linear encoding helps.
     Large positive cells = concept present but not linearly readable.
  3. Best-alpha heatmap — which (task, layer) cells needed strong L2.
     A proxy for representation noise / probe difficulty.
  4. Layer-wise F1 curves with 95% CI bands — publication-ready version
     of the existing layer_wise_*.pdf plots.
  5. Hero plate (2×2) — the above four panels combined into one figure
     suitable for a dissertation chapter opener.

Invocation:
  python3 -m backend.evaluation.probing_figures \
    --results eval_output/dissertation/probing/probing_results.json \
    --output  eval_output/dissertation/probing/figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

logger = logging.getLogger(__name__)

# ── Academic slate / navy palette ───────────────────────────────────────────

PALETTE = {
    "d":     "#1e3a8a",  # blue-900  — navy (text modality)
    "v":     "#b45309",  # amber-700 — earth (visual modality)
    "d+v":   "#047857",  # emerald-700 — jade (both)
    "ink":   "#0f172a",  # slate-900 — primary text
    "slate": "#475569",  # slate-600 — secondary text
    "grid":  "#e2e8f0",  # slate-200
    "bg":    "#ffffff",
    "accent":"#be123c",  # rose-700 — highlight
}

MODALITY_LABEL = {
    "d": "text (d)", "v": "visual (v)", "d+v": "both (d+v)",
    "p": "prose (p)", "pi": "prose+index (pi)",
    "n": "narration (n)", "ne": "narration+expl (ne)",
    "nc": "narration-clean (nc)", "nec": "narration-clean-expl (nec)",
    "b": "binary (b)", "m": "minimap (m)", "o": "overlay (o)",
    "va": "visual-annot (va)", "sp": "sparkline (sp)",
    "v2": "visual-2pane (v2)", "vh": "visual-heatmap (vh)",
    "vfix": "visual-fixed (vfix)",
}

TASK_LABEL = {
    "pressing_type": "Pressing Type",
    "compactness_trend": "Compactness Trend",
    "possession_phase": "Possession Phase",
    "territorial_dominance": "Territorial Dominance",
}


def _slate_navy_cmap() -> mcolors.LinearSegmentedColormap:
    """Sequential cmap: slate-50 → navy-900. Reads well in print."""
    return mcolors.LinearSegmentedColormap.from_list(
        "slate_navy",
        ["#f8fafc", "#cbd5e1", "#64748b", "#1e3a8a", "#0f172a"],
    )


def _gap_cmap() -> mcolors.LinearSegmentedColormap:
    """Diverging cmap centered on zero: rose ↔ slate ↔ emerald."""
    return mcolors.LinearSegmentedColormap.from_list(
        "rose_emerald",
        ["#881337", "#fda4af", "#f1f5f9", "#86efac", "#14532d"],
    )


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.edgecolor": PALETTE["slate"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["bg"],
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
    })


# ── Data extraction ─────────────────────────────────────────────────────────


def _extract_layer_grid(
    results: dict,
    modality: str,
    key: str,   # "linear_f1" or "mlp_f1"
) -> tuple[list[str], list[int], np.ndarray]:
    """Return (tasks, layers, grid[task, layer])."""
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    layers: set[int] = set()
    for t in tasks:
        lw = results[t].get(modality, {}).get("layer_wise", {})
        for k in lw:
            try:
                layers.add(int(k))
            except Exception:
                pass
    layers_sorted = sorted(layers)
    grid = np.full((len(tasks), len(layers_sorted)), np.nan)
    for i, t in enumerate(tasks):
        lw = results[t].get(modality, {}).get("layer_wise", {})
        for j, L in enumerate(layers_sorted):
            v = lw.get(str(L), lw.get(L))
            if isinstance(v, dict):
                grid[i, j] = v.get(key) if v.get(key) is not None else np.nan
            elif isinstance(v, (int, float)) and key == "linear_f1":
                grid[i, j] = float(v)
    return tasks, layers_sorted, grid


def _extract_best_alpha_grid(
    results: dict,
    modality: str,
) -> tuple[list[str], list[int], np.ndarray]:
    """Return (tasks, layers, grid) where cell = log10(best alpha) from MLP grid search."""
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    layers: set[int] = set()
    for t in tasks:
        lw = results[t].get(modality, {}).get("layer_wise", {})
        for k in lw:
            try:
                layers.add(int(k))
            except Exception:
                pass
    layers_sorted = sorted(layers)
    grid = np.full((len(tasks), len(layers_sorted)), np.nan)
    for i, t in enumerate(tasks):
        lw = results[t].get(modality, {}).get("layer_wise", {})
        for j, L in enumerate(layers_sorted):
            cell = lw.get(str(L), lw.get(L))
            if isinstance(cell, dict):
                bp = cell.get("mlp_best_params") or cell.get("best_params") or {}
                alpha = bp.get("alpha") if isinstance(bp, dict) else None
                if alpha is not None and alpha > 0:
                    grid[i, j] = float(np.log10(alpha))
    return tasks, layers_sorted, grid


# ── Individual panels ───────────────────────────────────────────────────────


def fig_f1_heatmap(
    results: dict,
    modality: str,
    out_dir: Path,
    probe_type: str = "linear",  # linear | mlp
) -> Path:
    key = f"{probe_type}_f1"
    tasks, layers, grid = _extract_layer_grid(results, modality, key)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    cmap = _slate_navy_cmap()
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks])
    ax.set_xlabel("Transformer layer")
    ax.set_title(f"{probe_type.capitalize()} probe F1 — modality: {MODALITY_LABEL[modality]}",
                 color=PALETTE["ink"])
    ax.grid(False)

    # Annotate cells
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.5 else PALETTE["ink"]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Macro-F1", color=PALETTE["ink"])
    cbar.ax.tick_params(labelsize=8)

    path = out_dir / f"heatmap_{probe_type}_{modality.replace('+','_')}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_mlp_minus_linear_gap(
    results: dict,
    modality: str,
    out_dir: Path,
) -> Path:
    _, _, lin = _extract_layer_grid(results, modality, "linear_f1")
    tasks, layers, mlp = _extract_layer_grid(results, modality, "mlp_f1")
    gap = mlp - lin

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    vmax = float(np.nanmax(np.abs(gap))) if np.isfinite(np.nanmax(np.abs(gap))) else 0.1
    vmax = max(vmax, 0.05)
    im = ax.imshow(gap, aspect="auto", cmap=_gap_cmap(), vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks])
    ax.set_xlabel("Transformer layer")
    ax.set_title(f"MLP − Linear F1 gap — modality: {MODALITY_LABEL[modality]}\n"
                 "positive = non-linear encoding helps  ·  near zero = linearly readable",
                 color=PALETTE["ink"])
    ax.grid(False)

    for i in range(gap.shape[0]):
        for j in range(gap.shape[1]):
            v = gap[i, j]
            if not np.isnan(v):
                color = PALETTE["ink"] if abs(v) < 0.5 * vmax else "white"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("ΔF1 (MLP − Linear)", color=PALETTE["ink"])
    cbar.ax.tick_params(labelsize=8)

    path = out_dir / f"gap_mlp_linear_{modality.replace('+','_')}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_best_alpha(
    results: dict,
    modality: str,
    out_dir: Path,
) -> Path:
    tasks, layers, grid = _extract_best_alpha_grid(results, modality)
    if np.all(np.isnan(grid)):
        logger.info("skipping best-alpha (no mlp best_params found in results)")
        return None  # type: ignore[return-value]

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    cmap = plt.get_cmap("cividis")
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-4, vmax=-1)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks])
    ax.set_xlabel("Transformer layer")
    ax.set_title(f"MLP grid-search best α (log₁₀) — modality: {MODALITY_LABEL[modality]}\n"
                 "higher = more regularisation needed (noisier representation)",
                 color=PALETTE["ink"])
    ax.grid(False)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{10**v:.0e}", ha="center", va="center",
                        color="white" if v > -2.5 else PALETTE["ink"], fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log₁₀ α", color=PALETTE["ink"])
    cbar.ax.tick_params(labelsize=8)

    path = out_dir / f"best_alpha_{modality.replace('+','_')}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_curves_with_bands(
    results: dict,
    out_dir: Path,
    modalities: list[str],
) -> Path:
    """4-panel figure (one per task) with linear (solid) + MLP (dashed) curves
    for each modality, shared axis style. Dissertation-ready replacement for
    the basic layer_wise_* plots."""
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    n = len(tasks)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        for mod in modalities:
            lw = results[task].get(mod, {}).get("layer_wise", {})
            if not lw:
                continue
            layers = sorted(int(k) for k in lw if lw[k] is not None)
            lin_vals = []
            mlp_vals = []
            for L in layers:
                cell = lw.get(str(L), lw.get(L))
                if isinstance(cell, dict):
                    lin_vals.append(cell.get("linear_f1", np.nan))
                    mlp_vals.append(cell.get("mlp_f1", np.nan))
                else:
                    lin_vals.append(cell if cell is not None else np.nan)
                    mlp_vals.append(np.nan)

            c = PALETTE.get(mod, PALETTE["slate"])
            ax.plot(layers, lin_vals, color=c, linewidth=2.0,
                    marker="o", markersize=4.5, label=f"Linear — {mod}")
            if any(v is not None and not np.isnan(v) for v in mlp_vals):
                ax.plot(layers, mlp_vals, color=c, linewidth=1.5,
                        linestyle="--", marker="s", markersize=4,
                        alpha=0.85, label=f"MLP — {mod}")

        ax.set_title(TASK_LABEL.get(task, task))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Transformer layer")
        if idx % 2 == 0:
            ax.set_ylabel("Probe F1 (macro)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.legend(loc="upper left", ncol=2, fontsize=8)

    fig.suptitle("Layer-wise linear vs MLP probe F1 — Qwen2-VL-7B",
                 color=PALETTE["ink"], fontsize=12, fontweight="semibold", y=0.995)
    fig.tight_layout()
    path = out_dir / "layerwise_curves_all_tasks.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_hero_plate(
    results: dict,
    out_dir: Path,
    modality: str = "d",
) -> Path:
    """2×2 hero plate for dissertation chapter opener.
    Panels:
      A. Layer-wise curves (all tasks, single modality, linear solid + MLP dashed)
      B. Linear F1 heatmap (tasks × layers)
      C. MLP − Linear gap heatmap
      D. Best-α heatmap (or peak-layer bar if alpha missing)
    """
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    _, layers_l, lin = _extract_layer_grid(results, modality, "linear_f1")
    _, _, mlp = _extract_layer_grid(results, modality, "mlp_f1")
    _, _, alpha_grid = _extract_best_alpha_grid(results, modality)
    gap = mlp - lin

    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)

    # Panel A — curves
    axA = fig.add_subplot(gs[0, 0])
    for task in tasks:
        lw = results[task].get(modality, {}).get("layer_wise", {})
        if not lw:
            continue
        layers = sorted(int(k) for k in lw if lw[k] is not None)
        lin_vals = [lw[str(L)].get("linear_f1") if isinstance(lw.get(str(L)), dict)
                    else lw.get(str(L)) for L in layers]
        mlp_vals = [lw[str(L)].get("mlp_f1") if isinstance(lw.get(str(L)), dict)
                    else None for L in layers]
        task_color = {
            "pressing_type":        "#1e3a8a",
            "compactness_trend":    "#b45309",
            "possession_phase":     "#047857",
            "territorial_dominance":"#be123c",
        }.get(task, PALETTE["slate"])
        axA.plot(layers, lin_vals, color=task_color, linewidth=2.0,
                 marker="o", markersize=4, label=TASK_LABEL.get(task, task))
        if any(v is not None for v in mlp_vals):
            axA.plot(layers, mlp_vals, color=task_color, linewidth=1.3,
                     linestyle="--", alpha=0.7)
    axA.set_ylim(0, 1)
    axA.set_xlabel("Transformer layer")
    axA.set_ylabel("Probe F1 (macro)")
    axA.set_title(f"A. Layer-wise F1 ({MODALITY_LABEL[modality]})  —  solid: linear · dashed: MLP")
    axA.legend(fontsize=8, loc="lower center", ncol=2)
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

    # Panel B — Linear heatmap
    axB = fig.add_subplot(gs[0, 1])
    im = axB.imshow(lin, aspect="auto", cmap=_slate_navy_cmap(), vmin=0, vmax=1)
    axB.set_xticks(range(len(layers_l))); axB.set_xticklabels(layers_l)
    axB.set_yticks(range(len(tasks)))
    axB.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks])
    axB.set_title("B. Linear probe F1")
    axB.grid(False)
    for i in range(lin.shape[0]):
        for j in range(lin.shape[1]):
            v = lin[i, j]
            if not np.isnan(v):
                axB.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.5 else PALETTE["ink"], fontsize=8)
    fig.colorbar(im, ax=axB, fraction=0.035, pad=0.02)

    # Panel C — gap
    axC = fig.add_subplot(gs[1, 0])
    vmax = float(np.nanmax(np.abs(gap))) if np.isfinite(np.nanmax(np.abs(gap))) else 0.1
    vmax = max(vmax, 0.05)
    im2 = axC.imshow(gap, aspect="auto", cmap=_gap_cmap(), vmin=-vmax, vmax=vmax)
    axC.set_xticks(range(len(layers_l))); axC.set_xticklabels(layers_l)
    axC.set_yticks(range(len(tasks)))
    axC.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks])
    axC.set_title("C. MLP − Linear F1  (positive ⇒ non-linear encoding helps)")
    axC.set_xlabel("Transformer layer")
    axC.grid(False)
    for i in range(gap.shape[0]):
        for j in range(gap.shape[1]):
            v = gap[i, j]
            if not np.isnan(v):
                axC.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         color="white" if abs(v) > 0.6 * vmax else PALETTE["ink"],
                         fontsize=8)
    fig.colorbar(im2, ax=axC, fraction=0.035, pad=0.02)

    # Panel D — best alpha (if present)
    axD = fig.add_subplot(gs[1, 1])
    if not np.all(np.isnan(alpha_grid)):
        im3 = axD.imshow(alpha_grid, aspect="auto", cmap="cividis", vmin=-4, vmax=-1)
        axD.set_xticks(range(len(layers_l))); axD.set_xticklabels(layers_l)
        axD.set_yticks(range(len(tasks)))
        axD.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks])
        axD.set_title("D. MLP grid-search best α (log₁₀)  —  higher = more regularisation")
        axD.set_xlabel("Transformer layer")
        axD.grid(False)
        for i in range(alpha_grid.shape[0]):
            for j in range(alpha_grid.shape[1]):
                v = alpha_grid[i, j]
                if not np.isnan(v):
                    axD.text(j, i, f"{10**v:.0e}", ha="center", va="center",
                             color="white" if v > -2.5 else PALETTE["ink"],
                             fontsize=7)
        fig.colorbar(im3, ax=axD, fraction=0.035, pad=0.02)
    else:
        # Fallback: peak-F1 layer per task bar chart
        peak_layers = []
        peak_f1s = []
        for i, t in enumerate(tasks):
            row = lin[i]
            if np.all(np.isnan(row)):
                peak_layers.append(np.nan); peak_f1s.append(np.nan); continue
            j = int(np.nanargmax(row))
            peak_layers.append(layers_l[j])
            peak_f1s.append(row[j])
        bars = axD.bar(range(len(tasks)), peak_f1s,
                       color=[PALETTE["d"], PALETTE["v"], PALETTE["d+v"], PALETTE["accent"]][: len(tasks)])
        for b, L, f1 in zip(bars, peak_layers, peak_f1s):
            axD.text(b.get_x() + b.get_width()/2, f1 + 0.02,
                     f"L{L}\n{f1:.2f}", ha="center", va="bottom", fontsize=9)
        axD.set_xticks(range(len(tasks)))
        axD.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks], rotation=12)
        axD.set_ylim(0, 1.05)
        axD.set_ylabel("Peak F1")
        axD.set_title("D. Best layer per task")

    fig.suptitle("Qwen2-VL-7B tactical probing  —  layer-wise linear vs MLP",
                 fontsize=13, fontweight="semibold", color=PALETTE["ink"], y=0.995)
    path = out_dir / f"hero_plate_{modality.replace('+','_')}.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


# ── Advanced analyses (consume analyses.json from probing_analyses.py) ──────


def _dict_to_series(d: dict) -> tuple[list[int], list[float]]:
    keys = sorted(int(k) for k in d.keys())
    return keys, [d[str(k)] for k in keys]


def fig_cka_matrix(analyses: dict, task: str, modality: str, out_dir: Path) -> Path | None:
    entry = analyses.get("per_task_modality", {}).get(task, {}).get(modality)
    if not entry or "cka" not in entry:
        return None
    layers = entry.get("cka_layers", list(range(len(entry["cka"]))))
    mat = np.array(entry["cka"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cmap = _slate_navy_cmap()
    im = ax.imshow(mat, aspect="equal", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(0, len(layers), max(1, len(layers) // 10)))
    ax.set_xticklabels([layers[i] for i in ax.get_xticks()])
    ax.set_yticks(range(0, len(layers), max(1, len(layers) // 10)))
    ax.set_yticklabels([layers[i] for i in ax.get_yticks()])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(f"Cross-layer CKA — {TASK_LABEL.get(task, task)} · {MODALITY_LABEL[modality]}")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="Linear CKA")
    path = out_dir / f"cka_{task}_{modality.replace('+','_')}.pdf"
    fig.savefig(path); fig.savefig(path.with_suffix(".png")); plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_separability_panel(analyses: dict, modality: str, out_dir: Path) -> Path | None:
    """3-row panel per task: silhouette, intrinsic dim, anisotropy across layers."""
    tasks = list(analyses.get("per_task_modality", {}).keys())
    if not tasks:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.2), sharex=True)

    for task in tasks:
        entry = analyses["per_task_modality"].get(task, {}).get(modality)
        if not entry:
            continue
        c = {
            "pressing_type":        "#1e3a8a",
            "compactness_trend":    "#b45309",
            "possession_phase":     "#047857",
            "territorial_dominance":"#be123c",
        }.get(task, PALETTE["slate"])
        for ax, key, ylab in (
            (axes[0], "silhouette",    "Silhouette (cosine)"),
            (axes[1], "intrinsic_dim", "95%-variance dims (PCA)"),
            (axes[2], "anisotropy",    "Mean pair cos-sim"),
        ):
            xs, ys = _dict_to_series(entry[key])
            ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=1.8,
                    color=c, label=TASK_LABEL.get(task, task))
            ax.set_ylabel(ylab)

    axes[0].set_title(f"Representation separability across depth — modality: {MODALITY_LABEL[modality]}")
    axes[-1].set_xlabel("Transformer layer")
    axes[0].legend(fontsize=8, loc="best", ncol=2)
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = out_dir / f"separability_{modality.replace('+','_')}.pdf"
    fig.savefig(path); fig.savefig(path.with_suffix(".png")); plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_selectivity(analyses: dict, modality: str, out_dir: Path) -> Path | None:
    """Selectivity = F1(real) − F1(label-shuffle control) per layer. Tasks stacked."""
    tasks = list(analyses.get("per_task_modality", {}).keys())
    if not tasks:
        return None
    n = len(tasks)
    fig, axes = plt.subplots(n, 1, figsize=(8.6, 2.2 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        entry = analyses["per_task_modality"].get(task, {}).get(modality, {})
        sel = entry.get("selectivity", {})
        if not sel:
            continue
        layers = sorted(int(k) for k in sel.keys())
        real = [sel[str(L)]["f1_real"]         for L in layers]
        shuf = [sel[str(L)]["f1_shuffle_mean"] for L in layers]
        selv = [sel[str(L)]["selectivity"]     for L in layers]

        ax.fill_between(layers, shuf, real, alpha=0.15, color=PALETTE["d"],
                        label="F1 uplift over control")
        ax.plot(layers, real, marker="o", markersize=4.5, linewidth=2.0,
                color=PALETTE["d"], label="F1 (real labels)")
        ax.plot(layers, shuf, marker="s", markersize=4, linewidth=1.3,
                color=PALETTE["accent"], linestyle="--",
                label="F1 (shuffled — control)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"{TASK_LABEL.get(task, task)}  "
                     f"(mean selectivity = {np.mean(selv):.2f})")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("Transformer layer")
    fig.suptitle(f"Probe selectivity — control task (Hewitt & Liang 2019) · {MODALITY_LABEL[modality]}",
                 color=PALETTE["ink"], fontsize=12, fontweight="semibold", y=0.995)
    fig.tight_layout()
    path = out_dir / f"selectivity_{modality.replace('+','_')}.pdf"
    fig.savefig(path); fig.savefig(path.with_suffix(".png")); plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_extraction_gap(analyses: dict, modality: str, out_dir: Path) -> Path | None:
    """Linear F1 − zero-shot prompting F1 per layer, stacked tasks."""
    tasks = list(analyses.get("per_task_modality", {}).keys())
    if not tasks:
        return None
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.axhline(0, color=PALETTE["slate"], linewidth=0.8)
    for task in tasks:
        gap = analyses["per_task_modality"].get(task, {}).get(modality, {}).get("extraction_gap", {})
        if not gap:
            continue
        xs = sorted(int(k) for k in gap.keys())
        ys = [gap[str(k)] for k in xs]
        c = {
            "pressing_type":        "#1e3a8a",
            "compactness_trend":    "#b45309",
            "possession_phase":     "#047857",
            "territorial_dominance":"#be123c",
        }.get(task, PALETTE["slate"])
        ax.plot(xs, ys, marker="o", markersize=4.5, linewidth=2.0,
                color=c, label=TASK_LABEL.get(task, task))
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Probe F1  −  Prompting F1")
    ax.set_title(f"Extraction gap across depth — {MODALITY_LABEL[modality]}\n"
                 "positive ⇒ the model encodes the concept but cannot say it",
                 color=PALETTE["ink"])
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = out_dir / f"extraction_gap_{modality.replace('+','_')}.pdf"
    fig.savefig(path); fig.savefig(path.with_suffix(".png")); plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_modality_synergy(analyses: dict, out_dir: Path) -> Path | None:
    """(d+v) − max(d, v) per layer, per task. Positive ⇒ multimodal fusion helps."""
    per_task = analyses.get("per_task", {})
    if not per_task:
        return None
    tasks = list(per_task.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3.6), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        syn = per_task[task].get("modality_synergy", {}).get("per_layer", {})
        if not syn:
            continue
        layers = sorted(int(k) for k in syn.keys())
        ax.axhline(0, color=PALETTE["slate"], linewidth=0.8)
        d  = [syn[str(L)]["linear_d"]   for L in layers]
        v  = [syn[str(L)]["linear_v"]   for L in layers]
        dv = [syn[str(L)]["linear_d+v"] for L in layers]
        sy = [syn[str(L)]["synergy"]    for L in layers]
        ax.plot(layers, d,  color=PALETTE["d"],   linewidth=1.4, marker="o", markersize=3.5, label="d")
        ax.plot(layers, v,  color=PALETTE["v"],   linewidth=1.4, marker="o", markersize=3.5, label="v")
        ax.plot(layers, dv, color=PALETTE["d+v"], linewidth=1.6, marker="o", markersize=4.0, label="d+v")
        ax.fill_between(layers, 0, sy, alpha=0.2, color=PALETTE["accent"],
                        label="synergy\n(d+v − max(d,v))")
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylim(-0.1, 1.0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("F1 (macro)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Modality synergy — where multimodal fusion helps",
                 color=PALETTE["ink"], fontsize=12, fontweight="semibold", y=1.02)
    fig.tight_layout()
    path = out_dir / "modality_synergy.pdf"
    fig.savefig(path); fig.savefig(path.with_suffix(".png")); plt.close(fig)
    logger.info("wrote %s", path)
    return path


def fig_tsne_strip(
    cache_dir: Path,
    task: str,
    modality: str,
    layers: list[int],
    out_dir: Path,
) -> Path | None:
    """Layer-evolution scatter strip. Uses shrinkage LDA (supervised, max
    inter-class variance, matches the dissertation's §4.5 fig06 projection)
    with a PCA-50 pre-reduction per Schumacher-style probing conventions.
    Robust to TSNE's MacOS sklearn segfault.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError:
        return None
    safe_mod = modality.replace("+", "_")
    npz_path = cache_dir / f"hidden_{task}_{safe_mod}.npz"
    if not npz_path.exists():
        logger.info("tsne strip: cache missing %s", npz_path)
        return None
    data = np.load(npz_path, allow_pickle=True)
    hidden = data["hidden"]
    labels = np.array(data["labels"])
    classes = list(dict.fromkeys(labels))
    if len(classes) < 2:
        return None
    le = LabelEncoder()
    y = le.fit_transform(labels)
    colors = {
        c: ("#1e3a8a", "#b45309", "#047857", "#be123c")[i % 4]
        for i, c in enumerate(classes)
    }

    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 3.4),
                             sharey=False)
    if len(layers) == 1:
        axes = [axes]
    for ax, L in zip(axes, layers):
        x = hidden[:, L, :]
        if x.shape[0] < 5:
            ax.set_visible(False); continue
        # Standardise + PCA-50 guard against degenerate directions.
        xs = StandardScaler(with_std=True).fit_transform(x)
        n_pca = min(50, xs.shape[0] - 1, xs.shape[1])
        xp = PCA(n_components=n_pca, random_state=42).fit_transform(xs)
        n_comp = min(2, len(classes) - 1) or 1
        try:
            lda = LinearDiscriminantAnalysis(
                n_components=n_comp, solver="lsqr", shrinkage="auto",
            )
            emb = lda.fit(xp, y).transform(xp)
            if emb.shape[1] == 1:
                emb = np.hstack([emb, np.zeros_like(emb)])
        except Exception as e:
            logger.warning("LDA failed L=%d: %s — falling back to PCA-2", L, e)
            emb = PCA(n_components=2, random_state=42).fit_transform(xp)
        for c in classes:
            mask = labels == c
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       color=colors[c], s=22, alpha=0.85,
                       edgecolors="white", linewidths=0.5, label=c)
        ax.set_title(f"Layer {L}")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(PALETTE["slate"])
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle(f"Layer evolution (shrinkage-LDA projection) — "
                 f"{TASK_LABEL.get(task, task)} · {MODALITY_LABEL[modality]}",
                 color=PALETTE["ink"], fontsize=11, fontweight="semibold", y=1.04)
    fig.tight_layout()
    path = out_dir / f"lda_strip_{task}_{safe_mod}.pdf"
    fig.savefig(path); fig.savefig(path.with_suffix(".png")); plt.close(fig)
    logger.info("wrote %s", path)
    return path


# ── Rigor-study dissertation figures ─────────────────────────────────────────


def fig_transfer_matrix(rigor_path: Path, out_dir: Path) -> Path | None:
    """4×4 cross-task transfer heatmap. Visually makes the case that tactical
    concepts occupy orthogonal subspaces — near-zero off-diagonal cells."""
    if not rigor_path.exists():
        return None
    d = json.loads(rigor_path.read_text())
    tasks = d.get("tasks", [])
    mat = d.get("matrix") or d.get("cosine_similarity")
    if not tasks or not mat:
        return None
    n = len(tasks)
    M = np.zeros((n, n))
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            v = mat[a].get(b) if isinstance(mat[a], dict) else None
            M[i, j] = v if isinstance(v, (int, float)) else np.nan
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    cmap = _gap_cmap()
    vmax = float(np.nanmax(np.abs(M)))
    im = ax.imshow(M, aspect="equal", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks], rotation=18, fontsize=9)
    ax.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks], fontsize=9)
    ax.set_xlabel("Test on", fontsize=10)
    ax.set_ylabel("Train on", fontsize=10)
    ax.set_title("Cross-task probe transfer  —  diagonal = in-task; off-diagonal ≈ 0 ⇒ orthogonal tactical subspaces",
                 color=PALETTE["ink"], fontsize=10.5)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isnan(v): continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.5 * vmax else PALETTE["ink"],
                    fontsize=10, fontweight="semibold" if i == j else "normal")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    p = out_dir / "transfer_matrix.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_confusion_grid(confusion_path: Path, out_dir: Path,
                       modality: str = "p") -> Path | None:
    """Multi-panel confusion matrices per task at the `modality` (default p).
    Exposes classes that collapse (all-zeros row) — honest reporting."""
    if not confusion_path.exists():
        return None
    d = json.loads(confusion_path.read_text())
    tasks = [t for t in d if modality in d[t]]
    if not tasks:
        return None
    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        entry = d[task][modality]
        classes = entry["classes"]
        cm = np.asarray(entry["confusion"])
        # Row-normalised (per true class)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_n = np.where(row_sum > 0, cm / row_sum, 0)
        im = ax.imshow(cm_n, aspect="equal", cmap=_slate_navy_cmap(), vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=15, fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{TASK_LABEL.get(task, task)}  ({modality})", fontsize=10)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = cm_n[i, j]
                ax.text(j, i, f"{cm[i,j]}\n({v:.0%})", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.5 else PALETTE["ink"])
    fig.suptitle(f"Confusion matrices — modality {modality}, last layer  ·  "
                 f"collapsed class rows flag honest-reporting risk",
                 fontsize=11, color=PALETTE["ink"], y=1.02)
    fig.tight_layout()
    p = out_dir / f"confusion_grid_{modality.replace('+','_')}.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_multiseed_strip(multiseed_path: Path, out_dir: Path) -> Path | None:
    """Strip plot with 5-seed bootstrap CI bars per (task × modality).
    The honest picture of cell stability — single-seed F1s that looked firm
    can reveal wide error bars here."""
    if not multiseed_path.exists():
        return None
    d = json.loads(multiseed_path.read_text())
    tasks = [t for t in d if isinstance(d[t], dict)]
    # Keep a canonical modality ordering
    mods_order = ["d", "v", "d+v", "p", "pi", "n", "ne", "b", "m", "o"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 4.2),
                             sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        entry = d[task]
        mods = [m for m in mods_order if m in entry]
        means = [entry[m]["mean_f1"] for m in mods]
        stds  = [entry[m]["std_f1"]  for m in mods]
        xs = list(range(len(mods)))
        colours = [
            {"d":"#1e3a8a","v":"#b45309","d+v":"#047857","p":"#be123c",
             "pi":"#7c3aed","n":"#0891b2","ne":"#0e7490",
             "b":"#6b7280","m":"#9ca3af","o":"#374151"}.get(m, "#475569")
            for m in mods
        ]
        ax.errorbar(xs, means, yerr=stds, fmt="none",
                    ecolor=PALETTE["slate"], capsize=5, elinewidth=1.2, zorder=2)
        ax.scatter(xs, means, s=120, c=colours, edgecolors=PALETTE["ink"],
                   linewidths=0.8, zorder=3)
        for xi, m, v, s in zip(xs, mods, means, stds):
            ax.text(xi, v + s + 0.02, f"{v:.2f}", ha="center", fontsize=8,
                    color=PALETTE["ink"])
        ax.set_xticks(xs)
        ax.set_xticklabels(mods, rotation=0, fontsize=9)
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Macro-F1 (5-seed mean ± std)")
    fig.suptitle("Probe robustness across 5 random train/test splits",
                 color=PALETTE["ink"], fontsize=12, fontweight="semibold", y=1.02)
    fig.tight_layout()
    p = out_dir / "multiseed_strip.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_perclass_evolution(perclass_path: Path, out_dir: Path,
                           modality: str = "p") -> Path | None:
    """Per-class F1 across layers for a given modality — one subplot per task.
    Reveals which class drives macro-F1 emergence and which collapses."""
    if not perclass_path.exists():
        return None
    d = json.loads(perclass_path.read_text())
    tasks = [t for t in d if modality in d[t]]
    if not tasks:
        return None
    n = len(tasks)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, sharey=True)
    axes = axes.flatten()
    class_palette = ["#1e3a8a", "#b45309", "#047857", "#be123c"]
    for ax, task in zip(axes, tasks):
        entry = d[task][modality]
        classes = entry["classes"]
        per_layer = entry["per_layer"]
        layers = sorted(int(k) for k in per_layer)
        for ci, cls in enumerate(classes):
            ys = [per_layer[str(L)].get(cls, 0) for L in layers]
            ax.plot(layers, ys, marker="o", markersize=4.5, linewidth=1.8,
                    color=class_palette[ci % len(class_palette)],
                    label=cls)
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10.5)
        ax.set_ylim(0, 1); ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Per-class F1")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="best")
    # Hide any unused axes
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Per-class F1 evolution across depth — modality {modality}",
                 color=PALETTE["ink"], fontsize=12, fontweight="semibold", y=1.00)
    fig.tight_layout()
    p = out_dir / f"perclass_evolution_{modality.replace('+','_')}.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_prompting_prescription(results: dict, multiseed_path: Path,
                               heuristic_path: Path, stab_path: Path,
                               out_dir: Path) -> Path | None:
    """Per-task dissertation prescription figure — the single practical
    'change what you feed the LLM' takeaway.

    For each task shows four horizontal bars:
        (1) current baseline (prior)
        (2) digit-space prompting F1 (what the current pipeline uses)
        (3) best probe-readable format F1 (what the model knows)
        (4) recommended practical input (best stratified F1 that's clean)

    Annotates the gap at each step and the recommended modality label.
    """
    if not multiseed_path.exists():
        return None
    ms = json.loads(multiseed_path.read_text())
    heur = json.loads(heuristic_path.read_text()) if heuristic_path.exists() else {}
    stab = json.loads(stab_path.read_text()) if stab_path.exists() else {}
    tasks = [t for t in ms if isinstance(ms[t], dict)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    # Exclude leaky modalities from "recommended" picks (ne has label leak)
    LEAKY = {"ne", "ne+v", "ne+va", "ne+sp"}
    for ax, task in zip(axes, tasks):
        prior = heur.get(task, {}).get("prior", {}).get("f1_macro", 0.0)
        # Current baseline: digit-space prompt F1
        entry_d = results.get(task, {}).get("d", {})
        pr = entry_d.get("prompting", {}) if isinstance(entry_d, dict) else {}
        current = pr.get("f1_macro", 0.0) if isinstance(pr, dict) else 0.0

        # Best probe regardless (shows model ceiling)
        best_mod, best_v = max(ms[task].items(), key=lambda kv: kv[1]["mean_f1"])
        # Best CLEAN probe (excludes leaky narration)
        clean_items = [(m, v) for m, v in ms[task].items() if m not in LEAKY]
        clean_mod, clean_v = max(clean_items, key=lambda kv: kv[1]["mean_f1"])

        stages = [
            ("Prior baseline\n(random)", prior, "#94a3b8", ""),
            (f"Current pipeline\nd prompting", current, "#be123c", "d"),
            (f"Recommended input\nfor probing", clean_v["mean_f1"], "#047857", clean_mod),
            (f"Model's ceiling\nbest probe", best_v["mean_f1"], "#1e3a8a", best_mod),
        ]
        ys = np.arange(len(stages))
        vals = [s[1] for s in stages]
        names = [s[0] for s in stages]
        colours = [s[2] for s in stages]
        ax.barh(ys, vals, color=colours, edgecolor=PALETTE["ink"], linewidth=0.7,
                height=0.7)
        for yi, (label, v, col, mod_name) in enumerate(stages):
            annot = f"{v:.2f}"
            if mod_name:
                annot += f"  [{mod_name}]"
            ax.text(min(v + 0.02, 0.95), yi, annot, va="center",
                    fontsize=10, color=PALETTE["ink"], fontweight="semibold")
        # Gap arrows
        for i in range(1, len(stages)):
            delta = stages[i][1] - stages[i-1][1]
            if abs(delta) < 0.01:
                continue
            y_mid = (ys[i-1] + ys[i]) / 2
            ax.annotate(f"{delta:+.2f}",
                        xy=(max(stages[i-1][1], 0.02), y_mid),
                        ha="left", va="center", fontsize=9,
                        color="#047857" if delta > 0 else "#be123c",
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="#cbd5e1", lw=0.6))
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Macro-F1")
        ax.set_title(TASK_LABEL.get(task, task), fontsize=11,
                     color=PALETTE["ink"], fontweight="semibold")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("Dissertation prescription — how to change LLM input per task for football tactics",
                 fontsize=13, color=PALETTE["ink"], fontweight="semibold", y=0.995)
    fig.tight_layout()
    p = out_dir / "prompting_prescription.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_hidden_state_flow(analyses_path: Path, out_dir: Path,
                          modality: str = "pi") -> Path | None:
    """Line chart of representation geometry across layers per task for a
    given modality. Three panels: silhouette, intrinsic dimensionality,
    anisotropy. Academic line chart style with serif fonts, muted colours
    per task. Shows how the hidden-state geometry evolves with depth."""
    if not analyses_path.exists():
        return None
    a = json.loads(analyses_path.read_text())
    tasks = [t for t in a.get("per_task_modality", {})
             if modality in a["per_task_modality"][t]]
    if not tasks:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    task_colours = {
        "pressing_type":"#1e3a8a","compactness_trend":"#b45309",
        "possession_phase":"#047857","territorial_dominance":"#be123c",
    }
    for panel, (metric, title, ylab) in enumerate([
        ("silhouette",     "Class silhouette (cosine)", "Silhouette score"),
        ("intrinsic_dim",  "Intrinsic dimensionality (PCA-95%)", "Dims"),
        ("anisotropy",     "Representation anisotropy", "Mean cos-sim of random pairs"),
    ]):
        ax = axes[panel]
        for task in tasks:
            entry = a["per_task_modality"][task][modality].get(metric, {})
            if not entry:
                continue
            xs = sorted(int(k) for k in entry.keys())
            ys = [entry[str(k)] for k in xs]
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=2.0,
                    color=task_colours.get(task, PALETTE["slate"]),
                    label=TASK_LABEL.get(task, task))
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5, color=PALETTE["ink"])
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if panel == 0:
            ax.legend(fontsize=9, loc="best", frameon=False)
    fig.suptitle(f"Hidden-state geometry across layers  ·  modality {modality}",
                 fontsize=12, color=PALETTE["ink"], fontweight="semibold", y=1.01)
    fig.tight_layout()
    p = out_dir / f"hidden_state_flow_{modality.replace('+','_')}.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_prompt_stability_boxplot(stab_path: Path, out_dir: Path,
                                  modalities: "list[str] | None" = None) -> Path | None:
    """Dissertation-grade prompt-stability boxplot. Macro-F1 distribution
    across 10 prompt variants per modality, pooled across tasks. Includes
    mean diamonds, a prior-baseline reference line, per-modality Δ annotation
    above each box, and modern slate/sage/rose colour palette.

    Args:
        modalities: Override the default modality set. Pass the 6-modality
            narrative subset (``["d", "v", "d+v"]`` in practice because the
            stability run only covers those three) to match the chapter.
    """
    if not stab_path.exists():
        return None
    d = json.loads(stab_path.read_text())
    tasks = list(d.keys())
    if not tasks:
        return None
    mod_order = modalities or ["d", "v", "d+v", "nec+v", "pt", "sf"]
    dists: dict[str, list[float]] = {m: [] for m in mod_order}
    per_cell_deltas: dict[str, list[float]] = {m: [] for m in mod_order}
    for task in tasks:
        for m in mod_order:
            entry = d[task].get(m, {})
            if not entry:
                continue
            vals = list(entry.get("variants", {}).values())
            dists[m].extend(vals)
            if vals:
                per_cell_deltas[m].append(max(vals) - min(vals))
    mods = [m for m in mod_order if dists[m]]
    data = [dists[m] for m in mods]

    # Muted modern palette — dissertation academic tones
    box_face = {
        "d":     "#94a3b8",  # slate
        "v":     "#e0b793",  # warm sand
        "d+v":   "#a7c4a0",  # sage
        "nec+v": "#b9a5d0",  # muted lavender
        "pt":    "#a2b9d9",  # steel blue
        "sf":    "#d9a5b3",  # dusty rose
    }
    box_edge = {k: _darken(v, 0.45) for k, v in box_face.items()}

    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    bp = ax.boxplot(
        data, tick_labels=mods, patch_artist=True,
        widths=0.55,
        medianprops={"color": PALETTE["ink"], "linewidth": 1.8},
        boxprops={"linewidth": 1.1},
        whiskerprops={"linewidth": 1.0, "color": "#334155"},
        capprops={"linewidth": 1.0, "color": "#334155"},
        flierprops={
            "marker": "o", "markersize": 5.5,
            "markerfacecolor": "white",
            "markeredgecolor": "#334155",
            "markeredgewidth": 0.9,
            "linestyle": "none",
        },
        showmeans=True,
        meanprops={
            "marker": "D", "markersize": 6.5,
            "markerfacecolor": "#f59e0b",
            "markeredgecolor": PALETTE["ink"],
            "markeredgewidth": 0.9,
        },
    )
    for patch, m in zip(bp["boxes"], mods):
        patch.set_facecolor(box_face.get(m, "#e5e7eb"))
        patch.set_edgecolor(box_edge.get(m, "#334155"))
        patch.set_alpha(0.92)

    # Reference line at prior-baseline (≈ 0.30 across tasks). Labelled in subtitle.
    ax.axhline(0.30, color="#6b7280", linewidth=1.1,
               linestyle=(0, (5, 4)), zorder=1)

    # Δ annotation per modality, placed above its whisker
    for i, m in enumerate(mods):
        mean_delta = float(np.mean(per_cell_deltas[m])) if per_cell_deltas[m] else 0.0
        y_top = max(dists[m]) + 0.02
        ax.text(
            i + 1, y_top,
            f"Δ={mean_delta:.2f}",
            ha="center", va="bottom",
            fontsize=9, color=box_edge.get(m, "#334155"),
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec=box_edge.get(m, "#334155"),
                      lw=0.8, alpha=0.95),
        )

    ax.set_ylabel("Macro-F1", fontsize=11.5, color=PALETTE["ink"])
    ax.set_xlabel("Modality", fontsize=11.5, color=PALETTE["ink"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(axis="both", labelsize=10, colors="#4b5563")

    for lbl in ax.get_xticklabels():
        lbl.set_style("italic")
        lbl.set_fontweight("semibold")
        lbl.set_color(PALETTE["ink"])

    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6, color="#9ca3af")
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, min(1.05, max(max(dists[m]) for m in mods) + 0.12))

    total_delta = float(np.mean([np.mean(per_cell_deltas[m])
                                  for m in mods if per_cell_deltas[m]]))
    fig.suptitle(
        "Prompt-wording sensitivity across 10 meaning-preserving variants",
        fontsize=13.5, color=PALETTE["ink"], fontweight="semibold", y=1.00,
    )
    fig.text(0.5, 0.925,
             f"mean Δ (max − min) per cell = {total_delta:.3f}  ·  "
             "◆ mean   ▬ median   ○ outlier   ╌ prior baseline = 0.30  ·  "
             "pooled across 4 tactical tasks",
             ha="center", fontsize=9.8, color="#4b5563", style="italic")

    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    p = out_dir / "prompt_stability_boxplot.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    return p


def _darken(hex_colour: str, factor: float) -> str:
    """Return `hex_colour` darkened by `factor` in [0,1]. 0 = no change, 1 = black."""
    c = hex_colour.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r = int(r * (1 - factor)); g = int(g * (1 - factor)); b = int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def fig_tsne_emergence(cache_dir: Path, out_dir: Path,
                       task: str = "compactness_trend",
                       modality: str = "d",
                       layers: "list[int] | None" = None,
                       panels: "list[tuple[str, int]] | None" = None) -> Path | None:
    """Schumacher Fig. 5 replication: side-by-side t-SNE scatter of hidden
    states coloured by class. Default shows L0 (input embedding) vs. the
    probe's best layer — demonstrates emergence of class structure with depth.

    `panels` lets the caller override with (title, layer) pairs; e.g.
    [("L0 input", 0), ("L4 probe peak", 4)].
    """
    from sklearn.decomposition import PCA
    try:
        import umap  # UMAP is stable on macOS where sklearn TSNE segfaults
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
    try:
        from sklearn.manifold import TSNE
        HAS_TSNE = True
    except ImportError:
        HAS_TSNE = False

    safe_mod = modality.replace("+", "_")
    npz_path = Path(cache_dir) / f"hidden_{task}_{safe_mod}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    hidden = data["hidden"]
    labels = np.array([str(l) for l in data["labels"]])
    if panels is None:
        # Default: L0 vs best probed layer from dense probe if available, else
        # first and last probed layer.
        probed = sorted(int(x) for x in data["layers_probed"])
        panels = [(f"Layer 0  (input embedding)", 0),
                  (f"Layer {probed[-1]}  (final)", probed[-1])]

    classes = sorted(set(labels.tolist()))
    class_colour = {
        c: col for c, col in zip(
            classes,
            ["#1e3a8a", "#b45309", "#047857", "#be123c", "#6b21a8"],
        )
    }

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.4),
                              sharey=False)
    if len(panels) == 1:
        axes = [axes]
    rng = np.random.default_rng(0)

    for ax, (title, L) in zip(axes, panels):
        X = hidden[:, L, :]
        # PCA pre-reduction for dimension stability on Mac (min of samples/dims)
        k = min(50, X.shape[0] - 1, X.shape[1])
        if X.shape[1] > k and k >= 2:
            X_pca = PCA(n_components=k, random_state=0).fit_transform(X)
        else:
            X_pca = X
        emb = None
        if HAS_UMAP:
            try:
                emb = umap.UMAP(
                    n_components=2, n_neighbors=min(15, max(5, X.shape[0]//8)),
                    min_dist=0.15, metric="euclidean", random_state=0,
                ).fit_transform(X_pca)
            except Exception:
                emb = None
        if emb is None:
            emb = PCA(n_components=2, random_state=0).fit_transform(X_pca)

        for c in classes:
            mask = labels == c
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       s=45, color=class_colour[c], alpha=0.85,
                       edgecolor="white", linewidth=0.8,
                       label=c, zorder=3)
        ax.set_title(title, fontsize=11, color=PALETTE["ink"])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_linewidth(0.7); sp.set_color("#9ca3af")

    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=10, title="class",
                    title_fontsize=10)
    fig.suptitle(
        f"Representation emergence across depth — task: {TASK_LABEL.get(task, task)}  "
        f"·  modality: {modality}",
        fontsize=13, color=PALETTE["ink"], fontweight="semibold", y=1.02,
    )
    fig.tight_layout()
    p = out_dir / f"tsne_emergence_{task}_{safe_mod}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    return p


def fig_prompt_stability_table(stab_path: Path, out_dir: Path) -> Path | None:
    """Dissertation-grade prompt-stability table. One grouped block per task
    (task name spans its rows), modality-coloured band, orange Δ column,
    blue ΔP@K column, footer explaining the ΔP@K = 0 finding. Clean serif
    typography, wide first column for full task names."""
    if not stab_path.exists():
        return None
    d = json.loads(stab_path.read_text())
    tasks = list(d.keys())
    # Narrative-relevant modalities only: d (baseline), v / d+v (vision replication
    # of Schumacher Table 3), p (prose), ne (leaky narration for §5.3 retraction).
    # Drops pi and n (not in narrative).
    mod_order = ["d", "v", "d+v", "p", "ne"]
    TASK_PRETTY = {
        "pressing_type": "Pressing type",
        "compactness_trend": "Compactness trend",
        "possession_phase": "Possession phase",
        "territorial_dominance": "Territorial dominance",
    }
    MOD_COLOUR = {
        "d":   "#eef2f7",
        "v":   "#fdf4e7",
        "d+v": "#eef7ef",
        "p":   "#fdf2f5",
        "ne":  "#eff9f2",
    }

    # Build rows, grouped per task (no blank separators — use band shading)
    rows: list[tuple[str, str, list[str], str]] = []
    for task in tasks:
        first = True
        for m in mod_order:
            e = d[task].get(m)
            if not e:
                continue
            task_label = TASK_PRETTY.get(task, task) if first else ""
            first = False
            rows.append((task, task_label, [
                m,
                f"{e['variant_min']:.3f}",
                f"{e['variant_max']:.3f}",
                f"{e['variant_mean']:.3f}",
                f"{e['variant_median']:.3f}",
                f"{e['variant_delta']:.3f}",
                f"{e.get('p_at_1', 0.0):.3f}",
                f"{e.get('p_at_20', 0.0):.3f}",
                f"{e.get('delta_p_at_k', 0.0):+.3f}",
            ], m))

    cols = ["task", "modality", "min", "max", "mean", "median",
            "Δ", "P@1", "P@20", "ΔP@K"]
    # Figure geometry — first column wider for task names
    col_widths = [0.18, 0.09, 0.07, 0.07, 0.07, 0.08, 0.08, 0.07, 0.07, 0.08]

    n_rows = len(rows)
    row_h = 0.36
    fig_h = max(5.2, 1.4 + row_h * (n_rows + 1))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    cell_text = [[lbl, *vals] for (_, lbl, vals, _) in rows]
    table = ax.table(
        cellText=cell_text, colLabels=cols, loc="center",
        cellLoc="center", colLoc="center", colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Header row styling
    for j, _ in enumerate(cols):
        c = table[(0, j)]
        c.set_text_props(fontweight="bold", color="white", fontsize=10.5)
        c.set_facecolor(PALETTE["ink"])
        c.set_edgecolor("#1f2937")
        c.set_linewidth(1.0)
        c.set_height(0.04)

    # Row shading by modality + band separators between tasks
    prev_task = None
    for i, (task, _, _, mod) in enumerate(rows, start=1):
        base = MOD_COLOUR.get(mod, "#ffffff")
        for j in range(len(cols)):
            c = table[(i, j)]
            c.set_facecolor(base)
            c.set_edgecolor("#e5e7eb")
            c.set_linewidth(0.4)
            c.set_height(0.038)
            c.set_text_props(color=PALETTE["ink"], fontsize=9.8)
        # First-column task name in bold, left-aligned
        tc = table[(i, 0)]
        tc.set_text_props(fontweight="bold", color=PALETTE["ink"], fontsize=10.2)
        tc._loc = "left"
        tc.PAD = 0.04
        # Modality cell italic
        mc = table[(i, 1)]
        mc.set_text_props(style="italic", fontweight="semibold",
                          color=PALETTE["ink"], fontsize=10)
        # Δ column: amber
        table[(i, 6)].set_facecolor("#fed7aa")
        table[(i, 6)].set_text_props(color="#7c2d12", fontweight="bold", fontsize=10)
        # ΔP@K column: blue
        table[(i, 9)].set_facecolor("#bfdbfe")
        table[(i, 9)].set_text_props(color="#1e3a8a", fontweight="bold", fontsize=10)
        # Thicker top border on first row of a new task
        if task != prev_task:
            for j in range(len(cols)):
                c = table[(i, j)]
                c.visible_edges = "TBLR"
                c.set_edgecolor("#9ca3af")
        prev_task = task

    # Title + subtitle
    fig.suptitle(
        "Prompt stability across tasks and modalities",
        fontsize=14.5, color=PALETTE["ink"], fontweight="semibold", y=0.98,
    )
    fig.text(
        0.5, 0.945,
        "10 meaning-preserving prompt variants (greedy decode) + 20 sampled "
        "generations per prompt at T=0.7",
        ha="center", fontsize=10, color="#4b5563", style="italic",
    )

    # Colour key for Δ / ΔP@K columns
    fig.text(0.5, 0.065,
             "orange   Δ (max − min)  = sensitivity to prompt wording          "
             "blue   ΔP@K (P@20 − P@1)  = benefit of repeated sampling",
             ha="center", fontsize=10, color="#334155")

    # ΔP@K explanatory footer (so readers don't think the zero column is a bug)
    fig.text(
        0.5, 0.02,
        "Note — ΔP@K = 0.000 across every cell is not a bug.  It is the "
        "expected floor of the unbiased pass@K estimator when every one of the "
        "20 sampled completions returns the same class.  Combined with the "
        "rank-of-class-token finding (§5.2), it means the model's rank-1 "
        "continuation prior out-weighs stochastic sampling at T=0.7 — the model "
        "cannot be pushed off its single prediction by temperature alone.",
        ha="center", va="bottom", fontsize=8.5, color="#4b5563",
        style="italic", wrap=True,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    p = out_dir / "prompt_stability_table.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    return p


def fig_extraction_gap_slope(results: dict, multiseed_path: Path,
                             out_dir: Path) -> Path | None:
    """Slope graph: left column = prompting F1 (d), right column = best
    stratified probe F1, lines connect per task. Classic 'model knows but
    cannot say' visualisation — wider slope = bigger extraction gap.
    """
    if not multiseed_path.exists():
        return None
    ms = json.loads(multiseed_path.read_text())
    tasks = [t for t in ms if isinstance(ms[t], dict)]
    if not tasks:
        return None
    fig, ax = plt.subplots(figsize=(10, 6.8))
    x_prompt, x_probe = 0.0, 1.0
    for task in tasks:
        # Prompting F1 from results (d modality = default prompt)
        entry = results.get(task, {}).get("d", {})
        pr = entry.get("prompting", {}) if isinstance(entry, dict) else {}
        p_val = pr.get("f1_macro") if isinstance(pr, dict) else None
        if p_val is None:
            continue
        # Best stratified probe
        ranked = sorted(ms[task].items(), key=lambda kv: -kv[1]["mean_f1"])
        winner, stats = ranked[0]
        probe_val = stats["mean_f1"]
        colour = TASK_CMAP.get(task, PALETTE["slate"])
        ax.plot([x_prompt, x_probe], [p_val, probe_val],
                color=colour, linewidth=2.6, alpha=0.9,
                marker="o", markersize=11,
                markeredgecolor=PALETTE["ink"], markeredgewidth=1.0)
        ax.text(x_prompt - 0.04, p_val,
                f"{TASK_LABEL.get(task, task)}\nprompt {p_val:.2f}",
                ha="right", va="center", fontsize=9.5,
                color=PALETTE["ink"])
        ax.text(x_probe + 0.04, probe_val,
                f"probe ({winner}) {probe_val:.2f}",
                ha="left", va="center", fontsize=9.5,
                color=colour, fontweight="semibold")
        # Gap annotation
        gap = probe_val - p_val
        ax.annotate(f"+{gap:.2f}", xy=(0.5, (p_val + probe_val) / 2),
                    ha="center", va="center", fontsize=10.5,
                    color=colour, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec=colour, lw=1.0))
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([x_prompt, x_probe])
    ax.set_xticklabels(["Zero-shot prompting F1\n(model speaks)",
                        "Best probe F1 (stratified 10-seed)\n(model knows)"],
                       fontsize=10.5, color=PALETTE["ink"])
    ax.set_ylabel("Macro-F1")
    ax.set_title("The extraction gap — what the model knows (right) "
                 "vs what it can say via prompting (left)",
                 fontsize=12, color=PALETTE["ink"], fontweight="semibold", pad=14)
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = out_dir / "extraction_gap_slope.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_extraction_gap_waterfall(results: dict, multiseed_path: Path,
                                 heuristic_path: Path,
                                 out_dir: Path) -> Path | None:
    """Per-task waterfall: Prior baseline → Prompting F1 → Best text probe
    → Best composite probe. Each step shows the cumulative gain from
    (1) having any data, (2) adding probing, (3) adding composite fusion.
    Dramatic dissertation figure for the extraction-gap section.
    """
    if not multiseed_path.exists():
        return None
    ms = json.loads(multiseed_path.read_text())
    tasks = [t for t in ms if isinstance(ms[t], dict)]
    if not tasks:
        return None
    heuristic = {}
    if heuristic_path.exists():
        heuristic = json.loads(heuristic_path.read_text())
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.2), sharey=True)

    for ax, task in zip(axes, tasks):
        stages = []
        # 1. Prior baseline
        prior_f1 = heuristic.get(task, {}).get("prior", {}).get("f1_macro", 0.0)
        stages.append(("Prior\nbaseline", prior_f1, "#94a3b8"))
        # 2. Prompting F1 (d modality)
        entry = results.get(task, {}).get("d", {})
        pr = entry.get("prompting", {}) if isinstance(entry, dict) else {}
        prompt_f1 = pr.get("f1_macro") if isinstance(pr, dict) else 0.0
        stages.append(("Zero-shot\nprompt", prompt_f1 or 0.0, "#be123c"))
        # 3. Best single-text probe under stratification
        text_mods_only = {m: v for m, v in ms[task].items() if "+" not in m}
        if text_mods_only:
            best_text, best_text_v = max(text_mods_only.items(),
                                         key=lambda kv: kv[1]["mean_f1"])
            stages.append((f"Best\nprobe ({best_text})",
                           best_text_v["mean_f1"], "#1e3a8a"))
        # 4. Best composite probe (single-seed because stratified composites
        #    aren't in ms if we only ran multiseed on the old 9 modalities).
        best_comp_f1 = 0.0
        best_comp_name = None
        for m, entry in results.get(task, {}).items():
            if "+" not in m or not isinstance(entry, dict):
                continue
            probe = entry.get("probe", {})
            f1 = probe.get("f1_macro") if isinstance(probe, dict) else None
            if f1 is not None and f1 > best_comp_f1:
                best_comp_f1 = f1; best_comp_name = m
        if best_comp_name:
            stages.append((f"Best composite\n({best_comp_name})", best_comp_f1, "#047857"))

        xs = np.arange(len(stages))
        vals = [s[1] for s in stages]
        names = [s[0] for s in stages]
        colours = [s[2] for s in stages]
        ax.bar(xs, vals, color=colours, edgecolor=PALETTE["ink"], linewidth=0.7)
        # Annotate deltas on top of each bar
        for i, (name, val, _) in enumerate(stages):
            ax.text(i, val + 0.02, f"{val:.2f}", ha="center",
                    fontsize=9.5, fontweight="semibold", color=PALETTE["ink"])
            if i > 0:
                delta = val - stages[i-1][1]
                y_mid = (val + stages[i-1][1]) / 2
                ax.annotate(f"{delta:+.2f}",
                            xy=(i - 0.5, y_mid),
                            ha="center", va="center", fontsize=9.5,
                            color="#047857" if delta > 0 else "#be123c",
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                      ec="#94a3b8", lw=0.6))
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.set_ylim(0, 1.02)
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10.5,
                     color=PALETTE["ink"])
        if ax is axes[0]:
            ax.set_ylabel("Macro-F1")
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Extraction-gap waterfall  —  Prior → Prompting → Best probe → Best composite",
                 fontsize=12.5, color=PALETTE["ink"], fontweight="semibold", y=1.02)
    fig.tight_layout()
    p = out_dir / "extraction_gap_waterfall.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_layerwise_best_modalities(results: dict, multiseed_path: Path,
                                  out_dir: Path, top_k: int = 4) -> Path | None:
    """Per task: layer-wise F1 line chart of the top-k modalities (ranked by
    stratified 10-seed mean). Clean dissertation-style: serif fonts, muted
    grid, distinguishing line colours and markers per modality."""
    if not multiseed_path.exists():
        return None
    ms = json.loads(multiseed_path.read_text())
    tasks = [t for t in ms if isinstance(ms[t], dict)]
    if not tasks:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)
    axes = axes.flatten()
    colours = ["#047857", "#1e3a8a", "#b45309", "#be123c", "#7c3aed"]
    markers = ["o", "s", "^", "D", "v"]
    for ax, task in zip(axes, tasks):
        # Rank modalities by stratified mean
        ranked = sorted(ms[task].items(), key=lambda kv: -kv[1]["mean_f1"])
        top_mods = [m for m, _ in ranked[:top_k]]
        for ki, mod in enumerate(top_mods):
            lw = results.get(task, {}).get(mod, {}).get("layer_wise", {})
            if not lw:
                continue
            layers = sorted(int(k) for k in lw if lw[k] is not None)
            f1s = [lw[str(L)].get("linear_f1") if isinstance(lw.get(str(L)), dict)
                   else lw.get(str(L)) for L in layers]
            ax.plot(layers, f1s, color=colours[ki % len(colours)],
                    marker=markers[ki % len(markers)], markersize=6,
                    linewidth=2.0, alpha=0.9,
                    label=f"{mod}  (final {ms[task][mod]['mean_f1']:.2f}±{ms[task][mod]['std_f1']:.2f})")
        ax.set_ylim(0, 1.02)
        ax.set_title(TASK_LABEL.get(task, task), fontsize=11.5,
                     color=PALETTE["ink"])
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Probe F1 (macro)")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    fig.suptitle("Layer-wise probe F1 for the top-4 modalities per task "
                 "(ranked by stratified 10-seed)",
                 fontsize=13, color=PALETTE["ink"], fontweight="semibold", y=0.998)
    fig.tight_layout()
    p = out_dir / "layerwise_best_modalities.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_2d_best_cells(cache_dir: Path, multiseed_path: Path,
                      out_dir: Path) -> Path | None:
    """2x2 grid: one task per panel, UMAP 2D projection of the winning
    modality's last-layer hidden states. The cleanest demonstration that
    the best modality actually separates classes."""
    if not multiseed_path.exists():
        return None
    ms = json.loads(multiseed_path.read_text())
    tasks = [t for t in ms if isinstance(ms[t], dict)]
    if not tasks:
        return None
    from sklearn.preprocessing import LabelEncoder
    try:
        import umap
        method = "UMAP"
    except ImportError:
        method = "LDA"

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))
    axes = axes.flatten()
    class_palette = ["#1e3a8a", "#b45309", "#047857", "#be123c"]
    for ax, task in zip(axes, tasks):
        # Winner from stratified
        ranked = sorted(ms[task].items(), key=lambda kv: -kv[1]["mean_f1"])
        if not ranked:
            continue
        winner, stats = ranked[0]
        safe_mod = winner.replace("+", "_")
        npz = Path(cache_dir) / f"hidden_{task}_{safe_mod}.npz"
        if not npz.exists():
            # Fallback to second-best if winner's cache is missing
            for m, s in ranked[1:]:
                safe_mod = m.replace("+", "_")
                npz = Path(cache_dir) / f"hidden_{task}_{safe_mod}.npz"
                if npz.exists():
                    winner = m; stats = s; break
            if not npz.exists():
                continue
        data = np.load(npz, allow_pickle=True)
        L = int(data["layers_probed"][-1])
        X = data["hidden"][:, L, :]
        labels = np.array([str(l) for l in data["labels"]])
        classes = list(dict.fromkeys(labels))
        le = LabelEncoder(); y = le.fit_transform(labels)

        # PCA-50 then UMAP or LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        xs = StandardScaler().fit_transform(X.astype(np.float64))
        xp = PCA(n_components=min(50, xs.shape[0] - 1, xs.shape[1]),
                 random_state=42).fit_transform(xs)
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=min(15, xp.shape[0] // 3),
                                min_dist=0.3, random_state=42, metric="cosine")
            emb = reducer.fit_transform(xp)
            used = "UMAP"
        except Exception:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis(n_components=min(2, len(classes) - 1) or 1,
                                             solver="lsqr", shrinkage="auto")
            emb = lda.fit(xp, y).transform(xp)
            if emb.shape[1] == 1:
                emb = np.hstack([emb, np.zeros_like(emb)])
            used = "LDA"

        for ci, cls in enumerate(classes):
            mask = labels == cls
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       color=class_palette[ci % len(class_palette)],
                       s=58, alpha=0.85, edgecolors="white", linewidths=0.7,
                       label=cls)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{TASK_LABEL.get(task, task)} — "
                     f"winner: {winner} ({stats['mean_f1']:.2f} ± {stats['std_f1']:.2f})",
                     fontsize=11, color=PALETTE["ink"])
        for s in ax.spines.values():
            s.set_color(PALETTE["slate"])
        ax.legend(fontsize=9, loc="best", frameon=False)
    fig.suptitle(f"2D class separation at last layer, best modality per task  ·  {used} projection",
                 fontsize=12.5, color=PALETTE["ink"], fontweight="semibold", y=1.00)
    fig.tight_layout()
    p = out_dir / "best_cells_2d.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_task_winners_podium(multiseed_path: Path, out_dir: Path,
                            top_k: int = 5) -> Path | None:
    """Clean 2x2 dissertation figure: per task, top-k modalities by
    stratified 10-seed mean F1, with CI error bars. Shows which modality
    actually wins each task when honesty is enforced."""
    if not multiseed_path.exists():
        return None
    d = json.loads(multiseed_path.read_text())
    tasks = [t for t in d if isinstance(d[t], dict)]
    if not tasks:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    axes = axes.flatten()
    colours = ["#047857", "#1e3a8a", "#b45309", "#be123c", "#7c3aed"]
    for ax, task in zip(axes, tasks):
        entry = d[task]
        # Sort modalities by mean_f1 desc, take top_k
        ranked = sorted(entry.items(),
                        key=lambda kv: -kv[1]["mean_f1"])[:top_k]
        mods = [m for m, _ in ranked]
        means = [v["mean_f1"] for _, v in ranked]
        stds = [v["std_f1"] for _, v in ranked]
        xs = np.arange(len(mods))
        bars = ax.bar(xs, means, yerr=stds,
                      color=colours[:len(mods)],
                      edgecolor=PALETTE["ink"], linewidth=0.7,
                      capsize=6, error_kw={"elinewidth": 1.3})
        # Gold-medal crown on the winner
        ax.text(xs[0], means[0] + stds[0] + 0.055, "★",
                ha="center", fontsize=18, color="#facc15")
        for xi, m, v, s in zip(xs, mods, means, stds):
            ax.text(xi, v + s + 0.015, f"{v:.2f}", ha="center", fontsize=9.5,
                    color=PALETTE["ink"], fontweight="semibold")
        ax.set_xticks(xs)
        ax.set_xticklabels(mods, fontsize=11, fontweight="semibold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro-F1 (10-seed stratified mean ± std)")
        ax.set_title(TASK_LABEL.get(task, task), fontsize=11.5, color=PALETTE["ink"])
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Best probe modalities per task — stratified 10-seed ranking",
                 fontsize=13, color=PALETTE["ink"], fontweight="semibold", y=0.995)
    fig.tight_layout()
    p = out_dir / "task_winners_podium.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_improvement_over_d(multiseed_path: Path, out_dir: Path) -> Path | None:
    """Per task: bar chart showing Δ F1 of each modality over the digit-space
    `d` baseline. Positive = better than d; the story of how much each
    alternative serialisation actually beats the Gruver default."""
    if not multiseed_path.exists():
        return None
    d = json.loads(multiseed_path.read_text())
    tasks = [t for t in d if isinstance(d[t], dict)]
    if not tasks:
        return None
    fig, ax = plt.subplots(figsize=(13, 5.6))
    mods_order = ["pi", "p", "n", "v", "d+v", "m", "ne", "va", "sp", "vh", "o", "b"]
    present = sorted({m for t in tasks for m in d[t]
                      if m in mods_order and m != "d"},
                     key=lambda m: mods_order.index(m))
    xs = np.arange(len(present))
    w = 0.19
    for ti, task in enumerate(tasks):
        d_baseline = d[task].get("d", {}).get("mean_f1", 0.0)
        deltas = []
        for m in present:
            if m in d[task]:
                deltas.append(d[task][m]["mean_f1"] - d_baseline)
            else:
                deltas.append(np.nan)
        offset = (ti - len(tasks) / 2 + 0.5) * w
        mask = ~np.isnan(deltas)
        ax.bar(xs[mask] + offset, np.array(deltas)[mask], w,
               color=TASK_CMAP.get(task, PALETTE["slate"]),
               edgecolor=PALETTE["ink"], linewidth=0.5,
               label=TASK_LABEL.get(task, task))
    ax.axhline(0, color=PALETTE["ink"], linewidth=1)
    ax.set_xticks(xs); ax.set_xticklabels(present, fontsize=10)
    ax.set_xlabel("Modality")
    ax.set_ylabel("Δ F1 vs digit-space (d) baseline")
    ax.set_title("How much does each modality beat digit-space? "
                 "(stratified 10-seed mean)  —  positive = better than d",
                 fontsize=11.5, color=PALETTE["ink"])
    ax.legend(fontsize=9, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = out_dir / "improvement_over_d.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_best_impacts_showcase(multiseed_path: Path, results: dict,
                              out_dir: Path) -> Path | None:
    """2x2 dissertation hero plate combining the four biggest §4.8 findings:
    (A) binning collapses every task, (B) extraction gap everywhere,
    (C) stratified vs single-seed correction, (D) ne+visual rescue on
    possession_phase. One figure that tells the whole story."""
    if not multiseed_path.exists():
        return None
    ms = json.loads(multiseed_path.read_text())
    tasks = [t for t in ms if isinstance(ms[t], dict)]
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.24)

    # A. Binning collapse — 'b' vs task-winner per task
    axA = fig.add_subplot(gs[0, 0])
    winners = {}
    for task in tasks:
        ranked = sorted(ms[task].items(), key=lambda kv: -kv[1]["mean_f1"])
        winners[task] = ranked[0] if ranked else ("?", {"mean_f1": 0, "std_f1": 0})
    b_vals = [ms[t].get("b", {}).get("mean_f1", 0) for t in tasks]
    b_stds = [ms[t].get("b", {}).get("std_f1", 0) for t in tasks]
    w_vals = [v[1]["mean_f1"] for v in winners.values()]
    w_stds = [v[1]["std_f1"] for v in winners.values()]
    w_names = [f"{v[0]}" for v in winners.values()]
    xs = np.arange(len(tasks))
    axA.bar(xs - 0.2, b_vals, 0.4, yerr=b_stds, capsize=4,
            color="#94a3b8", label="b (binned)",
            edgecolor=PALETTE["ink"], linewidth=0.7)
    axA.bar(xs + 0.2, w_vals, 0.4, yerr=w_stds, capsize=4,
            color="#047857", label="Task winner",
            edgecolor=PALETTE["ink"], linewidth=0.7)
    for xi, wn, wv in zip(xs, w_names, w_vals):
        axA.text(xi + 0.2, wv + 0.03, wn, ha="center", fontsize=8.5,
                 color=PALETTE["ink"], fontweight="semibold")
    axA.set_xticks(xs)
    axA.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks], fontsize=8.5, rotation=12)
    axA.set_ylim(0, 1)
    axA.set_ylabel("Macro-F1")
    axA.set_title("A. Binning (b) collapses to chance —  magnitude is causally essential",
                  fontsize=10.5, color=PALETTE["ink"])
    axA.legend(fontsize=9, loc="upper right", frameon=False)
    axA.grid(True, alpha=0.25, axis="y")
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)

    # B. Probe vs Prompt per task
    axB = fig.add_subplot(gs[0, 1])
    probe_vals = [winners[t][1]["mean_f1"] for t in tasks]
    prompt_vals = []
    for task in tasks:
        entry = results.get(task, {}).get("d", {})
        pr = entry.get("prompting", {}) if isinstance(entry, dict) else {}
        prompt_vals.append(pr.get("f1_macro", 0.0) if isinstance(pr, dict) else 0.0)
    axB.bar(xs - 0.2, prompt_vals, 0.4, color="#be123c",
            edgecolor=PALETTE["ink"], linewidth=0.7, label="Prompt F1 (d)")
    axB.bar(xs + 0.2, probe_vals, 0.4, color="#1e3a8a",
            edgecolor=PALETTE["ink"], linewidth=0.7, label="Best probe F1")
    for xi, pv, qv in zip(xs, prompt_vals, probe_vals):
        gap = qv - pv
        axB.annotate(f"+{gap:.2f}", xy=(xi, qv + 0.03), ha="center",
                     fontsize=9, color="#047857", fontweight="semibold")
    axB.set_xticks(xs)
    axB.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks], fontsize=8.5, rotation=12)
    axB.set_ylim(0, 1)
    axB.set_ylabel("Macro-F1")
    axB.set_title("B. Extraction gap: probe knows, prompt doesn't say",
                  fontsize=10.5, color=PALETTE["ink"])
    axB.legend(fontsize=9, loc="upper right", frameon=False)
    axB.grid(True, alpha=0.25, axis="y")
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # C. Single-seed vs stratified correction (use single-seed from results)
    axC = fig.add_subplot(gs[1, 0])
    mods_c = ["d", "p", "pi", "d+v"]
    single_vals = np.full((len(tasks), len(mods_c)), np.nan)
    strat_vals = np.full_like(single_vals, np.nan)
    for i, task in enumerate(tasks):
        for j, m in enumerate(mods_c):
            s = results.get(task, {}).get(m, {}).get("probe", {})
            if isinstance(s, dict) and "f1_macro" in s:
                single_vals[i, j] = s["f1_macro"]
            if m in ms.get(task, {}):
                strat_vals[i, j] = ms[task][m]["mean_f1"]
    deltas = strat_vals - single_vals
    im = axC.imshow(deltas, aspect="auto", cmap=_gap_cmap(),
                    vmin=-0.4, vmax=0.4)
    axC.set_xticks(range(len(mods_c)))
    axC.set_xticklabels(mods_c, fontsize=10)
    axC.set_yticks(range(len(tasks)))
    axC.set_yticklabels([TASK_LABEL.get(t, t) for t in tasks], fontsize=9)
    for i in range(len(tasks)):
        for j in range(len(mods_c)):
            v = deltas[i, j]
            if not np.isnan(v):
                axC.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=9, color="white" if abs(v) > 0.2 else PALETTE["ink"])
    axC.set_title("C. Stratified − single-seed correction per cell",
                  fontsize=10.5, color=PALETTE["ink"])
    axC.grid(False)
    fig.colorbar(im, ax=axC, fraction=0.04, pad=0.03)

    # D. ne+visual rescue on possession
    axD = fig.add_subplot(gs[1, 1])
    # Use single-seed composite results for this (we have them)
    labels_ = ["ne (alone)", "ne+v", "ne+va", "ne+sp", "p+v", "p+va"]
    vals = []
    for m in ["ne", "ne+v", "ne+va", "ne+sp", "p+v", "p+va"]:
        entry = results.get("possession_phase", {}).get(m, {})
        probe = entry.get("probe", {}) if isinstance(entry, dict) else {}
        vals.append(probe.get("f1_macro", 0.0) if isinstance(probe, dict) else 0.0)
    colours = ["#94a3b8"] + ["#047857"] * 3 + ["#be123c"] * 2
    bars = axD.bar(range(len(labels_)), vals, color=colours,
                   edgecolor=PALETTE["ink"], linewidth=0.7)
    for xi, v in enumerate(vals):
        axD.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9,
                 color=PALETTE["ink"], fontweight="semibold")
    axD.set_xticks(range(len(labels_)))
    axD.set_xticklabels(labels_, fontsize=9, rotation=10)
    axD.set_ylim(0, 1.05)
    axD.set_ylabel("Macro-F1")
    axD.set_title("D. Possession rhythm: narration+visual synergy "
                  "(green) vs digit+visual collapse (red)",
                  fontsize=10.5, color=PALETTE["ink"])
    axD.grid(True, alpha=0.25, axis="y")
    axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)

    fig.suptitle("Four biggest §4.8 findings in one figure",
                 fontsize=13, color=PALETTE["ink"], fontweight="semibold", y=1.00)
    fig.tight_layout()
    p = out_dir / "best_impacts_showcase.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_text_visual_pairing(results: dict, out_dir: Path) -> Path | None:
    """Per task: heatmap where rows = text modality, cols = visual modality,
    cells = composite F1. Shows which (text, visual) pairings build synergy
    and which destroy the representation."""
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    if not tasks:
        return None
    text_mods = ["d", "p", "pi", "n", "ne"]
    visual_mods = ["v", "va", "sp", "v2", "vh"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    axes = axes.flatten()
    cmap = _gap_cmap()
    for ax, task in zip(axes, tasks):
        M = np.full((len(text_mods), len(visual_mods)), np.nan)
        for i, tm in enumerate(text_mods):
            for j, vm in enumerate(visual_mods):
                key = f"{tm}+{vm}"
                entry = results[task].get(key, {})
                probe = entry.get("probe", {}) if isinstance(entry, dict) else {}
                f1 = probe.get("f1_macro") if isinstance(probe, dict) else None
                # Fusion delta: composite − max(text_alone, visual_alone)
                t_entry = results[task].get(tm, {}).get("probe", {})
                v_entry = results[task].get(vm, {}).get("probe", {})
                t_f1 = t_entry.get("f1_macro") if isinstance(t_entry, dict) else None
                v_f1 = v_entry.get("f1_macro") if isinstance(v_entry, dict) else None
                if f1 is not None and t_f1 is not None and v_f1 is not None:
                    M[i, j] = f1 - max(t_f1, v_f1)
        vmax = max(0.15, float(np.nanmax(np.abs(M))) if not np.all(np.isnan(M)) else 0.15)
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(visual_mods)))
        ax.set_xticklabels(visual_mods, fontsize=9)
        ax.set_yticks(range(len(text_mods)))
        ax.set_yticklabels(text_mods, fontsize=9)
        ax.set_xlabel("Visual modality")
        ax.set_ylabel("Text modality")
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10.5)
        ax.grid(False)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color=PALETTE["slate"])
                else:
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if abs(v) > 0.6 * vmax else PALETTE["ink"])
    fig.suptitle("Fusion Δ F1 = (text+visual) − max(text alone, visual alone)  —  "
                 "green cells = synergy; red cells = fusion destroys signal",
                 fontsize=11.5, color=PALETTE["ink"], fontweight="semibold", y=0.995)
    fig.tight_layout()
    p = out_dir / "text_visual_pairing.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_ne_rescue(results: dict, out_dir: Path) -> Path | None:
    """Bar chart showing ne alone vs ne+v, ne+va, ne+sp per task.
    Visualises the "tactical narration + any visual" rescue effect."""
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    if not tasks:
        return None
    mod_order = ["ne", "ne+v", "ne+va", "ne+sp"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    xs = np.arange(len(tasks))
    w = 0.19
    colours = ["#94a3b8", "#1e3a8a", "#0e7490", "#be185d"]
    for ki, mod in enumerate(mod_order):
        vals = []
        for task in tasks:
            entry = results[task].get(mod, {}).get("probe", {})
            vals.append(entry.get("f1_macro", 0.0) if isinstance(entry, dict) else 0.0)
        ax.bar(xs + (ki - 1.5) * w, vals, w, color=colours[ki],
               edgecolor=PALETTE["ink"], linewidth=0.6, label=mod)
        for xi, v in zip(xs, vals):
            if v > 0.01:
                ax.text(xi + (ki - 1.5) * w, v + 0.01, f"{v:.2f}",
                        ha="center", fontsize=7.5, color=PALETTE["ink"])
    ax.set_xticks(xs)
    ax.set_xticklabels([TASK_LABEL.get(t, t) for t in tasks], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probe F1 (macro)")
    ax.set_title("Extended narration rescue: ne alone vs ne + visual composite",
                 fontsize=11, color=PALETTE["ink"])
    ax.legend(fontsize=9, ncol=4, loc="upper center", frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = out_dir / "ne_visual_rescue.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_fusion_helps_vs_hurts(results: dict, out_dir: Path) -> Path | None:
    """Split visualisation: for each task, show every composite as a point
    on the plane (single-modality-best, composite-F1). Points above y = x
    are synergies, below are destructions. Colour = text base."""
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    if not tasks:
        return None
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.8 * len(tasks), 4.2),
                             sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    text_colour = {"d": "#be123c", "p": "#1e3a8a", "pi": "#0e7490",
                   "n": "#b45309", "ne": "#047857"}
    for ax, task in zip(axes, tasks):
        ax.plot([0, 1], [0, 1], color=PALETTE["ink"], linewidth=1,
                linestyle="--", alpha=0.6)
        for mod, entry in results[task].items():
            if "+" not in mod or not isinstance(entry, dict):
                continue
            text_mod, visual_mod = mod.split("+", 1)
            composite = entry.get("probe", {}).get("f1_macro") if isinstance(entry.get("probe"), dict) else None
            t_alone = results[task].get(text_mod, {}).get("probe", {})
            v_alone = results[task].get(visual_mod, {}).get("probe", {})
            t_f1 = t_alone.get("f1_macro") if isinstance(t_alone, dict) else None
            v_f1 = v_alone.get("f1_macro") if isinstance(v_alone, dict) else None
            if None in (composite, t_f1, v_f1):
                continue
            best = max(t_f1, v_f1)
            ax.scatter(best, composite, s=80,
                       color=text_colour.get(text_mod, PALETTE["slate"]),
                       alpha=0.85, edgecolors=PALETTE["ink"], linewidths=0.6)
            ax.annotate(f"{text_mod}+{visual_mod}", xy=(best, composite),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=PALETTE["ink"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("max(text, visual) alone")
        if ax is axes[0]:
            ax.set_ylabel("Composite F1")
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Composite vs best-single F1  —  above dashed line = synergy, below = destruction",
                 fontsize=11, color=PALETTE["ink"], y=1.01)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=9, label=m)
               for m, c in text_colour.items()]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    p = out_dir / "fusion_helps_vs_hurts.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_single_vs_stratified(single_results: dict, multiseed_path: Path,
                             out_dir: Path) -> Path | None:
    """Per-task bar chart showing single-seed headline F1 vs stratified
    10-seed mean ± std for each modality. Visualises the rigor correction
    — where single-seed was misleading, and where it was accurate.
    """
    if not multiseed_path.exists():
        return None
    ms = json.loads(multiseed_path.read_text())
    tasks = [t for t in ms if isinstance(ms[t], dict)]
    if not tasks:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    axes = axes.flatten()
    mods_order = ["d", "v", "d+v", "p", "pi", "n", "m", "o", "b"]
    for ax, task in zip(axes, tasks):
        ms_task = ms[task]
        mods = [m for m in mods_order if m in ms_task]
        single_vals = []
        strat_means = []
        strat_stds = []
        for m in mods:
            # Single-seed from results JSON
            s = single_results.get(task, {}).get(m, {})
            probe = s.get("probe", {}) if isinstance(s, dict) else {}
            s_f1 = probe.get("f1_macro") if isinstance(probe, dict) else None
            single_vals.append(s_f1 if s_f1 is not None else 0.0)
            strat_means.append(ms_task[m]["mean_f1"])
            strat_stds.append(ms_task[m]["std_f1"])
        xs = np.arange(len(mods))
        w = 0.38
        ax.bar(xs - w/2, single_vals, w, color="#cbd5e1",
               edgecolor=PALETTE["ink"], linewidth=0.6, label="Single-seed (head-tail split)")
        ax.bar(xs + w/2, strat_means, w, yerr=strat_stds,
               color="#1e3a8a", edgecolor=PALETTE["ink"], linewidth=0.6,
               capsize=4, error_kw={"elinewidth": 1.1}, label="Stratified 10-seed (mean ± std)")
        ax.set_xticks(xs)
        ax.set_xticklabels(mods, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(TASK_LABEL.get(task, task), fontsize=10.5)
        ax.set_ylabel("Macro-F1")
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.suptitle("Single-seed vs stratified 10-seed probe F1 — rigor correction",
                 fontsize=13, color=PALETTE["ink"], fontweight="semibold", y=0.995)
    fig.tight_layout()
    p = out_dir / "single_vs_stratified.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_mdl_compression(mdl_path: Path, out_dir: Path) -> Path | None:
    """Grouped bar chart of MDL compression ratio per (task × modality).
    Alternative to F1 — answers 'how much does the representation compress
    the labels' rather than 'does the probe classify well'.
    Positive bars = representation contains information; negative = uniform
    code beats the online probe (small-sample artefact for some tasks).
    """
    if not mdl_path.exists():
        return None
    d = json.loads(mdl_path.read_text())
    tasks = [t for t in d if isinstance(d[t], dict)]
    if not tasks:
        return None
    mods_order = ["d", "v", "d+v", "p", "pi", "n", "m", "o", "b"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.09
    for ti, task in enumerate(tasks):
        mods = [m for m in mods_order if m in d[task]]
        vals = [d[task][m]["compression_ratio"] for m in mods]
        xs = np.arange(len(mods))
        offset = (ti - len(tasks) / 2 + 0.5) * w
        ax.bar(xs + offset, vals, w, color=TASK_CMAP.get(task, PALETTE["slate"]),
               edgecolor="white", linewidth=0.4,
               label=TASK_LABEL.get(task, task))
    ax.axhline(0, color=PALETTE["ink"], linewidth=0.8)
    ax.set_xticks(np.arange(len(mods_order)))
    ax.set_xticklabels(mods_order, fontsize=10)
    ax.set_xlabel("Modality")
    ax.set_ylabel("MDL compression ratio (1 − online / uniform)")
    ax.set_title("Information-theoretic representation quality (Voita & Titov MDL)",
                 fontsize=11, color=PALETTE["ink"])
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, ncol=4, loc="best", frameon=False)
    fig.tight_layout()
    p = out_dir / "mdl_compression.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


# Task colour palette for the multi-task plots (matches radar + emergence).
TASK_CMAP = {
    "pressing_type":        "#1e3a8a",
    "compactness_trend":    "#b45309",
    "possession_phase":     "#047857",
    "territorial_dominance":"#be123c",
}


def fig_probe_vs_prompt_scatter(results: dict, out_dir: Path) -> Path | None:
    """Scatter plot: x = prompting F1, y = probe F1. One dot per (task,
    modality). Points above y=x line = extraction-gap cells. Points in the
    top-left = strong probe but weak prompt — the chapter's central
    'model-knows-but-can't-say-it' finding, visualised at a glance."""
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    xs, ys, cs, labels_ = [], [], [], []
    for task in tasks:
        for mod, entry in results[task].items():
            if not isinstance(entry, dict):
                continue
            probe = entry.get("probe", {}).get("f1_macro") if isinstance(entry.get("probe"), dict) else None
            prompt = entry.get("prompting", {}).get("f1_macro") if isinstance(entry.get("prompting"), dict) else None
            if probe is None or prompt is None:
                continue
            xs.append(prompt); ys.append(probe)
            cs.append(TASK_CMAP.get(task, PALETTE["slate"]))
            labels_.append(f"{task[:6]}·{mod}")
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color=PALETTE["ink"], linewidth=1, linestyle="--",
            alpha=0.6, label="probe = prompt")
    ax.fill_between([0, 1], [0, 1], 1, color="#bfdbfe", alpha=0.18,
                    label="extraction gap region")
    ax.scatter(xs, ys, c=cs, s=55, edgecolors="white", linewidths=0.8, alpha=0.9)
    # Label top-10 biggest gaps
    gaps = sorted(enumerate(zip(xs, ys, labels_)),
                  key=lambda kv: -(kv[1][1] - kv[1][0]))[:10]
    for _, (px, py, lab) in gaps:
        ax.annotate(lab, xy=(px, py), xytext=(4, 3),
                    textcoords="offset points", fontsize=7,
                    color=PALETTE["ink"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Prompting F1 (macro)")
    ax.set_ylabel("Probe F1 (macro)")
    ax.set_title("Probe-vs-Prompt per (task × modality)  —  all points above dashed line = "
                 "model knows but cannot say it",
                 fontsize=10.5, color=PALETTE["ink"])
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=TASK_CMAP[t], markersize=10,
                      label=TASK_LABEL.get(t, t)) for t in tasks]
    ax.legend(handles=handles, fontsize=9, loc="lower right", frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    p = out_dir / "probe_vs_prompt_scatter.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_emergence_diagram(results: dict, out_dir: Path,
                          threshold: float = 0.5) -> Path | None:
    """Concept-emergence diagram. For every (task, modality), plots the
    *earliest* transformer layer at which layer-wise linear-probe F1 crosses
    the threshold. Shows where tactical concepts become linearly decodable.
    The dot size encodes the final-layer F1; the x-position is the emergence
    layer; rows are modalities and columns are tasks.
    """
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    if not tasks:
        return None
    mod_order = ["d", "p", "pi", "n", "ne", "v", "d+v", "m", "o", "b"]
    mods_present: list[str] = []
    for m in mod_order:
        for t in tasks:
            lw = results[t].get(m, {}).get("layer_wise", {})
            if lw:
                if m not in mods_present:
                    mods_present.append(m)
                break
    n = len(mods_present)
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * n + 2.2))

    task_colours = {
        "pressing_type":        "#1e3a8a",
        "compactness_trend":    "#b45309",
        "possession_phase":     "#047857",
        "territorial_dominance":"#be123c",
    }
    yticks = list(range(n))
    max_layer_seen = 0
    for yi, m in enumerate(mods_present):
        for task in tasks:
            lw = results[task].get(m, {}).get("layer_wise", {})
            if not lw:
                continue
            layers = sorted(int(k) for k in lw if lw[k] is not None)
            def _f1(v):
                return v["linear_f1"] if isinstance(v, dict) else v
            f1s = [_f1(lw[str(L)]) for L in layers]
            if not f1s:
                continue
            crossing = None
            for L, v in zip(layers, f1s):
                if v is not None and v >= threshold:
                    crossing = L
                    break
            # If never crosses, place at max layer with low alpha
            last_f1 = f1s[-1] if f1s[-1] is not None else 0.0
            c = task_colours[task]
            max_layer_seen = max(max_layer_seen, max(layers))
            if crossing is not None:
                ax.scatter([crossing], [yi], s=50 + 220 * last_f1, c=c,
                           alpha=0.85, edgecolors=PALETTE["ink"], linewidths=0.8,
                           zorder=3)
            else:
                ax.scatter([max(layers) + 1], [yi], s=25, c=c, alpha=0.35,
                           marker="x", zorder=3)
    ax.set_yticks(yticks)
    ax.set_yticklabels(mods_present, fontsize=10)
    ax.set_xlim(-1, max_layer_seen + 3)
    ax.set_xlabel("Layer at which F1 ≥ 0.5  (dot size ∝ final-layer F1)")
    ax.set_title("Concept emergence across layers — earlier = concept encoded in shallower features",
                 color=PALETTE["ink"], fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w",
                     markerfacecolor=task_colours[t],
                     markersize=10, label=TASK_LABEL.get(t, t)) for t in tasks]
    legend.append(Line2D([0], [0], marker="x", color=PALETTE["slate"],
                         linestyle="", label="never reaches 0.5"))
    ax.legend(handles=legend, fontsize=9, loc="upper right", frameon=False)
    fig.tight_layout()
    p = out_dir / "emergence_diagram.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_modality_radar(results: dict, out_dir: Path,
                       modalities: list[str] | None = None) -> Path | None:
    """Radar / polar chart: one axis per modality, one coloured polygon per
    task. All-in-one visual for the nine-modality study."""
    tasks = [t for t in results if isinstance(results.get(t), dict)]
    mods = modalities or ["d", "v", "d+v", "p", "pi", "n", "ne", "b"]
    angles = np.linspace(0, 2 * np.pi, len(mods), endpoint=False).tolist()
    angles += angles[:1]  # close the loop
    fig, ax = plt.subplots(figsize=(7.8, 7.8),
                           subplot_kw=dict(projection="polar"))
    task_colours = {
        "pressing_type": "#1e3a8a",
        "compactness_trend": "#b45309",
        "possession_phase": "#047857",
        "territorial_dominance": "#be123c",
    }
    for task in tasks:
        vals = []
        for m in mods:
            entry = results[task].get(m, {})
            probe = entry.get("probe") if isinstance(entry, dict) else None
            v = probe.get("f1_macro") if isinstance(probe, dict) else None
            vals.append(v if isinstance(v, (int, float)) else 0.0)
        vals += vals[:1]
        c = task_colours.get(task, PALETTE["slate"])
        ax.plot(angles, vals, color=c, linewidth=2.0, label=TASK_LABEL.get(task, task))
        ax.fill(angles, vals, color=c, alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(mods, fontsize=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.4)
    ax.set_title("Probe F1 across modalities — polar view",
                 color=PALETTE["ink"], fontsize=12, fontweight="semibold", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=9, frameon=False)
    fig.tight_layout()
    p = out_dir / "modality_radar.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png")); plt.close(fig)
    return p


def fig_rank_descent_heatmap(rank_path: Path, out_dir: Path,
                             vocab_size: int = 152064,
                             modality: str = "d") -> Path | None:
    """Heatmap of class-name-token rank across layers. Rows = (task, token),
    columns = layers, cell colour = log10(rank). Makes the three-stage
    extraction mechanism visible as one image."""
    if not rank_path.exists():
        return None
    d = json.loads(rank_path.read_text())
    rows: list[tuple[str, str, str, dict[int, int]]] = []
    # (task, class, token_text, {layer -> rank})
    layers_seen: set[int] = set()
    for task, mods in d.items():
        if modality not in mods:
            continue
        for layer_s, cls_map in mods[modality].items():
            layers_seen.add(int(layer_s))
        # collate rows
        for cls in next(iter(mods[modality].values())).keys():
            # find all tokens for this class across layers
            toks = set()
            for layer_s, cmap in mods[modality].items():
                toks.update(cmap.get(cls, {}).keys())
            for tok in sorted(toks):
                row: dict[int, int] = {}
                for layer_s, cmap in mods[modality].items():
                    r = cmap.get(cls, {}).get(tok)
                    if r is not None:
                        row[int(layer_s)] = int(r)
                rows.append((task, cls, tok.strip(), row))
    if not rows:
        return None
    layers = sorted(layers_seen)
    # Build matrix — use log10(rank); missing = vocab_size (worst)
    mat = np.full((len(rows), len(layers)), np.log10(vocab_size))
    for i, (_, _, _, r) in enumerate(rows):
        for j, L in enumerate(layers):
            if L in r:
                mat[i, j] = np.log10(max(r[L], 1))
    row_labels = [f"{TASK_LABEL.get(t, t).split()[0][:11]} · {cls} · {tok}"
                  for t, cls, tok, _ in rows]
    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.25 * len(rows) + 1.4)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=np.log10(vocab_size))
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{L}" for L in layers], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    # Annotate each cell with rank (compact)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            r = 10 ** mat[i, j]
            if r >= vocab_size * 0.99:
                txt = "—"
            elif r >= 10000:
                txt = f"{int(round(r/1000))}k"
            else:
                txt = f"{int(round(r))}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color="white" if mat[i, j] > 3.8 else "#111827",
                    fontweight="semibold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("log₁₀(rank in 152 k vocab)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(
        f"Rank-descent of class-name tokens across depth — modality '{modality}'\n"
        "green = class word is high-probability next token · red = buried deep in vocab",
        fontsize=11, color=PALETTE["ink"], fontweight="semibold", pad=10,
    )
    ax.set_xlabel("Transformer layer")
    fig.tight_layout()
    p = out_dir / f"rank_descent_heatmap_{modality.replace('+','_')}.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    return p


def fig_layerwise_gap(dense_path: Path, results_path: Path,
                     out_dir: Path) -> Path | None:
    """For each task plots (probe F1 − best zero-shot prompt F1) across all
    layers. Positive area = extraction gap; the higher the curve, the more
    knowledge the probe reads that greedy decoding cannot surface.
    One line per task, one shared panel so comparisons are immediate."""
    if not (dense_path.exists() and results_path.exists()):
        return None
    d = json.loads(dense_path.read_text())
    r = json.loads(results_path.read_text())
    task_colours = {
        "pressing_type":     "#1e3a8a",
        "compactness_trend": "#b45309",
        "possession_phase":  "#047857",
        "territorial_dominance": "#be123c",
    }
    fig, ax = plt.subplots(figsize=(11, 5.6))
    max_gap = 0.0
    for task in sorted(d.keys()):
        # Best probe F1 at each layer across all probed modalities
        best_by_layer: dict[int, float] = {}
        for mod, layers in d[task].items():
            for L_s, cell in layers.items():
                L = int(L_s)
                best_by_layer[L] = max(best_by_layer.get(L, 0.0),
                                        float(cell["mean_f1"]))
        # Best prompt F1 (modality-agnostic) for this task
        best_prompt = 0.0
        for m, cell in r.get(task, {}).items():
            if isinstance(cell, dict):
                pf = cell.get("prompting", {}).get("f1_macro")
                if pf is not None:
                    best_prompt = max(best_prompt, float(pf))
        xs = sorted(best_by_layer.keys())
        ys = [best_by_layer[L] - best_prompt for L in xs]
        max_gap = max(max_gap, max(ys))
        ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=2.3,
                 color=task_colours.get(task, PALETTE["slate"]),
                 label=f"{TASK_LABEL.get(task, task)} (prompt={best_prompt:.2f})",
                 markerfacecolor="white",
                 markeredgecolor=task_colours.get(task, PALETTE["slate"]),
                 markeredgewidth=1.2,
                 zorder=3)
        ax.fill_between(xs, 0, ys,
                         color=task_colours.get(task, PALETTE["slate"]),
                         alpha=0.08, zorder=1)
    ax.axhline(0, color=PALETTE["ink"], linewidth=0.8, zorder=0)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Extraction gap  (best probe F1 − best prompt F1)")
    ax.set_title(
        "Layer-wise extraction gap — where in the model knowledge exists "
        "that greedy decoding cannot surface",
        fontsize=12, color=PALETTE["ink"], fontweight="semibold", pad=10,
    )
    ax.set_xlim(-1, 29)
    ax.set_ylim(-0.05, max_gap + 0.08)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28])
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5,
              facecolor="#f9fafb", edgecolor="#d1d5db")
    fig.tight_layout()
    p = out_dir / "layerwise_extraction_gap.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return p


def fig_stability_vs_f1(stab_path: Path, multiseed_path: Path,
                        out_dir: Path) -> Path | None:
    """Scatter plot: x = prompt-variant Δ (sensitivity to rewording),
    y = probe F1 (what the representation could extract). Each point =
    (task, modality) cell. Upper-right quadrant = high-F1-but-brittle."""
    if not (stab_path.exists() and multiseed_path.exists()):
        return None
    stab = json.loads(stab_path.read_text())
    ms = json.loads(multiseed_path.read_text())
    mod_colour = {
        "d": "#1e3a8a", "v": "#c2410c", "d+v": "#047857",
        "nec+v": "#be123c", "p": "#6b21a8",
        "pi": "#0891b2", "n": "#be185d", "ne": "#be185d",
    }
    mod_marker = {
        "d": "o", "v": "s", "d+v": "D",
        "p": "^", "pi": "v", "n": "P", "ne": "X",
        "nec": "p", "nec+v": "H",
    }
    task_shape = {
        "pressing_type": 0, "compactness_trend": 1,
        "possession_phase": 2, "territorial_dominance": 3,
    }
    points = []
    for t, mods in stab.items():
        for m, e in mods.items():
            delta = float(e.get("variant_delta", 0.0))
            f1 = None
            if t in ms and m in ms[t]:
                f1 = float(ms[t][m]["mean_f1"])
            if f1 is None:
                continue
            points.append((t, m, delta, f1))
    if not points:
        return None
    fig, ax = plt.subplots(figsize=(10, 7))
    # Quadrant guides at median values
    deltas = [p[2] for p in points]; f1s = [p[3] for p in points]
    xmid = float(np.median(deltas)); ymid = float(np.median(f1s))
    ax.axvline(xmid, color="#d1d5db", linewidth=0.8, linestyle=":", zorder=0)
    ax.axhline(ymid, color="#d1d5db", linewidth=0.8, linestyle=":", zorder=0)
    # Quadrant labels
    xr = max(deltas) - min(deltas); yr = max(f1s) - min(f1s)
    ax.text(xmid + xr*0.02, max(f1s) - yr*0.02,
            "HIGH-F1 · BRITTLE\n(review with caution)",
            fontsize=9, color="#9ca3af", ha="left", va="top",
            fontweight="semibold")
    ax.text(xmid - xr*0.02, max(f1s) - yr*0.02,
            "HIGH-F1 · STABLE\n(recommended)",
            fontsize=9, color="#9ca3af", ha="right", va="top",
            fontweight="semibold")
    ax.text(xmid - xr*0.02, min(f1s) + yr*0.02,
            "low-F1 · stable",
            fontsize=8.5, color="#d1d5db", ha="right", va="bottom")
    ax.text(xmid + xr*0.02, min(f1s) + yr*0.02,
            "low-F1 · brittle",
            fontsize=8.5, color="#d1d5db", ha="left", va="bottom")
    # Draw points
    seen_mods = []
    for t, m, dlt, f1 in points:
        col = mod_colour.get(m, "#475569")
        mkr = mod_marker.get(m, "o")
        ax.scatter(dlt, f1, s=150, color=col, marker=mkr,
                    edgecolor="white", linewidth=1.6, alpha=0.92, zorder=4)
        ax.annotate(f"{m}  ·  {TASK_LABEL.get(t, t).split()[0][:8]}",
                    xy=(dlt, f1), xytext=(6, 6), textcoords="offset points",
                    fontsize=7.5, color=col, alpha=0.85, zorder=5)
        if m not in seen_mods:
            seen_mods.append(m)
    ax.set_xlabel("Prompt-rewording sensitivity   Δ = max − min F1 across 10 variants")
    ax.set_ylabel("Linear-probe macro-F1 (stratified 10-seed)")
    ax.set_title(
        "Brittleness vs. representational ceiling\n"
        "each point = (task × modality) cell · upper-right = best-F1 but most prompt-fragile",
        fontsize=12, color=PALETTE["ink"], fontweight="semibold", pad=8,
    )
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    # Legend for modalities (distinct from colour — shape also varies)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=mod_marker.get(m, "o"),
               color="white", markerfacecolor=mod_colour.get(m, "#475569"),
               markeredgecolor="white", markeredgewidth=1.2,
               markersize=10, label=m)
        for m in seen_mods
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True,
              facecolor="#f9fafb", edgecolor="#d1d5db",
              fontsize=9.5, title="modality",
              title_fontsize=9, ncol=len(seen_mods))
    fig.tight_layout()
    p = out_dir / "stability_vs_f1_scatter.pdf"
    fig.savefig(p); fig.savefig(p.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return p


def fig_dense_layer_curves(dense_path: Path, out_dir: Path,
                           modalities: "list[str] | None" = None,
                           results_path: "Path | None" = None) -> Path | None:
    """Dense layer-wise probe curves (Schumacher Fig. 4 replication). For each
    task × modality plots F1 vs. layer across all 29 transformer layers (0–28)
    with per-seed std band. One panel per task; modality colours; peak marker
    per curve. When `results_path` is supplied, overlays the best zero-shot
    prompting F1 per task as a horizontal dashed line, making the extraction
    gap visually immediate."""
    if not dense_path.exists():
        return None
    d = json.loads(dense_path.read_text())
    prompt_best: dict[str, tuple[float, str]] = {}
    if results_path and Path(results_path).exists():
        r = json.loads(Path(results_path).read_text())
        for t in d:
            best_v, best_m = -1.0, ""
            for m in (modalities or ["d", "v", "d+v", "nec+v"]):
                cell = r.get(t, {}).get(m, {})
                if isinstance(cell, dict):
                    pf = cell.get("prompting", {}).get("f1_macro")
                    if pf is not None and pf > best_v:
                        best_v, best_m = float(pf), m
            if best_v >= 0:
                prompt_best[t] = (best_v, best_m)
    tasks = sorted(d.keys())
    if not tasks:
        return None
    mods = modalities or ["d", "v", "d+v", "nec+v"]
    mod_colour = {
        "d": "#1e3a8a", "v": "#c2410c", "d+v": "#047857",
        "nec+v": "#be123c", "p": "#6b21a8",
        "pt": "#0891b2", "sf": "#9333ea",
    }
    mod_label = {
        "d": "d",
        "v": "v",
        "d+v": "d+v",
        "nec+v": "nec+v (narration+image)",
        "p": "p",
        "pt": "pt (patch)",
        "sf": "sf (stat)",
    }
    # Literature-grade extensions get a subdued secondary treatment so they
    # don't compete with the core modalities for visual weight.
    SECONDARY = {"pt", "sf"}
    SECONDARY_STYLE = {
        "pt": dict(linestyle=(0, (7, 3)), alpha=0.68),
        "sf": dict(linestyle=(0, (1.5, 2.5)), alpha=0.62),
    }
    fig = plt.figure(figsize=(13, 9.4))
    gs = fig.add_gridspec(
        2, 2, hspace=0.34, wspace=0.22,
        top=0.86, bottom=0.13, left=0.07, right=0.97,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    handles = []
    for ax, task in zip(axes, tasks):
        ax.axhline(0.3, color="#d1d5db", linewidth=0.8, linestyle=":",
                   zorder=0)
        if task in prompt_best:
            pf, pmod = prompt_best[task]
            ax.axhline(pf, color="#64748b", linewidth=1.3,
                       linestyle=(0, (6, 3)), zorder=0.5)
            ax.text(14, pf - 0.028,
                    f"zero-shot prompt ({pmod}) = {pf:.2f}",
                    ha="center", va="top", fontsize=8.5,
                    color="#334155", fontweight="bold",
                    zorder=15,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="#64748b",
                              lw=0.9, alpha=1.0))
        task_max_cache = 0.0
        # First pass: draw all curves; collect per-primary peaks.
        primary_peaks: list[tuple[str, int, float, str]] = []  # (mod, layer, f1, colour)
        for m in mods:
            layers_dict = d.get(task, {}).get(m, {})
            if not layers_dict:
                continue
            layers = sorted(int(k) for k in layers_dict.keys())
            means = np.array([layers_dict[str(L)]["mean_f1"] for L in layers])
            stds = np.array([layers_dict[str(L)]["std_f1"] for L in layers])
            colour = mod_colour.get(m, PALETTE["slate"])
            is_secondary = m in SECONDARY
            if is_secondary:
                style = SECONDARY_STYLE[m]
                ln, = ax.plot(
                    layers, means,
                    color=colour, linewidth=1.6,
                    linestyle=style["linestyle"], alpha=style["alpha"],
                    label=mod_label.get(m, m), zorder=2,
                    solid_capstyle="round",
                )
                if ax is axes[0]:
                    handles.append(ln)
                continue
            # Single discrete std band — very low alpha so overlapping
            # modalities don't mix into a muddy wash.
            ax.fill_between(layers, means - stds, means + stds,
                             color=colour, alpha=0.025, linewidth=0,
                             zorder=1)
            # Softer halo under the curve — thinner and slightly translucent
            # so the white separation doesn't look plasticky.
            ax.plot(layers, means, color="white", linewidth=3.2, alpha=0.9,
                    zorder=2,
                    solid_capstyle="round", solid_joinstyle="round")
            ln, = ax.plot(layers, means,
                          color=colour, linewidth=1.9,
                          marker="o", markersize=2.6,
                          markerfacecolor="white",
                          markeredgecolor=colour, markeredgewidth=1.0,
                          label=mod_label.get(m, m), zorder=3,
                          solid_capstyle="round",
                          solid_joinstyle="round")
            if ax is axes[0]:
                handles.append(ln)
            i_best = int(np.argmax(means))
            peak_layer = layers[i_best]
            peak_f1 = float(means[i_best])
            primary_peaks.append((m, peak_layer, peak_f1, colour))
            # Small star for every primary, but only the winner gets a boxed label.
            ax.scatter([peak_layer], [peak_f1],
                       marker="*", s=130,
                       color=colour, edgecolor="white",
                       linewidth=1.1, zorder=6)
            task_max_cache = max(task_max_cache, float(np.max(means + stds)))

        # Boxed label only for the panel-winning primary curve.
        if primary_peaks:
            winner = max(primary_peaks, key=lambda t: t[2])
            w_mod, w_layer, w_f1, w_colour = winner
            # Bigger star for the winner to distinguish it.
            ax.scatter([w_layer], [w_f1], marker="*", s=230,
                       color=w_colour, edgecolor="white",
                       linewidth=1.4, zorder=7)
            dx = 14 if w_layer <= 14 else -14
            dy = 22
            ax.annotate(
                f"L{w_layer}  {w_f1:.2f}",
                xy=(w_layer, w_f1),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9, color=w_colour, fontweight="bold",
                ha="left" if dx > 0 else "right",
                va="bottom",
                zorder=20,
                bbox=dict(
                    boxstyle="round,pad=0.32",
                    fc="white", ec=w_colour, lw=1.0, alpha=1.0,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color=w_colour, lw=0.8, alpha=0.55,
                    shrinkA=2, shrinkB=4,
                ),
            )

        ax.set_xlim(-1.5, 29.5)
        y_bottom = 0.02
        if task in prompt_best:
            y_bottom = min(y_bottom, prompt_best[task][0] - 0.12)
        ax.set_ylim(max(-0.02, y_bottom), min(1.05, task_max_cache + 0.14))
        ax.set_xlabel("Transformer layer", fontsize=10)
        ax.set_ylabel("Macro-F1", fontsize=10)
        ax.set_title(TASK_LABEL.get(task, task),
                     fontsize=12, color=PALETTE["ink"], fontweight="semibold",
                     pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9ca3af")
        ax.spines["bottom"].set_color("#9ca3af")
        ax.tick_params(colors="#6b7280", labelsize=9)
        ax.grid(True, which="both", alpha=0.18, linewidth=0.6,
                color="#9ca3af", zorder=0)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28])

    fig.suptitle(
        "Dense layer-wise probe F1 across Qwen2-VL-7B  (layers 0–28)",
        fontsize=14.5, color=PALETTE["ink"], fontweight="semibold",
        y=0.965,
    )
    fig.text(0.5, 0.905,
             "⋆ peak layer per curve  ·  faint band = ±1 std (5 stratified seeds)  ·  dashed grey line = best zero-shot prompting F1",
             ha="center", fontsize=9.8, color="#4b5563", style="italic")
    leg = fig.legend(
        handles=handles, loc="lower center",
        ncol=len(handles), bbox_to_anchor=(0.5, 0.025),
        frameon=True, fontsize=10.5,
        handlelength=2.4, handletextpad=0.55,
        columnspacing=1.8, borderpad=0.7,
        labelcolor=PALETTE["ink"],
    )
    frame = leg.get_frame()
    frame.set_facecolor("#f9fafb")
    frame.set_edgecolor("#d1d5db")
    frame.set_linewidth(0.8)
    for txt in leg.get_texts():
        txt.set_fontweight("semibold")
    p = out_dir / "dense_layer_curves.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    return p


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="Path to probing_results.json")
    ap.add_argument("--output", required=True,
                    help="Directory to save figures")
    ap.add_argument("--analyses", default=None,
                    help="Optional analyses.json from probing_analyses.py")
    ap.add_argument("--cache-dir", default=None,
                    help="Hidden-state cache dir (for t-SNE strips)")
    ap.add_argument("--modalities", nargs="+", default=["d", "v", "d+v"])
    ap.add_argument("--hero-modality", default="d",
                    help="Which modality to use for the hero plate")
    ap.add_argument("--rigor-dir", default=None,
                    help="Directory with multiseed/transfer/confusion/perclass JSONs")
    args = ap.parse_args()

    _apply_style()
    results = json.loads(Path(args.results).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for mod in args.modalities:
        fig_f1_heatmap(results, mod, out, "linear")
        fig_f1_heatmap(results, mod, out, "mlp")
        fig_mlp_minus_linear_gap(results, mod, out)
        fig_best_alpha(results, mod, out)

    fig_curves_with_bands(results, out, args.modalities)
    fig_hero_plate(results, out, args.hero_modality)

    # Advanced analyses (optional — requires analyses.json + cache dir)
    if args.analyses and Path(args.analyses).exists():
        analyses = json.loads(Path(args.analyses).read_text())
        for mod in args.modalities:
            fig_separability_panel(analyses, mod, out)
            fig_selectivity(analyses, mod, out)
            fig_extraction_gap(analyses, mod, out)
            for task in analyses.get("per_task_modality", {}).keys():
                fig_cka_matrix(analyses, task, mod, out)
        fig_modality_synergy(analyses, out)

        # t-SNE evolution needs the raw cache, not the JSON.
        if args.cache_dir and Path(args.cache_dir).exists():
            cache_dir = Path(args.cache_dir)
            # Match filenames against known modality suffixes (longest first so
            # 'd_v' is preferred over 'v' / 'd' when both tails match).
            safe_by_orig = {m: m.replace("+", "_") for m in args.modalities}
            safe_suffixes = sorted(safe_by_orig.values(), key=len, reverse=True)
            for npz in sorted(cache_dir.glob("hidden_*.npz")):
                stem = npz.stem[len("hidden_"):]
                matched: tuple[str, str] | None = None
                for safe in safe_suffixes:
                    if stem.endswith("_" + safe):
                        orig_mod = next(m for m, s in safe_by_orig.items() if s == safe)
                        matched = (stem[: -(len(safe) + 1)], orig_mod)
                        break
                if matched is None:
                    continue
                task, mod = matched
                data = np.load(npz, allow_pickle=True)
                probed = sorted(int(x) for x in data["layers_probed"])
                if len(probed) >= 4:
                    pick = [probed[0],
                            probed[len(probed) // 3],
                            probed[2 * len(probed) // 3],
                            probed[-1]]
                else:
                    pick = probed
                fig_tsne_strip(cache_dir, task, mod, pick, out)

    # Rigor-study figures
    if args.rigor_dir:
        rigor = Path(args.rigor_dir)
        fig_transfer_matrix(rigor / "transfer_d.json", out)
        fig_confusion_grid(rigor / "confusion.json", out, modality="p")
        fig_confusion_grid(rigor / "confusion.json", out, modality="d")
        fig_multiseed_strip(rigor / "multiseed.json", out)
        fig_perclass_evolution(rigor / "perclass.json", out, modality="p")
        fig_perclass_evolution(rigor / "perclass.json", out, modality="d")
    fig_modality_radar(results, out)
    fig_emergence_diagram(results, out)
    # Rigor-correction + academic-standard extras
    if args.rigor_dir:
        rigor = Path(args.rigor_dir)
        fig_single_vs_stratified(results, rigor / "multiseed_stratified.json", out)
        fig_mdl_compression(rigor / "mdl.json", out)
    fig_probe_vs_prompt_scatter(results, out)
    fig_text_visual_pairing(results, out)
    fig_ne_rescue(results, out)
    fig_fusion_helps_vs_hurts(results, out)
    if args.rigor_dir:
        ms_full = Path(args.rigor_dir) / "multiseed_stratified_full.json"
        ms_alt = Path(args.rigor_dir) / "multiseed_stratified.json"
        ms = ms_full if ms_full.exists() else ms_alt
        fig_task_winners_podium(ms, out)
        fig_improvement_over_d(ms, out)
        fig_best_impacts_showcase(ms, results, out)
    logger.info("all figures written to %s", out)


if __name__ == "__main__":
    main()
