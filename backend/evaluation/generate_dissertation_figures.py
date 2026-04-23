"""Generate dissertation figures from evaluation results.

Outputs to dissertation/figures/:
  - visual_condition_comparison.pdf/png  — grounding rates across all conditions
  - linear_probing_results.pdf/png       — probe vs prompting F1 by modality
  - random_baseline_comparison.pdf/png   — learned vs random probe F1
  - example_compactness_chart.pdf/png    — rendered compactness time-series
  - example_centroid_chart.pdf/png       — rendered centroid trajectory
  - example_pressing_chart.pdf/png       — rendered pressing dashboard
  - linear_probing_layerwise.pdf/png     — v2 layer-wise probe F1 curves

Usage:
    python3 -m backend.evaluation.generate_dissertation_figures [--figures-dir dissertation/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, path_stem: Path) -> None:
    fig.savefig(str(path_stem) + ".pdf", bbox_inches="tight", dpi=150)
    fig.savefig(str(path_stem) + ".png", bbox_inches="tight", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved {path_stem}.pdf/.png")


# ── Figure 1: Visual condition comparison ────────────────────────────────────

def fig_visual_conditions(
    perframe_dir: Path,
    figures_dir: Path,
    provider: str = "gemini",
) -> None:
    """Bar chart of grounding rates across all evaluation conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    condition_dir = perframe_dir / provider

    # Condition display labels and file stems
    conditions = [
        ("BASELINE",          "Baseline",          "baseline_results.json"),
        ("PERFRAME_V1",       "Perframe V1",        "perframe_v1_results.json"),
        ("PERFRAME_V2",       "Perframe V2",        "perframe_v2_results.json"),
        ("DIGIT_SPACE",       "Digit-Space",        "digit_space_results.json"),
        ("VISUAL",            "Visual (all charts)","visual_results.json"),
        ("VISUAL_FOCUSED",    "Visual Focused",     "visual_focused_results.json"),
        ("VISUAL_MULTIMODAL", "Visual Multimodal",  "visual_multimodal_results.json"),
        ("FINDINGS_INFORMED", "Findings-Informed",  "findings_informed_results.json"),
    ]

    ANALYSIS_TYPES = ["match_overview", "tactical_deep_dive", "event_analysis", "player_spotlight"]
    labels_atype = ["Match\nOverview", "Tactical\nDeep Dive", "Event\nAnalysis", "Player\nSpotlight"]

    # Load what exists
    loaded = []
    for cname, clabel, fname in conditions:
        fpath = condition_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            by_atype = data.get("by_analysis_type", {})
            rates = [by_atype.get(at, {}).get("grounding_rate", float("nan")) for at in ANALYSIS_TYPES]
            overall = data.get("overall_grounding_rate", float("nan"))
            loaded.append((clabel, rates, overall))
        else:
            print(f"  [skip] {fname} not found")

    if not loaded:
        print("  No condition results found — skipping figure 1.")
        return

    n_cond = len(loaded)
    n_atype = len(ANALYSIS_TYPES)
    x = np.arange(n_cond)
    width = 0.18
    offsets = np.linspace(-(n_atype - 1) / 2, (n_atype - 1) / 2, n_atype) * width

    colours = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
    fig, ax = plt.subplots(figsize=(13, 5))

    for i, (atype_label, colour, offset) in enumerate(zip(labels_atype, colours, offsets)):
        vals = [c[1][i] * 100 for c in loaded]
        ax.bar(x + offset, vals, width=width, label=atype_label, color=colour, alpha=0.85)

    # Overall grounding as black diamond
    overall_vals = [c[2] * 100 for c in loaded]
    ax.plot(x, overall_vals, "kD", markersize=7, zorder=5, label="Overall (mean)")
    ax.plot(x, overall_vals, "k--", linewidth=1, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in loaded], fontsize=9)
    ax.set_ylabel("Grounding Rate (%)", fontsize=11)
    ax.set_title(
        "Commentary Grounding Rate by Evaluation Condition (Gemini, Analysis 18, n=1)",
        fontsize=12, pad=12
    )
    ax.set_ylim(0, 80)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, figures_dir / "visual_condition_comparison")


# ── Figure 2: Linear probing — probe vs prompting F1 ─────────────────────────

def fig_linear_probing(probing_dir: Path, figures_dir: Path) -> None:
    """Grouped bar chart: probe F1 vs prompting F1 by modality and task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results_path = probing_dir / "probing_results.json"
    if not results_path.exists():
        print(f"  [skip] {results_path} not found")
        return
    with open(results_path) as f:
        data = json.load(f)

    tasks = ["pressing_type", "compactness_trend", "possession_phase", "territorial_dominance"]
    task_labels = ["Pressing\nType", "Compactness\nTrend", "Possession\nPhase", "Territorial\nDominance"]

    modalities = ["d", "v", "d+v"]
    mod_labels = ["Text (d)", "Visual (v)", "Combined (d+v)"]
    mod_colours = ["#4878CF", "#6ACC65", "#D65F5F"]

    prompt_colour = "#888888"

    x = np.arange(len(tasks))
    n_bars = len(modalities) + 1  # +1 for prompting
    width = 0.18
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (mod, label, colour, offset) in enumerate(zip(modalities, mod_labels, mod_colours, offsets)):
        vals = []
        for task in tasks:
            td = data.get(task, {}).get(mod, {})
            vals.append(td.get("probe", {}).get("f1_macro", float("nan")) * 100)
        ax.bar(x + offset, vals, width=width, label=f"Probe {label}", color=colour, alpha=0.85)

    # Prompting F1 (same across modalities for d, use d prompting)
    prompt_vals = []
    for task in tasks:
        td = data.get(task, {}).get("d", {})
        prompt_vals.append(td.get("prompting", {}).get("f1_macro", float("nan")) * 100)
    ax.bar(x + offsets[-1], prompt_vals, width=width, label="Prompting (d)", color=prompt_colour, alpha=0.85, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_ylabel("Macro F1 (%)", fontsize=11)
    ax.set_title(
        "Linear Probing vs Prompting F1 by Task and Modality\n"
        "(Qwen2-VL-7B-Instruct, Analyses 18+13+17, n_test=24–31)",
        fontsize=11, pad=10
    )
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, figures_dir / "linear_probing_results")


# ── Figure 3: Random baseline comparison ─────────────────────────────────────

def fig_random_baseline(probing_dir: Path, figures_dir: Path) -> None:
    """Stacked comparison: pretrained probe, random probe, prompting per task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results_path = probing_dir / "probing_results.json"
    random_path = probing_dir / "random_baseline.json"
    if not results_path.exists() or not random_path.exists():
        print(f"  [skip] probing_results.json or random_baseline.json not found")
        return

    with open(results_path) as f:
        data = json.load(f)
    with open(random_path) as f:
        rdata = json.load(f)

    tasks = ["pressing_type", "compactness_trend", "possession_phase", "territorial_dominance"]
    task_labels = ["Pressing\nType", "Compactness\nTrend", "Possession\nPhase", "Territorial\nDominance"]

    pretrained_d = [data.get(t, {}).get("d", {}).get("probe", {}).get("f1_macro", 0) * 100 for t in tasks]
    random_d = [rdata.get(t, {}).get("random_f1", rdata.get(t, {}).get("f1_macro", 0)) * 100 for t in tasks]
    prompting = [data.get(t, {}).get("d", {}).get("prompting", {}).get("f1_macro", 0) * 100 for t in tasks]

    x = np.arange(len(tasks))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, pretrained_d, width=width, label="Probe (pretrained, d)", color="#4878CF", alpha=0.85)
    ax.bar(x,          random_d,    width=width, label="Probe (random weights, d)", color="#D65F5F", alpha=0.65, hatch="xx")
    ax.bar(x + width,  prompting,   width=width, label="Prompting (d)",             color="#888888", alpha=0.85, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.set_ylabel("Macro F1 (%)", fontsize=11)
    ax.set_title(
        "Pretrained Probe vs Random-Weight Probe vs Prompting\n"
        "(Qwen2-VL-7B, d modality — confirms learned representation)",
        fontsize=11, pad=10
    )
    ax.set_ylim(0, 85)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Annotate territorial gap (key finding)
    ax.annotate(
        "Random > Pretrained\n(pretraining suppresses\nspatial info)",
        xy=(3, random_d[3]), xytext=(2.4, 65),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=8, ha="center",
    )

    _save(fig, figures_dir / "random_baseline_comparison")


# ── Figure 4: Layer-wise probe F1 ─────────────────────────────────────────────

def fig_layerwise(probing_dir: Path, figures_dir: Path) -> None:
    """Line plot of layer-wise probe F1 across transformer layers (v2, Qwen2.5-7B, d)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_path = probing_dir / "probing_results.json"
    if not results_path.exists():
        print(f"  [skip] {results_path} not found")
        return
    with open(results_path) as f:
        data = json.load(f)

    # Layer-wise only in v2 (probing/probing_results.json, not probing_vl/)
    # Try to load from the v2 path
    v2_path = probing_dir.parent / "probing" / "probing_results.json"
    if v2_path.exists():
        with open(v2_path) as f:
            lw_data = json.load(f)
    else:
        print(f"  [skip] layer-wise data not found at {v2_path}")
        return

    tasks = {
        "pressing_type": ("Pressing Type", "#4878CF"),
        "compactness_trend": ("Compactness Trend", "#6ACC65"),
        "possession_phase": ("Possession Phase", "#D65F5F"),
        "territorial_dominance": ("Territorial Dominance", "#B47CC7"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for task, (label, colour) in tasks.items():
        lw = lw_data.get(task, {}).get("d", {}).get("layer_wise", {})
        if not lw:
            continue
        layers = sorted(int(k) for k in lw.keys())
        f1s = [lw[str(l)] * 100 for l in layers]
        ax.plot(layers, f1s, marker="o", markersize=5, label=label, color=colour, linewidth=2)

    ax.set_xlabel("Transformer Layer", fontsize=11)
    ax.set_ylabel("Probe F1 (%)", fontsize=11)
    ax.set_title(
        "Layer-Wise Linear Probe F1 (Qwen2.5-7B-Instruct, d modality)\n"
        "Discrimination emerges at layers 4–8 across all tasks",
        fontsize=11, pad=10
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(4, color="grey", linestyle=":", alpha=0.6, linewidth=1)
    ax.axvline(8, color="grey", linestyle=":", alpha=0.6, linewidth=1)
    ax.text(4.3, 5, "layer 4", fontsize=8, color="grey")
    ax.text(8.3, 5, "layer 8", fontsize=8, color="grey")

    _save(fig, figures_dir / "linear_probing_layerwise")


# ── Figures 5–7: Example rendered charts from Analysis 18 ────────────────────

def fig_example_charts(gt_path: Path, figures_dir: Path) -> None:
    """Render and save example compactness, centroid, and pressing charts."""
    import sys, os
    sys.path.insert(0, "backend")
    sys.path.insert(0, "backend/api")
    from services.tactical import VisualTimeSeriesRenderer

    if not gt_path.exists():
        print(f"  [skip] ground truth not found at {gt_path}")
        return

    with open(gt_path) as f:
        gt = json.load(f)

    fm = gt.get("frame_metrics", {})
    fps = gt.get("analytics", {}).get("fps") or 25.0

    chart_map = {
        "example_compactness_chart": ("compactness", VisualTimeSeriesRenderer._render_compactness),
        "example_centroid_chart": ("centroid", VisualTimeSeriesRenderer._render_centroid_trajectory),
        "example_pressing_chart": ("pressing", VisualTimeSeriesRenderer._render_pressing_dashboard),
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    for stem, (name, fn) in chart_map.items():
        try:
            img_bytes = fn(fm, fps)
            if img_bytes is None:
                print(f"  [skip] {name} chart — no data")
                continue
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(img)
            ax.axis("off")
            _save(fig, figures_dir / stem)
        except Exception as e:
            print(f"  [warn] {name} chart failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend/api"))

    p = argparse.ArgumentParser(description="Generate dissertation figures")
    p.add_argument("--figures-dir", default="dissertation/figures")
    p.add_argument("--perframe-dir", default="eval_output/dissertation/perframe")
    p.add_argument("--probing-dir", default="eval_output/dissertation/probing_vl")
    p.add_argument("--gt", default="eval_output/dissertation/db_grounded/18_db_ground_truth.json")
    p.add_argument("--provider", default="gemini")
    args = p.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating dissertation figures...")

    print("\n[1/6] Visual condition comparison")
    fig_visual_conditions(Path(args.perframe_dir), figures_dir, provider=args.provider)

    print("\n[2/6] Linear probing results (probe vs prompting by modality)")
    fig_linear_probing(Path(args.probing_dir), figures_dir)

    print("\n[3/6] Random baseline comparison")
    fig_random_baseline(Path(args.probing_dir), figures_dir)

    print("\n[4/6] Layer-wise probe F1 curves")
    fig_layerwise(Path(args.probing_dir), figures_dir)

    print("\n[5–7/6] Example rendered charts (Analysis 18)")
    fig_example_charts(Path(args.gt), figures_dir)

    print(f"\nDone. Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
