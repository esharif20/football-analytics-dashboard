"""Ablation study: isolating the 3.5x grounding improvement from Phase 14 → Phase 12.

Measures contribution of two components:
  - Tactical metrics alone (Phase 12 data, fallback patched out)
  - Numeric fallback verifier on top of tactical metrics

Three conditions:
  A — Phase 14 baseline analytics (no tactical key)
  B — Phase 12 analytics + tactical metrics, fallback patched to (None, None)
  C — Phase 12 analytics, full pipeline

Usage:
    python3 -m evaluation.ablation_study \\
        --baseline-analytics ../eval_output/10_analytics.json \\
        --tactical-analytics ../eval_output/phase12/10_analytics.json \\
        --provider openai \\
        --output ../eval_output/phase15/ablation/
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

# Run from backend/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import (
    ensure_output_dir,
    load_analytics,
    save_figure,
    save_latex_table,
)
from .llm_grounding import _run_provider, _ANALYSIS_TYPES


# ── Condition runner ──────────────────────────────────────────────────────────

async def run_condition(
    label: str,
    analytics: dict,
    provider: str,
    out: str,
    patch_fallback: bool = False,
) -> dict[str, dict]:
    """Run _run_provider for one ablation condition.

    If patch_fallback=True, patches _search_tactical_summary to return
    (None, None), isolating the contribution of tactical metrics alone
    from the numeric fallback verifier.

    Returns fmt_scores dict keyed by format name -> {analysis_type -> score}.
    """
    condition_out = str(Path(out) / label.lower().replace(" ", "_"))
    ensure_output_dir(condition_out)

    print(f"\n--- Condition {label} (patch_fallback={patch_fallback}) ---")

    if patch_fallback:
        with patch(
            "evaluation.llm_grounding._search_tactical_summary",
            return_value=(None, None),
        ):
            fmt_scores = await _run_provider(provider, analytics, condition_out)
    else:
        fmt_scores = await _run_provider(provider, analytics, condition_out)

    return fmt_scores


# ── Delta computation ─────────────────────────────────────────────────────────

def compute_deltas(
    scores_a: dict,
    scores_b: dict,
    scores_c: dict,
) -> dict:
    """Compute per-analysis-type grounding rate deltas.

    tactical_contribution = B - A  (metrics alone, no fallback)
    fallback_contribution = C - B  (fallback added on top of metrics)

    All scores are extracted from the 'markdown' format, which is the
    production formatter being evaluated.

    Returns:
        {
            analysis_type: {
                "rate_a": float,
                "rate_b": float,
                "rate_c": float,
                "tactical_contribution": float,   # B - A
                "fallback_contribution": float,   # C - B
                "total_improvement": float,       # C - A
            }
        }
    """
    deltas: dict[str, dict] = {}
    for atype in _ANALYSIS_TYPES:
        ra = scores_a.get("markdown", {}).get(atype, {}).get("grounding_rate", 0.0)
        rb = scores_b.get("markdown", {}).get(atype, {}).get("grounding_rate", 0.0)
        rc = scores_c.get("markdown", {}).get(atype, {}).get("grounding_rate", 0.0)
        deltas[atype] = {
            "rate_a": ra,
            "rate_b": rb,
            "rate_c": rc,
            "tactical_contribution": rb - ra,
            "fallback_contribution": rc - rb,
            "total_improvement": rc - ra,
        }
    return deltas


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def build_comparison_table(scores: dict[str, dict], output_dir: str) -> None:
    """LaTeX table: rows=analysis_types, cols=A/B/C grounding rates (markdown format)."""
    headers = ["Analysis Type", "Condition A (%)", "Condition B (%)", "Condition C (%)"]
    rows: list[list[Any]] = []
    for atype in _ANALYSIS_TYPES:
        ra = scores["A"].get("markdown", {}).get(atype, {}).get("grounding_rate", 0.0)
        rb = scores["B"].get("markdown", {}).get(atype, {}).get("grounding_rate", 0.0)
        rc = scores["C"].get("markdown", {}).get(atype, {}).get("grounding_rate", 0.0)
        rows.append([
            atype.replace("_", " ").title(),
            f"{ra * 100:.1f}",
            f"{rb * 100:.1f}",
            f"{rc * 100:.1f}",
        ])
    save_latex_table(
        headers=headers,
        rows=rows,
        caption=(
            "Grounding rate (\\%) per analysis type across ablation conditions. "
            "A: Phase 14 baseline; B: Phase 12 metrics, fallback disabled; "
            "C: Full Phase 12 pipeline."
        ),
        name="ablation_comparison",
        output_dir=output_dir,
        label="tab:ablation_comparison",
    )
    print(f"  Saved ablation_comparison.tex")


def build_contribution_table(deltas: dict, output_dir: str) -> None:
    """LaTeX table: rows=analysis_types, cols=delta_B_minus_A and delta_C_minus_B."""
    headers = [
        "Analysis Type",
        "Tactical Metrics (B-A, pp)",
        "Fallback Verifier (C-B, pp)",
        "Total (C-A, pp)",
    ]
    rows: list[list[Any]] = []
    for atype in _ANALYSIS_TYPES:
        d = deltas.get(atype, {})
        rows.append([
            atype.replace("_", " ").title(),
            f"{d.get('tactical_contribution', 0.0) * 100:+.1f}",
            f"{d.get('fallback_contribution', 0.0) * 100:+.1f}",
            f"{d.get('total_improvement', 0.0) * 100:+.1f}",
        ])
    save_latex_table(
        headers=headers,
        rows=rows,
        caption=(
            "Component contribution to grounding improvement (percentage points). "
            "Tactical Metrics: B minus A isolates phase 12 metric impact. "
            "Fallback Verifier: C minus B isolates numeric fallback contribution."
        ),
        name="ablation_component_contribution",
        output_dir=output_dir,
        label="tab:ablation_component_contribution",
    )
    print(f"  Saved ablation_component_contribution.tex")


# ── Grouped bar chart ─────────────────────────────────────────────────────────

def plot_grouped_bar(scores: dict[str, dict], output_dir: str) -> None:
    """Grouped bar chart: x=analysis_types, groups=A/B/C conditions (markdown format)."""
    n_types = len(_ANALYSIS_TYPES)
    x = np.arange(n_types)
    width = 0.25
    labels = [at.replace("_", " ").title() for at in _ANALYSIS_TYPES]

    rates_a = [scores["A"].get("markdown", {}).get(at, {}).get("grounding_rate", 0.0) * 100
               for at in _ANALYSIS_TYPES]
    rates_b = [scores["B"].get("markdown", {}).get(at, {}).get("grounding_rate", 0.0) * 100
               for at in _ANALYSIS_TYPES]
    rates_c = [scores["C"].get("markdown", {}).get(at, {}).get("grounding_rate", 0.0) * 100
               for at in _ANALYSIS_TYPES]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_a = ax.bar(x - width, rates_a, width, label="A: Baseline (Ph.14)", color="#c0392b", edgecolor="white")
    bars_b = ax.bar(x,          rates_b, width, label="B: Metrics only (Ph.12, no fallback)", color="#2980b9", edgecolor="white")
    bars_c = ax.bar(x + width, rates_c, width, label="C: Full Ph.12 pipeline", color="#27ae60", edgecolor="white")

    ax.bar_label(bars_a, fmt="%.0f%%", padding=2, fontsize=7)
    ax.bar_label(bars_b, fmt="%.0f%%", padding=2, fontsize=7)
    ax.bar_label(bars_c, fmt="%.0f%%", padding=2, fontsize=7)

    ax.set_ylabel("Grounding Rate (%)")
    ax.set_title("Ablation Study: Component Contribution to Grounding Improvement")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    save_figure(fig, "ablation_grouped_bar", output_dir)
    print(f"  Saved ablation_grouped_bar.pdf/.png")


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(deltas: dict) -> None:
    """Print A → B → C progression per analysis type."""
    print("\n" + "=" * 68)
    print("ABLATION SUMMARY  (markdown format, grounding rate %)")
    print("=" * 68)
    header = f"{'Analysis Type':<28} {'A':>6} {'B':>6} {'C':>6}  {'B-A':>6} {'C-B':>6} {'C-A':>6}"
    print(header)
    print("-" * 68)
    for atype in _ANALYSIS_TYPES:
        d = deltas.get(atype, {})
        ra  = d.get("rate_a", 0.0) * 100
        rb  = d.get("rate_b", 0.0) * 100
        rc  = d.get("rate_c", 0.0) * 100
        dba = d.get("tactical_contribution", 0.0) * 100
        dcb = d.get("fallback_contribution", 0.0) * 100
        dca = d.get("total_improvement", 0.0) * 100
        name = atype.replace("_", " ").title()
        print(f"{name:<28} {ra:>5.1f}% {rb:>5.1f}% {rc:>5.1f}%  {dba:>+5.1f}p {dcb:>+5.1f}p {dca:>+5.1f}p")
    print("=" * 68)

    # Averages
    all_dba = [deltas[a]["tactical_contribution"] for a in _ANALYSIS_TYPES if a in deltas]
    all_dcb = [deltas[a]["fallback_contribution"] for a in _ANALYSIS_TYPES if a in deltas]
    all_dca = [deltas[a]["total_improvement"] for a in _ANALYSIS_TYPES if a in deltas]
    if all_dba:
        avg_dba = sum(all_dba) / len(all_dba) * 100
        avg_dcb = sum(all_dcb) / len(all_dcb) * 100
        avg_dca = sum(all_dca) / len(all_dca) * 100
        print(f"\nAverage across all types:")
        print(f"  Tactical metrics contribution (B-A): {avg_dba:+.1f} pp")
        print(f"  Fallback verifier contribution (C-B): {avg_dcb:+.1f} pp")
        print(f"  Total improvement (C-A):              {avg_dca:+.1f} pp")


# ── Main ──────────────────────────────────────────────────────────────────────

async def _main_async(args) -> None:
    output_dir = str(ensure_output_dir(args.output))

    print(f"Loading baseline analytics:  {args.baseline_analytics}")
    print(f"Loading tactical analytics:  {args.tactical_analytics}")
    baseline_analytics = load_analytics(args.baseline_analytics)
    tactical_analytics = load_analytics(args.tactical_analytics)

    provider = args.provider
    scores: dict[str, dict] = {}

    # Condition A — baseline analytics, no patch
    try:
        scores["A"] = await run_condition("A", baseline_analytics, provider, output_dir, patch_fallback=False)
    except Exception as exc:
        print(f"  WARNING: Condition A failed ({type(exc).__name__}: {exc}). Using empty scores.")
        scores["A"] = {}

    # Condition B — tactical analytics, fallback patched to (None, None)
    try:
        scores["B"] = await run_condition("B", tactical_analytics, provider, output_dir, patch_fallback=True)
    except Exception as exc:
        print(f"  WARNING: Condition B failed ({type(exc).__name__}: {exc}). Using empty scores.")
        scores["B"] = {}

    # Condition C — tactical analytics, full pipeline (no patch)
    try:
        scores["C"] = await run_condition("C", tactical_analytics, provider, output_dir, patch_fallback=False)
    except Exception as exc:
        print(f"  WARNING: Condition C failed ({type(exc).__name__}: {exc}). Using empty scores.")
        scores["C"] = {}

    # Persist raw condition scores
    (Path(output_dir) / "ablation_raw_scores.json").write_text(
        json.dumps(
            {cond: {fmt: {at: sc for at, sc in by_at.items()}
                    for fmt, by_at in by_fmt.items()}
             for cond, by_fmt in scores.items()},
            indent=2,
            default=str,
        )
    )

    # Compute deltas
    deltas = compute_deltas(scores["A"], scores["B"], scores["C"])

    # Save tables and chart
    build_comparison_table(scores, output_dir)
    build_contribution_table(deltas, output_dir)
    plot_grouped_bar(scores, output_dir)

    # Print summary
    print_summary(deltas)
    print(f"\nAll outputs written to: {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study isolating tactical metric vs. fallback contributions."
    )
    parser.add_argument(
        "--baseline-analytics",
        required=True,
        help="Phase 14 analytics JSON (no tactical key, or tactical=null).",
    )
    parser.add_argument(
        "--tactical-analytics",
        required=True,
        help="Phase 12 analytics JSON with tactical metrics populated.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["gemini", "openai", "huggingface"],
        help="LLM provider to use for commentary generation and judging.",
    )
    parser.add_argument(
        "--output",
        default="eval_output/phase15/ablation",
        help="Output directory for tables, charts, and raw scores.",
    )
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
