"""Data ablation study: proving CV perception data improves LLM tactical analysis.

Systematically removes categories of analytics input and measures grounding/QA
degradation at each level.  Commentary is generated from *stripped* data but
verified against the *full* analytics — so condition F (zero data) measures pure
hallucination while condition A (full data) measures the grounded baseline.

Methodology inspired by:
  - GameSight (2604.00057) — component-wise ablation
  - LLM-Commentator (Cook & Karakus 2024) — structured event data baseline
  - TimeSoccer (2504.17365) — motion-aware frame selection ablation
  - TacticAI (2310.10553) — expert evaluation protocol

Conditions:
  A — full analytics (all perception data)
  B — no events (remove CV event detection)
  C — no kinematics (remove speed/distance tracking)
  D — no tactical (remove team shape metrics)  [auto-skipped if absent]
  E — possession only (minimal perception)
  F — no analytics (zero grounding — hallucination baseline)
  G — vision only (raw video frames, no structured data)  [requires --video]

Usage:
    # Quick A vs F comparison
    python -m backend.evaluation.ablation_data \\
        --analytics eval_output/10_analytics.json \\
        --provider openai \\
        --output eval_output/ablation/ \\
        --conditions A,F --skip-qa

    # Full run with VLM condition
    python -m backend.evaluation.ablation_data \\
        --analytics eval_output/10_analytics.json \\
        --provider openai \\
        --video path/to/video.mp4 \\
        --output eval_output/ablation/
"""

import argparse
import asyncio
import copy
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import ensure_output_dir, load_analytics, save_figure, save_latex_table
from .llm_grounding import (
    Claim,
    VerificationResult,
    _ANALYSIS_TYPES,
    compute_grounding_score,
    extract_claims,
    verify_claim,
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class AblationCondition:
    id: str
    name: str
    description: str
    strip_fn: Callable[[dict], dict]
    requires_video: bool = False
    include_insights: bool = True  # False = ablation condition H (no MatchInsights)
    strip_fewshot: bool = False     # True = ablation condition I (no few-shot examples)
    strip_metric_defs: bool = False  # True = ablation condition J (no metric definitions)


@dataclass
class ConditionResult:
    condition_id: str
    condition_name: str
    scores: dict[str, dict] = field(default_factory=dict)      # {analysis_type -> score dict}
    commentaries: dict[str, str] = field(default_factory=dict)  # {analysis_type -> text}
    claim_counts: dict[str, int] = field(default_factory=dict)  # {analysis_type -> n_claims}


# ── Strip functions ──────────────────────────────────────────────────────────


def _strip_full(analytics: dict) -> dict:
    return copy.deepcopy(analytics)


def _strip_no_events(analytics: dict) -> dict:
    d = copy.deepcopy(analytics)
    d.pop("events", None)
    return d


def _strip_no_kinematics(analytics: dict) -> dict:
    d = copy.deepcopy(analytics)
    for k in ("player_kinematics", "ball_kinematics", "ball_metrics", "ball_path"):
        d.pop(k, None)
    return d


def _strip_no_tactical(analytics: dict) -> dict:
    d = copy.deepcopy(analytics)
    d.pop("tactical", None)
    return d


def _strip_possession_only(analytics: dict) -> dict:
    keep = {"possession", "fps", "team_colors", "homography_available"}
    d = copy.deepcopy(analytics)
    for k in list(d.keys()):
        if k not in keep:
            del d[k]
    return d


def _strip_no_analytics(_analytics: dict) -> dict:
    return {"fps": 25}


def _strip_vision_only(_analytics: dict) -> dict:
    """No structured data — VLM sees raw frames only."""
    return {"fps": 25}


# ── Condition registry ───────────────────────────────────────────────────────


ABLATION_CONDITIONS: dict[str, AblationCondition] = {
    "A": AblationCondition("A", "full", "Full analytics (all perception data)", _strip_full),
    "B": AblationCondition("B", "no_events", "No CV event detection", _strip_no_events),
    "C": AblationCondition("C", "no_kinematics", "No speed/distance tracking", _strip_no_kinematics),
    "D": AblationCondition("D", "no_tactical", "No team shape metrics", _strip_no_tactical),
    "E": AblationCondition("E", "possession_only", "Possession data only", _strip_possession_only),
    "F": AblationCondition("F", "no_analytics", "Zero grounding (hallucination baseline)", _strip_no_analytics),
    "G": AblationCondition("G", "vision_only", "Raw video frames only (VLM)", _strip_vision_only, requires_video=True),
    "H": AblationCondition("H", "no_insights", "Full data, no pre-interpretation (MatchInsights A/B baseline)",
                           _strip_full, include_insights=False),
    "I": AblationCondition("I", "no_fewshot", "Full data + insights, no few-shot examples in prompt",
                           _strip_full, strip_fewshot=True),
    "J": AblationCondition("J", "no_metric_defs", "Full data + insights, no metric definitions in prompt",
                           _strip_full, strip_metric_defs=True),
}


# ── Core runner ──────────────────────────────────────────────────────────────


async def run_condition(
    condition: AblationCondition,
    full_analytics: dict,
    provider_name: str,
    output_dir: str,
    video_path: str | None = None,
) -> ConditionResult:
    """Run one ablation condition across all analysis types.

    Generates commentary from stripped analytics (or video-only for G),
    then verifies claims against the full analytics.
    """
    from services.llm_providers import get_provider
    from services.tactical import TacticalAnalyzer

    provider = get_provider(provider_name)
    judge = get_provider(provider_name)
    stripped = condition.strip_fn(full_analytics)

    result = ConditionResult(condition.id, condition.name)
    cond_dir = Path(output_dir) / condition.name
    cond_dir.mkdir(parents=True, exist_ok=True)

    # For vision-only, pass video to TacticalAnalyzer with empty analytics
    use_video = video_path if condition.requires_video else None

    for atype in _ANALYSIS_TYPES:
        label = f"[{condition.id}:{condition.name}] {atype}"
        print(f"  {label} ...", end=" ", flush=True)

        # Generate commentary from stripped data
        # For prompt ablation conditions (I, J), strip sections from system prompt
        prompt_override = None
        if condition.strip_fewshot or condition.strip_metric_defs:
            import re
            from services.tactical import SYSTEM_PROMPTS
            base = SYSTEM_PROMPTS.get(atype, SYSTEM_PROMPTS["match_overview"])
            if condition.strip_fewshot:
                base = re.sub(r"\n## Example Output.*", "", base, flags=re.DOTALL)
            if condition.strip_metric_defs:
                base = re.sub(r"\n## Metric Definitions.*?(?=\n## |\nYour task|\Z)", "\n", base, flags=re.DOTALL)
            prompt_override = base

        analyzer = TacticalAnalyzer(provider=provider)
        response = await analyzer.analyze(
            stripped, atype, video_path=use_video,
            include_insights=condition.include_insights,
            system_prompt_override=prompt_override,
        )
        commentary = response["content"]

        # Extract claims and verify against FULL analytics
        claims = await extract_claims(commentary, judge)
        verdicts = [verify_claim(c, full_analytics) for c in claims]
        score = compute_grounding_score(verdicts, commentary=commentary, analytics=full_analytics)

        result.scores[atype] = score
        result.commentaries[atype] = commentary
        result.claim_counts[atype] = len(claims)

        gr = score.get("grounding_rate", 0)
        hr = score.get("hallucination_rate", 0)
        print(f"grounding={gr:.1%}  hallucination={hr:.1%}  claims={len(claims)}")

        # Save per-condition artifact
        artifact = {
            "condition": condition.name,
            "analysis_type": atype,
            "commentary": commentary,
            "claims": [{"text": c.text, "type": c.claim_type, "metric": c.referenced_metric} for c in claims],
            "verdicts": [{"claim": v.claim.text, "verdict": v.verdict, "explanation": v.explanation} for v in verdicts],
            "score": score,
        }
        (cond_dir / f"{atype}.json").write_text(json.dumps(artifact, indent=2, default=str))

    return result


# ── Delta computation ────────────────────────────────────────────────────────


def compute_deltas(
    results: dict[str, ConditionResult],
    baseline_id: str = "A",
) -> dict[str, dict]:
    """Compute grounding/hallucination deltas vs baseline for each condition."""
    baseline = results.get(baseline_id)
    if not baseline:
        return {}

    deltas = {}
    for cid, cr in results.items():
        if cid == baseline_id:
            continue
        per_type = {}
        for atype in _ANALYSIS_TYPES:
            base_gr = baseline.scores.get(atype, {}).get("grounding_rate", 0.0)
            base_hr = baseline.scores.get(atype, {}).get("hallucination_rate", 0.0)
            cond_gr = cr.scores.get(atype, {}).get("grounding_rate", 0.0)
            cond_hr = cr.scores.get(atype, {}).get("hallucination_rate", 0.0)
            per_type[atype] = {
                "baseline_grounding": base_gr,
                "condition_grounding": cond_gr,
                "grounding_delta_pp": (cond_gr - base_gr) * 100,
                "baseline_hallucination": base_hr,
                "condition_hallucination": cond_hr,
                "hallucination_delta_pp": (cond_hr - base_hr) * 100,
            }
        # Average across types
        avg_gr_delta = np.mean([v["grounding_delta_pp"] for v in per_type.values()])
        avg_hr_delta = np.mean([v["hallucination_delta_pp"] for v in per_type.values()])
        deltas[f"{cid}_vs_{baseline_id}"] = {
            "condition": cr.condition_name,
            "per_type": per_type,
            "avg_grounding_delta_pp": float(avg_gr_delta),
            "avg_hallucination_delta_pp": float(avg_hr_delta),
        }
    return deltas


# ── Visualization ────────────────────────────────────────────────────────────


def plot_grounding_bars(results: dict[str, ConditionResult], output_dir: str) -> None:
    """Grouped bar chart: conditions × analysis_types grounding rates."""
    cond_ids = sorted(results.keys())
    n_types = len(_ANALYSIS_TYPES)
    n_conds = len(cond_ids)
    x = np.arange(n_types)
    width = 0.8 / n_conds

    colors = ["#27ae60", "#2980b9", "#8e44ad", "#f39c12", "#e74c3c", "#95a5a6", "#1abc9c"]
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, cid in enumerate(cond_ids):
        cr = results[cid]
        rates = [cr.scores.get(at, {}).get("grounding_rate", 0) * 100 for at in _ANALYSIS_TYPES]
        offset = (i - n_conds / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=f"{cid}: {cr.condition_name}",
                      color=colors[i % len(colors)], edgecolor="white")
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=6)

    ax.set_ylabel("Grounding Rate (%)")
    ax.set_title("Data Ablation: Grounding Rate by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels([at.replace("_", " ").title() for at in _ANALYSIS_TYPES], rotation=12, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    save_figure(fig, "ablation_data_grounding_bars", output_dir)


def plot_degradation_curve(results: dict[str, ConditionResult], output_dir: str) -> None:
    """Line plot: average grounding rate as data is progressively removed."""
    # Order conditions by data richness: F → E → D → C → B → A
    richness_order = ["F", "E", "D", "C", "B", "A"]
    available = [c for c in richness_order if c in results]
    if len(available) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for atype in _ANALYSIS_TYPES:
        rates = [results[c].scores.get(atype, {}).get("grounding_rate", 0) * 100 for c in available]
        ax.plot(range(len(available)), rates, marker="o", linewidth=2,
                label=atype.replace("_", " ").title())

    # Average line
    avg_rates = [np.mean([results[c].scores.get(at, {}).get("grounding_rate", 0) * 100
                          for at in _ANALYSIS_TYPES]) for c in available]
    ax.plot(range(len(available)), avg_rates, marker="s", linewidth=3, color="black",
            linestyle="--", label="Average")

    labels = [f"{c}: {results[c].condition_name}" for c in available]
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Grounding Rate (%)")
    ax.set_xlabel("Data Richness →")
    ax.set_title("Grounding Degradation Curve: Less Data → More Hallucination")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "ablation_data_degradation_curve", output_dir)


def plot_hallucination_heatmap(results: dict[str, ConditionResult], output_dir: str) -> None:
    """Heatmap: conditions × analysis_types hallucination rates."""
    cond_ids = sorted(results.keys())
    matrix = np.array([
        [results[c].scores.get(at, {}).get("hallucination_rate", 0) * 100 for at in _ANALYSIS_TYPES]
        for c in cond_ids
    ])

    fig, ax = plt.subplots(figsize=(8, max(4, len(cond_ids) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(50, matrix.max() + 5))
    ax.set_xticks(range(len(_ANALYSIS_TYPES)))
    ax.set_xticklabels([at.replace("_", " ").title() for at in _ANALYSIS_TYPES], rotation=20, ha="right")
    ax.set_yticks(range(len(cond_ids)))
    ax.set_yticklabels([f"{c}: {results[c].condition_name}" for c in cond_ids])
    for i in range(len(cond_ids)):
        for j in range(len(_ANALYSIS_TYPES)):
            ax.text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > 25 else "black")
    fig.colorbar(im, ax=ax, label="Hallucination Rate (%)")
    ax.set_title("Hallucination Rate by Condition and Analysis Type")
    fig.tight_layout()
    save_figure(fig, "ablation_data_hallucination_heatmap", output_dir)


# ── LaTeX tables ─────────────────────────────────────────────────────────────


def build_tables(
    results: dict[str, ConditionResult],
    deltas: dict[str, dict],
    output_dir: str,
) -> None:
    """Build comparison + delta LaTeX tables."""
    cond_ids = sorted(results.keys())

    # Comparison table: rows=conditions, cols=avg grounding/hallucination/claims
    headers = ["Condition", "Avg Grounding (%)", "Avg Hallucination (%)", "Total Claims"]
    rows: list[list[Any]] = []
    for cid in cond_ids:
        cr = results[cid]
        avg_gr = np.mean([cr.scores.get(at, {}).get("grounding_rate", 0) for at in _ANALYSIS_TYPES]) * 100
        avg_hr = np.mean([cr.scores.get(at, {}).get("hallucination_rate", 0) for at in _ANALYSIS_TYPES]) * 100
        total_claims = sum(cr.claim_counts.get(at, 0) for at in _ANALYSIS_TYPES)
        rows.append([f"{cid}: {cr.condition_name}", f"{avg_gr:.1f}", f"{avg_hr:.1f}", str(total_claims)])

    save_latex_table(
        headers=headers, rows=rows,
        caption="Data ablation: average grounding and hallucination rates per condition.",
        name="ablation_data_comparison", output_dir=output_dir,
        label="tab:ablation_data_comparison",
    )

    # Delta table: rows=non-baseline conditions, cols=delta vs A
    if deltas:
        d_headers = ["Condition", "Grounding Delta (pp)", "Hallucination Delta (pp)"]
        d_rows = []
        for key, d in deltas.items():
            d_rows.append([
                d["condition"],
                f"{d['avg_grounding_delta_pp']:+.1f}",
                f"{d['avg_hallucination_delta_pp']:+.1f}",
            ])
        save_latex_table(
            headers=d_headers, rows=d_rows,
            caption="Grounding change (pp) when removing perception data components vs full pipeline.",
            name="ablation_data_deltas", output_dir=output_dir,
            label="tab:ablation_data_deltas",
        )


# ── Summary ──────────────────────────────────────────────────────────────────


def print_summary(results: dict[str, ConditionResult], deltas: dict[str, dict]) -> None:
    """Print compact ablation summary to console."""
    cond_ids = sorted(results.keys())
    print(f"\n{'=' * 76}")
    print("DATA ABLATION SUMMARY  (grounding rate %)")
    print(f"{'=' * 76}")
    header = f"{'Condition':<24}" + "".join(f"{at[:12]:>13}" for at in _ANALYSIS_TYPES) + f"{'AVG':>8}"
    print(header)
    print("-" * 76)

    for cid in cond_ids:
        cr = results[cid]
        rates = [cr.scores.get(at, {}).get("grounding_rate", 0) * 100 for at in _ANALYSIS_TYPES]
        avg = np.mean(rates)
        label = f"{cid}: {cr.condition_name}"
        line = f"{label:<24}" + "".join(f"{r:>12.1f}%" for r in rates) + f"{avg:>7.1f}%"
        print(line)

    if deltas:
        print(f"\n{'DELTAS vs A (pp)':<24}" + "".join(f"{at[:12]:>13}" for at in _ANALYSIS_TYPES) + f"{'AVG':>8}")
        print("-" * 76)
        for key, d in sorted(deltas.items()):
            name = d["condition"]
            vals = [d["per_type"].get(at, {}).get("grounding_delta_pp", 0) for at in _ANALYSIS_TYPES]
            avg = d["avg_grounding_delta_pp"]
            line = f"{name:<24}" + "".join(f"{v:>+12.1f}%" for v in vals) + f"{avg:>+7.1f}%"
            print(line)
    print("=" * 76)


# ── Orchestrator ─────────────────────────────────────────────────────────────


async def run_ablation(
    analytics_path: str,
    provider: str,
    output_dir: str,
    conditions: list[str] | None = None,
    video_path: str | None = None,
    skip_qa: bool = False,
    n_runs: int = 1,
) -> dict:
    """Run the full data ablation study.

    Args:
        analytics_path: Path to analytics JSON.
        provider: LLM provider name.
        output_dir: Output directory for results.
        conditions: Subset of condition IDs (e.g. ["A", "F"]). None = all.
        video_path: Optional video for VLM condition G.
        skip_qa: Skip QA benchmark per condition.
        n_runs: Number of runs per condition (default 1). >1 reports mean ± std
            across LLM non-determinism (Sports Intelligence benchmark methodology).

    Returns:
        Complete results dict with per-condition scores + deltas.
    """
    ensure_output_dir(output_dir)
    full_analytics = load_analytics(analytics_path)

    # Select conditions
    selected = list(ABLATION_CONDITIONS.values())
    if conditions:
        valid_ids = set(conditions)
        selected = [c for c in selected if c.id in valid_ids]

    # Auto-skip conditions that need missing data
    if "tactical" not in full_analytics:
        before = len(selected)
        selected = [c for c in selected if c.id != "D"]
        if len(selected) < before:
            print("  Note: Condition D (no_tactical) auto-skipped — 'tactical' key absent from analytics")

    if not video_path:
        before = len(selected)
        selected = [c for c in selected if not c.requires_video]
        if len(selected) < before:
            print("  Note: Condition G (vision_only) skipped — no --video provided")

    print(f"\n{'=' * 60}")
    print("Data Ablation Study")
    print(f"Analytics:  {analytics_path}")
    print(f"Provider:   {provider}")
    print(f"Conditions: {', '.join(c.id + ':' + c.name for c in selected)}")
    print(f"Output:     {output_dir}")
    print(f"{'=' * 60}")

    # Run each condition (optionally multiple times for statistical reliability)
    results: dict[str, ConditionResult] = {}
    run_stats: dict[str, dict] = {}  # {cond_id: {atype: {mean/std/ci95}}}

    for condition in selected:
        print(f"\n--- Condition {condition.id}: {condition.description} ---")
        if n_runs <= 1:
            cr = await run_condition(condition, full_analytics, provider, output_dir, video_path)
            results[condition.id] = cr
        else:
            print(f"  Running {n_runs} times for mean ± std (Sports Intelligence methodology)...")
            all_runs: list[ConditionResult] = []
            for run_i in range(n_runs):
                print(f"  [Run {run_i + 1}/{n_runs}]")
                run_dir = str(Path(output_dir) / f"run_{run_i + 1}")
                cr_i = await run_condition(condition, full_analytics, provider, run_dir, video_path)
                all_runs.append(cr_i)
            # Aggregate: use last run as canonical result, compute stats
            results[condition.id] = all_runs[-1]
            stats_per_type: dict[str, dict] = {}
            for atype in _ANALYSIS_TYPES:
                gr_vals = [r.scores.get(atype, {}).get("grounding_rate", 0) for r in all_runs]
                hr_vals = [r.scores.get(atype, {}).get("hallucination_rate", 0) for r in all_runs]
                mean_gr = float(np.mean(gr_vals))
                std_gr = float(np.std(gr_vals))
                mean_hr = float(np.mean(hr_vals))
                std_hr = float(np.std(hr_vals))
                # 95% CI (t-interval approximation for small n)
                from scipy import stats as scipy_stats
                ci_gr = float(scipy_stats.t.ppf(0.975, df=max(1, n_runs - 1)) * std_gr / (n_runs ** 0.5))
                ci_hr = float(scipy_stats.t.ppf(0.975, df=max(1, n_runs - 1)) * std_hr / (n_runs ** 0.5))
                stats_per_type[atype] = {
                    "grounding_mean": mean_gr, "grounding_std": std_gr, "grounding_ci95": ci_gr,
                    "hallucination_mean": mean_hr, "hallucination_std": std_hr, "hallucination_ci95": ci_hr,
                    "n_runs": n_runs,
                }
                print(f"  {atype}: grounding={mean_gr:.1%} ± {std_gr:.1%} (95%CI ±{ci_gr:.1%}), "
                      f"hallucination={mean_hr:.1%} ± {std_hr:.1%}")
            run_stats[condition.id] = stats_per_type

    # Compute deltas vs baseline
    deltas = compute_deltas(results, baseline_id="A")

    # Build outputs
    print_summary(results, deltas)
    plot_grounding_bars(results, output_dir)
    plot_degradation_curve(results, output_dir)
    plot_hallucination_heatmap(results, output_dir)
    build_tables(results, deltas, output_dir)

    # Save JSON results
    out_data = {
        "_meta": {
            "analytics_path": analytics_path,
            "provider": provider,
            "conditions_run": [c.id for c in selected],
            "n_runs": n_runs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "conditions": {
            cid: {
                "name": cr.condition_name,
                "scores": cr.scores,
                "claim_counts": cr.claim_counts,
            }
            for cid, cr in results.items()
        },
        "deltas": deltas,
        "multi_run_stats": run_stats,
    }
    results_path = Path(output_dir) / "ablation_data_results.json"
    results_path.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"\nResults saved to: {results_path}")

    return out_data


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env", override=True)
    except ImportError:
        pass
    parser = argparse.ArgumentParser(
        description="Data ablation study: does CV perception data improve LLM analysis?"
    )
    parser.add_argument("--analytics", required=True, help="Path to *_analytics.json")
    parser.add_argument("--provider", default="openai", choices=["gemini", "openai", "huggingface", "claude", "groq"])
    parser.add_argument("--output", default="eval_output/ablation/")
    parser.add_argument("--conditions", default=None,
                        help="Comma-separated subset: A,B,C,D,E,F,G (default: all)")
    parser.add_argument("--video", default=None, help="Video path for VLM condition G")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA benchmark per condition")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of runs per condition for mean ± std reporting (default: 1)")

    args = parser.parse_args()

    cond_list = None
    if args.conditions:
        cond_list = [c.strip().upper() for c in args.conditions.split(",")]
        invalid = set(cond_list) - set(ABLATION_CONDITIONS.keys())
        if invalid:
            parser.error(f"Unknown conditions: {invalid}. Valid: {set(ABLATION_CONDITIONS.keys())}")

    asyncio.run(run_ablation(
        analytics_path=args.analytics,
        provider=args.provider,
        output_dir=args.output,
        conditions=cond_list,
        video_path=args.video,
        skip_qa=args.skip_qa,
        n_runs=args.n_runs,
    ))


if __name__ == "__main__":
    main()
