"""Grounding analysis suite for dissertation evaluation.

Three analyses that tell the "grounding reduces hallucination" story:
  A. Claim Type Distribution Shift  — baseline vs grounded conditions
  B. Hallucination Stress Test      — perturbed analytics faithfulness
  C. Unverifiable Claim Categorisation — taxonomy + future-work map

Usage (from backend/ directory):
    python3 -m evaluation.grounding_analysis \\
        --baseline-dir ../eval_output/grounding/ \\
        --grounded-dir ../eval_output/phase15/grounding/10/ \\
        --analytics ../eval_output/phase12/10_analytics.json \\
        --provider openai \\
        --output ../eval_output/phase15/analysis/
"""

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup (runs from backend/ directory) ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation._common import (
    ensure_output_dir,
    load_analytics,
    save_figure,
    save_latex_table,
)
from evaluation.llm_grounding import (
    FORMATTERS,
    _ANALYSIS_TYPES,
    _run_provider,
    extract_claims,
    verify_claim,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis A — Claim Type Distribution Shift
# ═══════════════════════════════════════════════════════════════════════════════

_CLAIM_TYPES = ("numeric", "qualitative", "comparative", "entity_reference")


def load_claim_type_distribution(artifacts_dir: str) -> dict[str, int]:
    """Load all artifact JSONs from artifacts_dir, count claim_type occurrences.

    Returns {numeric: N, qualitative: N, comparative: N, entity_reference: N}.
    Missing or empty directories produce zero counts with a printed warning.
    """
    counts: dict[str, int] = {ct: 0 for ct in _CLAIM_TYPES}
    adir = Path(artifacts_dir)
    if not adir.exists():
        print(f"  [warn] artifacts directory not found: {adir} — skipping")
        return counts

    json_files = list(adir.glob("*.json"))
    if not json_files:
        print(f"  [warn] no JSON files in {adir} — skipping")
        return counts

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] could not read {jf.name}: {exc}")
            continue

        # Each artifact may contain claims at top level or nested under keys
        vr_list = data.get("verification_results", [])
        if not vr_list:
            # Try claims list
            vr_list = data.get("claims", [])
        for item in vr_list:
            ct = item.get("claim_type", "")
            if ct in counts:
                counts[ct] += 1

    return counts


def plot_claim_type_shift(
    baseline_dist: dict[str, int],
    grounded_dist: dict[str, int],
    output_dir: str,
) -> None:
    """Stacked bar chart: two bars (baseline vs grounded), stacked by claim type."""
    claim_types = list(_CLAIM_TYPES)
    colours = ["#4f86c6", "#e07b54", "#6aab6a", "#c471cd"]

    baseline_vals = np.array([baseline_dist.get(ct, 0) for ct in claim_types], dtype=float)
    grounded_vals = np.array([grounded_dist.get(ct, 0) for ct in claim_types], dtype=float)

    # Normalise to proportions (guard divide-by-zero)
    b_total = baseline_vals.sum() or 1.0
    g_total = grounded_vals.sum() or 1.0
    baseline_pct = baseline_vals / b_total * 100
    grounded_pct = grounded_vals / g_total * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.array([0.0, 1.0])
    bar_w = 0.55
    bottoms = np.zeros(2)

    for i, (ct, col) in enumerate(zip(claim_types, colours)):
        heights = np.array([baseline_pct[i], grounded_pct[i]])
        bars = ax.bar(x, heights, bar_w, bottom=bottoms, color=col, label=ct.replace("_", " ").title())
        for xi, (h, b) in enumerate(zip(heights, bottoms)):
            if h > 2:
                ax.text(xi, b + h / 2, f"{h:.1f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([f"Baseline\n(n={int(b_total)})", f"Grounded\n(n={int(g_total)})"], fontsize=11)
    ax.set_ylabel("Claim type proportion (%)")
    ax.set_title("Claim Type Distribution: Baseline vs Grounded")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    save_figure(fig, "claim_type_shift_stacked_bar", output_dir)

    # LaTeX table
    rows = []
    for ct in claim_types:
        b_n = baseline_dist.get(ct, 0)
        g_n = grounded_dist.get(ct, 0)
        rows.append([
            ct.replace("_", " ").title(),
            b_n, f"{b_n / b_total * 100:.1f}\\%",
            g_n, f"{g_n / g_total * 100:.1f}\\%",
        ])
    rows.append(["Total", int(b_total), "100\\%", int(g_total), "100\\%"])
    save_latex_table(
        headers=["Claim Type", "Baseline N", "Baseline \\%", "Grounded N", "Grounded \\%"],
        rows=rows,
        caption="Claim type distribution shift between baseline (Phase 14) and grounded (Phase 12) conditions.",
        name="claim_type_shift",
        output_dir=output_dir,
        label="tab:claim_type_shift",
    )
    print(f"  [A] Saved claim_type_shift.tex + claim_type_shift_stacked_bar.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis B — Hallucination Stress Test
# ═══════════════════════════════════════════════════════════════════════════════

def perturb_analytics(analytics: dict, noise_factor: float = 0.3, seed: int = 42) -> dict:
    """Deep-copy analytics and perturb selected numeric fields.

    Only walks:
      - tactical.summary.*  (all numeric values)
      - possession.team_1_percentage
      - possession.team_2_percentage

    Possession percentages are clamped to [0, 100].
    The original dict is never mutated.
    """
    rng = random.Random(seed)
    perturbed = copy.deepcopy(analytics)

    tactical = perturbed.get("tactical")
    if isinstance(tactical, dict):
        summary = tactical.get("summary", {})
        if isinstance(summary, dict):
            for key, val in summary.items():
                if isinstance(val, (int, float)):
                    factor = 1.0 + rng.uniform(-noise_factor, noise_factor)
                    summary[key] = val * factor

    possession = perturbed.get("possession")
    if isinstance(possession, dict):
        for pct_key in ("team_1_percentage", "team_2_percentage"):
            if pct_key in possession and isinstance(possession[pct_key], (int, float)):
                factor = 1.0 + rng.uniform(-noise_factor, noise_factor)
                raw = possession[pct_key] * factor
                possession[pct_key] = max(0.0, min(100.0, raw))

    return perturbed


def _count_perturbed_faithful(artifacts_dir: Path, perturbed_analytics: dict) -> dict[str, Any]:
    """Check how many numeric claims in a run match the *perturbed* values.

    Returns {grounding_rate, n_claims, faithfulness}.
    """
    import re

    def _parse_numeric(val_str: str) -> float | None:
        try:
            return float(re.sub(r"[,%a-zA-Z/\s]", "", str(val_str)))
        except (ValueError, TypeError):
            return None

    def _get_nested(d: dict, dotpath: str) -> Any:
        cur = d
        for p in dotpath.split("."):
            if isinstance(cur, dict):
                cur = cur.get(p)
            elif isinstance(cur, list) and p.isdigit():
                cur = cur[int(p)] if int(p) < len(cur) else None
            else:
                return None
            if cur is None:
                return None
        return cur

    json_files = list(artifacts_dir.glob("*.json")) if artifacts_dir.exists() else []
    total_numeric = 0
    faithful = 0
    total_claims = 0
    verified_sum = 0

    for jf in json_files:
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        score = data.get("score", {})
        total_claims += score.get("total_claims", 0)
        verified_sum += score.get("verified", 0)

        for item in data.get("verification_results", []):
            if item.get("claim_type") != "numeric":
                continue
            claimed_num = _parse_numeric(item.get("referenced_value", ""))
            if claimed_num is None:
                continue
            actual_perturbed = _get_nested(perturbed_analytics, item.get("referenced_metric", ""))
            if actual_perturbed is None or not isinstance(actual_perturbed, (int, float)):
                continue
            total_numeric += 1
            if abs(actual_perturbed - claimed_num) <= max(abs(actual_perturbed) * 0.05, 0.5):
                faithful += 1

    grounding_rate = (verified_sum / total_claims) if total_claims else 0.0
    faithfulness = (faithful / total_numeric) if total_numeric else 0.0
    return {
        "grounding_rate": grounding_rate,
        "n_claims": total_claims,
        "faithful_numeric": faithful,
        "total_numeric": total_numeric,
        "faithfulness": faithfulness,
    }


async def run_stress_test(
    analytics_path: str,
    provider_name: str,
    output_dir: str,
) -> dict[str, dict]:
    """Run three conditions and measure faithfulness under perturbation.

    Conditions:
      1. original       — real analytics
      2. perturbed      — analytics with 30% noise on tactical.summary + possession pct
      3. empty_tactical — analytics with tactical key set to None

    Returns {condition: {grounding_rate, n_claims, faithfulness (perturbed only)}}.
    """
    analytics = load_analytics(analytics_path)
    perturbed = perturb_analytics(analytics, noise_factor=0.3, seed=42)
    empty_tactical = copy.deepcopy(analytics)
    empty_tactical["tactical"] = None

    conditions: dict[str, dict] = {
        "original": analytics,
        "perturbed": perturbed,
        "empty_tactical": empty_tactical,
    }

    results: dict[str, dict] = {}

    for condition_name, cond_analytics in conditions.items():
        cond_out = str(Path(output_dir) / f"stress_{condition_name}")
        ensure_output_dir(cond_out)
        print(f"  [B] Running condition '{condition_name}' ...")
        try:
            await _run_provider(provider_name, cond_analytics, cond_out)
        except Exception as exc:
            print(f"  [B][warn] condition '{condition_name}' failed: {type(exc).__name__}: {exc!s:.120}")
            results[condition_name] = {"grounding_rate": 0.0, "n_claims": 0, "faithfulness": None}
            continue

        artifacts_dir = Path(cond_out) / "artifacts"
        if condition_name == "perturbed":
            stats = _count_perturbed_faithful(artifacts_dir, perturbed)
        else:
            # Compute grounding rate from saved artifacts
            json_files = list(artifacts_dir.glob("*.json")) if artifacts_dir.exists() else []
            total_claims = 0
            verified_sum = 0
            for jf in json_files:
                try:
                    data = json.loads(jf.read_text())
                    sc = data.get("score", {})
                    total_claims += sc.get("total_claims", 0)
                    verified_sum += sc.get("verified", 0)
                except (json.JSONDecodeError, OSError):
                    continue
            grounding_rate = (verified_sum / total_claims) if total_claims else 0.0
            stats = {"grounding_rate": grounding_rate, "n_claims": total_claims, "faithfulness": None}

        results[condition_name] = stats
        print(f"       grounding_rate={stats['grounding_rate']:.1%}  n_claims={stats['n_claims']}"
              + (f"  faithfulness={stats['faithfulness']:.1%}" if stats.get("faithfulness") is not None else ""))

    # LaTeX: summary table
    rows = []
    for cname, st in results.items():
        faith_str = f"{st['faithfulness']:.1%}" if st.get("faithfulness") is not None else "N/A"
        rows.append([
            cname.replace("_", " ").title(),
            st["n_claims"],
            f"{st['grounding_rate']:.1%}",
            faith_str,
        ])
    save_latex_table(
        headers=["Condition", "Claims", "Grounding Rate", "Faithfulness (perturbed)"],
        rows=rows,
        caption="Stress test results across three analytics conditions.",
        name="stress_test_results",
        output_dir=output_dir,
        label="tab:stress_test_results",
    )

    # LaTeX: faithfulness detail for perturbed condition
    perturbed_stats = results.get("perturbed", {})
    faith_rows = [[
        perturbed_stats.get("faithful_numeric", 0),
        perturbed_stats.get("total_numeric", 0),
        f"{perturbed_stats.get('faithfulness', 0):.1%}" if perturbed_stats.get("faithfulness") is not None else "N/A",
    ]]
    save_latex_table(
        headers=["Faithful Numeric Claims", "Total Numeric Claims", "Faithfulness Rate"],
        rows=faith_rows,
        caption="LLM faithfulness to perturbed analytics values (perturbed condition only).",
        name="stress_test_faithfulness",
        output_dir=output_dir,
        label="tab:stress_test_faithfulness",
    )
    print(f"  [B] Saved stress_test_results.tex + stress_test_faithfulness.tex")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis C — Unverifiable Claim Categorisation
# ═══════════════════════════════════════════════════════════════════════════════

UNVERIFIABLE_CATEGORIES: dict[str, list[str]] = {
    "spatial_zone": [
        "flank", "wide area", "final third", "half-space", "channel", "wing", "left", "right",
    ],
    "temporal_sequence": [
        "counter-attack", "transition", "build-up", "break", "recovery",
    ],
    "subjective_quality": [
        "effective", "dominant", "impressive", "dynamic", "solid", "strong", "poor",
    ],
    "tactical_intent": [
        "exploit", "utilize", "utilise", "strategy", "approach", "philosophy", "attempt",
    ],
    "event_specific": [
        "goal", "shot", "cross", "dribble", "tackle", "header", "save",
    ],
}

FUTURE_WORK_MAP: dict[str, str] = {
    "spatial_zone": "Pitch zone segmentation (divide pitch into thirds/channels)",
    "temporal_sequence": "Event sequence matching (chain possession events by timestamp)",
    "subjective_quality": "Relative metric ranking (compare against historical baselines)",
    "tactical_intent": "Intent inference from event sequences (not directly verifiable)",
    "event_specific": "Frame-precise event timestamps in pipeline output",
}


def categorize_unverifiable(claims: list[dict]) -> dict[str, list[dict]]:
    """Filter to verdict='unverifiable', keyword-match into categories.

    A claim can only match one category (first match wins).
    Returns {category_name: [claim, ...], "other": [...]}.
    """
    buckets: dict[str, list[dict]] = {cat: [] for cat in UNVERIFIABLE_CATEGORIES}
    buckets["other"] = []

    for claim in claims:
        if claim.get("verdict") != "unverifiable":
            continue
        text_lower = (claim.get("text", "") + " " + claim.get("source_sentence", "")).lower()
        matched = False
        for category, keywords in UNVERIFIABLE_CATEGORIES.items():
            if any(kw in text_lower for kw in keywords):
                buckets[category].append(claim)
                matched = True
                break
        if not matched:
            buckets["other"].append(claim)

    return buckets


def plot_unverifiable_pie(category_counts: dict[str, int], output_dir: str) -> None:
    """Pie chart of unverifiable claim categories."""
    labels = []
    sizes = []
    for cat, cnt in category_counts.items():
        if cnt > 0:
            labels.append(cat.replace("_", "\n").title())
            sizes.append(cnt)

    if not sizes:
        print("  [C][warn] no unverifiable claims found — skipping pie chart")
        return

    colours = ["#4f86c6", "#e07b54", "#6aab6a", "#c471cd", "#f0c040", "#888888"]
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colours[: len(sizes)],
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140,
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Unverifiable Claim Categories")
    fig.tight_layout()
    save_figure(fig, "unverifiable_pie", output_dir)


def run_analysis_c(artifacts_dirs: list[str], output_dir: str) -> None:
    """Load claims from all artifact dirs, categorise unverifiable ones, produce outputs."""
    all_claims: list[dict] = []
    for adir_str in artifacts_dirs:
        adir = Path(adir_str)
        if not adir.exists():
            print(f"  [C][warn] {adir} does not exist — skipping")
            continue
        for jf in adir.glob("*.json"):
            try:
                data = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # Merge claim info with verdict from verification_results
            vr_map = {
                vr.get("text", ""): vr
                for vr in data.get("verification_results", [])
            }
            for claim in data.get("claims", []):
                merged = dict(claim)
                vr = vr_map.get(claim.get("text", ""), {})
                merged["verdict"] = vr.get("verdict", "unverifiable")
                merged["source_sentence"] = merged.get("source_sentence", "") or vr.get("source_sentence", "")
                all_claims.append(merged)

    buckets = categorize_unverifiable(all_claims)
    category_counts = {cat: len(items) for cat, items in buckets.items()}
    total_unverifiable = sum(category_counts.values())
    print(f"  [C] Total unverifiable claims: {total_unverifiable}")

    # Breakdown table
    rows = []
    for cat in list(UNVERIFIABLE_CATEGORIES.keys()) + ["other"]:
        cnt = category_counts.get(cat, 0)
        pct = cnt / total_unverifiable * 100 if total_unverifiable else 0.0
        rows.append([
            cat.replace("_", " ").title(),
            cnt,
            f"{pct:.1f}\\%",
        ])
    rows.append(["Total", total_unverifiable, "100\\%"])
    save_latex_table(
        headers=["Category", "Count", "Proportion"],
        rows=rows,
        caption="Breakdown of unverifiable claims by linguistic category.",
        name="unverifiable_breakdown",
        output_dir=output_dir,
        label="tab:unverifiable_breakdown",
    )

    # Future work mapping table
    fw_rows = [
        [
            cat.replace("_", " ").title(),
            category_counts.get(cat, 0),
            FUTURE_WORK_MAP.get(cat, ""),
        ]
        for cat in UNVERIFIABLE_CATEGORIES
    ]
    save_latex_table(
        headers=["Category", "Count", "Recommended Extension"],
        rows=fw_rows,
        caption="Future work extensions that would make each unverifiable category verifiable.",
        name="future_work_mapping",
        output_dir=output_dir,
        label="tab:future_work_mapping",
    )

    plot_unverifiable_pie(category_counts, output_dir)
    print(f"  [C] Saved unverifiable_breakdown.tex + future_work_mapping.tex + unverifiable_pie.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

async def _main_async(args: argparse.Namespace) -> None:
    output_dir = args.output
    ensure_output_dir(output_dir)

    print("\n=== Grounding Analysis Suite ===")

    # ── Analysis A ────────────────────────────────────────────────────────────
    print("\n[Analysis A] Claim Type Distribution Shift")
    baseline_artifacts = str(Path(args.baseline_dir) / "artifacts")
    grounded_artifacts = str(Path(args.grounded_dir) / "artifacts")
    baseline_dist = load_claim_type_distribution(baseline_artifacts)
    grounded_dist = load_claim_type_distribution(grounded_artifacts)
    print(f"  Baseline: {baseline_dist}")
    print(f"  Grounded: {grounded_dist}")
    plot_claim_type_shift(baseline_dist, grounded_dist, output_dir)

    # ── Analysis B ────────────────────────────────────────────────────────────
    print("\n[Analysis B] Hallucination Stress Test")
    stress_results = await run_stress_test(
        analytics_path=args.analytics,
        provider_name=args.provider,
        output_dir=output_dir,
    )

    # ── Analysis C ────────────────────────────────────────────────────────────
    print("\n[Analysis C] Unverifiable Claim Categorisation")
    # Collect all artifact dirs from baseline and grounded inputs
    artifact_dirs = [baseline_artifacts, grounded_artifacts]
    run_analysis_c(artifact_dirs, output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grounding analysis suite — three analyses for dissertation evaluation",
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Directory of baseline (Phase 14) grounding run (contains artifacts/ subdir)",
    )
    parser.add_argument(
        "--grounded-dir",
        required=True,
        help="Directory of grounded (Phase 12) grounding run (contains artifacts/ subdir)",
    )
    parser.add_argument(
        "--analytics",
        required=True,
        help="Path to *_analytics.json for the stress test",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["gemini", "openai", "huggingface"],
        help="LLM provider for Analysis B stress test",
    )
    parser.add_argument(
        "--output",
        default="eval_output/phase15/analysis",
        help="Output directory for all tables and figures",
    )
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
