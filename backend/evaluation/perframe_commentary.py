"""Per-frame data commentary generation and evaluation.

Compares two conditions for Analysis 18 (83%/17% possession — richest tactical clip):

  BASELINE  — aggregate analytics only (equivalent to R-ALL in db_eval_runner.py)
  PERFRAME  — aggregate analytics + per-frame spatial evidence section

The per-frame section is built by PerFrameContextFormatter (tactical.py) from the
db_extractor ground truth JSON. Claim verification runs both the standard JSON layer
and the DB-grounded layer so the improvement in spatial grounding is measurable.

Usage:
    python3 -m backend.evaluation.perframe_commentary \\
        --analysis-id 18 \\
        --ground-truth eval_output/dissertation/db_grounded/18_db_ground_truth.json \\
        --provider openai \\
        --n-runs 1 \\
        --output eval_output/dissertation/perframe/
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))  # backend/api/

from ._common import load_db_ground_truth, ensure_output_dir
from .llm_grounding import extract_claims, extract_claims_stable, verify_claim
from .db_eval_runner import verify_claim_db_grounded

logger = logging.getLogger(__name__)

ANALYSIS_TYPES = [
    "match_overview",
    "tactical_deep_dive",
    "event_analysis",
    "player_spotlight",
]


# ── Context builders ──────────────────────────────────────────────────────────


def build_baseline_context(analytics: dict) -> str:
    """Aggregate-only context — identical to the R-ALL condition."""
    from services.tactical import GroundingFormatter
    return GroundingFormatter.format(analytics, include_insights=True)


def build_perframe_context(analytics: dict, db_ground_truth: dict) -> str:
    """Aggregate context + per-frame spatial evidence section."""
    from services.tactical import GroundingFormatter
    return GroundingFormatter.format(
        analytics,
        include_insights=True,
        per_frame_data=db_ground_truth,
    )


def build_perframe_system_prompt(analysis_type: str) -> str:
    """Standard system prompt augmented with per-frame usage guidance (V1)."""
    from services.tactical import SYSTEM_PROMPTS, augment_system_prompt_perframe
    base = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["match_overview"])
    return augment_system_prompt_perframe(base)


def build_perframe_v2_context(analytics: dict, db_ground_truth: dict) -> str:
    """PERFRAME_V2: aggregate + wordalised insights + raw per-frame tables.

    Implements Context Engineering (Sumpter 2025): PerFrameInsights wordalise
    the time series into analyst prose before the LLM sees raw evidence tables.
    """
    from services.tactical import GroundingFormatter
    return GroundingFormatter.format(
        analytics,
        include_insights=True,
        per_frame_data=db_ground_truth,
        per_frame_insights=True,
    )


def build_perframe_v2_system_prompt(analysis_type: str) -> str:
    """Expert-persona + chain-of-thought system prompt for PERFRAME_V2."""
    from services.tactical import SYSTEM_PROMPTS, augment_system_prompt_perframe_v2
    base = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["match_overview"])
    return augment_system_prompt_perframe_v2(base)


def build_digit_space_context(analytics: dict, db_ground_truth: dict) -> str:
    """DIGIT_SPACE: aggregate + per-frame tables with digit-space numeric encoding.

    Applies digit-space tokenization (Gruver et al. 2023) to compactness and
    centroid time series to reduce BPE fragmentation of decimal values.
    """
    from services.tactical import GroundingFormatter, PerFrameContextFormatter
    base = GroundingFormatter.format(analytics, include_insights=True)
    pf_section = PerFrameContextFormatter.format(db_ground_truth, digit_space=True)
    if pf_section:
        return base + "\n" + pf_section
    return base


def build_visual_context(
    analytics: dict, db_ground_truth: dict
) -> "tuple[str, list[bytes]]":
    """VISUAL: aggregate + wordalised insights (no raw tables) + all 4 charts.

    Returns (text_context, images) for vision-capable LLMs.
    """
    from services.tactical import GroundingFormatter, VisualTimeSeriesRenderer
    # Text: aggregate + MatchInsights only (no raw per-frame tables)
    text = GroundingFormatter.format(analytics, include_insights=True)
    images = VisualTimeSeriesRenderer.render_all(db_ground_truth)
    return text, images


def build_visual_focused_context(
    analytics: dict, db_ground_truth: dict
) -> "tuple[str, list[bytes]]":
    """VISUAL_FOCUSED: aggregate + wordalised insights + compactness chart only.

    Single-chart condition to test focused vs comprehensive visual context.
    """
    from services.tactical import GroundingFormatter, VisualTimeSeriesRenderer
    text = GroundingFormatter.format(analytics, include_insights=True)
    fm = db_ground_truth.get("frame_metrics", {})
    fps = db_ground_truth.get("analytics", {}).get("fps") or 25.0
    img = VisualTimeSeriesRenderer._render_compactness(fm, fps)
    images = [img] if img else []
    return text, images


def build_visual_multimodal_context(
    analytics: dict, db_ground_truth: dict
) -> "tuple[str, list[bytes]]":
    """VISUAL_MULTIMODAL: aggregate + raw per-frame tables + all 4 charts.

    Both text (d) and visual (v) modalities together — matches paper's d+v condition.
    """
    from services.tactical import GroundingFormatter, VisualTimeSeriesRenderer
    text = GroundingFormatter.format(
        analytics, include_insights=True, per_frame_data=db_ground_truth
    )
    images = VisualTimeSeriesRenderer.render_all(db_ground_truth)
    return text, images


# Probe-informed chart routing: maps each analysis type to the chart whose visual
# representation achieved the highest probe F1 in the Qwen2-VL-7B probing study
# (Schumacher et al. 2026 §4). See dissertation/findings/linear_probing_findings.md.
_FINDINGS_CHART_MAP: dict[str, str] = {
    "match_overview": "centroid",       # territorial v=0.444 >> d=0.182 (pretraining suppresses text)
    "tactical_deep_dive": "pressing",   # pressing dashboard encodes inter-team distance (v vs d gap)
    "event_analysis": "pressing",       # event grounding 0%→67% with visual; distance spikes = events
    "player_spotlight": "centroid",     # player position relative to team centroid trajectory
}

# FINDINGS_INFORMED_MC: minimum-sufficient chart set per analysis type.
# Event claims are multi-modal (temporal trigger + team shape + pitch location) so
# require ≥3 complementary modalities; other types still route to single best chart.
# Ablation baseline: FINDINGS_INFORMED_MC (1–3 charts) vs FINDINGS_INFORMED (1 chart)
# vs VISUAL (4 charts, fixed) isolates selection effect from richness effect.
_FINDINGS_CHART_MAP_MC: dict[str, list[str]] = {
    "match_overview": ["centroid"],
    "tactical_deep_dive": ["pressing", "compactness"],
    "event_analysis": ["pressing", "compactness", "centroid"],
    "player_spotlight": ["centroid"],
}


def build_findings_informed_context(
    analytics: dict, db_ground_truth: dict
) -> "tuple[dict[str, str], dict[str, list[bytes]]]":
    """FINDINGS_INFORMED: probe-study-informed per-analysis-type chart routing.

    Text layer: aggregate + wordalised insights only (identical to VISUAL_FOCUSED).
    Image layer: per-analysis-type chart selected from _FINDINGS_CHART_MAP.

    Returns (contexts_dict, images_dict) keyed by analysis type.
    """
    from services.tactical import GroundingFormatter, VisualTimeSeriesRenderer
    fm = db_ground_truth.get("frame_metrics", {})
    fps = db_ground_truth.get("analytics", {}).get("fps") or 25.0
    shared_text = GroundingFormatter.format(analytics, include_insights=True)
    contexts: dict = {}
    images: dict = {}
    for atype in ANALYSIS_TYPES:
        chart_name = _FINDINGS_CHART_MAP.get(atype, "compactness")
        img = VisualTimeSeriesRenderer.render_named(chart_name, fm, fps)
        contexts[atype] = shared_text
        images[atype] = [img] if img else []
    return contexts, images


def build_findings_informed_system_prompt(analysis_type: str) -> str:
    """System prompt with chart-specific guidance for FINDINGS_INFORMED condition."""
    from services.tactical import SYSTEM_PROMPTS, augment_system_prompt_findings_informed
    base = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["match_overview"])
    chart_name = _FINDINGS_CHART_MAP.get(analysis_type, "compactness")
    return augment_system_prompt_findings_informed(base, analysis_type, chart_name)


def build_findings_informed_mc_context(
    analytics: dict, db_ground_truth: dict
) -> "tuple[dict[str, str], dict[str, list[bytes]]]":
    """FINDINGS_INFORMED_MC: probe-informed minimum-sufficient chart set routing.

    Identical to FINDINGS_INFORMED except each analysis type receives the chart
    *set* from _FINDINGS_CHART_MAP_MC rather than a single chart. Event claims
    need ≥3 modalities; overview/spotlight are unchanged (single chart).
    """
    from services.tactical import GroundingFormatter, VisualTimeSeriesRenderer
    fm = db_ground_truth.get("frame_metrics", {})
    fps = db_ground_truth.get("analytics", {}).get("fps") or 25.0
    shared_text = GroundingFormatter.format(analytics, include_insights=True)
    contexts: dict = {}
    images: dict = {}
    for atype in ANALYSIS_TYPES:
        chart_names = _FINDINGS_CHART_MAP_MC.get(atype, ["compactness"])
        imgs = [
            img for cname in chart_names
            for img in [VisualTimeSeriesRenderer.render_named(cname, fm, fps)]
            if img is not None
        ]
        contexts[atype] = shared_text
        images[atype] = imgs
    return contexts, images


def build_findings_informed_mc_system_prompt(analysis_type: str) -> str:
    """System prompt with multi-chart guidance for FINDINGS_INFORMED_MC condition."""
    from services.tactical import SYSTEM_PROMPTS, augment_system_prompt_findings_informed
    base = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["match_overview"])
    chart_names = _FINDINGS_CHART_MAP_MC.get(analysis_type, ["compactness"])
    return augment_system_prompt_findings_informed(base, analysis_type, chart_names)


def build_visual_system_prompt(analysis_type: str) -> str:
    """System prompt augmented with visual chart reading guidance."""
    from services.tactical import SYSTEM_PROMPTS, augment_system_prompt_visual
    base = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["match_overview"])
    return augment_system_prompt_visual(base)


def build_visual_focused_system_prompt(analysis_type: str) -> str:
    """System prompt augmented with focused compactness-chart guidance."""
    from services.tactical import SYSTEM_PROMPTS, augment_system_prompt_visual_focused
    base = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["match_overview"])
    return augment_system_prompt_visual_focused(base)


# ── Single run ────────────────────────────────────────────────────────────────


async def _run_single(
    analysis_type: str,
    context: str,
    system_prompt: str,
    analytics: dict,
    db_ground_truth: dict,
    provider,
    images: "list[bytes] | None" = None,
) -> dict:
    """Generate + extract + verify one analysis type."""
    commentary = await provider.generate(system_prompt, context, images=images)
    claims = await extract_claims_stable(commentary, provider, n_extract=3)

    n_claims = len(claims)
    if n_claims == 0:
        return {
            "grounding_rate": None,   # None = extractor returned no claims; excluded from avg
            "hallucination_rate": None,
            "n_claims": 0,
            "db_resolution_rate": None,
            "commentary": commentary,
            "claims": [],
        }

    std_results = [verify_claim(c, analytics) for c in claims]
    db_results = [
        verify_claim_db_grounded(c, analytics, db_ground_truth) for c in claims
    ]

    verified = sum(1 for r in std_results if r.verdict == "verified")
    refuted = sum(1 for r in std_results if r.verdict == "refuted")
    db_resolved = sum(
        1 for r in db_results
        if r["resolution"] in ("resolved_verified", "resolved_refuted")
    )

    return {
        "grounding_rate": verified / n_claims,
        "hallucination_rate": refuted / n_claims,
        "n_claims": n_claims,
        "db_resolution_rate": db_resolved / n_claims,
        "commentary": commentary,
        "claims": [
            {
                "text": c.text,
                "claim_type": c.claim_type,
                "referenced_metric": c.referenced_metric,
                "std_verdict": r.verdict,
                "db_verdict": dr["db_verdict"],
                "resolution": dr["resolution"],
                "db_evidence": dr["db_evidence"],
            }
            for c, r, dr in zip(claims, std_results, db_results)
        ],
    }


# ── Condition runner ───────────────────────────────────────────────────────────


async def run_condition(
    condition_name: str,
    context: "str | dict",
    system_prompt_fn,  # callable(analysis_type) -> str
    analytics: dict,
    db_ground_truth: dict,
    provider,
    n_runs: int,
    output_dir: Path,
    images: "list[bytes] | dict | None" = None,
) -> dict:
    """Run all analysis types × n_runs for one condition and aggregate results.

    context and images may optionally be dicts keyed by analysis_type to support
    per-analysis-type routing (FINDINGS_INFORMED condition). When a plain str/list
    is passed the existing behaviour is preserved.
    """
    all_results: dict = {atype: [] for atype in ANALYSIS_TYPES}

    for run_idx in range(1, n_runs + 1):
        logger.info("[%s] Run %d/%d", condition_name, run_idx, n_runs)
        for atype in ANALYSIS_TYPES:
            ctx = context[atype] if isinstance(context, dict) else context
            imgs = images[atype] if isinstance(images, dict) else images
            result = await _run_single(
                atype,
                ctx,
                system_prompt_fn(atype),
                analytics,
                db_ground_truth,
                provider,
                images=imgs,
            )
            result["run"] = run_idx
            result["analysis_type"] = atype
            all_results[atype].append(result)
            gr_display = result["grounding_rate"]
            db_display = result["db_resolution_rate"]
            logger.info(
                "  %s: grounding=%s db_resolution=%s n_claims=%d",
                atype,
                f"{gr_display*100:.1f}%" if gr_display is not None else "N/A",
                f"{db_display*100:.1f}%" if db_display is not None else "N/A",
                result["n_claims"],
            )

    # Average across runs — skip None values (n_claims=0 runs excluded from rate averages)
    def _avg(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None

    summary: dict = {}
    for atype, runs in all_results.items():
        if not runs:
            continue
        summary[atype] = {
            "grounding_rate": _avg([r["grounding_rate"] for r in runs]),
            "hallucination_rate": _avg([r["hallucination_rate"] for r in runs]),
            "n_claims": sum(r["n_claims"] for r in runs) / len(runs),
            "db_resolution_rate": _avg([r["db_resolution_rate"] for r in runs]),
        }

    gr_vals = [s["grounding_rate"] for s in summary.values() if s["grounding_rate"] is not None]
    db_vals = [s["db_resolution_rate"] for s in summary.values() if s["db_resolution_rate"] is not None]
    overall_gr = sum(gr_vals) / len(gr_vals) if gr_vals else 0.0
    overall_db = sum(db_vals) / len(db_vals) if db_vals else 0.0

    output = {
        "condition": condition_name,
        "n_runs": n_runs,
        "overall_grounding_rate": overall_gr,
        "overall_db_resolution_rate": overall_db,
        "by_analysis_type": summary,
        "runs": {atype: all_results[atype] for atype in ANALYSIS_TYPES},
    }

    out_path = output_dir / f"{condition_name.lower()}_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("[%s] Overall grounding: %.1f%% | DB resolution: %.1f%%",
                condition_name, overall_gr * 100, overall_db * 100)
    return output


# ── Output formatters ─────────────────────────────────────────────────────────


def save_comparison_table(
    results_by_condition: "dict[str, dict]",
    output_dir: Path,
    analysis_id: int = 18,
    n_runs: int = 1,
) -> None:
    """Write a markdown comparison table for all evaluated conditions.

    Args:
        results_by_condition: {condition_name: run_condition() output}
        output_dir: Directory to write comparison_table.md into.
        analysis_id: Analysis clip ID (for header).
        n_runs: Number of runs per condition (for header).
    """
    conditions = list(results_by_condition.keys())
    baseline_name = conditions[0] if conditions else "BASELINE"
    baseline = results_by_condition.get(baseline_name, {})

    header = "| Analysis Type | " + " | ".join(conditions) + " |"
    sep = "|---" + "|---" * len(conditions) + "|"

    lines = [
        "# Per-Frame Commentary: Condition Comparison",
        "",
        f"Analysis {analysis_id} | Runs: {n_runs}",
        "",
        "## Grounding Rate by Analysis Type",
        "",
        header,
        sep,
    ]
    def _fmt_gr(gr, b_gr, is_baseline):
        """Format a grounding rate cell; None means no claims were extracted."""
        if gr is None:
            return "N/A (no claims)"
        if is_baseline:
            return f"{gr*100:.1f}%"
        delta = gr - (b_gr or 0.0)
        sign = "+" if delta >= 0 else ""
        return f"{gr*100:.1f}% ({sign}{delta*100:.1f}pp)"

    for atype in ANALYSIS_TYPES:
        label = atype.replace("_", " ").title()
        cols = [label]
        b_gr = baseline.get("by_analysis_type", {}).get(atype, {}).get("grounding_rate") or 0.0
        for cname in conditions:
            res = results_by_condition.get(cname, {})
            gr = res.get("by_analysis_type", {}).get(atype, {}).get("grounding_rate")
            cols.append(_fmt_gr(gr, b_gr, cname == baseline_name))
        lines.append("| " + " | ".join(cols) + " |")

    # Overall row
    b_ov = baseline.get("overall_grounding_rate", 0)
    overall_cols = ["**Overall**"]
    for cname in conditions:
        res = results_by_condition.get(cname, {})
        gr = res.get("overall_grounding_rate", 0)
        delta = gr - b_ov
        sign = "+" if delta >= 0 else ""
        if cname == baseline_name:
            overall_cols.append(f"**{gr*100:.1f}%**")
        else:
            overall_cols.append(f"**{gr*100:.1f}%** ({sign}{delta*100:.1f}pp)")
    lines.append("| " + " | ".join(overall_cols) + " |")

    lines += [
        "",
        "## DB Resolution Rate by Condition",
        "",
        "| Condition | Overall DB Resolution |",
        "|---|---|",
    ]
    for cname, res in results_by_condition.items():
        db_r = res.get("overall_db_resolution_rate", 0)
        lines.append(f"| {cname} | {db_r*100:.1f}% |")

    (output_dir / "comparison_table.md").write_text("\n".join(lines))
    logger.info("Comparison table written.")


def save_prompt_example(
    context_baseline: str,
    context_perframe: str,
    baseline_results: dict,
    perframe_results: dict,
    output_dir: Path,
    context_perframe_v2: "str | None" = None,
    perframe_v2_results: "dict | None" = None,
) -> None:
    """Save the full prompt and example commentary for dissertation inclusion."""
    from services.tactical import (
        SYSTEM_PROMPTS, augment_system_prompt_perframe,
        augment_system_prompt_perframe_v2,
    )

    base_prompt = SYSTEM_PROMPTS["match_overview"]
    augmented_v1 = augment_system_prompt_perframe(base_prompt)
    augmented_v2 = augment_system_prompt_perframe_v2(base_prompt)

    def _get_commentary(results: dict) -> str:
        runs = results.get("runs", {}).get("match_overview", [])
        return runs[0]["commentary"] if runs else "(no commentary generated)"

    bl_commentary = _get_commentary(baseline_results)
    pf_commentary = _get_commentary(perframe_results)

    lines = [
        "# Per-Frame Commentary: Prompt and Output Examples",
        "",
        "## V1 System Prompt — Augmentation (per-frame addition only)",
        "```",
        augmented_v1[len(base_prompt):].strip(),
        "```",
        "",
        "---",
        "",
        "## V2 System Prompt — Context Engineering (expert persona + chain-of-thought)",
        "```",
        augmented_v2[len(base_prompt):].strip(),
        "```",
        "",
        "---",
        "",
        "## User Prompt — Per-Frame V1 Context Section",
        "*(Raw tables appended after aggregate data)*",
        "",
        "```markdown",
    ]

    marker_pf = "## Per-Frame Spatial Evidence"
    pf_idx = context_perframe.find(marker_pf)
    lines.append(context_perframe[pf_idx:].strip() if pf_idx >= 0 else "[Not found]")
    lines += ["```", "", "---", ""]

    if context_perframe_v2:
        lines += [
            "## User Prompt — Per-Frame V2 Context Section",
            "*(Wordalised insights prepended before raw tables)*",
            "",
            "```markdown",
        ]
        marker_v2 = "## Per-Frame Tactical Insights"
        v2_idx = context_perframe_v2.find(marker_v2)
        lines.append(context_perframe_v2[v2_idx:].strip() if v2_idx >= 0 else "[Not found]")
        lines += ["```", "", "---", ""]

    lines += [
        "## Commentary Output Comparison (Match Overview, Analysis 18)",
        "",
        "### BASELINE — Aggregate Only",
        f"*(Grounding rate: {baseline_results['by_analysis_type'].get('match_overview', {}).get('grounding_rate', 0)*100:.1f}%)*",
        "",
        f"> {bl_commentary.replace(chr(10), chr(10) + '> ')}",
        "",
        "---",
        "",
        "### PERFRAME_V1 — Aggregate + Raw Per-Frame Tables",
        f"*(Grounding rate: {perframe_results['by_analysis_type'].get('match_overview', {}).get('grounding_rate', 0)*100:.1f}%)*",
        "",
        f"> {pf_commentary.replace(chr(10), chr(10) + '> ')}",
        "",
        "---",
        "",
    ]

    if perframe_v2_results:
        v2_commentary = _get_commentary(perframe_v2_results)
        lines += [
            "### PERFRAME_V2 — Wordalised Insights + Expert Persona + Chain-of-Thought",
            f"*(Grounding rate: {perframe_v2_results['by_analysis_type'].get('match_overview', {}).get('grounding_rate', 0)*100:.1f}%)*",
            "",
            f"> {v2_commentary.replace(chr(10), chr(10) + '> ')}",
            "",
            "---",
            "",
        ]

    lines += [
        "## Grounding Rate Comparison",
        "",
        "| Condition | Overall | Match Overview | Tactical | Event | Player |",
        "|---|---|---|---|---|---|",
    ]

    def _row(name, res):
        bt = res["by_analysis_type"]
        return (
            f"| {name} "
            f"| {res['overall_grounding_rate']*100:.1f}% "
            f"| {bt.get('match_overview', {}).get('grounding_rate', 0)*100:.1f}% "
            f"| {bt.get('tactical_deep_dive', {}).get('grounding_rate', 0)*100:.1f}% "
            f"| {bt.get('event_analysis', {}).get('grounding_rate', 0)*100:.1f}% "
            f"| {bt.get('player_spotlight', {}).get('grounding_rate', 0)*100:.1f}% |"
        )

    lines.append(_row("Baseline", baseline_results))
    lines.append(_row("PERFRAME_V1", perframe_results))
    if perframe_v2_results:
        lines.append(_row("PERFRAME_V2 (Wordalisation)", perframe_v2_results))

    (output_dir / "perframe_prompt_example.md").write_text("\n".join(lines))
    logger.info("Prompt example written.")


# All supported condition names (order = display order in comparison table)
ALL_CONDITIONS = [
    "BASELINE",
    "PERFRAME_V1",
    "PERFRAME_V2",
    "DIGIT_SPACE",
    "VISUAL",
    "VISUAL_FOCUSED",
    "VISUAL_MULTIMODAL",
    "FINDINGS_INFORMED",
    "FINDINGS_INFORMED_MC",
]


# ── Main ──────────────────────────────────────────────────────────────────────


async def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path="backend/.env", override=True)
        load_dotenv(dotenv_path=".env", override=False)
    except ImportError:
        pass

    output_dir = ensure_output_dir(args.output)

    # Determine which conditions to run
    requested = (
        [c.strip().upper() for c in args.conditions.split(",")]
        if args.conditions
        else ALL_CONDITIONS
    )
    invalid = [c for c in requested if c not in ALL_CONDITIONS]
    if invalid:
        logger.error("Unknown conditions: %s. Valid: %s", invalid, ALL_CONDITIONS)
        sys.exit(1)
    logger.info("Running conditions: %s", requested)

    # Load data
    gt = load_db_ground_truth(args.ground_truth)
    analytics = gt.get("analytics", {})
    if not analytics:
        logger.error("No 'analytics' key in ground truth JSON. Exiting.")
        sys.exit(1)

    # Get provider
    from api.services.llm_providers import get_provider
    from services.tactical import SYSTEM_PROMPTS
    provider = get_provider(args.provider)

    def std_prompt(atype: str) -> str:
        return SYSTEM_PROMPTS.get(atype, "")

    # ── Build context + images for each condition ─────────────────────────────
    # Lazily build only what is needed.
    _context_cache: dict = {}

    def _get(name: str):
        if name in _context_cache:
            return _context_cache[name]
        if name == "BASELINE":
            val = (build_baseline_context(analytics), None)
        elif name == "PERFRAME_V1":
            val = (build_perframe_context(analytics, gt), None)
        elif name == "PERFRAME_V2":
            val = (build_perframe_v2_context(analytics, gt), None)
        elif name == "DIGIT_SPACE":
            val = (build_digit_space_context(analytics, gt), None)
        elif name == "VISUAL":
            text, imgs = build_visual_context(analytics, gt)
            val = (text, imgs)
        elif name == "VISUAL_FOCUSED":
            text, imgs = build_visual_focused_context(analytics, gt)
            val = (text, imgs)
        elif name == "VISUAL_MULTIMODAL":
            text, imgs = build_visual_multimodal_context(analytics, gt)
            val = (text, imgs)
        elif name == "FINDINGS_INFORMED":
            ctxs, imgsd = build_findings_informed_context(analytics, gt)
            val = (ctxs, imgsd)
        elif name == "FINDINGS_INFORMED_MC":
            ctxs, imgsd = build_findings_informed_mc_context(analytics, gt)
            val = (ctxs, imgsd)
        else:
            val = (build_baseline_context(analytics), None)
        _context_cache[name] = val
        ctx0 = val[0]
        ctx_len = len(next(iter(ctx0.values()))) if isinstance(ctx0, dict) else len(ctx0)
        imgs0 = val[1]
        imgs_n = len(next(iter(imgs0.values()))) if isinstance(imgs0, dict) else (len(imgs0) if imgs0 else 0)
        logger.info("%s context: %d chars (%d images)", name, ctx_len, imgs_n)
        return val

    # System-prompt function per condition
    _prompt_fn: dict = {
        "BASELINE": std_prompt,
        "PERFRAME_V1": build_perframe_system_prompt,
        "PERFRAME_V2": build_perframe_v2_system_prompt,
        "DIGIT_SPACE": build_perframe_system_prompt,  # same guidance as V1
        "VISUAL": build_visual_system_prompt,
        "VISUAL_FOCUSED": build_visual_focused_system_prompt,
        "VISUAL_MULTIMODAL": build_visual_system_prompt,
        "FINDINGS_INFORMED": build_findings_informed_system_prompt,
        "FINDINGS_INFORMED_MC": build_findings_informed_mc_system_prompt,
    }

    # ── Run each condition sequentially ──────────────────────────────────────
    results_by_condition: dict = {}
    for cname in requested:
        logger.info("=== Running %s condition ===", cname)
        ctx, imgs = _get(cname)
        results_by_condition[cname] = await run_condition(
            cname,
            ctx,
            _prompt_fn[cname],
            analytics,
            gt,
            provider,
            args.n_runs,
            output_dir,
            images=imgs,
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_comparison_table(
        results_by_condition,
        output_dir,
        analysis_id=args.analysis_id,
        n_runs=args.n_runs,
    )

    # Save prompt examples for the three text-only conditions if available
    bl = results_by_condition.get("BASELINE")
    pf1 = results_by_condition.get("PERFRAME_V1")
    if bl and pf1:
        ctx_bl, _ = _get("BASELINE")
        ctx_pf1, _ = _get("PERFRAME_V1")
        ctx_pf2_text = _get("PERFRAME_V2")[0] if "PERFRAME_V2" in results_by_condition else None
        save_prompt_example(
            ctx_bl,
            ctx_pf1,
            bl,
            pf1,
            output_dir,
            context_perframe_v2=ctx_pf2_text,
            perframe_v2_results=results_by_condition.get("PERFRAME_V2"),
        )

    logger.info("Done. Outputs in %s", output_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-frame commentary evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available conditions: {', '.join(ALL_CONDITIONS)}",
    )
    p.add_argument("--analysis-id", type=int, default=18)
    p.add_argument(
        "--ground-truth",
        default="eval_output/dissertation/db_grounded/18_db_ground_truth.json",
        help="Path to db_extractor ground truth JSON",
    )
    p.add_argument("--provider", default="openai",
                   choices=["openai", "gemini", "claude", "groq", "stub"])
    p.add_argument("--n-runs", type=int, default=1)
    p.add_argument("--output", default="eval_output/dissertation/perframe/")
    p.add_argument(
        "--conditions",
        default=None,
        help=(
            "Comma-separated list of conditions to run, e.g. "
            "'BASELINE,VISUAL,VISUAL_FOCUSED'. Default: all 7 conditions."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
