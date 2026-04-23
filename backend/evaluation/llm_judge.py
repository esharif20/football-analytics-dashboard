"""LLM-as-a-Judge evaluation module for football tactical commentary.

Implements three evaluation paradigms from the NLP evaluation literature:

1. G-Eval Pointwise Scoring (Liu et al. 2023, NeurIPS)
   - Chain-of-thought LLM scoring on 5 quality dimensions (1-5 Likert)
   - Rubric-based for reproducibility
   - Multi-sample averaging for stability

2. Pairwise Comparison (Zheng et al. 2023, NeurIPS MT-Bench)
   - "Which commentary is better?" — both positions evaluated to detect bias
   - Position-bias coefficient computed over all comparisons
   - Win rate, tie rate, agreement rate per pair

3. Multi-Judge Agreement
   - Same commentary evaluated by OpenAI and Gemini judges
   - Krippendorff's alpha (ordinal) for inter-judge agreement
   - Calibration against expert_validation.py Likert ratings if available

Usage:
    # Pointwise G-Eval across conditions
    python3 -m backend.evaluation.llm_judge \\
        --analytics eval_output/10_analytics.json \\
        --judge-provider openai \\
        --conditions A,F,H,I,J \\
        --n-runs 3 \\
        --output eval_output/dissertation/judge_openai_10/

    # Pairwise comparison
    python3 -m backend.evaluation.llm_judge \\
        --analytics eval_output/10_analytics.json \\
        --judge-provider openai \\
        --pairwise A:H,A:I,A:J,A:F \\
        --output eval_output/dissertation/pairwise_openai_10/

    # Multi-judge agreement (requires both judge runs to be completed first)
    python3 -m backend.evaluation.llm_judge \\
        --analytics eval_output/10_analytics.json \\
        --multi-judge openai,gemini \\
        --judge-dirs eval_output/dissertation/judge_openai_10/,eval_output/dissertation/judge_gemini_10/ \\
        --output eval_output/dissertation/judge_agreement/

References:
    Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.
        NeurIPS 2023. arXiv:2303.16634
    Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
        NeurIPS 2023. arXiv:2306.05685
    Krippendorff (2011). Computing Krippendorff's Alpha-Reliability.
        Annenberg School for Communication.
"""

import argparse
import asyncio
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import ensure_output_dir, load_analytics, save_figure, save_latex_table
from .llm_grounding import _ANALYSIS_TYPES
from .ablation_data import ABLATION_CONDITIONS, run_condition


# ── G-Eval dimensions ─────────────────────────────────────────────────────────

GEVAL_DIMENSIONS: dict[str, dict] = {
    "coherence": {
        "description": (
            "Does the commentary have a logical flow? Are tactical observations "
            "connected and building toward a unified narrative?"
        ),
        "rubric": {
            1: "Disconnected claims with no logical structure — reads as a bullet dump",
            2: "Some related points but poor transitions; feels like a list",
            3: "Mostly coherent with occasional non-sequiturs",
            4: "Well-structured with clear logical progression",
            5: "Expert-quality narrative flow; every insight builds on the previous",
        },
        "expert_likert_map": "overall_usefulness",
    },
    "consistency": {
        "description": (
            "Are all claims internally consistent? No contradictions between statements "
            "(e.g. calling Team 1 dominant in one paragraph, then passive in the next)?"
        ),
        "rubric": {
            1: "Multiple direct contradictions between claims",
            2: "At least one clear factual contradiction",
            3: "No contradictions but some tension or ambiguity between claims",
            4: "Fully consistent with very minor ambiguity",
            5: "All claims reinforce each other; perfectly consistent picture",
        },
        "expert_likert_map": "tactical_accuracy",
    },
    "fluency": {
        "description": (
            "Is the language professional, clear, and suitable for a coaching-staff briefing? "
            "Free from repetition, filler, and awkward phrasing?"
        ),
        "rubric": {
            1: "Poor grammar or phrasing; repetitive; unsuitable for professional use",
            2: "Readable but with noticeable filler or repetition",
            3: "Professional but generic — could describe any match",
            4: "Clear, specific, and polished coaching-style language",
            5: "Publication-quality analytical prose with a distinctive voice",
        },
        "expert_likert_map": "language_quality",
    },
    "relevance": {
        "description": (
            "Does the commentary focus on tactically significant observations? "
            "Does it highlight what matters (formation, pressing, possession patterns) "
            "rather than trivial or generic statistics?"
        ),
        "rubric": {
            1: "Mostly irrelevant or trivially obvious observations",
            2: "Some relevant points buried in filler",
            3: "Relevant but misses key tactical features for this match",
            4: "Covers important tactical themes with good prioritisation",
            5: "Expert-level selection of the most tactically significant insights",
        },
        "expert_likert_map": "insight_depth",
    },
    "groundedness": {
        "description": (
            "Are statistical claims backed by specific numbers from the analytics data? "
            "Does the commentary cite concrete values rather than vague qualifiers like 'high' or 'strong'?"
        ),
        "rubric": {
            1: "No specific data cited; entirely vague claims",
            2: "Occasional numbers but mostly unsupported assertions",
            3: "Some grounded claims but key statistics missing",
            4: "Most claims cite specific data points from the analytics",
            5: "Every claim is grounded in specific, verifiable data values",
        },
        "expert_likert_map": "data_groundedness",
    },
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    dimension: str
    score: float          # 1-5 Likert (mean across samples)
    score_std: float      # std across n_samples
    reasoning: str        # chain-of-thought from median sample
    provider: str
    analysis_type: str
    condition: str
    n_samples: int


@dataclass
class PairwiseResult:
    winner: str           # "A", "B", or "TIE"
    reasoning: str
    confidence: str       # "high", "medium", "low"
    position_order: str   # "AB" or "BA"
    condition_a: str
    condition_b: str
    analysis_type: str
    provider: str


# ── Prompt templates ──────────────────────────────────────────────────────────

_GEVAL_PROMPT = """\
You are evaluating the quality of football tactical commentary generated from match data.

## Source Data Summary
{analytics_summary}

## Commentary to Evaluate
{commentary}

## Evaluation Dimension: {dimension_name}
{dimension_description}

## Scoring Rubric
1: {rubric_1}
2: {rubric_2}
3: {rubric_3}
4: {rubric_4}
5: {rubric_5}

## Instructions
1. Explain your reasoning step by step (chain-of-thought), citing specific parts of the commentary
2. Assign a score from 1 to 5 based on the rubric above
3. Be strict — a score of 5 should be genuinely excellent, not just acceptable

Output format (follow exactly):
Reasoning: <your step-by-step analysis>
Score: <integer 1-5>"""

_PAIRWISE_PROMPT = """\
You are comparing two football tactical commentaries generated from the same match data.

## Source Data Summary
{analytics_summary}

## Commentary {label_a}
{commentary_a}

## Commentary {label_b}
{commentary_b}

## Evaluation Criteria
Compare on ALL five criteria:
1. Tactical accuracy — are claims factually correct?
2. Data grounding — are statistics drawn from the source data?
3. Insight depth — are observations tactically meaningful beyond the obvious?
4. Language quality — is the prose professional and clear?
5. Overall utility — would a coaching staff find this useful in a briefing?

## Instructions
- Consider ALL five criteria, not just one
- If both commentaries are genuinely similar quality, output TIE
- Ignore surface differences like length or formatting — focus on substance
- Be decisive: only mark TIE when truly equivalent

Output format (follow exactly):
Winner: <{label_a}, {label_b}, or TIE>
Reasoning: <2-3 sentences explaining the decision, referencing specific content>
Confidence: <high, medium, or low>"""


# ── Analytics summary ─────────────────────────────────────────────────────────

def _build_analytics_summary(analytics: dict) -> str:
    """Build a short structured summary of analytics for judge context."""
    lines = ["Match Analytics Summary:"]

    poss = analytics.get("possession", {})
    if poss:
        t1 = poss.get("team_1_percentage", poss.get("team1_percentage"))
        t2 = poss.get("team_2_percentage", poss.get("team2_percentage"))
        if t1 is not None:
            lines.append(f"- Possession: Team 1 {t1:.1f}% / Team 2 {t2:.1f}%")

    summary = analytics.get("tactical", {}).get("summary", {})
    if summary:
        for k, v in list(summary.items())[:6]:
            if isinstance(v, (int, float)):
                lines.append(f"- {k}: {v:.2f}")

    kinematics = analytics.get("player_kinematics", {})
    if kinematics:
        lines.append(f"- Players tracked: {len(kinematics)}")
        distances = [p.get("total_distance_m") for p in kinematics.values() if isinstance(p, dict)]
        distances = [d for d in distances if isinstance(d, (int, float))]
        if distances:
            lines.append(f"- Max distance: {max(distances):.0f} m")

    events = analytics.get("events", [])
    if events:
        event_types = {}
        for e in events:
            et = e.get("event_type", "unknown")
            event_types[et] = event_types.get(et, 0) + 1
        event_str = ", ".join(f"{k}: {v}" for k, v in list(event_types.items())[:5])
        lines.append(f"- Events: {event_str}")

    return "\n".join(lines)


# ── Score parsing ─────────────────────────────────────────────────────────────

def _parse_score(response: str) -> tuple[float | None, str]:
    """Extract score and reasoning from G-Eval response."""
    reasoning = ""
    score = None

    reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\nScore:|$)", response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    score_match = re.search(r"Score:\s*([1-5])", response, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))

    return score, reasoning


def _parse_pairwise(response: str, label_a: str, label_b: str) -> tuple[str, str, str]:
    """Extract winner, reasoning, confidence from pairwise response."""
    winner = "TIE"
    reasoning = ""
    confidence = "medium"

    winner_match = re.search(
        rf"Winner:\s*({re.escape(label_a)}|{re.escape(label_b)}|TIE)",
        response, re.IGNORECASE
    )
    if winner_match:
        raw = winner_match.group(1).upper()
        if raw == label_a.upper():
            winner = label_a
        elif raw == label_b.upper():
            winner = label_b
        else:
            winner = "TIE"

    reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\nConfidence:|$)", response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    confidence_match = re.search(r"Confidence:\s*(high|medium|low)", response, re.IGNORECASE)
    if confidence_match:
        confidence = confidence_match.group(1).lower()

    return winner, reasoning, confidence


# ── G-Eval scoring ────────────────────────────────────────────────────────────

async def geval_score(
    commentary: str,
    analytics_summary: str,
    provider,
    condition: str,
    analysis_type: str,
    dimensions: list[str] | None = None,
    n_samples: int = 3,
) -> dict[str, JudgeResult]:
    """G-Eval pointwise scoring with chain-of-thought (Liu et al. 2023).

    Runs n_samples judgments per dimension. Returns mean score ± std and
    the reasoning string from the median-score sample.
    """
    dims = dimensions or list(GEVAL_DIMENSIONS.keys())
    results: dict[str, JudgeResult] = {}

    for dim in dims:
        dim_cfg = GEVAL_DIMENSIONS[dim]
        rubric = dim_cfg["rubric"]
        prompt = _GEVAL_PROMPT.format(
            analytics_summary=analytics_summary,
            commentary=commentary,
            dimension_name=dim,
            dimension_description=dim_cfg["description"],
            rubric_1=rubric[1],
            rubric_2=rubric[2],
            rubric_3=rubric[3],
            rubric_4=rubric[4],
            rubric_5=rubric[5],
        )

        system = (
            "You are a rigorous evaluator of AI-generated sports commentary. "
            "Follow the scoring rubric strictly. Output only the format specified."
        )

        scores_raw: list[float] = []
        reasoning_samples: list[tuple[float, str]] = []

        for _ in range(n_samples):
            try:
                response = await provider.generate(system, prompt)
                score, reasoning = _parse_score(response)
                if score is not None:
                    scores_raw.append(score)
                    reasoning_samples.append((score, reasoning))
            except Exception:
                pass

        if not scores_raw:
            scores_raw = [3.0]
            reasoning_samples = [(3.0, "Evaluation failed")]

        mean_score = float(np.mean(scores_raw))
        std_score = float(np.std(scores_raw)) if len(scores_raw) > 1 else 0.0

        # Pick reasoning from the sample closest to the mean
        median_reasoning = min(reasoning_samples, key=lambda x: abs(x[0] - mean_score))[1]

        results[dim] = JudgeResult(
            dimension=dim,
            score=round(mean_score, 3),
            score_std=round(std_score, 3),
            reasoning=median_reasoning,
            provider=type(provider).__name__,
            analysis_type=analysis_type,
            condition=condition,
            n_samples=len(scores_raw),
        )

    return results


# ── Pairwise comparison ───────────────────────────────────────────────────────

async def pairwise_compare(
    commentary_a: str,
    commentary_b: str,
    analytics_summary: str,
    provider,
    condition_a: str,
    condition_b: str,
    analysis_type: str,
    label_a: str = "A",
    label_b: str = "B",
) -> list[PairwiseResult]:
    """Pairwise comparison with position-bias detection (Zheng et al. 2023).

    Runs evaluation in both orders (A-B and B-A). Returns two PairwiseResults
    so the caller can compute flip rate and agreement rate.
    """
    results = []

    for order in [("AB", commentary_a, commentary_b, label_a, label_b),
                  ("BA", commentary_b, commentary_a, label_b, label_a)]:
        order_key, ca, cb, la, lb = order
        prompt = _PAIRWISE_PROMPT.format(
            analytics_summary=analytics_summary,
            commentary_a=ca,
            commentary_b=cb,
            label_a=la,
            label_b=lb,
        )
        system = (
            "You are an impartial evaluator comparing two AI-generated football tactical commentaries. "
            "Be decisive and follow the output format exactly."
        )
        try:
            response = await provider.generate(system, prompt)
            raw_winner, reasoning, confidence = _parse_pairwise(response, la, lb)
            # Normalise winner back to original labels regardless of order
            if order_key == "AB":
                winner = raw_winner  # A/B/TIE already in original labels
            else:
                # Swap back: in "BA" order, la=label_b and lb=label_a
                if raw_winner == la:
                    winner = condition_b  # what was shown as 'A' in BA order is actually B
                elif raw_winner == lb:
                    winner = condition_a
                else:
                    winner = "TIE"
                winner = raw_winner if raw_winner == "TIE" else (label_a if raw_winner == lb else label_b)
        except Exception as exc:
            raw_winner, reasoning, confidence, winner = "TIE", str(exc), "low", "TIE"

        results.append(PairwiseResult(
            winner=winner,
            reasoning=reasoning,
            confidence=confidence,
            position_order=order_key,
            condition_a=condition_a,
            condition_b=condition_b,
            analysis_type=analysis_type,
            provider=type(provider).__name__,
        ))

    return results


# ── Krippendorff's alpha ──────────────────────────────────────────────────────

def compute_krippendorff_alpha(ratings: np.ndarray) -> float:
    """Ordinal Krippendorff's alpha for inter-judge agreement.

    Args:
        ratings: shape (n_judges, n_items), values 1-5 (or NaN for missing)

    Returns:
        alpha in [-1, 1]; >0.667 = acceptable, >0.8 = strong agreement
    """
    n_judges, n_items = ratings.shape
    # Observed disagreement (ordinal)
    observed_disagreement = 0.0
    n_pairs = 0
    for item in range(n_items):
        vals = ratings[:, item]
        vals = vals[~np.isnan(vals)]
        if len(vals) < 2:
            continue
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                observed_disagreement += (vals[i] - vals[j]) ** 2
                n_pairs += 1

    if n_pairs == 0:
        return float("nan")
    observed_disagreement /= n_pairs

    # Expected disagreement (from marginal distribution)
    all_vals = ratings.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    if len(all_vals) < 2:
        return float("nan")

    expected_disagreement = 0.0
    n = len(all_vals)
    for i in range(n):
        for j in range(i + 1, n):
            expected_disagreement += (all_vals[i] - all_vals[j]) ** 2
    expected_disagreement /= (n * (n - 1) / 2)

    if expected_disagreement == 0:
        return 1.0
    return float(1.0 - observed_disagreement / expected_disagreement)


def compute_position_bias(pairwise_results: list[PairwiseResult]) -> dict:
    """Analyse position bias across all pairwise comparisons.

    Returns:
        flip_rate: fraction where swapping order changed the winner
        agreement_rate: fraction where both orders agreed
        first_position_preference: rate of preferring first-listed option
    """
    pairs: dict[tuple, list[PairwiseResult]] = {}
    for r in pairwise_results:
        key = (r.condition_a, r.condition_b, r.analysis_type)
        pairs.setdefault(key, []).append(r)

    flips = 0
    agreements = 0
    first_preferred = 0
    total = 0

    for results in pairs.values():
        if len(results) != 2:
            continue
        ab = next((r for r in results if r.position_order == "AB"), None)
        ba = next((r for r in results if r.position_order == "BA"), None)
        if ab is None or ba is None:
            continue
        total += 1
        if ab.winner == ba.winner:
            agreements += 1
        else:
            flips += 1
        # First-position preference: AB order selected 'A' or BA order selected 'B'
        if (ab.position_order == "AB" and ab.winner == "A") or \
           (ba.position_order == "BA" and ba.winner == "B"):
            first_preferred += 1

    if total == 0:
        return {"flip_rate": 0.0, "agreement_rate": 0.0, "first_position_preference": 0.0, "n_pairs": 0}

    return {
        "flip_rate": round(flips / total, 3),
        "agreement_rate": round(agreements / total, 3),
        "first_position_preference": round(first_preferred / total, 3),
        "n_pairs": total,
    }


# ── Full study runner ─────────────────────────────────────────────────────────

async def run_judge_study(
    analytics: dict,
    judge_provider_name: str,
    output_dir: str,
    conditions: list[str] = None,
    n_runs: int = 3,
    pairwise_pairs: list[tuple[str, str]] | None = None,
    include_pairwise: bool = True,
) -> dict:
    """Full LLM-as-Judge study: pointwise G-Eval + pairwise + bias analysis.

    Generates commentaries from the ablation conditions, then evaluates with G-Eval.
    """
    from services.llm_providers import get_provider
    from services.tactical import TacticalAnalyzer

    if conditions is None:
        conditions = ["A", "F", "H", "I", "J"]
    if pairwise_pairs is None:
        pairwise_pairs = [("A", "H"), ("A", "I"), ("A", "F"), ("A", "J")]

    out = ensure_output_dir(output_dir)
    gen_provider = get_provider(judge_provider_name)
    judge_provider = get_provider(judge_provider_name)
    analytics_summary = _build_analytics_summary(analytics)

    print(f"\nLLM-as-a-Judge Study")
    print(f"Judge:      {judge_provider_name}")
    print(f"Conditions: {conditions}")
    print(f"Dimensions: {list(GEVAL_DIMENSIONS.keys())}")
    print(f"N-runs:     {n_runs}")

    # ── Step 1: Generate commentaries for each condition ──────────────────────
    print("\n[1/3] Generating commentaries...")
    commentaries: dict[str, dict[str, str]] = {}  # {condition -> {atype -> text}}

    for cond_id in conditions:
        if cond_id not in ABLATION_CONDITIONS:
            print(f"  Warning: unknown condition {cond_id!r} — skipping")
            continue
        condition = ABLATION_CONDITIONS[cond_id]
        commentaries[cond_id] = {}

        # For prompt ablation conditions, prepare stripped prompt
        prompt_override_map: dict[str, str | None] = {}
        if condition.strip_fewshot or condition.strip_metric_defs:
            import re as _re
            from services.tactical import SYSTEM_PROMPTS
            for atype in _ANALYSIS_TYPES:
                base = SYSTEM_PROMPTS.get(atype, SYSTEM_PROMPTS["match_overview"])
                if condition.strip_fewshot:
                    base = _re.sub(r"\n## Example Output.*", "", base, flags=_re.DOTALL)
                if condition.strip_metric_defs:
                    base = _re.sub(r"\n## Metric Definitions.*?(?=\n## |\nYour task|\Z)", "\n", base, flags=_re.DOTALL)
                prompt_override_map[atype] = base
        else:
            prompt_override_map = {atype: None for atype in _ANALYSIS_TYPES}

        stripped = condition.strip_fn(analytics)

        for atype in _ANALYSIS_TYPES:
            print(f"  [{cond_id}] {atype} ...", end=" ", flush=True)
            try:
                analyzer = TacticalAnalyzer(provider=gen_provider)
                result = await analyzer.analyze(
                    stripped, atype,
                    include_insights=condition.include_insights,
                    system_prompt_override=prompt_override_map.get(atype),
                )
                commentaries[cond_id][atype] = result["content"]
                print("done")
            except Exception as exc:
                print(f"ERROR: {exc}")
                commentaries[cond_id][atype] = ""

    # ── Step 2: G-Eval pointwise scoring ─────────────────────────────────────
    print("\n[2/3] Running G-Eval pointwise scoring...")
    geval_results: dict[str, dict[str, dict[str, JudgeResult]]] = {}  # {cond -> {atype -> {dim -> result}}}

    for cond_id, atype_texts in commentaries.items():
        geval_results[cond_id] = {}
        for atype, text in atype_texts.items():
            if not text:
                continue
            print(f"  [{cond_id}] {atype} ...", end=" ", flush=True)
            try:
                scores = await geval_score(
                    text, analytics_summary, judge_provider,
                    condition=cond_id, analysis_type=atype,
                    n_samples=n_runs,
                )
                geval_results[cond_id][atype] = scores
                avg = np.mean([r.score for r in scores.values()])
                print(f"avg={avg:.2f}")
            except Exception as exc:
                print(f"ERROR: {exc}")

    # ── Step 3: Pairwise comparison ───────────────────────────────────────────
    all_pairwise: list[PairwiseResult] = []
    if include_pairwise and pairwise_pairs:
        print("\n[3/3] Running pairwise comparison...")
        for cond_a_id, cond_b_id in pairwise_pairs:
            if cond_a_id not in commentaries or cond_b_id not in commentaries:
                print(f"  Skipping {cond_a_id} vs {cond_b_id} — missing commentaries")
                continue
            for atype in _ANALYSIS_TYPES:
                text_a = commentaries.get(cond_a_id, {}).get(atype, "")
                text_b = commentaries.get(cond_b_id, {}).get(atype, "")
                if not text_a or not text_b:
                    continue
                print(f"  [{cond_a_id} vs {cond_b_id}] {atype} ...", end=" ", flush=True)
                try:
                    pair_results = await pairwise_compare(
                        text_a, text_b, analytics_summary, judge_provider,
                        condition_a=cond_a_id, condition_b=cond_b_id,
                        analysis_type=atype,
                    )
                    all_pairwise.extend(pair_results)
                    # Summarise: pick winner from AB order
                    ab = next((r for r in pair_results if r.position_order == "AB"), pair_results[0])
                    print(f"winner={ab.winner}")
                except Exception as exc:
                    print(f"ERROR: {exc}")

    # ── Compute aggregates ────────────────────────────────────────────────────
    geval_aggregated = _aggregate_geval(geval_results)
    pairwise_aggregated = _aggregate_pairwise(all_pairwise)
    bias_stats = compute_position_bias(all_pairwise)

    results = {
        "_meta": {
            "judge_provider": judge_provider_name,
            "conditions": conditions,
            "n_runs": n_runs,
            "n_dimensions": len(GEVAL_DIMENSIONS),
        },
        "geval_raw": {
            cond: {
                atype: {dim: {"score": r.score, "std": r.score_std, "reasoning": r.reasoning}
                        for dim, r in dim_results.items()}
                for atype, dim_results in atype_results.items()
            }
            for cond, atype_results in geval_results.items()
        },
        "geval_aggregated": geval_aggregated,
        "pairwise_raw": [
            {
                "winner": r.winner, "reasoning": r.reasoning,
                "confidence": r.confidence, "position_order": r.position_order,
                "condition_a": r.condition_a, "condition_b": r.condition_b,
                "analysis_type": r.analysis_type,
            }
            for r in all_pairwise
        ],
        "pairwise_aggregated": pairwise_aggregated,
        "position_bias": bias_stats,
    }

    # Save JSON
    (out / "judge_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {out / 'judge_results.json'}")

    # Save tables and figures
    _save_geval_table(geval_aggregated, str(out))
    _save_pairwise_table(pairwise_aggregated, bias_stats, str(out))
    _save_geval_radar(geval_aggregated, str(out))
    _save_pairwise_bar(pairwise_aggregated, str(out))

    return results


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _aggregate_geval(
    geval_results: dict[str, dict[str, dict[str, JudgeResult]]]
) -> dict[str, dict[str, dict]]:
    """Aggregate G-Eval scores: per-condition × per-dimension mean/std across analysis types."""
    aggregated: dict[str, dict[str, dict]] = {}

    for cond_id, atype_results in geval_results.items():
        aggregated[cond_id] = {}
        for dim in GEVAL_DIMENSIONS:
            scores = []
            for atype, dim_results in atype_results.items():
                if dim in dim_results:
                    scores.append(dim_results[dim].score)
            if scores:
                aggregated[cond_id][dim] = {
                    "mean": round(float(np.mean(scores)), 3),
                    "std": round(float(np.std(scores)), 3),
                    "n": len(scores),
                }

    return aggregated


def _aggregate_pairwise(pairwise_results: list[PairwiseResult]) -> dict:
    """Aggregate pairwise results: win rates per pair (using AB-order results only)."""
    # Use only AB-order results to avoid double-counting
    ab_results = [r for r in pairwise_results if r.position_order == "AB"]

    pairs: dict[str, list[str]] = {}
    for r in ab_results:
        key = f"{r.condition_a}_vs_{r.condition_b}"
        pairs.setdefault(key, []).append(r.winner)

    aggregated = {}
    for pair_key, winners in pairs.items():
        n = len(winners)
        parts = pair_key.split("_vs_")
        cond_a = parts[0] if len(parts) == 2 else "A"
        cond_b = parts[1] if len(parts) == 2 else "B"
        aggregated[pair_key] = {
            "condition_a": cond_a,
            "condition_b": cond_b,
            "n": n,
            "win_a": round(sum(1 for w in winners if w == cond_a) / n, 3) if n else 0.0,
            "win_b": round(sum(1 for w in winners if w == cond_b) / n, 3) if n else 0.0,
            "tie": round(sum(1 for w in winners if w == "TIE") / n, 3) if n else 0.0,
        }

    return aggregated


# ── Output: tables ────────────────────────────────────────────────────────────

def _save_geval_table(aggregated: dict, output_dir: str) -> None:
    """Save G-Eval scores as LaTeX table."""
    dims = list(GEVAL_DIMENSIONS.keys())
    headers = ["Condition"] + [d.capitalize() for d in dims] + ["Avg"]
    rows = []
    for cond_id in sorted(aggregated.keys()):
        dim_scores = aggregated[cond_id]
        vals = [f"{dim_scores.get(d, {}).get('mean', 0):.2f}" for d in dims]
        means = [dim_scores.get(d, {}).get("mean", 0) for d in dims]
        avg = f"{np.mean(means):.2f}" if means else "—"
        rows.append([cond_id] + vals + [avg])

    save_latex_table(
        headers, rows,
        caption="G-Eval LLM-as-Judge scores per condition (1-5 Likert, mean across analysis types). "
                "Evaluated using chain-of-thought prompting (Liu et al. 2023, NeurIPS).",
        name="geval_scores",
        output_dir=output_dir,
        label="tab:geval_scores",
    )
    print(f"  Saved: {output_dir}/geval_scores.tex")


def _save_pairwise_table(aggregated: dict, bias_stats: dict, output_dir: str) -> None:
    """Save pairwise win rates as LaTeX table."""
    headers = ["Comparison", "Win-A (%)", "Win-B (%)", "Tie (%)", "Flip Rate"]
    rows = []
    for pair_key, stats in aggregated.items():
        cond_a = stats["condition_a"]
        cond_b = stats["condition_b"]
        rows.append([
            f"{cond_a} vs {cond_b}",
            f"{stats['win_a']:.0%}",
            f"{stats['win_b']:.0%}",
            f"{stats['tie']:.0%}",
            f"{bias_stats.get('flip_rate', 0):.0%}",
        ])
    # Append bias summary row
    rows.append([
        "Position bias (overall)",
        "—", "—", "—",
        f"{bias_stats.get('flip_rate', 0):.0%} flips, "
        f"{bias_stats.get('first_position_preference', 0):.0%} 1st-pref",
    ])

    save_latex_table(
        headers, rows,
        caption="Pairwise preference win rates with position-bias analysis (Zheng et al. 2023, NeurIPS). "
                "Win-A = condition A wins; Flip Rate = fraction of comparisons where swapping positions changed the winner.",
        name="pairwise_results",
        output_dir=output_dir,
        label="tab:pairwise_results",
    )
    print(f"  Saved: {output_dir}/pairwise_results.tex")


# ── Output: figures ───────────────────────────────────────────────────────────

def _save_geval_radar(aggregated: dict, output_dir: str) -> None:
    """Save G-Eval radar (spider) chart — one polygon per condition."""
    dims = list(GEVAL_DIMENSIONS.keys())
    n_dims = len(dims)
    angles = [n / float(n_dims) * 2 * np.pi for n in range(n_dims)]
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    colors = plt.cm.Set2(np.linspace(0, 1, len(aggregated)))

    for (cond_id, dim_scores), color in zip(sorted(aggregated.items()), colors):
        values = [dim_scores.get(d, {}).get("mean", 0) for d in dims]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=cond_id, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.capitalize() for d in dims], size=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], size=8)
    ax.set_title("G-Eval Quality Dimensions by Condition\n(Liu et al. 2023, 1-5 Likert)",
                 size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    save_figure(fig, "geval_radar", output_dir)
    print(f"  Saved: {output_dir}/geval_radar.png")


def _save_pairwise_bar(aggregated: dict, output_dir: str) -> None:
    """Save pairwise win-rate grouped bar chart."""
    if not aggregated:
        return

    pair_keys = sorted(aggregated.keys())
    x = np.arange(len(pair_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(pair_keys) * 1.5), 5))

    win_a = [aggregated[k]["win_a"] * 100 for k in pair_keys]
    win_b = [aggregated[k]["win_b"] * 100 for k in pair_keys]
    tie_rates = [aggregated[k]["tie"] * 100 for k in pair_keys]

    ax.bar(x - width, win_a, width, label="Condition A wins", color="#2196F3")
    ax.bar(x, win_b, width, label="Condition B wins", color="#F44336")
    ax.bar(x + width, tie_rates, width, label="Tie", color="#9E9E9E")

    labels = [f"{aggregated[k]['condition_a']} vs {aggregated[k]['condition_b']}" for k in pair_keys]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Pairwise Preference Win Rates\n(Position-bias mitigated, Zheng et al. 2023)")
    ax.legend()
    ax.axhline(50, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    save_figure(fig, "pairwise_winrate", output_dir)
    print(f"  Saved: {output_dir}/pairwise_winrate.png")


# ── Multi-judge agreement ─────────────────────────────────────────────────────

def compute_judge_agreement(judge_results: list[dict], output_dir: str) -> dict:
    """Compute inter-judge agreement (Krippendorff's alpha) across multiple judge result files.

    Args:
        judge_results: list of loaded judge_results.json dicts (one per judge)
        output_dir: where to save agreement table

    Returns:
        {dimension: alpha, "overall": mean_alpha}
    """
    dims = list(GEVAL_DIMENSIONS.keys())
    alphas: dict[str, float] = {}

    for dim in dims:
        # Collect all scores per (condition, atype) from each judge
        all_items: list[dict] = []  # list of {judge_idx -> score}
        for judge_idx, jr in enumerate(judge_results):
            raw = jr.get("geval_raw", {})
            for cond, atype_results in raw.items():
                for atype, dim_results in atype_results.items():
                    if dim in dim_results:
                        item_key = f"{cond}_{atype}"
                        # Find or create entry for this item
                        found = next((x for x in all_items if x.get("_key") == item_key), None)
                        if found is None:
                            found = {"_key": item_key}
                            all_items.append(found)
                        found[judge_idx] = dim_results[dim]["score"]

        if not all_items:
            alphas[dim] = float("nan")
            continue

        n_judges = len(judge_results)
        n_items = len(all_items)
        ratings = np.full((n_judges, n_items), np.nan)
        for item_idx, item in enumerate(all_items):
            for judge_idx in range(n_judges):
                if judge_idx in item:
                    ratings[judge_idx, item_idx] = item[judge_idx]

        alphas[dim] = round(compute_krippendorff_alpha(ratings), 3)

    all_finite = [v for v in alphas.values() if not np.isnan(v)]
    alphas["overall"] = round(float(np.mean(all_finite)), 3) if all_finite else float("nan")

    # Save table
    headers = ["Dimension", "Krippendorff α", "Interpretation"]
    rows = []
    for dim, alpha in alphas.items():
        if np.isnan(alpha):
            interp = "insufficient data"
        elif alpha >= 0.8:
            interp = "strong agreement"
        elif alpha >= 0.667:
            interp = "acceptable agreement"
        elif alpha >= 0.0:
            interp = "weak agreement"
        else:
            interp = "below chance"
        rows.append([dim.capitalize(), f"{alpha:.3f}", interp])

    ensure_output_dir(output_dir)
    save_latex_table(
        headers, rows,
        caption="Inter-judge agreement (Krippendorff's alpha, ordinal) across LLM judges per G-Eval dimension. "
                "Alpha > 0.667 indicates acceptable agreement; > 0.8 is strong (Krippendorff 2011).",
        name="judge_agreement",
        output_dir=output_dir,
        label="tab:judge_agreement",
    )
    print(f"  Saved: {output_dir}/judge_agreement.tex")

    # Save heatmap
    _save_agreement_heatmap(alphas, output_dir)

    return alphas


def _save_agreement_heatmap(alphas: dict, output_dir: str) -> None:
    """Save Krippendorff's alpha heatmap (single judge pair for now)."""
    dims = [d for d in GEVAL_DIMENSIONS if d in alphas]
    values = [[alphas.get(d, 0)] for d in dims]

    fig, ax = plt.subplots(figsize=(3, len(dims) * 0.7 + 1))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["Krippendorff α"])
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([d.capitalize() for d in dims])
    for i, dim in enumerate(dims):
        ax.text(0, i, f"{alphas.get(dim, 0):.3f}", ha="center", va="center",
                color="white" if abs(alphas.get(dim, 0)) > 0.5 else "black", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.1)
    ax.set_title("Inter-Judge Agreement\n(Krippendorff's α)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "judge_agreement_heatmap", output_dir)
    print(f"  Saved: {output_dir}/judge_agreement_heatmap.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env", override=True)
    except ImportError:
        pass
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge evaluation: G-Eval (Liu 2023) + Pairwise (Zheng 2023)"
    )
    parser.add_argument("--analytics", required=True, help="Path to *_analytics.json")
    parser.add_argument("--judge-provider", default="openai", choices=["openai", "gemini", "huggingface", "claude", "groq"])
    parser.add_argument("--conditions", default="A,F,H,I,J",
                        help="Comma-separated condition IDs for G-Eval (default: A,F,H,I,J)")
    parser.add_argument("--pairwise", default=None,
                        help="Comma-separated pairs for pairwise comparison, e.g. A:H,A:I,A:F")
    parser.add_argument("--n-runs", type=int, default=3, help="Samples per G-Eval dimension")
    parser.add_argument("--output", default="eval_output/judge", help="Output directory")
    parser.add_argument("--no-pairwise", action="store_true", help="Skip pairwise comparison")
    parser.add_argument(
        "--multi-judge", default=None,
        help="Compute inter-judge agreement: comma-separated providers e.g. openai,gemini"
    )
    parser.add_argument(
        "--judge-dirs", default=None,
        help="Comma-separated paths to completed judge result directories (for --multi-judge)"
    )
    args = parser.parse_args()

    analytics = load_analytics(args.analytics)

    # Multi-judge agreement mode
    if args.multi_judge and args.judge_dirs:
        judge_paths = [p.strip() for p in args.judge_dirs.split(",")]
        judge_results = []
        for jp in judge_paths:
            result_file = Path(jp) / "judge_results.json"
            if result_file.exists():
                judge_results.append(json.loads(result_file.read_text()))
            else:
                print(f"Warning: {result_file} not found — skipping")
        if judge_results:
            alphas = compute_judge_agreement(judge_results, args.output)
            print(f"\nInter-judge agreement (Krippendorff's α):")
            for dim, alpha in alphas.items():
                print(f"  {dim:<20} {alpha:.3f}")
        return

    # Standard G-Eval + pairwise mode
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    pairwise_pairs = None
    if args.pairwise:
        pairwise_pairs = [
            tuple(p.strip().split(":")) for p in args.pairwise.split(",")
            if ":" in p
        ]

    asyncio.run(run_judge_study(
        analytics=analytics,
        judge_provider_name=args.judge_provider,
        output_dir=args.output,
        conditions=conditions,
        n_runs=args.n_runs,
        pairwise_pairs=pairwise_pairs,
        include_pairwise=not args.no_pairwise,
    ))


if __name__ == "__main__":
    main()
