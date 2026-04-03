"""LLM commentary grounding rate evaluator.

Generates tactical commentary from analytics JSON, then uses an LLM-as-judge
to extract factual claims and verifies each against the source analytics data.

Also compares three grounding formats:
  1. Structured markdown (current GroundingFormatter — the design choice)
  2. Raw JSON (what the assessor warned against)
  3. Minimal prose (natural language summaries)

This directly addresses the assessor feedback:
  "LLMs are notoriously poor with JSON ... may want to consider other markup formats"

Usage:
    python -m backend.evaluation.llm_grounding \\
        --analytics path/to/analytics.json \\
        --provider gemini \\
        --output eval_output/grounding/
"""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Add parent dirs to path so imports work when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))  # backend/api/

from ._common import (
    EvalConfig,
    ensure_output_dir,
    load_analytics,
    save_figure,
    save_latex_table,
)


@dataclass
class Claim:
    text: str
    claim_type: str          # "numeric", "comparative", "qualitative", "entity_reference"
    referenced_metric: str   # e.g. "possession.team_1_percentage"
    referenced_value: str    # e.g. "63.2%"
    source_sentence: str


@dataclass
class VerificationResult:
    claim: Claim
    verdict: str             # "verified", "refuted", "unverifiable", "plausible"
    actual_value: Any
    explanation: str



def format_as_markdown(analytics: dict) -> str:
    """Current production formatter — structured markdown tables."""
    from services.tactical import GroundingFormatter
    return GroundingFormatter.format(analytics)


def format_as_raw_json(analytics: dict) -> str:
    """Raw JSON — what the assessor warned against (the control condition)."""
    # Strip very large arrays to avoid token limits (same as production does)
    stripped = {
        k: v for k, v in analytics.items()
        if k not in ("ball_path_positions",)
    }
    return json.dumps(stripped, indent=2)


def format_as_prose(analytics: dict) -> str:
    """Natural language prose summaries of the analytics data."""
    lines = ["The following is a summary of match analytics data.\n"]
    poss = analytics.get("possession", {})
    if poss:
        t1, t2 = poss.get("team_1_percentage", 0), poss.get("team_2_percentage", 0)
        lines.append(f"Possession: Team 1 held the ball for {t1:.1f}% of the match, "
                     f"Team 2 for {t2:.1f}%. Possession changed hands {poss.get('possession_changes', 0)} times.")
    players = analytics.get("player_kinematics", {})
    if players:
        lines.append("\nPlayer performance:")
        team_players: dict[int, list] = {1: [], 2: []}
        for tid, s in players.items():
            tid_int = s.get("team_id")
            if tid_int in team_players:
                d, av, mx = s.get("total_distance_m"), s.get("avg_speed_m_per_sec"), s.get("max_speed_m_per_sec")
                team_players[tid_int].append(f"    Player #{tid}: {f'{d:.0f}m' if d else 'N/A'} covered, "
                                              f"avg {f'{av*3.6:.1f} km/h' if av else 'N/A'}, "
                                              f"max {f'{mx*3.6:.1f} km/h' if mx else 'N/A'}")
        for t, plist in team_players.items():
            if plist:
                lines.append(f"  Team {t}:")
                lines.extend(plist)
    ball = analytics.get("ball_kinematics", {})
    if ball:
        d, av = ball.get("total_distance_m"), ball.get("avg_speed_m_per_sec")
        lines.append(f"\nBall: total distance {d:.0f}m" if d else "\nBall: distance not available")
        if av:
            lines.append(f"  Average ball speed: {av * 3.6:.1f} km/h")
    events = analytics.get("events", [])
    if events:
        counts: dict[str, int] = {}
        for ev in events:
            counts[ev.get("event_type", "unknown")] = counts.get(ev.get("event_type", "unknown"), 0) + 1
        lines.append(f"\nEvents detected: {len(events)} total.")
        lines.extend(f"  {et.capitalize()}: {cnt}" for et, cnt in sorted(counts.items()))
    return "\n".join(lines)


FORMATTERS = {
    "markdown": format_as_markdown,
    "json": format_as_raw_json,
    "prose": format_as_prose,
}



EXTRACTION_PROMPT = """\
You are a fact-checking assistant for football tactical analysis.
Given the following tactical commentary, extract every factual claim that
could in principle be verified against match data.

For each claim output a JSON object with:
  - "text": the exact claim text (verbatim from the commentary)
  - "claim_type": one of "numeric", "comparative", "qualitative", "entity_reference"
  - "referenced_metric": the analytics field it relates to (e.g. "possession.team_1_percentage",
    "player_kinematics.4.total_distance_m", "events.pass.count")
  - "referenced_value": the value the commentary claims (e.g. "63%", "8,200m", "Team 1")
  - "source_sentence": the full sentence containing the claim

Output ONLY a valid JSON array of these objects. No explanation, no markdown fences."""


async def extract_claims(commentary: str, provider) -> list[Claim]:
    """Use LLM-as-judge to extract structured claims from commentary."""
    raw = await provider.generate(EXTRACTION_PROMPT, commentary)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return [
            Claim(text=it.get("text", ""), claim_type=it.get("claim_type", "qualitative"),
                  referenced_metric=it.get("referenced_metric", ""),
                  referenced_value=it.get("referenced_value", ""),
                  source_sentence=it.get("source_sentence", ""))
            for it in json.loads(raw)
            if isinstance(it, dict) and it.get("text")
        ]
    except (json.JSONDecodeError, TypeError):
        return _regex_fallback_extract(commentary)


def _find_sentence(text: str, pos: int) -> str:
    start = max(0, text.rfind(".", 0, pos) + 1)
    end = text.find(".", pos)
    return text[start:(len(text) if end == -1 else end + 1)].strip()


def _regex_fallback_extract(commentary: str) -> list[Claim]:
    """Regex fallback when LLM judge fails to produce valid JSON."""
    claims = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", commentary):
        claims.append(Claim(text=m.group(0), claim_type="numeric",
                            referenced_metric="unknown_percentage", referenced_value=m.group(0),
                            source_sentence=_find_sentence(commentary, m.start())))
    for m in re.finditer(r"(\d[\d,]+)\s*(?:m\b|metres?|meters?)", commentary, re.IGNORECASE):
        claims.append(Claim(text=m.group(0), claim_type="numeric",
                            referenced_metric="distance_m", referenced_value=m.group(0),
                            source_sentence=_find_sentence(commentary, m.start())))
    return claims



def _get_nested(d: dict, dotpath: str) -> Any:
    """Navigate a dot-separated path in a nested dict."""
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


def _parse_numeric(val_str: str) -> float | None:
    """Parse a value string like '63.2%', '8,200m', '12.3 km/h'."""
    try:
        return float(re.sub(r"[,%a-zA-Z/\s]", "", val_str))
    except (ValueError, TypeError):
        return None


# Qualitative claim rules: (keyword pattern, analytics check function)
_QUALITATIVE_RULES = [
    ("press", lambda a: (a.get("possession", {}).get("possession_changes", 0) > 10, "possession_changes")),
    ("dominat", lambda a: (
        abs(a.get("possession", {}).get("team_1_percentage", 50) - 50) > 15, "possession imbalance > 15%"
    )),
    ("fast", lambda a: (
        any(
            s.get("max_speed_m_per_sec", 0) * 3.6 > 20
            for s in a.get("player_kinematics", {}).values()
        ), "player max_speed > 20 km/h"
    )),
]


def verify_claim(claim: Claim, analytics: dict) -> VerificationResult:  # noqa: C901
    """Verify a single claim against the source analytics dict."""
    def _unverifiable(msg: str, val: Any = None) -> VerificationResult:
        return VerificationResult(claim=claim, verdict="unverifiable", actual_value=val, explanation=msg)

    if claim.claim_type == "numeric":
        actual = _get_nested(analytics, claim.referenced_metric)
        claimed_num = _parse_numeric(claim.referenced_value)
        if actual is None or claimed_num is None:
            return _unverifiable(f"Could not resolve metric path '{claim.referenced_metric}'", actual)
        display = actual * 3.6 if ("speed" in claim.referenced_metric and "per_sec" in claim.referenced_metric) else actual
        within = abs(display - claimed_num) <= max(abs(display) * 0.05, 0.5)
        verdict = "verified" if within else "refuted"
        suffix = " (within 5% tolerance)" if within else ""
        return VerificationResult(claim=claim, verdict=verdict, actual_value=round(display, 2),
                                  explanation=f"Claimed {claimed_num}, actual {display:.2f}{suffix}")

    if claim.claim_type == "comparative":
        text_lower = claim.text.lower()
        actual = _get_nested(analytics, claim.referenced_metric)
        if actual is None:
            return _unverifiable(f"Metric '{claim.referenced_metric}' not found")
        poss = analytics.get("possession", {})
        t1, t2 = poss.get("team_1_percentage", 50), poss.get("team_2_percentage", 50)
        if "team 1" in text_lower and "team 2" in text_lower:
            pos = "more" in text_lower or "higher" in text_lower or "greater" in text_lower
            verdict = "verified" if (t1 > t2) == pos else "refuted"
            return VerificationResult(claim=claim, verdict=verdict,
                                      actual_value={"team_1_pct": t1, "team_2_pct": t2},
                                      explanation=f"Team 1: {t1:.1f}%, Team 2: {t2:.1f}%")
        return _unverifiable("Comparative claim could not be automatically verified", actual)

    if claim.claim_type == "entity_reference":
        kinematics = analytics.get("player_kinematics", {})
        m = re.search(r"#?(\d+)", claim.text)
        if m:
            tid = m.group(1)
            exists = tid in kinematics or int(tid) in kinematics
            verdict = "verified" if exists else "refuted"
            explanation = (f"Track #{tid} exists in player_kinematics" if exists
                           else f"Track #{tid} not found in player_kinematics")
            return VerificationResult(claim=claim, verdict=verdict,
                                      actual_value=tid if exists else None, explanation=explanation)
        return _unverifiable("No track ID found in claim text")

    if claim.claim_type == "qualitative":
        text_lower = claim.text.lower()
        for keyword, check_fn in _QUALITATIVE_RULES:
            if keyword in text_lower:
                is_plausible, evidence = check_fn(analytics)
                return VerificationResult(claim=claim,
                                          verdict="plausible" if is_plausible else "unverifiable",
                                          actual_value=evidence,
                                          explanation=f"Qualitative: '{keyword}' claim checked against {evidence}")
        return _unverifiable("No rule matched for qualitative claim")

    return _unverifiable(f"Unknown claim type: {claim.claim_type}")



def compute_grounding_score(results: list[VerificationResult]) -> dict:
    """Compute grounding rate and hallucination rate."""
    total = len(results)
    if total == 0:
        return {"grounding_rate": 0.0, "hallucination_rate": 0.0, "total_claims": 0}
    counts = {v: 0 for v in ("verified", "refuted", "unverifiable", "plausible")}
    by_type: dict[str, dict] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
        ct = r.claim.claim_type
        by_type.setdefault(ct, {"verified": 0, "refuted": 0, "unverifiable": 0, "plausible": 0})
        by_type[ct][r.verdict] = by_type[ct].get(r.verdict, 0) + 1
    denom = counts["verified"] + counts["refuted"] + counts["unverifiable"]
    return {"total_claims": total, "grounding_rate": counts["verified"] / denom if denom else 0.0,
            "hallucination_rate": counts["refuted"] / total, **counts, "by_claim_type": by_type}



def _plot_format_comparison(format_scores: dict[str, dict], analysis_type: str, output_dir: str) -> None:
    formats = list(format_scores.keys())
    rates = [format_scores[f].get("grounding_rate", 0) * 100 for f in formats]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(formats, rates, color=["#4f86c6", "#e07b54", "#6aab6a"][:len(formats)],
                  edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Grounding rate (%)")
    ax.set_title(f"Grounding Rate by Input Format\n({analysis_type})")
    fig.tight_layout()
    save_figure(fig, f"grounding_format_comparison_{analysis_type}", output_dir)


def _plot_grounding_summary(type_scores: dict[str, dict], fmt_name: str, output_dir: str) -> None:
    analysis_types = list(type_scores.keys())
    rates = [type_scores[t].get("grounding_rate", 0) * 100 for t in analysis_types]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(analysis_types, rates, color=["#4f86c6"] * len(analysis_types), edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Grounding rate (%)")
    ax.set_title(f"Grounding Rate by Analysis Type ({fmt_name} format)")
    ax.set_xticklabels(analysis_types, rotation=15, ha="right")
    fig.tight_layout()
    save_figure(fig, f"grounding_by_type_{fmt_name}", output_dir)


async def _run_single_format(analytics: dict, fmt_name: str, fmt_fn, analysis_type: str,
                             provider, judge_provider) -> dict:
    """Generate commentary, extract claims, verify. Returns score dict."""
    from services.tactical import TacticalAnalyzer
    grounded_input = fmt_fn(analytics)
    if fmt_name == "markdown":
        commentary = (await TacticalAnalyzer(provider=provider).analyze(analytics, analysis_type))["content"]
    else:
        from services.tactical import SYSTEM_PROMPTS
        commentary = await provider.generate(
            SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["match_overview"]), grounded_input
        )
    claims = await extract_claims(commentary, judge_provider)
    results = [verify_claim(c, analytics) for c in claims]
    score = compute_grounding_score(results)
    return {
        "format": fmt_name, "analysis_type": analysis_type, "commentary": commentary,
        "claims": [asdict(c) for c in claims],
        "verification_results": [{**asdict(r.claim), "verdict": r.verdict,
                                   "actual_value": str(r.actual_value), "explanation": r.explanation}
                                  for r in results],
        "score": score,
    }


_REAL_PROVIDERS = ["gemini", "openai", "huggingface"]
_ANALYSIS_TYPES = ["match_overview", "tactical_deep_dive", "event_analysis", "player_spotlight"]


async def _run_provider(provider_name: str, analytics: dict, out: str) -> dict[str, dict[str, dict]]:
    """Run all format x analysis_type combos for one provider.

    Returns format_type_scores: {fmt -> {analysis_type -> score}}.
    Skips and returns {} if provider is not available.
    """
    from services.llm_providers import get_provider

    provider = get_provider(provider_name)
    if not provider.is_available():
        print(f"  Skipping {provider_name} (no API key)")
        return {}

    judge_provider = get_provider(provider_name)
    formats_to_run = list(FORMATTERS.keys())
    all_results: dict[str, dict] = {}
    fmt_scores: dict[str, dict[str, dict]] = {f: {} for f in formats_to_run}

    for atype in _ANALYSIS_TYPES:
        for fmt_name, fmt_fn in FORMATTERS.items():
            print(f"  [{provider_name}][{fmt_name}] {atype} ...", end=" ", flush=True)
            result = await _run_single_format(analytics, fmt_name, fmt_fn, atype, provider, judge_provider)
            sc = result["score"]
            print(f"grounding={sc['grounding_rate']:.1%}  claims={sc['total_claims']}  verified={sc['verified']}")
            all_results[f"{provider_name}_{fmt_name}_{atype}"] = result
            fmt_scores[fmt_name][atype] = sc

    # Save artifacts
    artifacts_dir = Path(out) / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for key, res in all_results.items():
        (artifacts_dir / f"{key}.json").write_text(json.dumps(res, indent=2))

    # Format comparison table
    ptype = "match_overview"
    save_latex_table(
        headers=["Format", "Claims", "Verified", "Refuted", "Unverifiable", "Grounding Rate", "Hallucination Rate"],
        rows=[[fn, *[fmt_scores[fn].get(ptype, {}).get(k, 0) for k in ("total_claims", "verified", "refuted", "unverifiable")],
               f"{fmt_scores[fn].get(ptype, {}).get('grounding_rate', 0):.1%}",
               f"{fmt_scores[fn].get(ptype, {}).get('hallucination_rate', 0):.1%}"] for fn in formats_to_run],
        caption=f"Grounding rate by input format — {provider_name} ({ptype})",
        name=f"{provider_name}_grounding_format_comparison", output_dir=out,
        label=f"tab:grounding_format_{provider_name}",
    )
    # Per-type table (markdown format)
    save_latex_table(
        headers=["Analysis Type", "Claims", "Verified", "Refuted", "Grounding Rate"],
        rows=[[at.replace("_", " ").title(), *[fmt_scores["markdown"].get(at, {}).get(k, 0) for k in ("total_claims", "verified", "refuted")],
               f"{fmt_scores['markdown'].get(at, {}).get('grounding_rate', 0):.1%}"] for at in _ANALYSIS_TYPES],
        caption=f"Grounding rate by analysis type — {provider_name} (markdown format)",
        name=f"{provider_name}_grounding_by_type", output_dir=out,
        label=f"tab:grounding_by_type_{provider_name}",
    )
    _plot_format_comparison({f: fmt_scores[f].get(ptype, {}) for f in formats_to_run}, f"{provider_name}_{ptype}", out)
    _plot_grounding_summary(fmt_scores["markdown"], f"{provider_name}_markdown", out)

    # Example claims for dissertation
    verified_examples: list = []
    refuted_examples: list = []
    for key, res in all_results.items():
        if "markdown" not in key:
            continue
        for vr in res.get("verification_results", []):
            if vr["verdict"] == "verified" and len(verified_examples) < 5:
                verified_examples.append(vr)
            elif vr["verdict"] == "refuted" and len(refuted_examples) < 5:
                refuted_examples.append(vr)
    (Path(out) / f"{provider_name}_example_claims.json").write_text(
        json.dumps({"verified": verified_examples, "refuted": refuted_examples}, indent=2)
    )
    return fmt_scores


async def run_async(config: EvalConfig) -> dict:
    out = str(config.output_dir)
    ensure_output_dir(out)
    analytics = load_analytics(config.analytics_path)

    providers_to_run = _REAL_PROVIDERS if config.provider == "all" else [config.provider]

    print(f"\n=== LLM Grounding Evaluator ===")
    print(f"Provider(s): {', '.join(providers_to_run)}")
    print(f"Formats:     {', '.join(FORMATTERS.keys())}")
    print(f"Analysis types: {', '.join(_ANALYSIS_TYPES)}\n")

    # provider_name -> format -> analysis_type -> score
    all_provider_scores: dict[str, dict[str, dict[str, dict]]] = {}
    for provider_name in providers_to_run:
        fmt_type_scores = await _run_provider(provider_name, analytics, out)
        if fmt_type_scores:
            all_provider_scores[provider_name] = fmt_type_scores

    # Cross-provider comparison table (grounding_rate per provider per analysis_type)
    if len(all_provider_scores) > 1:
        cp_rows = []
        for atype in _ANALYSIS_TYPES:
            row = [atype.replace("_", " ").title()]
            for pname in all_provider_scores:
                rate = all_provider_scores[pname].get("markdown", {}).get(atype, {}).get("grounding_rate", 0)
                row.append(f"{rate:.1%}")
            cp_rows.append(row)
        save_latex_table(
            headers=["Analysis Type"] + list(all_provider_scores.keys()),
            rows=cp_rows,
            caption="Cross-provider grounding rate comparison (markdown format)",
            name="grounding_provider_comparison",
            output_dir=out,
            label="tab:grounding_provider_comparison",
        )

    print(f"\nOutputs saved to: {out}/")
    return {"provider_scores": all_provider_scores}


def run(config: EvalConfig) -> dict:
    return asyncio.run(run_async(config))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM commentary grounding rate evaluator")
    parser.add_argument("--analytics", required=True, help="Path to *_analytics.json")
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "huggingface", "all", "stub"],
    )
    parser.add_argument("--output", default="eval_output/grounding")
    args = parser.parse_args()

    config = EvalConfig(
        analytics_path=args.analytics,
        provider=args.provider,
        output_dir=args.output,
    )
    run(config)


if __name__ == "__main__":
    main()
