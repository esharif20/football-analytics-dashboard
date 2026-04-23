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



def format_as_markdown(analytics: dict, include_insights: bool = True) -> str:
    """Current production formatter — structured markdown tables.

    Args:
        include_insights: If True (default), prepends MatchInsights pre-interpreted
            findings (Key Findings section). Set to False for the "structured markdown
            only" condition in the 3-way format comparison.
    """
    from services.tactical import GroundingFormatter
    return GroundingFormatter.format(analytics, include_insights=include_insights)


def format_as_markdown_no_insights(analytics: dict) -> str:
    """Structured markdown tables WITHOUT the MatchInsights interpretation layer.

    Used as the middle condition in the 3-way format comparison:
      Raw JSON  →  Structured Markdown  →  Markdown + MatchInsights
    """
    return format_as_markdown(analytics, include_insights=False)


def format_as_raw_json(analytics: dict) -> str:
    """Raw JSON — what the assessor warned against (the control condition)."""
    # Strip per-frame arrays to stay within LLM context limits
    _PER_FRAME_PLAYER_KEYS = {"speeds_px_per_frame", "speeds_m_per_sec"}
    _DROP_TOP_LEVEL = {"ball_path_positions"}
    stripped: dict = {}
    for k, v in analytics.items():
        if k in _DROP_TOP_LEVEL:
            continue
        if k == "player_kinematics" and isinstance(v, dict):
            stripped[k] = {
                pid: {pk: pv for pk, pv in pdata.items() if pk not in _PER_FRAME_PLAYER_KEYS}
                for pid, pdata in v.items()
            }
        elif k == "ball_path" and isinstance(v, dict):
            stripped[k] = {pk: pv for pk, pv in v.items()
                           if pk not in ("positions", "pitch_positions")}
        elif k == "possession" and isinstance(v, dict):
            stripped[k] = {pk: pv for pk, pv in v.items() if pk != "events"}
        else:
            stripped[k] = v
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
    "markdown_no_insights": format_as_markdown_no_insights,
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


async def extract_claims_stable(
    commentary: str, provider, n_extract: int = 3
) -> list[Claim]:
    """Majority-vote claim extraction: run extractor n_extract times, union results.

    Reduces LLM-variance-induced n_claims=0 edge cases that arise with narrative
    event prose. Claims are deduplicated by normalised text (lowercase, stripped).
    """
    seen: dict[str, Claim] = {}
    for _ in range(n_extract):
        batch = await extract_claims(commentary, provider)
        for c in batch:
            key = c.text.lower().strip()[:120]
            if key and key not in seen:
                seen[key] = c
    return list(seen.values())


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



def _get_nested(d: dict, dotpath: str | None) -> Any:
    """Navigate a dot-separated path in a nested dict."""
    if not dotpath:
        return None
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


# ── Tactical metric helpers for qualitative rules ─────────────────────────

def _get_tactical_summary(a: dict) -> dict:
    """Get tactical.summary from analytics, handling both nested and flat forms."""
    t = a.get("tactical", {})
    return t.get("summary", {}) if isinstance(t, dict) else {}


def _check_compact(a: dict) -> tuple:
    """Check 'compact' claims against convex hull compactness."""
    s = _get_tactical_summary(a)
    t1 = s.get("team_1_avg_compactness_m2")
    t2 = s.get("team_2_avg_compactness_m2")
    if t1 is not None and t2 is not None:
        is_compact = min(t1, t2) < 1000  # < 1000m² is compact
        return (is_compact, f"compactness T1={t1:.0f}m² T2={t2:.0f}m²")
    # Fallback to possession changes
    return (a.get("possession", {}).get("possession_changes", 0) > 10, "no tactical data, used possession_changes")


def _check_high_line(a: dict) -> tuple:
    """Check 'high line' / 'deep block' claims against defensive line height."""
    s = _get_tactical_summary(a)
    t1 = s.get("team_1_avg_defensive_line_m")
    t2 = s.get("team_2_avg_defensive_line_m")
    if t1 is not None and t2 is not None:
        high = max(t1, t2) > 40  # > 40m from goal = high line
        return (high, f"def_line T1={t1:.1f}m T2={t2:.1f}m")
    return (False, "no defensive line data")


def _check_press(a: dict) -> tuple:
    """Check pressing claims against pressing intensity + PPDA."""
    s = _get_tactical_summary(a)
    t1_pi = s.get("team_1_avg_pressing_intensity")
    t2_pi = s.get("team_2_avg_pressing_intensity")
    ppda_1 = s.get("ppda_team_1")
    ppda_2 = s.get("ppda_team_2")
    if t1_pi is not None and t2_pi is not None:
        high_press = max(t1_pi, t2_pi) > 0.3
        evidence = f"pressing T1={t1_pi:.2f} T2={t2_pi:.2f}"
        if ppda_1 is not None and ppda_2 is not None:
            evidence += f", PPDA T1={ppda_1:.1f} T2={ppda_2:.1f}"
        return (high_press, evidence)
    # Fallback
    return (a.get("possession", {}).get("possession_changes", 0) > 10, "possession_changes")


def _check_stretched(a: dict) -> tuple:
    """Check 'stretched' / 'spread' claims against stretch index."""
    s = _get_tactical_summary(a)
    t1 = s.get("team_1_avg_stretch_index_m")
    t2 = s.get("team_2_avg_stretch_index_m")
    if t1 is not None and t2 is not None:
        stretched = max(t1, t2) > 20  # > 20m mean dist = stretched
        return (stretched, f"stretch_index T1={t1:.1f}m T2={t2:.1f}m")
    return (False, "no stretch index data")


def _check_event(a: dict, event_type: str) -> tuple:
    """Check event-type claims (pass, shot, challenge) against the events array."""
    events = a.get("events", [])
    matched = [e for e in events if e.get("event_type") == event_type]
    found = len(matched) > 0
    return (found, f"{len(matched)} '{event_type}' events in analytics")


# Qualitative claim rules: (keyword pattern, analytics check function)
_QUALITATIVE_RULES = [
    ("compact", _check_compact),
    ("stretched", _check_stretched),
    ("spread", _check_stretched),
    ("high line", _check_high_line),
    ("deep block", _check_high_line),
    ("deep defen", _check_high_line),
    ("press", _check_press),
    ("dominat", lambda a: (
        abs(a.get("possession", {}).get("team_1_percentage", 50) - 50) > 15, "possession imbalance > 15%"
    )),
    ("fast", lambda a: (
        any(
            s.get("max_speed_m_per_sec", 0) * 3.6 > 20
            for s in a.get("player_kinematics", {}).values()
        ), "player max_speed > 20 km/h"
    )),
    # Event-type rules — verify claims like "Player #X passes to #Y" or "shot attempt"
    ("pass",         lambda a: _check_event(a, "pass")),
    ("cross",        lambda a: _check_event(a, "pass")),      # crosses are pass subtypes
    ("through ball", lambda a: _check_event(a, "pass")),
    ("shot",         lambda a: _check_event(a, "shot")),
    ("challeng",     lambda a: _check_event(a, "challenge")),
    ("tackl",        lambda a: _check_event(a, "challenge")),
    ("duel",         lambda a: _check_event(a, "challenge")),
    ("intercept",    lambda a: _check_event(a, "challenge")),
]


def _search_tactical_summary(analytics: dict, claimed_num: float, claim_text: str) -> tuple:
    """Search tactical.summary and possession for a value matching the claim.

    Returns (actual_value, key_path) or (None, None).
    """
    text_lower = claim_text.lower()

    # Determine which team the claim references
    team_hint = None
    if "team 1" in text_lower or "team 0" in text_lower:
        team_hint = "team_1"
    elif "team 2" in text_lower:
        team_hint = "team_2"

    # Search tactical summary
    tactical = analytics.get("tactical", {})
    summary = tactical.get("summary", {}) if isinstance(tactical, dict) else {}
    for key, val in summary.items():
        if val is None or not isinstance(val, (int, float)):
            continue
        # If team hint, prefer matching key
        if team_hint and team_hint not in key:
            continue
        if abs(val - claimed_num) <= max(abs(val) * 0.05, 0.5):
            return val, f"tactical.summary.{key}"

    # Also search without team filter if no exact match
    if team_hint:
        for key, val in summary.items():
            if val is None or not isinstance(val, (int, float)):
                continue
            if abs(val - claimed_num) <= max(abs(val) * 0.05, 0.5):
                return val, f"tactical.summary.{key}"

    # Search possession stats
    poss = analytics.get("possession", {})
    for key, val in poss.items():
        if val is None or not isinstance(val, (int, float)):
            continue
        if abs(val - claimed_num) <= max(abs(val) * 0.05, 0.5):
            return val, f"possession.{key}"

    return None, None


def verify_claim(claim: Claim, analytics: dict) -> VerificationResult:  # noqa: C901
    """Verify a single claim against the source analytics dict."""
    def _unverifiable(msg: str, val: Any = None) -> VerificationResult:
        return VerificationResult(claim=claim, verdict="unverifiable", actual_value=val, explanation=msg)

    if claim.claim_type == "numeric":
        actual = _get_nested(analytics, claim.referenced_metric)
        claimed_num = _parse_numeric(claim.referenced_value)

        # Fallback: if dot-path failed, search tactical summary + top-level for matching value
        if actual is None and claimed_num is not None:
            actual, found_key = _search_tactical_summary(analytics, claimed_num, claim.text)
            if actual is not None:
                claim.referenced_metric = found_key  # Update for reporting

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

    # Event-timing / event-participant resolver: runs for ANY claim type when
    # the claim text mentions a player track ID + event keyword. Promotes
    # otherwise-qualitative event claims to "verified" via analytics.events.
    text_lower_ev = claim.text.lower()
    event_kw = next(
        (k for k in ("challenge", "tackle", "pass", "shot", "interception", "clearance")
         if k in text_lower_ev),
        None,
    )
    player_ids = re.findall(r"#(\d+)", claim.text)
    if event_kw and player_ids:
        events = analytics.get("events", [])
        if not events:
            # Try nested path analytics.events (some schemas nest it)
            events = _get_nested(analytics, "events") or []
        matching = [
            e for e in events
            if isinstance(e, dict) and (
                e.get("event_type", "").lower() == event_kw
                or event_kw in e.get("event_type", "").lower()
            )
        ]
        if player_ids and matching:
            pid_set = {int(p) for p in player_ids}
            player_match = [
                e for e in matching
                if e.get("player_id") in pid_set
                or e.get("track_id") in pid_set
                or int(e.get("player_id", -1)) in pid_set
                or int(e.get("track_id", -1)) in pid_set
            ]
            if player_match:
                return VerificationResult(
                    claim=claim,
                    verdict="verified",
                    actual_value=f"{len(player_match)} matching event(s)",
                    explanation=(
                        f"Event resolver: found {len(player_match)} '{event_kw}' "
                        f"event(s) for player(s) {player_ids} in analytics.events"
                    ),
                )
        if matching and not player_ids:
            # Event type exists but no specific player — treat as plausible
            return VerificationResult(
                claim=claim, verdict="plausible",
                actual_value=f"{len(matching)} {event_kw} events exist",
                explanation=f"Event resolver: {len(matching)} '{event_kw}' events found",
            )

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



# ── FActScore + Wiseman content selection metrics (B2) ────────────────────────

def classify_hallucination_type(result: VerificationResult) -> str:
    """Classify hallucination following FActScore framework (Min et al. 2023).

    - "intrinsic": claim directly contradicts source data (refuted)
    - "extrinsic": claim cannot be verified from source data (unverifiable)
    - "grounded": claim is supported (verified or plausible)
    """
    if result.verdict == "verified" or result.verdict == "plausible":
        return "grounded"
    if result.verdict == "refuted":
        return "intrinsic"
    return "extrinsic"


def compute_factscore_breakdown(results: list[VerificationResult]) -> dict:
    """FActScore-aligned breakdown (Min et al. 2023 EMNLP).

    Returns:
        {
            "grounded": int,
            "intrinsic_hallucinations": int,
            "extrinsic_hallucinations": int,
            "factscore": float,  # grounded / total
            "intrinsic_rate": float,
            "extrinsic_rate": float,
        }
    """
    if not results:
        return {}
    cats = [classify_hallucination_type(r) for r in results]
    grounded = cats.count("grounded")
    intrinsic = cats.count("intrinsic")
    extrinsic = cats.count("extrinsic")
    total = len(results)
    return {
        "grounded": grounded,
        "intrinsic_hallucinations": intrinsic,
        "extrinsic_hallucinations": extrinsic,
        "factscore": round(grounded / total, 4) if total else 0.0,
        "intrinsic_rate": round(intrinsic / total, 4) if total else 0.0,
        "extrinsic_rate": round(extrinsic / total, 4) if total else 0.0,
    }


# ── Error Taxonomy (Chirkova MME 2026 + Sports Intelligence Yang 2025) ─────────


def classify_error_type(result: VerificationResult) -> str:
    """Fine-grained error taxonomy for failed/unverifiable claims.

    Five categories (Chirkova et al. MME 2026, adapted for football analytics):
      - fabricated_statistic: LLM invented a number not in the data at all
      - wrong_attribution: correct metric value but attributed to wrong team/player
      - unsupported_inference: plausible tactical claim with no data backing
      - magnitude_error: correct metric, wrong value (within same order of magnitude)
      - context_confusion: misinterprets what a metric means (e.g. PPDA direction)
      - grounded: not an error

    Returns one of the above strings.
    """
    if result.verdict in ("verified", "plausible"):
        return "grounded"

    claim = result.claim
    text_lower = claim.text.lower()
    explained = (result.explanation or "").lower()

    # context_confusion: PPDA direction, compactness direction errors
    _CONFUSION_PATTERNS = [
        ("ppda", "higher", "aggressive"),      # higher PPDA ≠ more aggressive
        ("ppda", "more", "aggressive"),
        ("compact", "higher", "compact"),      # higher compactness m² ≠ more compact
        ("compact", "larger", "compact"),
    ]
    for metric, wrong_dir, wrong_interp in _CONFUSION_PATTERNS:
        if metric in text_lower and wrong_dir in text_lower and wrong_interp in text_lower:
            return "context_confusion"

    # magnitude_error: claim mentions a number in the right range but wrong value
    # Check this BEFORE wrong_attribution — same-metric wrong-value errors should not
    # be confused with team attribution errors.
    if claim.claim_type == "numeric" and result.verdict == "refuted":
        claimed_num = _parse_numeric(claim.referenced_value)
        actual = result.actual_value
        if claimed_num is not None and actual is not None:
            try:
                actual_f = float(actual)
                ratio = abs(claimed_num - actual_f) / max(abs(actual_f), 1e-6)
                if ratio < 0.50:  # within 50% — same ballpark, wrong value
                    return "magnitude_error"
            except (TypeError, ValueError):
                pass

    # wrong_attribution: claim names one team for a value that belongs to the other
    # Only applies when both teams are mentioned OR one team is mentioned with a value
    # that is actually the other team's — indicated by the referenced_metric being
    # team-specific and the verdict being refuted after passing the magnitude check.
    if claim.claim_type in ("numeric", "comparative"):
        has_team1 = "team 1" in text_lower
        has_team2 = "team 2" in text_lower
        if result.verdict == "refuted" and (has_team1 ^ has_team2):
            if result.actual_value is not None and "%" in claim.referenced_value:
                return "wrong_attribution"

    # unsupported_inference: qualitative claims that have no data support
    if claim.claim_type == "qualitative" and result.verdict == "unverifiable":
        return "unsupported_inference"

    # fabricated_statistic: numeric claim, but metric path not found at all
    if claim.claim_type == "numeric" and result.verdict == "unverifiable":
        if "could not resolve" in explained or "not found" in explained:
            return "fabricated_statistic"

    # Default for refuted numeric/comparative
    if result.verdict == "refuted":
        return "fabricated_statistic"

    return "unsupported_inference"


def compute_error_taxonomy(results: list[VerificationResult]) -> dict:
    """Compute error distribution across taxonomy categories.

    Returns:
        {
            "total": int,
            "grounded": int,
            "error_counts": {category: count},
            "error_rates": {category: float},
            "dominant_error": str,   # most common error type
        }
    """
    if not results:
        return {}
    cats = [classify_error_type(r) for r in results]
    total = len(cats)
    error_types = [
        "fabricated_statistic", "wrong_attribution", "unsupported_inference",
        "magnitude_error", "context_confusion",
    ]
    error_counts = {et: cats.count(et) for et in error_types}
    grounded = cats.count("grounded")
    error_total = total - grounded
    error_rates = {
        et: round(c / total, 4) for et, c in error_counts.items()
    }
    dominant = max(error_counts, key=lambda k: error_counts[k]) if error_total > 0 else "none"
    return {
        "total": total,
        "grounded": grounded,
        "error_total": error_total,
        "error_counts": error_counts,
        "error_rates": error_rates,
        "dominant_error": dominant,
    }


def _extract_numeric_values(text: str) -> list[float]:
    """Extract all numbers from commentary text."""
    return [float(m) for m in re.findall(r"\d+(?:\.\d+)?", text)]


def _flatten_analytics_numerics(analytics: dict) -> dict[str, float]:
    """Flatten all numeric scalars from analytics into a flat key→value dict."""
    flat: dict[str, float] = {}
    poss = analytics.get("possession", {})
    for k, v in poss.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[f"possession.{k}"] = float(v)
    tac = analytics.get("tactical", {})
    summary = tac.get("summary", {}) if isinstance(tac, dict) else {}
    for k, v in summary.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[f"tactical.{k}"] = float(v)
    for pid, pk in analytics.get("player_kinematics", {}).items():
        for k in ("total_distance_m", "avg_speed_m_per_sec", "max_speed_m_per_sec"):
            v = pk.get(k) if isinstance(pk, dict) else None
            if v is not None:
                flat[f"player.{pid}.{k}"] = float(v)
    return flat


def compute_content_selection_metrics(
    commentary: str,
    analytics: dict,
) -> dict:
    """Wiseman et al. (2017) Rotowire content selection metrics.

    RG (Relation Generation) precision: of all numbers in the commentary,
    what fraction match a value in the analytics source?

    CS (Content Selection):
        Precision = |mentioned ∩ available| / |mentioned|
        Recall    = |mentioned ∩ available| / |available|

    Here, "available" = all numeric fields in analytics,
          "mentioned" = numeric values extracted from commentary.

    Returns:
        {rg_precision, cs_precision, cs_recall, cs_f1, mentioned_count, available_count}
    """
    available = _flatten_analytics_numerics(analytics)
    available_values = list(available.values())
    mentioned = _extract_numeric_values(commentary)

    if not mentioned:
        return {"rg_precision": 0.0, "cs_precision": 0.0, "cs_recall": 0.0, "cs_f1": 0.0,
                "mentioned_count": 0, "available_count": len(available_values)}

    # RG precision: fraction of mentioned numbers that exist in analytics (5% tolerance)
    def _matches_any(val: float, candidates: list[float]) -> bool:
        return any(abs(val - c) <= max(abs(c) * 0.05, 0.5) for c in candidates)

    rg_matched = [v for v in mentioned if _matches_any(v, available_values)]
    rg_precision = len(rg_matched) / len(mentioned) if mentioned else 0.0

    # CS: use metric keys mentioned vs available (by matching key name in commentary)
    available_keys = set(available.keys())
    # A key is "mentioned" if a substring of its name appears in the commentary
    # Map short names to the longer key names
    _SHORT_NAMES = {
        "possession": "possession.",
        "compactness": "tactical.team",
        "stretch": "tactical.team",
        "pressing": "tactical.team",
        "ppda": "tactical.ppda",
        "distance": "player.",
        "speed": "player.",
        "territory": "tactical.team",
    }
    mentioned_keys: set[str] = set()
    commentary_lower = commentary.lower()
    for short, prefix in _SHORT_NAMES.items():
        if short in commentary_lower:
            for k in available_keys:
                if k.startswith(prefix):
                    mentioned_keys.add(k)

    intersect = mentioned_keys & available_keys
    cs_p = len(intersect) / len(mentioned_keys) if mentioned_keys else 0.0
    cs_r = len(intersect) / len(available_keys) if available_keys else 0.0
    cs_f1 = (2 * cs_p * cs_r / (cs_p + cs_r)) if (cs_p + cs_r) > 0 else 0.0

    return {
        "rg_precision": round(rg_precision, 4),
        "rg_matched_count": len(rg_matched),
        "cs_precision": round(cs_p, 4),
        "cs_recall": round(cs_r, 4),
        "cs_f1": round(cs_f1, 4),
        "mentioned_count": len(mentioned),
        "available_count": len(available_values),
    }


def compute_grounding_score(
    results: list[VerificationResult],
    commentary: str = "",
    analytics: dict | None = None,
) -> dict:
    """Compute grounding rate, hallucination rate, FActScore, and content selection."""
    total = len(results)
    if total == 0:
        return {"grounding_rate": 0.0, "hallucination_rate": 0.0, "total_claims": 0,
                "verified": 0, "refuted": 0, "unverifiable": 0, "plausible": 0, "by_claim_type": {}}
    counts = {v: 0 for v in ("verified", "refuted", "unverifiable", "plausible")}
    by_type: dict[str, dict] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
        ct = r.claim.claim_type
        by_type.setdefault(ct, {"verified": 0, "refuted": 0, "unverifiable": 0, "plausible": 0})
        by_type[ct][r.verdict] = by_type[ct].get(r.verdict, 0) + 1
    denom = counts["verified"] + counts["refuted"] + counts["unverifiable"]
    base = {"total_claims": total, "grounding_rate": counts["verified"] / denom if denom else 0.0,
            "hallucination_rate": counts["refuted"] / total, **counts, "by_claim_type": by_type}

    # FActScore breakdown (Min et al. 2023)
    base["factscore"] = compute_factscore_breakdown(results)

    # Error taxonomy (Chirkova MME 2026)
    base["error_taxonomy"] = compute_error_taxonomy(results)

    # Content selection (Wiseman et al. 2017)
    if commentary and analytics:
        base["content_selection"] = compute_content_selection_metrics(commentary, analytics)

    return base



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
    # Truncate context to ~12k chars to avoid API request size limits
    _MAX_CONTEXT_CHARS = 12_000
    if len(grounded_input) > _MAX_CONTEXT_CHARS:
        grounded_input = grounded_input[:_MAX_CONTEXT_CHARS] + "\n... [truncated]"
    if fmt_name == "markdown":
        commentary = (await TacticalAnalyzer(provider=provider).analyze(analytics, analysis_type))["content"]
    else:
        from services.tactical import SYSTEM_PROMPTS
        commentary = await provider.generate(
            SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["match_overview"]), grounded_input
        )
    claims = await extract_claims(commentary, judge_provider)
    results = [verify_claim(c, analytics) for c in claims]
    score = compute_grounding_score(results, commentary=commentary, analytics=analytics)
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
    artifacts_dir.mkdir(parents=True, exist_ok=True)
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
        try:
            fmt_type_scores = await _run_provider(provider_name, analytics, out)
        except Exception as exc:
            print(f"  Skipping {provider_name} (error: {type(exc).__name__}: {exc!s:.120})")
            continue
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
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env", override=True)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="LLM commentary grounding rate evaluator")
    parser.add_argument("--analytics", required=True, help="Path to *_analytics.json")
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "huggingface", "claude", "groq", "all", "stub"],
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
