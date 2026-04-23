"""DB-grounded claim verification — 5 additional verification dimensions.

Extends the FActScore-style claim verification from llm_grounding.py with
five new dimensions that use per-frame database data as ground truth, allowing
previously "unverifiable" (extrinsic) claims to be resolved.

Dimensions:
  1. Spatial    — player mean position vs. team centroid, pitch thirds
  2. Temporal   — possession/event patterns in named time windows
  3. Event-Spatial — event startX/endX vs. claimed pitch zones
  4. Trajectory — ball circulation location and variance (quick transitions)
  5. Cross-Frame Consistency — monotonic ordering of time references in commentary

Usage:
    python3 -m backend.evaluation.db_grounding \\
        --ground-truth eval_output/18_db_ground_truth.json \\
        --claims eval_output/claims.json \\
        --commentary eval_output/commentary.txt \\
        --output eval_output/dissertation/db_grounded/
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean, variance
from typing import Any

# Add backend/ to sys.path for relative imports when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Pitch constants (playerPositions.pitchX/pitchY are in metres) ─────────────

PITCH_LENGTH_M = 105.0   # X direction
PITCH_WIDTH_M  =  68.0   # Y direction

DEFENSIVE_THIRD_MAX_M  = 35.0
MIDDLE_THIRD_MAX_M     = 70.0

SPATIAL_OFFSET_M = 10.0  # 10m threshold for "deep" / "high" classification


# ── Keyword lists ─────────────────────────────────────────────────────────────

_SPATIAL_KEYWORDS = [
    "deep", "high line", "defensive line", "pushed up", "dropped back",
    "defensive position", "attacking position", "defensive third", "middle third",
    "attacking third", "final third", "own half", "opposition half",
    "wide", "narrow", "flank", "channel", "deep-lying", "advanced",
]

_TEMPORAL_KEYWORDS = [
    "first half", "second half", "early", "later", "throughout",
    "minute", "opening", "closing", "beginning", "end of",
    "dominated possession early", "dominated early", "controlled later",
    "in the first", "in the second", "throughout the",
]

_EVENT_SPATIAL_KEYWORDS = [
    "progressive pass", "pass from midfield", "pass into final third",
    "pass into the final third", "pass from defensive", "challenge in defensive",
    "cross from the flank", "cross from flank", "shot from outside",
    "clearance from", "pass from own half",
]

_TRAJECTORY_KEYWORDS = [
    "ball circulation", "kept possession in their half", "pushed the ball forward",
    "defensive build-up", "build-up", "quick transitions", "transition",
    "patient build-up", "patient play", "high tempo", "direct play",
    "long ball", "played out from the back",
]


# ── Low-level helpers ─────────────────────────────────────────────────────────


def _classify_pitch_zone(pitch_x_m: float) -> str:
    """Return 'defensive', 'middle', or 'attacking' based on pitchX (metres)."""
    if pitch_x_m < DEFENSIVE_THIRD_MAX_M:
        return "defensive"
    if pitch_x_m < MIDDLE_THIRD_MAX_M:
        return "middle"
    return "attacking"


def _detect_spatial_keywords(text: str) -> list[str]:
    """Return spatial keywords found in the claim text."""
    tl = text.lower()
    return [kw for kw in _SPATIAL_KEYWORDS if kw in tl]


def _detect_temporal_keywords(text: str) -> list[str]:
    """Return temporal keywords found in the claim text."""
    tl = text.lower()
    return [kw for kw in _TEMPORAL_KEYWORDS if kw in tl]


def _detect_event_spatial_keywords(text: str) -> list[str]:
    """Return event-spatial trigger keywords found in the claim text."""
    tl = text.lower()
    return [kw for kw in _EVENT_SPATIAL_KEYWORDS if kw in tl]


def _detect_trajectory_keywords(text: str) -> list[str]:
    """Return trajectory trigger keywords found in the claim text."""
    tl = text.lower()
    return [kw for kw in _TRAJECTORY_KEYWORDS if kw in tl]


def _extract_player_ids(text: str) -> list[int]:
    """Extract player track IDs from claim text — e.g. '#7' -> 7."""
    return [int(m) for m in re.findall(r"#(\d+)", text)]


def _extract_team_id(text: str) -> int | None:
    """Extract team ID (1 or 2) from claim text."""
    tl = text.lower()
    if "team 1" in tl or "team one" in tl:
        return 1
    if "team 2" in tl or "team two" in tl:
        return 2
    return None


def _get_player_mean_position(
    player_id: int, frames: list[dict]
) -> tuple[float, float] | None:
    """Compute mean (pitchX, pitchY) for a player across all frames where they appear.

    Expects frames in the shape produced by db_extractor:
        frame["playerPositions"][str(player_id)] == {"pitchX": ..., "pitchY": ...}
    """
    xs: list[float] = []
    ys: list[float] = []
    pid_str = str(player_id)
    for frame in frames:
        positions = frame.get("playerPositions") or {}
        entry = positions.get(pid_str) or positions.get(player_id)
        if not entry:
            continue
        px = entry.get("pitchX")
        py = entry.get("pitchY")
        if px is not None and py is not None:
            try:
                xs.append(float(px))
                ys.append(float(py))
            except (TypeError, ValueError):
                pass
    if not xs:
        return None
    return (mean(xs), mean(ys))


def _get_team_centroid_mean(
    team_id: int, frames: list[dict]
) -> tuple[float, float] | None:
    """Compute mean team centroid (pitchX, pitchY) across all frames.

    Looks for frame["teamCentroids"][str(team_id)] or ["teamCentroids"][team_id].
    Falls back to averaging all players with matching teamId in playerPositions.
    """
    xs: list[float] = []
    ys: list[float] = []
    tid_str = str(team_id)
    for frame in frames:
        # Preferred: explicit centroid data
        centroids = frame.get("teamCentroids") or {}
        centroid = centroids.get(tid_str) or centroids.get(team_id)
        if centroid:
            cx = centroid.get("pitchX")
            cy = centroid.get("pitchY")
            if cx is not None and cy is not None:
                try:
                    xs.append(float(cx))
                    ys.append(float(cy))
                    continue
                except (TypeError, ValueError):
                    pass

        # Fallback: average players whose teamId matches
        positions = frame.get("playerPositions") or {}
        frame_xs: list[float] = []
        frame_ys: list[float] = []
        for entry in positions.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("teamId") != team_id and entry.get("teamId") != tid_str:
                continue
            px = entry.get("pitchX")
            py = entry.get("pitchY")
            if px is not None and py is not None:
                try:
                    frame_xs.append(float(px))
                    frame_ys.append(float(py))
                except (TypeError, ValueError):
                    pass
        if frame_xs:
            xs.append(mean(frame_xs))
            ys.append(mean(frame_ys))

    if not xs:
        return None
    return (mean(xs), mean(ys))


# ── Dimension 1: Spatial Claim Verification ───────────────────────────────────


def _spatial_direction_from_keywords(keywords: list[str]) -> str | None:
    """Infer expected spatial direction from keyword list.

    Returns 'deep' (defensive), 'high' (attacking), 'wide', or None.
    """
    deep_kw = {"deep", "dropped back", "defensive position", "defensive third",
               "defensive line", "own half", "deep-lying"}
    high_kw = {"pushed up", "high line", "attacking position", "attacking third",
               "final third", "opposition half", "advanced"}
    wide_kw = {"wide", "flank", "channel"}

    found_kw = set(keywords)
    if found_kw & high_kw:
        return "high"
    if found_kw & deep_kw:
        return "deep"
    if found_kw & wide_kw:
        return "wide"
    return None


def _verify_spatial_claim(
    claim_dict: dict, frames: list[dict]
) -> dict:
    """Verify a single spatial claim against per-frame position data."""
    text = claim_dict.get("text", "")
    keywords = _detect_spatial_keywords(text)

    if not keywords:
        return {
            "claim_text": text,
            "verdict": "unresolvable",
            "evidence": "no spatial keywords found",
        }

    player_ids = _extract_player_ids(text)
    if not player_ids:
        return {
            "claim_text": text,
            "verdict": "unresolvable",
            "evidence": "no player ID (#N) found in claim text",
        }

    pid = player_ids[0]
    player_pos = _get_player_mean_position(pid, frames)
    if player_pos is None:
        return {
            "claim_text": text,
            "verdict": "unresolvable",
            "evidence": f"no pitch coordinates available for player #{pid}",
        }

    player_x, _ = player_pos
    direction = _spatial_direction_from_keywords(keywords)
    zone = _classify_pitch_zone(player_x)

    if direction is None:
        # Pitch-zone claim (defensive/middle/attacking third)
        claimed_zone = None
        tl = text.lower()
        if "defensive third" in tl or "own half" in tl:
            claimed_zone = "defensive"
        elif "middle third" in tl:
            claimed_zone = "middle"
        elif "attacking third" in tl or "final third" in tl or "opposition half" in tl:
            claimed_zone = "attacking"

        if claimed_zone is None:
            return {
                "claim_text": text,
                "verdict": "unresolvable",
                "evidence": f"could not infer expected direction from keywords: {keywords}",
            }
        verdict = "verified" if zone == claimed_zone else "refuted"
        evidence = (
            f"player #{pid} mean pitchX={player_x:.1f}m → zone='{zone}'; "
            f"claim expects '{claimed_zone}'"
        )
        return {"claim_text": text, "verdict": verdict, "evidence": evidence}

    if direction == "wide":
        # "Wide" means pitchY near 0 or near 68m; define flanks as outer 25%
        _, player_y = player_pos
        is_wide = player_y < PITCH_WIDTH_M * 0.25 or player_y > PITCH_WIDTH_M * 0.75
        verdict = "verified" if is_wide else "refuted"
        evidence = (
            f"player #{pid} mean pitchY={player_y:.1f}m; "
            f"flank threshold <{PITCH_WIDTH_M*0.25:.1f}m or >{PITCH_WIDTH_M*0.75:.1f}m"
        )
        return {"claim_text": text, "verdict": verdict, "evidence": evidence}

    # "deep" or "high" — compare player X to team centroid X
    # Infer team from text, fall back to trying both
    team_id = _extract_team_id(text)
    centroid_pos = None
    if team_id is not None:
        centroid_pos = _get_team_centroid_mean(team_id, frames)
    if centroid_pos is None:
        # Try both teams, use the one closer in X
        for tid in (1, 2):
            cp = _get_team_centroid_mean(tid, frames)
            if cp and centroid_pos is None:
                centroid_pos = cp

    if centroid_pos is None:
        evidence = (
            f"player #{pid} mean pitchX={player_x:.1f}m; "
            f"no team centroid data available for comparison"
        )
        return {"claim_text": text, "verdict": "unresolvable", "evidence": evidence}

    centroid_x, _ = centroid_pos
    delta = player_x - centroid_x

    if direction == "deep":
        # Player behind (lower X) than centroid by >10m
        verdict = "verified" if delta < -SPATIAL_OFFSET_M else "refuted"
    else:  # "high"
        # Player ahead (higher X) than centroid by >10m
        verdict = "verified" if delta > SPATIAL_OFFSET_M else "refuted"

    evidence = (
        f"player #{pid} mean pitchX={player_x:.1f}m; "
        f"team centroid pitchX={centroid_x:.1f}m; "
        f"delta={delta:+.1f}m; threshold=±{SPATIAL_OFFSET_M}m; "
        f"claim='{direction}'"
    )
    return {"claim_text": text, "verdict": verdict, "evidence": evidence}


def run_spatial_dimension(
    claims: list[dict], frames: list[dict]
) -> dict:
    """Run Dim 1: Spatial Claim Verification across all claims."""
    details: list[dict] = []
    triggered = 0
    verified = 0
    refuted = 0
    unresolvable = 0

    for claim in claims:
        keywords = _detect_spatial_keywords(claim.get("text", ""))
        if not keywords:
            continue
        triggered += 1
        result = _verify_spatial_claim(claim, frames)
        details.append(result)
        v = result["verdict"]
        if v == "verified":
            verified += 1
        elif v == "refuted":
            refuted += 1
        else:
            unresolvable += 1

    rate = round(verified / triggered, 4) if triggered else 0.0
    return {
        "total_spatial_claims": triggered,
        "verified": verified,
        "refuted": refuted,
        "unresolvable": unresolvable,
        "rate": rate,
        "details": details,
    }


# ── Dimension 2: Temporal Claim Verification ──────────────────────────────────


def _infer_frame_window(keyword: str, total_frames: int) -> tuple[int, int] | None:
    """Map a temporal keyword to an approximate (start, end) frame index range."""
    if not total_frames:
        return None
    if "first half" in keyword:
        return (0, total_frames // 2)
    if "second half" in keyword:
        return (total_frames // 2, total_frames)
    if "early" in keyword or "opening" in keyword or "beginning" in keyword:
        return (0, total_frames // 4)
    if "later" in keyword or "closing" in keyword or "end of" in keyword:
        return (total_frames * 3 // 4, total_frames)
    if "throughout" in keyword:
        return (0, total_frames)
    # "minute N" pattern
    m = re.search(r"minute\s+(\d+)", keyword)
    if m:
        minute = int(m.group(1))
        # Approximate: assume 25fps, 60s/min; locate window of ±30s
        center = minute * 25 * 60
        return (max(0, center - 750), min(total_frames, center + 750))
    return None


def _possession_in_window(
    frames: list[dict], start: int, end: int, team_id: int | None
) -> float | None:
    """Compute fraction of frames in [start, end) where team_id has possession.

    Looks for frame["possessionTeam"] == team_id or team_id_str.
    """
    window = frames[start:end]
    if not window:
        return None
    if team_id is None:
        return None
    tid_str = str(team_id)
    poss_count = sum(
        1 for f in window
        if f.get("possessionTeam") in (team_id, tid_str)
    )
    return round(poss_count / len(window), 4)


def _verify_temporal_claim(
    claim_dict: dict, frames: list[dict]
) -> dict:
    """Verify a single temporal claim against per-frame possession data."""
    text = claim_dict.get("text", "")
    keywords = _detect_temporal_keywords(text)
    if not keywords:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": "no temporal keywords found"}

    total = len(frames)
    if total == 0:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": "no frame data available"}

    team_id = _extract_team_id(text)
    tl = text.lower()

    # Determine what the claim says (dominate = >60%, controlled = >55%)
    claims_dominant = any(w in tl for w in ("dominat", "controlled", "dominated"))
    claims_low = any(w in tl for w in ("struggled", "lacked possession", "gave away"))

    # Match temporal window from the most specific keyword
    window = None
    for kw in keywords:
        window = _infer_frame_window(kw, total)
        if window is not None:
            break

    if window is None:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": f"could not map temporal keyword(s) to frame window: {keywords}"}

    start, end = window
    poss_frac = _possession_in_window(frames, start, end, team_id)
    if poss_frac is None:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": (
                    f"no possession data in frame window [{start}, {end}] "
                    f"for team {team_id}"
                )}

    if claims_dominant:
        verdict = "verified" if poss_frac >= 0.55 else "refuted"
    elif claims_low:
        verdict = "verified" if poss_frac <= 0.45 else "refuted"
    else:
        # Generic temporal + team mention — just check they had majority
        verdict = "verified" if poss_frac >= 0.50 else "refuted"

    evidence = (
        f"team {team_id} possession fraction in window [{start}, {end}] = "
        f"{poss_frac:.2%} (frames sampled: {end - start})"
    )
    return {"claim_text": text, "verdict": verdict, "evidence": evidence}


def run_temporal_dimension(claims: list[dict], frames: list[dict]) -> dict:
    """Run Dim 2: Temporal Claim Verification."""
    details: list[dict] = []
    triggered = 0
    verified = 0
    refuted = 0
    unresolvable = 0

    for claim in claims:
        if not _detect_temporal_keywords(claim.get("text", "")):
            continue
        triggered += 1
        result = _verify_temporal_claim(claim, frames)
        details.append(result)
        v = result["verdict"]
        if v == "verified":
            verified += 1
        elif v == "refuted":
            refuted += 1
        else:
            unresolvable += 1

    rate = round(verified / triggered, 4) if triggered else 0.0
    return {
        "total_temporal_claims": triggered,
        "verified": verified,
        "refuted": refuted,
        "unresolvable": unresolvable,
        "rate": rate,
        "details": details,
    }


# ── Dimension 3: Event-Spatial Verification ───────────────────────────────────


def _classify_event_zone(event: dict) -> dict:
    """Return start/end zone labels for an event based on its coordinates.

    Event startX/endX are stored in cm by the pipeline — convert to metres
    before zone classification.
    """
    start_x = event.get("startX") or event.get("start_x")
    end_x   = event.get("endX")   or event.get("end_x")
    result: dict[str, Any] = {}
    if start_x is not None:
        try:
            start_m = float(start_x) / 100.0  # cm → metres
            result["start_zone"] = _classify_pitch_zone(start_m)
            result["start_x_m"] = start_m
        except (TypeError, ValueError):
            pass
    if end_x is not None:
        try:
            end_m = float(end_x) / 100.0  # cm → metres
            result["end_zone"] = _classify_pitch_zone(end_m)
            result["end_x_m"] = end_m
        except (TypeError, ValueError):
            pass
    return result


def _expected_zones_from_keywords(keywords: list[str]) -> dict:
    """Infer expected start_zone and/or end_zone from event-spatial keywords."""
    expected: dict[str, str] = {}
    for kw in keywords:
        if "from midfield" in kw or "pass from midfield" in kw:
            expected["start_zone"] = "middle"
        if "from defensive" in kw or "from own half" in kw or "from the back" in kw:
            expected["start_zone"] = "defensive"
        if "into final third" in kw or "into the final third" in kw:
            expected["end_zone"] = "attacking"
        if "challenge in defensive" in kw:
            expected["start_zone"] = "defensive"
        if "progressive pass" in kw:
            # Progressive: end_x > start_x, no strict zone requirement
            expected["progressive"] = True
        if "cross from" in kw and "flank" in kw:
            expected["start_zone"] = "middle"  # Flanks in middle/attacking third
        if "shot from outside" in kw:
            expected["start_zone"] = "middle"
    return expected


def _match_event_type(text: str, event: dict) -> bool:
    """Check whether an event type is consistent with the claim text."""
    tl = text.lower()
    etype = (event.get("event_type") or event.get("type") or "").lower()
    if any(w in tl for w in ("pass", "cross", "through ball", "progressive pass")):
        return "pass" in etype
    if any(w in tl for w in ("shot", "attempt")):
        return "shot" in etype
    if any(w in tl for w in ("challenge", "tackle", "duel", "intercept")):
        return "challenge" in etype
    return True  # No type filter — accept any


def _verify_event_spatial_claim(
    claim_dict: dict, events_db: list[dict]
) -> dict:
    """Verify a single event-spatial claim against events_db coordinates."""
    text = claim_dict.get("text", "")
    keywords = _detect_event_spatial_keywords(text)
    if not keywords:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": "no event-spatial keywords found"}

    if not events_db:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": "no events_db data available"}

    expected = _expected_zones_from_keywords(keywords)
    if not expected:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": f"could not infer expected zones from keywords: {keywords}"}

    # Filter events by type
    candidate_events = [e for e in events_db if _match_event_type(text, e)]
    if not candidate_events:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": "no matching event type found in events_db"}

    matches_found = 0
    mismatches_found = 0
    checked = 0

    for event in candidate_events:
        zones = _classify_event_zone(event)
        if not zones:
            continue
        checked += 1
        ok = True

        if "start_zone" in expected and "start_zone" in zones:
            if zones["start_zone"] != expected["start_zone"]:
                ok = False

        if "end_zone" in expected and "end_zone" in zones:
            if zones["end_zone"] != expected["end_zone"]:
                ok = False

        if expected.get("progressive") and "start_x" in zones and "end_x" in zones:
            if zones["end_x"] <= zones["start_x"]:
                ok = False

        if ok:
            matches_found += 1
        else:
            mismatches_found += 1

    if checked == 0:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": "events found but none had spatial coordinates"}

    match_rate = matches_found / checked
    verdict = "verified" if match_rate >= 0.5 else "refuted"
    evidence = (
        f"checked {checked} candidate events; {matches_found} matched spatial claim "
        f"({match_rate:.0%}); expected={expected}"
    )
    return {"claim_text": text, "verdict": verdict, "evidence": evidence}


def run_event_spatial_dimension(
    claims: list[dict], events_db: list[dict]
) -> dict:
    """Run Dim 3: Event-Spatial Verification."""
    details: list[dict] = []
    triggered = 0
    verified = 0
    refuted = 0
    unresolvable = 0

    for claim in claims:
        if not _detect_event_spatial_keywords(claim.get("text", "")):
            continue
        triggered += 1
        result = _verify_event_spatial_claim(claim, events_db)
        details.append(result)
        v = result["verdict"]
        if v == "verified":
            verified += 1
        elif v == "refuted":
            refuted += 1
        else:
            unresolvable += 1

    rate = round(verified / triggered, 4) if triggered else 0.0
    return {
        "total_event_spatial_claims": triggered,
        "verified": verified,
        "refuted": refuted,
        "unresolvable": unresolvable,
        "rate": rate,
        "details": details,
    }


# ── Dimension 4: Trajectory Consistency ───────────────────────────────────────


def _extract_ball_x_sequence(
    frames: list[dict],
) -> list[float]:
    """Extract ordered ball pitchX values from per-frame data.

    Looks for frame["ballPosition"]["pitchX"] or frame["ball"]["pitchX"].
    """
    xs: list[float] = []
    for frame in frames:
        ball = frame.get("ballPosition") or frame.get("ball") or {}
        px = ball.get("pitchX")
        if px is not None:
            try:
                xs.append(float(px))
            except (TypeError, ValueError):
                pass
    return xs


def _extract_ball_x_from_trajectory(ball_trajectory: Any) -> list[float]:
    """Extract X values from ball_trajectory (various possible structures)."""
    xs: list[float] = []
    if isinstance(ball_trajectory, list):
        for point in ball_trajectory:
            if isinstance(point, dict):
                px = point.get("pitchX") or point.get("x")
                if px is not None:
                    try:
                        xs.append(float(px))
                    except (TypeError, ValueError):
                        pass
            elif isinstance(point, (int, float)):
                xs.append(float(point))
    elif isinstance(ball_trajectory, dict):
        # May have "pitch_positions" as a list of [x, y] pairs
        positions = ball_trajectory.get("pitch_positions") or ball_trajectory.get("positions")
        if isinstance(positions, list):
            for pt in positions:
                if isinstance(pt, (list, tuple)) and len(pt) >= 1:
                    try:
                        xs.append(float(pt[0]))
                    except (TypeError, ValueError):
                        pass
                elif isinstance(pt, dict):
                    px = pt.get("pitchX") or pt.get("x")
                    if px is not None:
                        try:
                            xs.append(float(px))
                        except (TypeError, ValueError):
                            pass
    return xs


def _direction_change_rate(xs: list[float]) -> float:
    """Compute fraction of consecutive pairs that reverse direction (0..1)."""
    if len(xs) < 3:
        return 0.0
    changes = 0
    for i in range(1, len(xs) - 1):
        prev_dir = xs[i] - xs[i - 1]
        curr_dir = xs[i + 1] - xs[i]
        if prev_dir * curr_dir < 0:
            changes += 1
    return changes / (len(xs) - 2)


def _verify_trajectory_claim(
    claim_dict: dict, ball_xs: list[float]
) -> dict:
    """Verify a single trajectory claim against ball pitchX sequence."""
    text = claim_dict.get("text", "")
    keywords = _detect_trajectory_keywords(text)
    if not keywords:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": "no trajectory keywords found"}

    if len(ball_xs) < 5:
        return {"claim_text": text, "verdict": "unresolvable",
                "evidence": f"insufficient ball position data ({len(ball_xs)} points)"}

    tl = text.lower()
    mean_x = mean(ball_xs)
    var_x = variance(ball_xs) if len(ball_xs) > 1 else 0.0
    dir_change_rate = _direction_change_rate(ball_xs)
    mean_zone = _classify_pitch_zone(mean_x)

    # Determine what the claim asserts
    claims_defensive_circ  = any(w in tl for w in (
        "kept possession in their half", "defensive build-up", "played out from the back",
        "build-up", "ball circulation"
    ))
    claims_high_possession  = any(w in tl for w in ("pushed the ball forward",))
    claims_transitions      = any(w in tl for w in (
        "quick transitions", "transition", "high tempo", "direct play", "long ball"
    ))
    claims_patient          = any(w in tl for w in ("patient build-up", "patient play"))

    team_id = _extract_team_id(text)

    if claims_transitions:
        # High direction-change rate (>0.4) and/or high variance → quick transitions
        is_quick = dir_change_rate > 0.4 or var_x > (PITCH_LENGTH_M * 0.1) ** 2
        verdict = "verified" if is_quick else "refuted"
        evidence = (
            f"ball dir_change_rate={dir_change_rate:.2f} (threshold 0.4); "
            f"pitchX variance={var_x:.1f}; mean_x={mean_x:.1f}m"
        )
    elif claims_patient:
        # Low direction-change rate and low variance → patient build-up
        is_patient = dir_change_rate <= 0.3 and var_x <= (PITCH_LENGTH_M * 0.15) ** 2
        verdict = "verified" if is_patient else "refuted"
        evidence = (
            f"ball dir_change_rate={dir_change_rate:.2f} (≤0.3 for patient); "
            f"pitchX variance={var_x:.1f}"
        )
    elif claims_defensive_circ:
        # Ball mean X < 35m → claim of defensive zone circulation verified
        verdict = "verified" if mean_zone == "defensive" else "refuted"
        evidence = (
            f"ball mean pitchX={mean_x:.1f}m → zone='{mean_zone}'; "
            f"claim expects 'defensive' zone; team={team_id}"
        )
    elif claims_high_possession:
        verdict = "verified" if mean_zone in ("middle", "attacking") else "refuted"
        evidence = (
            f"ball mean pitchX={mean_x:.1f}m → zone='{mean_zone}'; "
            f"claim expects forward possession; team={team_id}"
        )
    else:
        # Generic trajectory keyword — just report zone
        verdict = "verified"
        evidence = (
            f"trajectory data available; ball mean pitchX={mean_x:.1f}m → "
            f"zone='{mean_zone}'; dir_change_rate={dir_change_rate:.2f}"
        )

    return {"claim_text": text, "verdict": verdict, "evidence": evidence}


def run_trajectory_dimension(
    claims: list[dict],
    frames: list[dict],
    db_ground_truth: dict,
) -> dict:
    """Run Dim 4: Trajectory Consistency."""
    # Build ball X sequence from per_frame and/or ball_trajectory in analytics
    ball_xs = _extract_ball_x_sequence(frames)
    if not ball_xs:
        # Fallback to analytics.ball_path.pitch_positions
        analytics = db_ground_truth.get("analytics") or {}
        ball_traj = analytics.get("ball_path") or analytics.get("ball_trajectory")
        if ball_traj:
            ball_xs = _extract_ball_x_from_trajectory(ball_traj)

    log.info("Trajectory dimension: %d ball pitchX samples loaded", len(ball_xs))

    details: list[dict] = []
    triggered = 0
    verified = 0
    refuted = 0
    unresolvable = 0

    for claim in claims:
        if not _detect_trajectory_keywords(claim.get("text", "")):
            continue
        triggered += 1
        result = _verify_trajectory_claim(claim, ball_xs)
        details.append(result)
        v = result["verdict"]
        if v == "verified":
            verified += 1
        elif v == "refuted":
            refuted += 1
        else:
            unresolvable += 1

    rate = round(verified / triggered, 4) if triggered else 0.0
    return {
        "total_trajectory_claims": triggered,
        "verified": verified,
        "refuted": refuted,
        "unresolvable": unresolvable,
        "rate": rate,
        "details": details,
    }


# ── Dimension 5: Cross-Frame Consistency Score ────────────────────────────────


def _extract_time_references(text: str) -> list[float]:
    """Extract time references (in seconds) from commentary text.

    Handles:
      - "minute 12" / "12th minute" → 720s
      - "MM:SS" timestamp → float seconds
      - "at 45" (bare number when 'minute' is nearby in context) → 2700s
    """
    refs: list[float] = []

    # "minute N" or "N th minute" or "N-minute" patterns
    for m in re.finditer(r"\bminute\s+(\d+)\b|\b(\d+)(?:st|nd|rd|th)?\s+minute\b", text, re.I):
        n = m.group(1) or m.group(2)
        refs.append(float(n) * 60)

    # "MM:SS" timestamps
    for m in re.finditer(r"\b(\d{1,3}):(\d{2})\b", text):
        mins, secs = int(m.group(1)), int(m.group(2))
        refs.append(float(mins * 60 + secs))

    # "at 45" / "at 90" when adjacent to obvious match-time context — conservative pattern
    for m in re.finditer(r"\bat\s+(\d{1,2})\b", text, re.I):
        val = int(m.group(1))
        if 1 <= val <= 90:
            refs.append(float(val) * 60)

    return refs


def run_cross_frame_consistency(commentary: str) -> dict:
    """Run Dim 5: Cross-Frame Consistency Score.

    Checks if all time references in the commentary appear in monotonically
    increasing order (i.e. the narrative follows chronological order).

    Returns:
        {
            "score": float | None — 1.0 (all in order), 0.5 (partially), 0.0 (reversed),
                     None if no time refs found,
            "n_references": int,
            "in_order": bool | None,
            "time_refs_seconds": list[float],
        }
    """
    refs = _extract_time_references(commentary)

    if not refs:
        log.info("Cross-frame consistency: no time references found in commentary")
        return {
            "score": None,
            "n_references": 0,
            "in_order": None,
            "time_refs_seconds": [],
        }

    log.info(
        "Cross-frame consistency: found %d time reference(s): %s",
        len(refs), [f"{r:.0f}s" for r in refs],
    )

    if len(refs) == 1:
        return {
            "score": 1.0,
            "n_references": 1,
            "in_order": True,
            "time_refs_seconds": refs,
        }

    # Check monotonic increase
    violations = sum(1 for i in range(len(refs) - 1) if refs[i] > refs[i + 1])
    total_pairs = len(refs) - 1

    if violations == 0:
        score = 1.0
        in_order = True
    elif violations == total_pairs:
        score = 0.0
        in_order = False
    else:
        score = round(1.0 - violations / total_pairs, 4)
        in_order = None  # Partial

    return {
        "score": score,
        "n_references": len(refs),
        "in_order": in_order,
        "time_refs_seconds": refs,
    }


# ── Resolution summary ────────────────────────────────────────────────────────


def _build_resolution_summary(
    claims: list[dict],
    dim_results: dict,
) -> dict:
    """Compute how many previously-unverifiable claims were resolved by DB data.

    Counts claims that originally carried verdict="unverifiable" and were
    resolved (to verified or refuted) by any of the four claim-level dimensions.
    """
    # Collect all claim texts that were originally unverifiable
    previously_unverifiable = {
        c.get("text", "") for c in claims
        if c.get("verdict") == "unverifiable"
    }

    # Collect resolved verdicts across all 4 claim-level dimensions
    now_verified: set[str] = set()
    now_refuted: set[str] = set()

    for dim_key in ("spatial", "temporal", "event_spatial", "trajectory"):
        dim = dim_results.get(dim_key, {})
        for detail in dim.get("details", []):
            ct = detail.get("claim_text", "")
            if ct not in previously_unverifiable:
                continue
            if detail.get("verdict") == "verified":
                now_verified.add(ct)
            elif detail.get("verdict") == "refuted":
                now_refuted.add(ct)

    now_verified_by_db = len(now_verified)
    now_refuted_by_db  = len(now_refuted)
    still_unresolvable = max(0, len(previously_unverifiable) - now_verified_by_db - now_refuted_by_db)

    return {
        "previously_unverifiable": len(previously_unverifiable),
        "now_verified_by_db": now_verified_by_db,
        "now_refuted_by_db": now_refuted_by_db,
        "still_unresolvable": still_unresolvable,
    }


# ── Main entry point ──────────────────────────────────────────────────────────


def run_db_grounding(
    commentary: str,
    claims: list[dict],
    db_ground_truth: dict,
    analytics: dict,
) -> dict:
    """Run 5-dimension DB-grounded verification on extracted claims.

    Args:
        commentary:      Full commentary text (used for Dim 5 only).
        claims:          List of Claim objects serialised as dicts — output of
                         extract_claims() → [asdict(c), ...].  Each dict must
                         have at least "text", "claim_type", "verdict" (optional).
        db_ground_truth: Output of db_extractor.extract_analysis_data().
                         Expected top-level keys: "analytics", "per_frame",
                         "events_db", "statistics", "frame_metrics", "formations".
        analytics:       Original aggregated analytics JSON (for fallback lookups).

    Returns:
        Dict with keys: "spatial", "temporal", "event_spatial", "trajectory",
        "cross_frame_consistency", "resolution_summary".
    """
    per_frame_data = db_ground_truth.get("per_frame") or {}
    frames: list[dict] = per_frame_data.get("frames") or []
    events_db: list[dict] = db_ground_truth.get("events_db") or []

    log.info(
        "DB grounding: %d claims, %d frames, %d events",
        len(claims), len(frames), len(events_db),
    )

    # Run each dimension independently — failures in one must not affect others
    try:
        spatial_result = run_spatial_dimension(claims, frames)
        log.info(
            "Dim 1 (spatial): %d triggered, %d verified, %d refuted, %d unresolvable",
            spatial_result["total_spatial_claims"],
            spatial_result["verified"],
            spatial_result["refuted"],
            spatial_result["unresolvable"],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Dim 1 (spatial) failed: %s", exc)
        spatial_result = {
            "total_spatial_claims": 0, "verified": 0, "refuted": 0,
            "unresolvable": 0, "rate": 0.0, "details": [],
            "error": str(exc),
        }

    try:
        temporal_result = run_temporal_dimension(claims, frames)
        log.info(
            "Dim 2 (temporal): %d triggered, %d verified, %d refuted, %d unresolvable",
            temporal_result["total_temporal_claims"],
            temporal_result["verified"],
            temporal_result["refuted"],
            temporal_result["unresolvable"],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Dim 2 (temporal) failed: %s", exc)
        temporal_result = {
            "total_temporal_claims": 0, "verified": 0, "refuted": 0,
            "unresolvable": 0, "rate": 0.0, "details": [],
            "error": str(exc),
        }

    try:
        event_spatial_result = run_event_spatial_dimension(claims, events_db)
        log.info(
            "Dim 3 (event-spatial): %d triggered, %d verified, %d refuted, %d unresolvable",
            event_spatial_result["total_event_spatial_claims"],
            event_spatial_result["verified"],
            event_spatial_result["refuted"],
            event_spatial_result["unresolvable"],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Dim 3 (event-spatial) failed: %s", exc)
        event_spatial_result = {
            "total_event_spatial_claims": 0, "verified": 0, "refuted": 0,
            "unresolvable": 0, "rate": 0.0, "details": [],
            "error": str(exc),
        }

    try:
        trajectory_result = run_trajectory_dimension(claims, frames, db_ground_truth)
        log.info(
            "Dim 4 (trajectory): %d triggered, %d verified, %d refuted, %d unresolvable",
            trajectory_result["total_trajectory_claims"],
            trajectory_result["verified"],
            trajectory_result["refuted"],
            trajectory_result["unresolvable"],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Dim 4 (trajectory) failed: %s", exc)
        trajectory_result = {
            "total_trajectory_claims": 0, "verified": 0, "refuted": 0,
            "unresolvable": 0, "rate": 0.0, "details": [],
            "error": str(exc),
        }

    try:
        consistency_result = run_cross_frame_consistency(commentary)
        log.info(
            "Dim 5 (cross-frame consistency): score=%s, n_refs=%d",
            consistency_result["score"],
            consistency_result["n_references"],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Dim 5 (cross-frame consistency) failed: %s", exc)
        consistency_result = {
            "score": None, "n_references": 0, "in_order": None,
            "time_refs_seconds": [], "error": str(exc),
        }

    dim_results = {
        "spatial": spatial_result,
        "temporal": temporal_result,
        "event_spatial": event_spatial_result,
        "trajectory": trajectory_result,
    }

    try:
        resolution = _build_resolution_summary(claims, dim_results)
    except Exception as exc:  # noqa: BLE001
        log.warning("Resolution summary failed: %s", exc)
        resolution = {
            "previously_unverifiable": 0,
            "now_verified_by_db": 0,
            "now_refuted_by_db": 0,
            "still_unresolvable": 0,
            "error": str(exc),
        }

    return {
        "spatial": spatial_result,
        "temporal": temporal_result,
        "event_spatial": event_spatial_result,
        "trajectory": trajectory_result,
        "cross_frame_consistency": consistency_result,
        "resolution_summary": resolution,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DB-grounded claim verification — 5 additional dimensions"
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to db_extractor output JSON (e.g. eval_output/18_db_ground_truth.json)",
    )
    parser.add_argument(
        "--claims",
        required=True,
        help="Path to JSON file with claims list (output of extract_claims())",
    )
    parser.add_argument(
        "--commentary",
        required=True,
        help="Path to plain-text file containing the LLM commentary",
    )
    parser.add_argument(
        "--output",
        default="eval_output/dissertation/db_grounded/",
        help="Directory to write results JSON (default: eval_output/dissertation/db_grounded/)",
    )
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        gt = json.load(f)
    with open(args.claims) as f:
        claims_data = json.load(f)
    with open(args.commentary) as f:
        commentary_text = f.read()

    results = run_db_grounding(
        commentary=commentary_text,
        claims=claims_data,
        db_ground_truth=gt,
        analytics=gt.get("analytics", {}),
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "db_grounding_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("Results written to %s", out_path)
    print(json.dumps(results, indent=2))
