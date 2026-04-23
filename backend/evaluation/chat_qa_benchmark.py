"""Chat QA benchmark -- auto-generated question-answer pairs grounded in analytics.

Implements RAGAS-style faithfulness evaluation (Shahul Es et al. 2023) and
SQuAD 2.0-style unanswerable question detection (Rajpurkar et al. 2018).

Auto-generates up to 100 deterministic QA pairs from analytics JSON across 8 categories:
  1. Numeric (deterministic, easy to verify)
  2. Comparative (team A vs team B)
  3. Tactical (phase-based, requires reasoning)
  4. Temporal ("When?") — RAG-Football Tzikas 2025
  5. Spatial ("Where?") — RAG-Football Tzikas 2025
  6. Entity ("Who?") — RAG-Football Tzikas 2025
  7. Multi-hop (cross-metric reasoning) — Sports Intelligence Yang 2025
  8. Unanswerable (question cannot be answered from available data)

Usage:
    python3 -m evaluation.chat_qa_benchmark \\
        --analytics ../eval_output/phase12/10_analytics.json \\
        --provider openai \\
        --output ../eval_output/phase16/chat_qa/
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import ensure_output_dir, load_analytics, load_db_ground_truth, save_figure, save_latex_table
import matplotlib.pyplot as plt


# ── QA pair generation ────────────────────────────────────────────────────────


def _safe_pct(v: Any) -> str:
    return f"{float(v):.1f}%" if v is not None else "N/A"


def _safe_m(v: Any) -> str:
    return f"{float(v):.0f}m" if v is not None else "N/A"


def _safe_kmh(v: Any) -> str:
    return f"{float(v) * 3.6:.1f} km/h" if v is not None else "N/A"


def generate_qa_pairs(analytics: dict) -> list[dict]:
    """Generate deterministic QA pairs from analytics JSON.

    Returns list of:
        {
            "question": str,
            "expected_answer": str,      # Ground-truth from analytics
            "category": str,             # numeric|comparative|tactical|unanswerable
            "verifiable": bool,
            "source_path": str,          # dot-path to analytics value
        }
    """
    pairs: list[dict] = []
    poss = analytics.get("possession", {})
    tac_summary = analytics.get("tactical", {}).get("summary", {})
    players = analytics.get("player_kinematics", {})
    events = analytics.get("events", [])

    # ── NUMERIC (15 questions) ────────────────────────────────────────────────

    # Possession
    t1_pct = poss.get("team_1_percentage")
    pairs.append({
        "question": "What percentage of possession did Team 1 have?",
        "expected_answer": _safe_pct(t1_pct),
        "category": "numeric",
        "verifiable": t1_pct is not None,
        "source_path": "possession.team_1_percentage",
    })

    t2_pct = poss.get("team_2_percentage")
    pairs.append({
        "question": "What percentage of possession did Team 2 have?",
        "expected_answer": _safe_pct(t2_pct),
        "category": "numeric",
        "verifiable": t2_pct is not None,
        "source_path": "possession.team_2_percentage",
    })

    changes = poss.get("possession_changes")
    pairs.append({
        "question": "How many times did possession change hands during the match?",
        "expected_answer": str(changes) if changes is not None else "N/A",
        "category": "numeric",
        "verifiable": changes is not None,
        "source_path": "possession.possession_changes",
    })

    # Top player distance
    sorted_players = sorted(
        [(tid, s) for tid, s in players.items() if s.get("total_distance_m") is not None],
        key=lambda x: x[1].get("total_distance_m", 0),
        reverse=True,
    )
    if sorted_players:
        top_tid, top_stats = sorted_players[0]
        pairs.append({
            "question": f"How far did player #{top_tid} cover during the match?",
            "expected_answer": _safe_m(top_stats.get("total_distance_m")),
            "category": "numeric",
            "verifiable": True,
            "source_path": f"player_kinematics.{top_tid}.total_distance_m",
        })
        pairs.append({
            "question": f"What was the maximum speed of player #{top_tid}?",
            "expected_answer": _safe_kmh(top_stats.get("max_speed_m_per_sec")),
            "category": "numeric",
            "verifiable": top_stats.get("max_speed_m_per_sec") is not None,
            "source_path": f"player_kinematics.{top_tid}.max_speed_m_per_sec",
        })

    # Tactical metrics
    for qtext, key, fmt in [
        ("What was Team 1's average compactness?", "team_1_avg_compactness_m2", lambda v: f"{float(v):.1f}m2"),
        ("What was Team 2's average compactness?", "team_2_avg_compactness_m2", lambda v: f"{float(v):.1f}m2"),
        ("What was Team 1's average defensive line height?", "team_1_avg_defensive_line_m", lambda v: f"{float(v):.1f}m"),
        ("What was Team 2's average defensive line height?", "team_2_avg_defensive_line_m", lambda v: f"{float(v):.1f}m"),
        ("What was Team 1's pressing intensity?", "team_1_avg_pressing_intensity", lambda v: f"{float(v):.2f}"),
        ("What was Team 2's pressing intensity?", "team_2_avg_pressing_intensity", lambda v: f"{float(v):.2f}"),
        ("What was Team 1's PPDA (passes per defensive action)?", "ppda_team_1", lambda v: f"{float(v):.1f}"),
        ("What was the average inter-team distance?", "avg_inter_team_distance_m", lambda v: f"{float(v):.1f}m"),
    ]:
        val = tac_summary.get(key)
        pairs.append({
            "question": qtext,
            "expected_answer": fmt(val) if val is not None else "N/A",
            "category": "numeric",
            "verifiable": val is not None,
            "source_path": f"tactical.summary.{key}",
        })

    # Events
    pass_events = [e for e in events if e.get("event_type") == "pass"]
    shot_events = [e for e in events if e.get("event_type") == "shot"]
    t1_passes = len([e for e in pass_events if e.get("team_id") == 1])
    t2_passes = len([e for e in pass_events if e.get("team_id") == 2])
    pairs.append({
        "question": "How many passes did Team 1 complete?",
        "expected_answer": str(t1_passes),
        "category": "numeric",
        "verifiable": True,
        "source_path": "events[event_type=pass,team_id=1].count",
    })
    pairs.append({
        "question": "How many shots did Team 2 take?",
        "expected_answer": str(len([e for e in shot_events if e.get("team_id") == 2])),
        "category": "numeric",
        "verifiable": True,
        "source_path": "events[event_type=shot,team_id=2].count",
    })

    # ── COMPARATIVE (10 questions) ────────────────────────────────────────────

    if t1_pct is not None and t2_pct is not None:
        dominant = "Team 1" if t1_pct > t2_pct else "Team 2"
        pairs.append({
            "question": "Which team had more possession?",
            "expected_answer": f"{dominant} ({_safe_pct(max(t1_pct, t2_pct))})",
            "category": "comparative",
            "verifiable": True,
            "source_path": "possession.team_1_percentage,possession.team_2_percentage",
        })

    t1_comp = tac_summary.get("team_1_avg_compactness_m2")
    t2_comp = tac_summary.get("team_2_avg_compactness_m2")
    if t1_comp is not None and t2_comp is not None:
        more_compact = "Team 1" if t1_comp < t2_comp else "Team 2"
        pairs.append({
            "question": "Which team was more compact?",
            "expected_answer": f"{more_compact} ({min(t1_comp, t2_comp):.0f}m2 vs {max(t1_comp, t2_comp):.0f}m2)",
            "category": "comparative",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_avg_compactness_m2",
        })

    t1_press = tac_summary.get("team_1_avg_pressing_intensity")
    t2_press = tac_summary.get("team_2_avg_pressing_intensity")
    if t1_press is not None and t2_press is not None:
        higher = "Team 1" if t1_press > t2_press else "Team 2"
        pairs.append({
            "question": "Which team pressed more intensely?",
            "expected_answer": f"{higher} ({max(t1_press, t2_press):.2f} vs {min(t1_press, t2_press):.2f})",
            "category": "comparative",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_avg_pressing_intensity",
        })

    t1_terr = tac_summary.get("team_1_avg_territory_pct")
    t2_terr = tac_summary.get("team_2_avg_territory_pct")
    if t1_terr is not None and t2_terr is not None:
        terr_lead = "Team 1" if t1_terr > t2_terr else "Team 2"
        pairs.append({
            "question": "Which team controlled more pitch territory?",
            "expected_answer": f"{terr_lead} ({max(t1_terr, t2_terr)*100:.1f}%)",
            "category": "comparative",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_avg_territory_pct",
        })

    # Compare two players
    if len(sorted_players) >= 2:
        tid1, s1 = sorted_players[0]
        tid2, s2 = sorted_players[1]
        d1 = s1.get("total_distance_m", 0)
        d2 = s2.get("total_distance_m", 0)
        pairs.append({
            "question": f"Who covered more distance: player #{tid1} or player #{tid2}?",
            "expected_answer": f"Player #{tid1} ({d1:.0f}m vs {d2:.0f}m)",
            "category": "comparative",
            "verifiable": True,
            "source_path": f"player_kinematics.{tid1}.total_distance_m",
        })

    t1_dl = tac_summary.get("team_1_avg_defensive_line_m")
    t2_dl = tac_summary.get("team_2_avg_defensive_line_m")
    if t1_dl is not None and t2_dl is not None:
        high_line = "Team 1" if t1_dl > t2_dl else "Team 2"
        pairs.append({
            "question": "Which team defended with a higher defensive line?",
            "expected_answer": f"{high_line} ({max(t1_dl, t2_dl):.1f}m vs {min(t1_dl, t2_dl):.1f}m)",
            "category": "comparative",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_avg_defensive_line_m",
        })

    ppda1 = tac_summary.get("ppda_team_1")
    ppda2 = tac_summary.get("ppda_team_2")
    if ppda1 is not None and ppda2 is not None:
        agg_team = "Team 1" if ppda1 < ppda2 else "Team 2"
        pairs.append({
            "question": "Which team was more aggressive in pressing based on PPDA?",
            "expected_answer": f"{agg_team} (PPDA {min(ppda1, ppda2):.1f})",
            "category": "comparative",
            "verifiable": True,
            "source_path": "tactical.summary.ppda_team_1",
        })

    pairs.append({
        "question": "Which team made more passes?",
        "expected_answer": f"Team {'1' if t1_passes >= t2_passes else '2'} ({max(t1_passes, t2_passes)} passes)",
        "category": "comparative",
        "verifiable": True,
        "source_path": "events[event_type=pass].count",
    })

    t1_sprints = sum(s.get("sprint_count", 0) or 0 for s in players.values() if s.get("team_id") == 1)
    t2_sprints = sum(s.get("sprint_count", 0) or 0 for s in players.values() if s.get("team_id") == 2)
    pairs.append({
        "question": "Which team performed more sprints?",
        "expected_answer": f"Team {'1' if t1_sprints >= t2_sprints else '2'} ({max(t1_sprints, t2_sprints)} sprints)",
        "category": "comparative",
        "verifiable": True,
        "source_path": "player_kinematics[team_id=1].sprint_count.sum",
    })

    # Progressive passes
    prog_t1 = len([e for e in events if e.get("event_type") == "pass" and e.get("team_id") == 1 and e.get("is_progressive")])
    prog_t2 = len([e for e in events if e.get("event_type") == "pass" and e.get("team_id") == 2 and e.get("is_progressive")])
    if prog_t1 + prog_t2 > 0:
        pairs.append({
            "question": "Which team made more progressive passes?",
            "expected_answer": f"Team {'1' if prog_t1 >= prog_t2 else '2'} ({max(prog_t1, prog_t2)} progressive passes)",
            "category": "comparative",
            "verifiable": True,
            "source_path": "events[event_type=pass,is_progressive=true].count",
        })

    # ── TACTICAL (10 questions) ───────────────────────────────────────────────

    windows = analytics.get("tactical", {}).get("windows", [])
    ip_windows = [w for w in windows if w.get("phase_team_1") == "ip"]
    oop_windows = [w for w in windows if w.get("phase_team_1") == "oop"]

    def _mean_window(ws, key):
        vals = [w.get(key) for w in ws if w.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    pairs.append({
        "question": "How compact was Team 1 when in possession?",
        "expected_answer": f"{_mean_window(ip_windows, 'team_1_compactness'):.0f}m2" if ip_windows and _mean_window(ip_windows, 'team_1_compactness') else "N/A",
        "category": "tactical",
        "verifiable": bool(ip_windows),
        "source_path": "tactical.windows[phase_team_1=ip].team_1_compactness.mean",
    })

    pairs.append({
        "question": "How compact was Team 1 when out of possession?",
        "expected_answer": f"{_mean_window(oop_windows, 'team_1_compactness'):.0f}m2" if oop_windows and _mean_window(oop_windows, 'team_1_compactness') else "N/A",
        "category": "tactical",
        "verifiable": bool(oop_windows),
        "source_path": "tactical.windows[phase_team_1=oop].team_1_compactness.mean",
    })

    t1_press_dist = tac_summary.get("team_1_press_type_distribution", {})
    if t1_press_dist:
        dominant_press = max(t1_press_dist, key=lambda k: t1_press_dist.get(k, 0)) if t1_press_dist else None
        pairs.append({
            "question": "What type of pressing block did Team 1 predominantly use?",
            "expected_answer": f"{dominant_press} press ({t1_press_dist.get(dominant_press, 0)} windows)" if dominant_press else "N/A",
            "category": "tactical",
            "verifiable": bool(t1_press_dist),
            "source_path": "tactical.summary.team_1_press_type_distribution",
        })

    cp_t1 = tac_summary.get("team_1_counter_press_windows", 0)
    pairs.append({
        "question": "How many windows did Team 1 engage in counter-pressing?",
        "expected_answer": str(cp_t1),
        "category": "tactical",
        "verifiable": True,
        "source_path": "tactical.summary.team_1_counter_press_windows",
    })

    t1_opp_half = tac_summary.get("team_1_avg_opp_half_territory_pct")
    pairs.append({
        "question": "What percentage of the opponent's half did Team 1 control on average?",
        "expected_answer": _safe_pct(t1_opp_half * 100 if t1_opp_half else None),
        "category": "tactical",
        "verifiable": t1_opp_half is not None,
        "source_path": "tactical.summary.team_1_avg_opp_half_territory_pct",
    })

    # IP vs OOP compactness change
    ip_comp = _mean_window(ip_windows, "team_1_compactness")
    oop_comp = _mean_window(oop_windows, "team_1_compactness")
    if ip_comp and oop_comp:
        pairs.append({
            "question": "Was Team 1 more compact when in possession or out of possession?",
            "expected_answer": f"{'In possession' if ip_comp < oop_comp else 'Out of possession'} (IP: {ip_comp:.0f}m2, OOP: {oop_comp:.0f}m2)",
            "category": "tactical",
            "verifiable": True,
            "source_path": "tactical.windows.phase_comparison",
        })

    pairs.append({
        "question": "How did Team 1's pressing intensity change in transition phases?",
        "expected_answer": _get_phase_press_change(tac_summary),
        "category": "tactical",
        "verifiable": bool(tac_summary.get("team_1_dat_pressing_intensity")),
        "source_path": "tactical.summary.team_1_dat_pressing_intensity",
    })

    pairs.append({
        "question": "What was the dominant pressing style of Team 2 based on their defensive line?",
        "expected_answer": _infer_press_style(tac_summary, team=2),
        "category": "tactical",
        "verifiable": bool(t2_dl),
        "source_path": "tactical.summary.team_2_avg_defensive_line_m",
    })

    pairs.append({
        "question": "In which phase of play did Team 1 cover the most pitch territory?",
        "expected_answer": _best_territory_phase(tac_summary, team=1),
        "category": "tactical",
        "verifiable": bool(tac_summary.get("team_1_ip_territory_pct")),
        "source_path": "tactical.summary.team_1_ip_territory_pct",
    })

    pairs.append({
        "question": "How many progressive passes did Team 1 make?",
        "expected_answer": str(prog_t1),
        "category": "tactical",
        "verifiable": True,
        "source_path": "events[event_type=pass,is_progressive=true,team_id=1].count",
    })

    # ── TEMPORAL (8 questions) — "When?" (RAG-Football Tzikas 2025) ──────────

    shot_events = [e for e in events if e.get("event_type") == "shot"]
    chall_events = [e for e in events if e.get("event_type") == "challenge"]

    if shot_events:
        first_shot = min(shot_events, key=lambda e: e.get("timestamp_sec", float("inf")))
        ts = first_shot.get("timestamp_sec", 0)
        pairs.append({
            "question": "At what timestamp (approximately) did the first shot occur?",
            "expected_answer": f"{int(ts // 60)}:{int(ts % 60):02d}",
            "category": "temporal",
            "verifiable": True,
            "source_path": "events[event_type=shot].min(timestamp_sec)",
        })

    if len(shot_events) >= 2:
        last_shot = max(shot_events, key=lambda e: e.get("timestamp_sec", 0))
        ts = last_shot.get("timestamp_sec", 0)
        pairs.append({
            "question": "When (approximately) did the last shot occur?",
            "expected_answer": f"{int(ts // 60)}:{int(ts % 60):02d}",
            "category": "temporal",
            "verifiable": True,
            "source_path": "events[event_type=shot].max(timestamp_sec)",
        })

    if pass_events:
        first_pass = min(pass_events, key=lambda e: e.get("timestamp_sec", float("inf")))
        ts = first_pass.get("timestamp_sec", 0)
        pairs.append({
            "question": "When (approximately) was the first pass detected?",
            "expected_answer": f"{int(ts // 60)}:{int(ts % 60):02d}",
            "category": "temporal",
            "verifiable": True,
            "source_path": "events[event_type=pass].min(timestamp_sec)",
        })

    if chall_events:
        first_chall = min(chall_events, key=lambda e: e.get("timestamp_sec", float("inf")))
        ts = first_chall.get("timestamp_sec", 0)
        pairs.append({
            "question": "When did the first challenge occur?",
            "expected_answer": f"{int(ts // 60)}:{int(ts % 60):02d}",
            "category": "temporal",
            "verifiable": True,
            "source_path": "events[event_type=challenge].min(timestamp_sec)",
        })

    # Phase durations
    ip_dur = tac_summary.get("team_1_total_ip_duration_s")
    oop_dur = tac_summary.get("team_1_total_oop_duration_s")
    if ip_dur is not None:
        pairs.append({
            "question": "How many seconds did Team 1 spend in possession overall?",
            "expected_answer": f"{ip_dur:.0f}s",
            "category": "temporal",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_total_ip_duration_s",
        })
    if oop_dur is not None:
        pairs.append({
            "question": "How many seconds did Team 1 spend out of possession?",
            "expected_answer": f"{oop_dur:.0f}s",
            "category": "temporal",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_total_oop_duration_s",
        })

    # ── SPATIAL (8 questions) — "Where?" (RAG-Football Tzikas 2025) ──────────

    def _zone(x: float, pitch_len: float = 10500.0) -> str:
        """Classify x-coord into defensive/middle/attacking third."""
        frac = x / pitch_len
        if frac < 0.33:
            return "defensive third"
        if frac < 0.66:
            return "middle third"
        return "attacking third"

    if pass_events:
        zone_counts: dict[str, int] = {}
        for ev in pass_events:
            ps = ev.get("pitch_start")
            if ps and len(ps) >= 1:
                z = _zone(ps[0])
                zone_counts[z] = zone_counts.get(z, 0) + 1
        if zone_counts:
            busiest_zone = max(zone_counts, key=lambda k: zone_counts[k])
            pairs.append({
                "question": "Which pitch zone had the most pass activity?",
                "expected_answer": f"{busiest_zone} ({zone_counts[busiest_zone]} passes)",
                "category": "spatial",
                "verifiable": True,
                "source_path": "events[event_type=pass].pitch_start.zone_distribution",
            })

        # Passes by team entering attacking third
        t1_attacking = sum(
            1 for e in pass_events
            if e.get("team_id") == 1
            and e.get("pitch_end") and len(e["pitch_end"]) >= 1
            and _zone(e["pitch_end"][0]) == "attacking third"
        )
        t2_attacking = sum(
            1 for e in pass_events
            if e.get("team_id") == 2
            and e.get("pitch_end") and len(e["pitch_end"]) >= 1
            and _zone(e["pitch_end"][0]) == "attacking third"
        )
        if t1_attacking + t2_attacking > 0:
            pairs.append({
                "question": "Which team played more passes into the attacking third?",
                "expected_answer": f"Team {'1' if t1_attacking >= t2_attacking else '2'} ({max(t1_attacking, t2_attacking)} passes)",
                "category": "spatial",
                "verifiable": True,
                "source_path": "events[event_type=pass,pitch_end.zone=attacking_third].count",
            })

    if shot_events:
        # Shot zones — where are shots being taken from?
        shot_zones: dict[str, int] = {}
        for ev in shot_events:
            ps = ev.get("pitch_start")
            if ps and len(ps) >= 1:
                z = _zone(ps[0])
                shot_zones[z] = shot_zones.get(z, 0) + 1
        if shot_zones:
            main_shot_zone = max(shot_zones, key=lambda k: shot_zones[k])
            pairs.append({
                "question": "From which zone were most shots taken?",
                "expected_answer": f"{main_shot_zone} ({shot_zones[main_shot_zone]} shots)",
                "category": "spatial",
                "verifiable": True,
                "source_path": "events[event_type=shot].pitch_start.zone_distribution",
            })

    # Territory by zone
    t1_opp_half = tac_summary.get("team_1_avg_opp_half_territory_pct")
    if t1_opp_half is not None:
        pairs.append({
            "question": "Did Team 1 spend more time in their own half or the opponent's half?",
            "expected_answer": (
                f"Opponent's half ({t1_opp_half*100:.1f}% of time in opp half)"
                if t1_opp_half > 0.5
                else f"Own half (only {t1_opp_half*100:.1f}% of time in opp half)"
            ),
            "category": "spatial",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_avg_opp_half_territory_pct",
        })

    # ── SPATIAL from tactical summary (compactness, defensive line, team shape) ─
    # These use the tactical window summary which is computed when homography is
    # available. Questions are skipped if the tactical summary is absent (e.g.,
    # when the tactical module didn't run on the clip).
    t1_compact = tac_summary.get("team_1_avg_compactness_m2")
    t2_compact = tac_summary.get("team_2_avg_compactness_m2")
    if t1_compact is not None and t2_compact is not None:
        more_compact_team = "1" if t1_compact < t2_compact else "2"
        less_compact_team = "2" if t1_compact < t2_compact else "1"
        min_c = min(t1_compact, t2_compact)
        max_c = max(t1_compact, t2_compact)
        pairs.append({
            "question": "Which team was more compact (occupied less area on the pitch)?",
            "expected_answer": f"Team {more_compact_team} ({min_c:.0f} m² vs {max_c:.0f} m²)",
            "category": "spatial",
            "verifiable": True,
            "source_path": "tactical.summary.team_X_avg_compactness_m2",
        })
        pairs.append({
            "question": "Which team covered more pitch area on average?",
            "expected_answer": f"Team {less_compact_team} ({max_c:.0f} m²) — larger convex hull area",
            "category": "spatial",
            "verifiable": True,
            "source_path": "tactical.summary.team_X_avg_compactness_m2",
        })

    inter_team_dist = tac_summary.get("avg_inter_team_distance_m")
    if inter_team_dist is not None:
        pairs.append({
            "question": "What was the average distance between the two teams during the match?",
            "expected_answer": f"{inter_team_dist:.1f}m",
            "category": "spatial",
            "verifiable": True,
            "source_path": "tactical.summary.avg_inter_team_distance_m",
        })

    t1_def_line = tac_summary.get("team_1_avg_defensive_line_m")
    t2_def_line = tac_summary.get("team_2_avg_defensive_line_m")
    if t1_def_line is not None and t2_def_line is not None:
        higher_team = "1" if t1_def_line > t2_def_line else "2"
        pairs.append({
            "question": "Which team maintained a higher defensive line?",
            "expected_answer": f"Team {higher_team} ({max(t1_def_line, t2_def_line):.1f}m vs {min(t1_def_line, t2_def_line):.1f}m from goal)",
            "category": "spatial",
            "verifiable": True,
            "source_path": "tactical.summary.team_X_avg_defensive_line_m",
        })

    t1_width = tac_summary.get("team_1_avg_width_m")
    t1_length = tac_summary.get("team_1_avg_length_m")
    if t1_width is not None:
        pairs.append({
            "question": "How wide was Team 1's average shape across the pitch?",
            "expected_answer": f"{t1_width:.1f}m",
            "category": "spatial",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_avg_width_m",
        })
    if t1_length is not None:
        pairs.append({
            "question": "What was Team 1's average team length (front-to-back depth)?",
            "expected_answer": f"{t1_length:.1f}m",
            "category": "spatial",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_avg_length_m",
        })

    # ── ENTITY (8 questions) — "Who?" (RAG-Football Tzikas 2025) ─────────────

    if sorted_players:
        # Who ran most distance
        top_dist_tid, top_dist_stats = sorted_players[0]
        pairs.append({
            "question": "Which player covered the most total distance?",
            "expected_answer": f"Player #{top_dist_tid} (Team {top_dist_stats.get('team_id', '?')}) with {top_dist_stats.get('total_distance_m', 0):.0f}m",
            "category": "entity",
            "verifiable": True,
            "source_path": "player_kinematics.max(total_distance_m)",
        })

        # Who had highest top speed
        by_speed = sorted(
            [(tid, s) for tid, s in players.items() if s.get("max_speed_m_per_sec") is not None],
            key=lambda x: x[1].get("max_speed_m_per_sec", 0),
            reverse=True,
        )
        if by_speed:
            spd_tid, spd_stats = by_speed[0]
            spd_kmh = (spd_stats.get("max_speed_m_per_sec") or 0) * 3.6
            pairs.append({
                "question": "Which player recorded the highest top speed?",
                "expected_answer": f"Player #{spd_tid} (Team {spd_stats.get('team_id', '?')}) at {spd_kmh:.1f} km/h",
                "category": "entity",
                "verifiable": True,
                "source_path": "player_kinematics.max(max_speed_m_per_sec)",
            })

        # Who had highest avg speed
        by_avg = sorted(
            [(tid, s) for tid, s in players.items() if s.get("avg_speed_m_per_sec") is not None],
            key=lambda x: x[1].get("avg_speed_m_per_sec", 0),
            reverse=True,
        )
        if by_avg:
            avg_tid, avg_stats = by_avg[0]
            avg_kmh = (avg_stats.get("avg_speed_m_per_sec") or 0) * 3.6
            pairs.append({
                "question": "Which player had the highest average speed?",
                "expected_answer": f"Player #{avg_tid} (Team {avg_stats.get('team_id', '?')}) at {avg_kmh:.1f} km/h average",
                "category": "entity",
                "verifiable": True,
                "source_path": "player_kinematics.max(avg_speed_m_per_sec)",
            })

    # Who made most passes (from events)
    player_passes: dict = {}
    for ev in pass_events:
        pid = ev.get("player_track_id")
        tid = ev.get("team_id")
        if pid is not None:
            player_passes[pid] = (player_passes.get(pid, (0, tid))[0] + 1, tid)
    if player_passes:
        top_passer = max(player_passes, key=lambda k: player_passes[k][0])
        pass_count, pass_team = player_passes[top_passer]
        pairs.append({
            "question": "Which player made the most passes?",
            "expected_answer": f"Player #{top_passer} (Team {pass_team}) with {pass_count} passes",
            "category": "entity",
            "verifiable": True,
            "source_path": "events[event_type=pass].player_track_id.count",
        })

    # Who made most sprints
    by_sprints = sorted(
        [(tid, s) for tid, s in players.items() if s.get("sprint_count") is not None],
        key=lambda x: x[1].get("sprint_count", 0),
        reverse=True,
    )
    if by_sprints:
        spr_tid, spr_stats = by_sprints[0]
        pairs.append({
            "question": "Which player performed the most sprints?",
            "expected_answer": f"Player #{spr_tid} (Team {spr_stats.get('team_id', '?')}) with {spr_stats.get('sprint_count', 0)} sprints",
            "category": "entity",
            "verifiable": True,
            "source_path": "player_kinematics.max(sprint_count)",
        })

    # ── MULTI-HOP (5 questions) — cross-metric (Sports Intelligence Yang 2025) ─

    # Multi-hop 1: possession leader → also more shots?
    if t1_pct is not None and t2_pct is not None:
        poss_leader = 1 if t1_pct > t2_pct else 2
        t1_shots = len([e for e in shot_events if e.get("team_id") == 1])
        t2_shots = len([e for e in shot_events if e.get("team_id") == 2])
        shot_leader = 1 if t1_shots >= t2_shots else 2
        same = poss_leader == shot_leader
        pairs.append({
            "question": "Did the team with more possession also have more shots?",
            "expected_answer": (
                f"Yes — Team {poss_leader} had both more possession ({max(t1_pct, t2_pct):.1f}%) "
                f"and more shots ({max(t1_shots, t2_shots)})"
                if same else
                f"No — Team {poss_leader} had more possession ({max(t1_pct, t2_pct):.1f}%) "
                f"but Team {shot_leader} had more shots ({max(t1_shots, t2_shots)})"
            ),
            "category": "multi_hop",
            "verifiable": True,
            "source_path": "possession.team_1_percentage + events[event_type=shot].count",
        })

    # Multi-hop 2: more compact → also less possession?
    if t1_comp is not None and t2_comp is not None and t1_pct is not None and t2_pct is not None:
        compact_team = 1 if t1_comp < t2_comp else 2
        compact_poss = t1_pct if compact_team == 1 else t2_pct
        pairs.append({
            "question": "Did the more compact team also have less possession (consistent with a defensive setup)?",
            "expected_answer": (
                f"Yes — Team {compact_team} was more compact ({min(t1_comp, t2_comp):.0f}m²) "
                f"and had less possession ({compact_poss:.1f}%)"
                if compact_poss < 50
                else f"No — Team {compact_team} was more compact ({min(t1_comp, t2_comp):.0f}m²) "
                f"but still had {compact_poss:.1f}% possession (more attack-minded compact shape)"
            ),
            "category": "multi_hop",
            "verifiable": True,
            "source_path": "tactical.summary.team_1_avg_compactness_m2 + possession.team_1_percentage",
        })

    # Multi-hop 3: most sprints player → also highest top speed?
    if by_sprints and by_speed:
        sprint_leader = by_sprints[0][0]
        speed_leader = by_speed[0][0]
        pairs.append({
            "question": "Was the player with the most sprints also the fastest player on the pitch?",
            "expected_answer": (
                f"Yes — Player #{sprint_leader} had both the most sprints and highest top speed"
                if str(sprint_leader) == str(speed_leader) else
                f"No — Player #{sprint_leader} had the most sprints but Player #{speed_leader} had the higher top speed"
            ),
            "category": "multi_hop",
            "verifiable": True,
            "source_path": "player_kinematics.max(sprint_count) + player_kinematics.max(max_speed_m_per_sec)",
        })

    # ── UNANSWERABLE (10 questions) ───────────────────────────────────────────
    # SQuAD 2.0: questions that cannot be answered from the available data.
    # Correct answer: model should say "I cannot answer this from the available data."

    unanswerable = [
        ("What are the real names of the players?", "player_names"),
        ("What was the score at half-time?", "halftime_score"),
        ("Which players were injured during the match?", "injuries"),
        ("What formation did Team 1 use?", "formation"),  # not directly computed
        ("How many yellow cards were shown?", "cards"),
        ("What was the expected goals (xG) for each team?", "xg"),
        ("Who was the man of the match?", "man_of_the_match"),
        ("What was Team 1's win probability at the start of the second half?", "win_probability"),
        ("How many corners did each team win?", "corners"),
        ("What was the attendance at the match?", "attendance"),
    ]
    for question, source in unanswerable:
        pairs.append({
            "question": question,
            "expected_answer": "This information is not available in the match analytics data.",
            "category": "unanswerable",
            "verifiable": False,
            "source_path": f"N/A ({source} not in analytics)",
        })

    return pairs[:100]  # cap at 100 (expanded from 45)


def generate_db_qa_pairs(db_ground_truth: dict) -> list[dict]:
    """Generate QA pairs sourced from per-frame DB ground truth (db_extractor.py output).

    Covers three new categories:
      - db_spatial   : player-position zone occupancy, centroid distances
      - db_temporal  : ball speed statistics, frame-level metrics
      - db_event_spatial : event location classification from spatial coordinates

    All event coordinates are stored in cm in the DB — divide by 100 for metres.
    Player centroid coords are stored in metres directly.
    """
    pairs: list[dict] = []
    fm = db_ground_truth.get("frame_metrics", {})
    events_db = db_ground_truth.get("events_db", [])
    formations = db_ground_truth.get("formations", {})

    # frame_metrics structure (from db_extractor.py):
    #   inter_team_distance_m  → list of {frame, distance_m}
    #   team_centroids         → list of {frame, team_1: [x,y], team_2: [x,y]}
    #   ball_trajectory        → list of {frame, pitchX, pitchY, speed_m_per_s}
    #   pitch_zone_occupancy   → {team_1: {defensive, middle, attacking}, team_2: ...}
    #   compactness_m2         → list of {frame, team_1_m2, team_2_m2}

    # ── DB SPATIAL (player-position derived) ────────────────────────────────────

    zone_occ = fm.get("pitch_zone_occupancy", {})

    # Dominant pitch third per team (from per-frame zone occupancy)
    for team_key, label in [("team_1", "Team 1"), ("team_2", "Team 2")]:
        team_zones = zone_occ.get(team_key, {})
        if team_zones:
            def_frac = team_zones.get("defensive", 0)
            mid_frac = team_zones.get("middle", 0)
            atk_frac = team_zones.get("attacking", 0)
            dominant_zone = max(
                [("defensive third", def_frac), ("middle third", mid_frac), ("attacking third", atk_frac)],
                key=lambda x: x[1],
            )
            pairs.append({
                "question": f"In which pitch third did {label} players spend the most time?",
                "expected_answer": (
                    f"{dominant_zone[0].title()} "
                    f"(def: {def_frac*100:.1f}%, mid: {mid_frac*100:.1f}%, atk: {atk_frac*100:.1f}%)"
                ),
                "category": "db_spatial",
                "verifiable": True,
                "source_path": f"db.frame_metrics.pitch_zone_occupancy.{team_key}",
            })

    # Average inter-team centroid distance (from per-frame positions)
    itd_records = fm.get("inter_team_distance_m", [])
    valid_itd = [r["distance_m"] for r in itd_records if isinstance(r, dict) and (r.get("distance_m") or 0) > 0]
    if valid_itd:
        mean_itd = sum(valid_itd) / len(valid_itd)
        pairs.append({
            "question": "What was the average distance between the two teams' centres of mass, as computed from per-frame player tracking?",
            "expected_answer": f"{mean_itd:.1f}m",
            "category": "db_spatial",
            "verifiable": True,
            "source_path": "db.frame_metrics.inter_team_distance_m.mean",
        })

    # Estimated formations
    t1_form = formations.get("team_1", {})
    t2_form = formations.get("team_2", {})
    t1_f = t1_form.get("formation") if isinstance(t1_form, dict) else None
    t2_f = t2_form.get("formation") if isinstance(t2_form, dict) else None
    t1_conf = t1_form.get("confidence", 0.0) if isinstance(t1_form, dict) else 0.0
    t2_conf = t2_form.get("confidence", 0.0) if isinstance(t2_form, dict) else 0.0

    if t1_f and t1_f != "unknown":
        pairs.append({
            "question": "What formation did Team 1 use, based on the average player positions tracked by the CV pipeline?",
            "expected_answer": f"{t1_f} (confidence: {t1_conf:.2f})",
            "category": "db_spatial",
            "verifiable": t1_conf > 0.2,
            "source_path": "db.formations.team_1.formation",
        })
    if t2_f and t2_f != "unknown":
        pairs.append({
            "question": "What formation did Team 2 use, based on the average player positions tracked by the CV pipeline?",
            "expected_answer": f"{t2_f} (confidence: {t2_conf:.2f})",
            "category": "db_spatial",
            "verifiable": t2_conf > 0.2,
            "source_path": "db.formations.team_2.formation",
        })

    # Team 1 mean centroid position
    centroid_records = fm.get("team_centroids", [])
    t1_xs = [r["team_1"][0] for r in centroid_records if isinstance(r, dict) and r.get("team_1")]
    t1_ys = [r["team_1"][1] for r in centroid_records if isinstance(r, dict) and r.get("team_1")]
    if t1_xs:
        mean_x = sum(t1_xs) / len(t1_xs)
        mean_y = sum(t1_ys) / len(t1_ys)
        pairs.append({
            "question": "What was Team 1's average centroid position on the pitch (in metres)?",
            "expected_answer": f"pitchX={mean_x:.1f}m, pitchY={mean_y:.1f}m",
            "category": "db_spatial",
            "verifiable": True,
            "source_path": "db.frame_metrics.team_centroids.team_1.mean",
        })

    # ── DB TEMPORAL (frame-level speed / trajectory) ─────────────────────────────

    ball_traj = fm.get("ball_trajectory", [])
    ball_speeds = [r["speed_m_per_s"] for r in ball_traj if isinstance(r, dict) and r.get("speed_m_per_s") is not None and r["speed_m_per_s"] >= 0]
    if ball_speeds:
        mean_spd = sum(ball_speeds) / len(ball_speeds)
        max_spd = max(ball_speeds)
        pairs.append({
            "question": "What was the average ball speed across the clip, as measured from per-frame tracking?",
            "expected_answer": f"{mean_spd:.1f} m/s ({mean_spd * 3.6:.1f} km/h)",
            "category": "db_temporal",
            "verifiable": True,
            "source_path": "db.frame_metrics.ball_trajectory.speed_m_per_s.mean",
        })
        pairs.append({
            "question": "What was the peak ball speed recorded during the clip?",
            "expected_answer": f"{max_spd:.1f} m/s ({max_spd * 3.6:.1f} km/h)",
            "category": "db_temporal",
            "verifiable": True,
            "source_path": "db.frame_metrics.ball_trajectory.speed_m_per_s.max",
        })
        slow_frames = sum(1 for v in ball_speeds if v < 5.0)
        slow_pct = slow_frames / len(ball_speeds) * 100
        pairs.append({
            "question": "What proportion of tracked frames had the ball moving at below 5 m/s (slow play)?",
            "expected_answer": f"{slow_pct:.1f}% of frames",
            "category": "db_temporal",
            "verifiable": True,
            "source_path": "db.frame_metrics.ball_trajectory.speed_m_per_s.pct_below_5",
        })

    # Inter-team distance trend (first quarter vs last quarter of clip)
    if len(valid_itd) >= 10:
        first_q = valid_itd[:len(valid_itd) // 4]
        last_q = valid_itd[3 * len(valid_itd) // 4:]
        mean_first = sum(first_q) / len(first_q)
        mean_last = sum(last_q) / len(last_q)
        trend = "increased" if mean_last > mean_first else "decreased"
        pairs.append({
            "question": "Did the distance between the two teams increase or decrease over the course of the clip?",
            "expected_answer": f"Distance {trend} (first quarter mean: {mean_first:.1f}m → last quarter mean: {mean_last:.1f}m)",
            "category": "db_temporal",
            "verifiable": True,
            "source_path": "db.frame_metrics.inter_team_distance_m.trend",
        })

    # ── DB EVENT-SPATIAL (events with spatial coordinates) ───────────────────────

    _PITCH_LEN_CM = 10500.0

    def _zone_cm(x_cm: float) -> str:
        frac = x_cm / _PITCH_LEN_CM
        if frac < 0.33:
            return "defensive third"
        if frac < 0.66:
            return "middle third"
        return "attacking third"

    # events_db uses camelCase keys: startX, startY, endX, endY
    events_with_coords = [
        e for e in events_db
        if (e.get("startX") or 0) > 0
    ]
    challenge_events = [e for e in events_with_coords if e.get("type") in ("challenge", "duel")]
    pass_events_db = [e for e in events_with_coords if e.get("type") == "pass"]

    # Challenges by zone
    if challenge_events:
        chall_zones: dict[str, int] = {}
        for ev in challenge_events:
            z = _zone_cm(ev["startX"])
            chall_zones[z] = chall_zones.get(z, 0) + 1
        dominant_chall_zone = max(chall_zones, key=lambda k: chall_zones[k])
        pairs.append({
            "question": "In which pitch third did the majority of challenges/duels occur?",
            "expected_answer": (
                f"{dominant_chall_zone.title()} "
                f"({chall_zones[dominant_chall_zone]} of {sum(chall_zones.values())} challenges)"
            ),
            "category": "db_event_spatial",
            "verifiable": True,
            "source_path": "db.events_db[type=challenge].start_x.zone",
        })

        # Challenges in defensive third (proxy for defensive pressure)
        def_challs = chall_zones.get("defensive third", 0)
        pairs.append({
            "question": "How many challenges occurred in the defensive third?",
            "expected_answer": str(def_challs),
            "category": "db_event_spatial",
            "verifiable": True,
            "source_path": "db.events_db[type=challenge,zone=defensive_third].count",
        })

    # Passes by zone
    if pass_events_db:
        pass_zones: dict[str, int] = {}
        for ev in pass_events_db:
            z = _zone_cm(ev["startX"])
            pass_zones[z] = pass_zones.get(z, 0) + 1
        dominant_pass_zone = max(pass_zones, key=lambda k: pass_zones[k])
        pairs.append({
            "question": "From which pitch third were most passes played?",
            "expected_answer": (
                f"{dominant_pass_zone.title()} "
                f"({pass_zones[dominant_pass_zone]} of {sum(pass_zones.values())} passes)"
            ),
            "category": "db_event_spatial",
            "verifiable": True,
            "source_path": "db.events_db[type=pass].start_x.zone",
        })
    elif events_with_coords:
        # No passes specifically, but we have events with coords
        all_zones: dict[str, int] = {}
        for ev in events_with_coords:
            z = _zone_cm(ev["startX"])
            all_zones[z] = all_zones.get(z, 0) + 1
        pairs.append({
            "question": "In which pitch third did most detected events occur?",
            "expected_answer": f"{max(all_zones, key=lambda k: all_zones[k]).title()} ({max(all_zones.values())} events)",
            "category": "db_event_spatial",
            "verifiable": True,
            "source_path": "db.events_db.start_x.zone_distribution",
        })

    return pairs


def _get_phase_press_change(summary: dict) -> str:
    dat_press = summary.get("team_1_dat_pressing_intensity")
    ip_press = summary.get("team_1_ip_pressing_intensity")
    if dat_press is not None and ip_press is not None:
        return f"Increased to {dat_press:.2f} in transition-to-attack vs {ip_press:.2f} in-possession"
    return "N/A"


def _infer_press_style(summary: dict, team: int) -> str:
    dl = summary.get(f"team_{team}_avg_defensive_line_m")
    if dl is None:
        return "N/A"
    if dl > 60:
        return f"High press (defensive line at {dl:.1f}m)"
    if dl >= 35:
        return f"Mid-block (defensive line at {dl:.1f}m)"
    return f"Low block (defensive line at {dl:.1f}m)"


def _best_territory_phase(summary: dict, team: int) -> str:
    phases = {
        "in-possession": summary.get(f"team_{team}_ip_territory_pct"),
        "out-of-possession": summary.get(f"team_{team}_oop_territory_pct"),
        "transition attack": summary.get(f"team_{team}_dat_territory_pct"),
    }
    valid = {k: v for k, v in phases.items() if v is not None}
    if not valid:
        return "N/A"
    best = max(valid, key=lambda k: valid[k])
    return f"{best.title()} ({valid[best]*100:.1f}%)"


# ── Evaluation ────────────────────────────────────────────────────────────────


async def evaluate_qa_pair(
    qa: dict,
    analytics: dict,
    provider,
    grounded_markdown: str,
) -> dict:
    """Ask the chat system the question and evaluate the answer.

    Returns:
        {question, expected, model_answer, category, correct, unanswerable_detected}
    """
    system = (
        "You are an expert football tactical analyst. Answer questions about this specific match "
        "based ONLY on the data provided. If the question cannot be answered from the data, "
        "say exactly: 'This information is not available in the match analytics data.'\n\n"
        f"Match data:\n{grounded_markdown}"
    )
    try:
        answer = await provider.chat(system, [{"role": "user", "content": qa["question"]}])
    except Exception as e:
        answer = f"[ERROR: {e}]"

    expected = qa["expected_answer"]
    category = qa["category"]

    if category == "unanswerable":
        # SQuAD 2.0: correct if model refuses to answer
        unanswerable_detected = (
            "not available" in answer.lower()
            or "cannot answer" in answer.lower()
            or "don't have" in answer.lower()
            or "no information" in answer.lower()
        )
        return {**qa, "model_answer": answer, "correct": unanswerable_detected,
                "unanswerable_detected": unanswerable_detected}

    if category == "numeric":
        # Extract numbers and compare (5% tolerance)
        def _nums(s):
            return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
        exp_nums = _nums(expected)
        ans_nums = _nums(answer)
        correct = any(
            any(abs(a - e) <= max(abs(e) * 0.05, 0.5) for a in ans_nums)
            for e in exp_nums
        ) if exp_nums and ans_nums else False
        return {**qa, "model_answer": answer, "correct": correct, "unanswerable_detected": False}

    # If data was missing (expected = "N/A" / "0" / not verifiable), correct answer
    # is the model acknowledging the data isn't available (like an unanswerable question).
    _unavailable = ("not available", "cannot answer", "don't have",
                    "no information", "not provided", "no data", "n/a")
    if expected in ("N/A", "0") or not qa.get("verifiable", True):
        detected = any(p in answer.lower() for p in _unavailable)
        return {**qa, "model_answer": answer, "correct": detected, "unanswerable_detected": detected}

    # Comparative / tactical: fuzzy keyword match on expected answer keywords.
    # Strip punctuation/units that corrupt keyword matching (e.g. "(defensive", "45.2m)").
    exp_keywords = [re.sub(r"[(),%°]", "", w).lower() for w in expected.split() if len(w) > 3]
    exp_keywords = [k for k in exp_keywords if len(k) > 2]  # re-filter after stripping
    ans_lower = answer.lower()
    matches = sum(1 for kw in exp_keywords if kw in ans_lower)
    correct = matches >= max(1, len(exp_keywords) // 3)
    return {**qa, "model_answer": answer, "correct": correct, "unanswerable_detected": False}


async def run_benchmark(
    analytics: dict,
    provider_name: str,
    output_dir: str,
    db_ground_truth: dict | None = None,
) -> dict:
    """Run the full QA benchmark.

    Args:
        analytics: Pipeline analytics JSON (from load_analytics).
        provider_name: LLM provider name.
        output_dir: Directory for output files.
        db_ground_truth: Optional per-frame DB ground truth (from load_db_ground_truth).
            When provided, adds db_spatial / db_temporal / db_event_spatial QA categories.
    """
    from services.llm_providers import get_provider
    from services.tactical import GroundingFormatter

    provider = get_provider(provider_name)
    if not provider.is_available():
        print(f"  Provider {provider_name} not available")
        return {}

    grounded_md = GroundingFormatter.format(analytics)
    pairs = generate_qa_pairs(analytics)

    if db_ground_truth is not None:
        db_pairs = generate_db_qa_pairs(db_ground_truth)
        pairs = (pairs + db_pairs)[:120]  # cap raised from 100 to 120 when DB pairs added
        print(f"  Added {len(db_pairs)} DB-sourced QA pairs (db_spatial/db_temporal/db_event_spatial)")

    print(f"  Running {len(pairs)} QA pairs with {provider_name}...")
    results = []
    for i, qa in enumerate(pairs):
        print(f"    [{i+1}/{len(pairs)}] {qa['category']}: {qa['question'][:60]}...")
        result = await evaluate_qa_pair(qa, analytics, provider, grounded_md)
        results.append(result)

    # Aggregate metrics
    by_category: dict[str, dict] = {}
    for cat in (
        "numeric", "comparative", "tactical", "temporal", "spatial", "entity", "multi_hop",
        "unanswerable", "db_spatial", "db_temporal", "db_event_spatial",
    ):
        cat_results = [r for r in results if r["category"] == cat]
        correct = sum(1 for r in cat_results if r["correct"])
        n = len(cat_results)
        by_category[cat] = {
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
        }

    total = len(results)
    total_correct = sum(1 for r in results if r["correct"])

    # Unanswerable precision/recall (SQuAD 2.0)
    unans = [r for r in results if r["category"] == "unanswerable"]
    ans = [r for r in results if r["category"] != "unanswerable"]
    tp = sum(1 for r in unans if r.get("unanswerable_detected"))
    fp = sum(1 for r in ans if r.get("unanswerable_detected"))
    fn = sum(1 for r in unans if not r.get("unanswerable_detected"))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    summary = {
        "provider": provider_name,
        "total_questions": total,
        "overall_accuracy": round(total_correct / total, 4) if total else 0.0,
        "by_category": by_category,
        "unanswerable_detection": {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)},
    }

    # Save results
    out = ensure_output_dir(output_dir)
    (out / f"{provider_name}_qa_results.json").write_text(json.dumps(results, indent=2))
    (out / f"{provider_name}_qa_summary.json").write_text(json.dumps(summary, indent=2))

    # LaTeX tables
    rows = [
        [cat.replace("_", " ").capitalize(), v["n"], v["correct"],
         f"{v['accuracy']*100:.1f}\\%"]
        for cat, v in by_category.items()
        if v["n"] > 0
    ]
    rows.append(["\\textbf{Total}", total, total_correct, f"\\textbf{{{total_correct/total*100:.1f}\\%}}"])
    save_latex_table(
        headers=["Category", "N", "Correct", "Accuracy"],
        rows=rows,
        caption=f"Chat QA benchmark accuracy by category -- {provider_name} (RAGAS/SQuAD 2.0 methodology)",
        name=f"{provider_name}_qa_accuracy",
        output_dir=output_dir,
        label=f"tab:qa_accuracy_{provider_name}",
    )

    unanswerable_rows = [
        ["Unanswerable detection precision", f"{prec*100:.1f}\\%"],
        ["Unanswerable detection recall", f"{rec*100:.1f}\\%"],
        ["Unanswerable detection F1", f"{f1*100:.1f}\\%"],
    ]
    save_latex_table(
        headers=["Metric", "Value"],
        rows=unanswerable_rows,
        caption=f"Unanswerable question detection metrics (SQuAD 2.0) -- {provider_name}",
        name=f"{provider_name}_unanswerable_detection",
        output_dir=output_dir,
        label=f"tab:unanswerable_{provider_name}",
    )

    # Bar chart
    _CAT_COLORS = {
        "numeric": "#4f86c6", "comparative": "#e07b54", "tactical": "#6aab6a",
        "temporal": "#f0c040", "spatial": "#e06090", "entity": "#60c0e0",
        "db_spatial": "#c06090", "db_temporal": "#d0a020", "db_event_spatial": "#509070",
        "multi_hop": "#a070d0", "unanswerable": "#b89fc8",
    }
    fig, ax = plt.subplots(figsize=(10, 4))
    cats = [c for c in by_category if by_category[c]["n"] > 0]
    accs = [by_category[c]["accuracy"] * 100 for c in cats]
    colors = [_CAT_COLORS.get(c, "#888888") for c in cats]
    bars = ax.bar([c.replace("_", " ") for c in cats], accs, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Chat QA Benchmark -- {provider_name}")
    fig.tight_layout()
    save_figure(fig, f"{provider_name}_qa_accuracy", output_dir)

    print(f"\n  Overall accuracy: {total_correct}/{total} = {summary['overall_accuracy']:.1%}")
    print(f"  Unanswerable F1: {f1:.1%}")
    for cat, v in by_category.items():
        print(f"    {cat}: {v['correct']}/{v['n']} = {v['accuracy']:.1%}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat QA benchmark (RAGAS + SQuAD 2.0)")
    parser.add_argument("--analytics", required=True)
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Optional path to per-frame DB ground truth JSON (from db_extractor.py). "
             "When provided, adds db_spatial / db_temporal / db_event_spatial QA categories.",
    )
    parser.add_argument("--provider", default="openai", choices=["gemini", "openai", "huggingface", "claude", "groq"])
    parser.add_argument("--output", default="eval_output/phase16/chat_qa")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env", override=True)
    except ImportError:
        pass

    analytics = load_analytics(args.analytics)
    db_ground_truth = None
    if args.ground_truth:
        db_ground_truth = load_db_ground_truth(args.ground_truth)

    asyncio.run(run_benchmark(analytics, args.provider, args.output, db_ground_truth=db_ground_truth))


if __name__ == "__main__":
    main()
