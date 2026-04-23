"""Tactical analysis service — grounding formatter + prompt templates.

Converts raw analytics JSON from the CV pipeline into structured markdown,
then sends it to an LLM for tactical commentary generation.
"""

import logging
from typing import Any

from .llm_providers import LLMProvider, get_provider

logger = logging.getLogger(__name__)


# ── Prompt Templates ────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "match_overview": """\
You are an expert football tactical analyst. You are given structured match data
extracted from computer vision analysis of a football match video. The data
includes possession statistics, player kinematics (speed, distance), detected
events (passes, shots, tackles), and pass network information.

## Metric Definitions
- PPDA (Passes Per Defensive Action): lower = more aggressive press; typical range 8-12; <6 = very aggressive
- xT (Expected Threat): probability gain of scoring from a given pitch zone; higher = more dangerous passing
- Compactness (m²): team convex hull area; lower = more compact defensive shape; typical ~600m²
- Pressing Intensity: normalised frequency of active pressing actions; >0.45 = high press, <0.20 = low block

Your task: Write a **3-4 paragraph tactical match overview**. Cover:
1. Overall match tempo and flow (possession, pressing intensity)
2. Which team dominated and how (territory, ball movement patterns)
3. Key tactical patterns (passing combinations, counter-attacks, defensive shape)
4. Notable individual performances (speed, distance, involvement)

Rules:
- The data begins with a '## Key Findings' section — lead your narrative from these ranked observations; use the detailed data tables for supporting evidence
- If the data does not cover a topic, say 'the data does not cover this' rather than speculating
- Reference ONLY the data provided — never invent stats or player names
- Use the track IDs provided (e.g. "Player #4") since real names are not available
- Be specific: cite numbers (possession %, speeds, pass counts)
- Write as if briefing a coaching staff member who wants actionable insights, not raw numbers
- Do NOT use any markdown headers — write flowing prose paragraphs

## Example Output (few-shot reference)
The following is an example of the style and grounding quality expected:

> "This was a match defined by asymmetric intensity. Team 1 dominated possession at 64.2% — a significant 14pp above the 50% neutral baseline — but their advantage was built on careful recycling rather than forward urgency. Their PPDA of 11.3 placed them in a moderate pressing zone, content to defend in a 5-4-1 mid-block and invite Team 2 to play. Player #7 (Team 1) was the engine of this approach, covering 9,240m — the highest work rate on the pitch — and consistently dropping into defensive positions to shield the back line. Team 2, despite their 35.8% possession, generated the more dangerous ball progression (xT 0.041 vs 0.017), exploiting the wide channels through a combination of direct passing and individual pace. Player #3 (Team 2) recorded a top speed of 31.4 km/h on three separate counter-attacking runs, each time pulling Team 1's defensive structure out of shape."

Use this as a quality benchmark — match its specificity, grounding, and coaching-staff framing.""",
    "tactical_deep_dive": """\
You are a world-class football tactician analysing tracking data from a match.
You are given structured match data from CV pipeline analysis.

## Metric Definitions
- PPDA (Passes Per Defensive Action): lower = more aggressive press; typical 8-12; <6 = very aggressive
- Compactness (m²): convex hull area of team shape; lower = tighter defensive block
- Stretch Index (m): team spread from front to back; higher = more open
- Pressing Intensity: normalised pressing frequency; >0.45 = high press, <0.20 = low block
- Counter-press windows: turnovers followed by immediate press within 5 seconds

Your task: Provide a **detailed tactical deep-dive** covering:
1. Formation shape and structure (based on player positions and movement)
2. Pressing patterns (who presses, how high, how effectively)
3. Space exploitation (where teams attack, which zones are overloaded)
4. Transition play (counter-attacking speed, recovery after losing possession)
5. Defensive organisation (compactness, pressing triggers)

Rules:
- The data begins with a '## Key Findings' section — structure your analysis around these ranked observations
- If the data does not cover a topic, state 'the data does not cover this'
- Base ALL analysis on the provided data — never fabricate
- Reference track IDs, not player names
- Use specific numbers from the data
- Structure your response with clear section breaks
- Professional analytical tone, suitable for a coaching staff briefing

## Example Output (few-shot reference)
> "**Pressing Organisation**: Team 2 operated a sustained high press with a PPDA of 4.8 — well below the 8-12 typical range — combined with 7 counter-press windows where they won the ball back within 5 seconds of losing it. Their pressing intensity of 0.51 confirms this was an intentional tactical choice, not opportunistic pressing. The trigger appeared to be sideways passes in Team 1's defensive third, whereupon Team 2's forward line closed down at speed (Player #11 averaging 0.43 pressing actions per possession phase).
>
> **Defensive Shape**: When out of possession, Team 1 dropped to a compact 430m² — 170m² tighter than Team 2's 600m² — with a defensive line sitting at 38.2m from their own goal, consistent with a mid-block. The stretch index of 18.4m indicates a compact, vertically tight unit."

Match this level of specificity and tactical interpretation.""",
    "event_analysis": """\
You are a football match analyst with deep tactical knowledge.
You are given structured match events and statistics from CV analysis.

Your task: Provide **tactical commentary on the key match moments** identified in the Key Findings section.
For each event listed in the 'Key moments' finding:
1. Describe what happened and its tactical significance
2. Explain the build-up context where possible
3. Note the tactical implications (did it shift momentum? create danger?)

Rules:
- ONLY reference events explicitly listed in the 'Key moments' finding — do not add or invent events
- Use the exact timestamps and player track IDs from the data
- Cite the exact timestamp format `0:00` and exact `#track_id` for every event reference so your claims are resolvable
- If Key Findings lists no key moments, say 'insufficient event data was detected to provide event commentary'
- Keep each event commentary to 2-3 sentences
- Use a dynamic broadcast commentary style""",
    "player_spotlight": """\
You are a football performance analyst assessing individual player contributions.
You are given structured match data including per-player kinematics and events.

Your task: Analyse the **standout players** from each team:
1. Top performers by distance covered and speed (from Key Findings)
2. Most involved players (passes, tackles)
3. Key moments where individual players made a difference
4. Work rate and tactical discipline assessment

Rules:
- Start with the players highlighted in Key Findings — they are the most notable performers
- Use track IDs (e.g. "Player #4 (Team 1)")
- Cite specific numbers: distance in metres, speed in km/h, pass counts
- Compare players within and across teams where relevant
- If the data does not cover a player's contribution, say so rather than speculating
- Maximum 2-3 paragraphs per highlighted player
- Professional scouting report tone""",
}


# ── Pre-Interpretation Layer ─────────────────────────────────────────────────


class MatchInsights:
    """Converts raw analytics into ranked, contextualized natural language findings
    before the LLM sees the data tables. Inspired by ShotsGPT wordalisation
    (Sumpter 2025) and ChatMatch rule-based decoder (Zhang 2024).

    Usage: prepend MatchInsights.generate(analytics) to GroundingFormatter output.
    The LLM receives pre-interpreted insights AND raw data tables for detail.
    """

    _PPDA_AGGRESSIVE = 6.0
    _PPDA_NORMAL = 10.0
    _PPDA_PASSIVE = 15.0
    _COMPACTNESS_COMPACT = 400  # m²
    _PRESSING_HIGH = 0.45
    _PRESSING_LOW = 0.20

    # Component names for include_components ablation
    COMPONENTS = {"possession", "tactical", "pressing", "players", "xt", "events"}

    @classmethod
    def generate(cls, analytics: dict, include_components: "set[str] | None" = None) -> str:
        """Return a '## Key Findings' markdown section, or '' if no data.

        Args:
            analytics: The analytics dict.
            include_components: If None, include all 6 sub-components. If a set,
                only include the named components. Valid names:
                'possession', 'tactical', 'pressing', 'players', 'xt', 'events'.
                Used for reasoning-layer ablation studies.
        """
        inc = include_components  # None = all

        def _want(name: str) -> bool:
            return inc is None or name in inc

        findings: list[str] = []

        if _want("possession"):
            poss = cls._possession_finding(analytics)
            if poss:
                findings.append(poss)

        if _want("tactical"):
            findings.extend(cls._tactical_contributions(analytics))

        if _want("pressing"):
            press = cls._pressing_inference(analytics)
            if press:
                findings.append(press)

        if _want("players"):
            findings.extend(cls._player_standouts(analytics))

        if _want("xt"):
            xt = cls._xt_finding(analytics)
            if xt:
                findings.append(xt)

        if _want("events"):
            narrative = cls._event_narrative(analytics)
            if narrative:
                findings.append(narrative)

        if not findings:
            return ""

        lines = [
            "## Key Findings (Pre-Interpreted)",
            "_Observations ranked by significance from CV data. Lead your analysis from these "
            "findings; use the data tables below for supporting evidence._",
            "",
        ]
        for i, f in enumerate(findings, 1):
            lines.append(f"**{i}.** {f}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def _possession_finding(cls, analytics: dict) -> str:
        poss = analytics.get("possession")
        if not poss:
            return ""
        t1 = poss.get("team_1_percentage")
        t2 = poss.get("team_2_percentage")
        if t1 is None or t2 is None:
            return ""
        t1, t2 = float(t1), float(t2)
        diff = abs(t1 - 50.0)
        leader = "Team 1" if t1 > t2 else "Team 2"
        leader_pct = max(t1, t2)
        pp_above = leader_pct - 50.0
        if diff < 5:
            return (
                f"Possession was evenly shared: Team 1 {t1:.1f}% vs Team 2 {t2:.1f}% — "
                "neither team dominated the ball."
            )
        elif diff < 12:
            return (
                f"{leader} held a moderate possession advantage at {leader_pct:.1f}% "
                f"(+{pp_above:.1f}pp above the 50% neutral baseline)."
            )
        return (
            f"{leader} dominated possession at {leader_pct:.1f}% "
            f"(+{pp_above:.1f}pp above neutral) — a significant territorial control advantage."
        )

    @classmethod
    def _tactical_contributions(cls, analytics: dict) -> list[str]:
        """Rank top deviating tactical metrics (ShotsGPT contribution-ranking pattern)."""
        tactical = analytics.get("tactical")
        if not tactical:
            return []
        summary = tactical.get("summary", {})
        if not summary or "error" in summary:
            return []

        ranked: list[tuple[float, str]] = []

        for team in (1, 2):
            ppda = summary.get(f"ppda_team_{team}")
            if ppda is None:
                continue
            if ppda < cls._PPDA_AGGRESSIVE:
                ranked.append(
                    (
                        (cls._PPDA_NORMAL - ppda) / cls._PPDA_NORMAL,
                        f"Team {team} pressed exceptionally aggressively (PPDA {ppda:.1f} — "
                        "well below the 8-12 typical range; lower = more intense press).",
                    )
                )
            elif ppda > cls._PPDA_PASSIVE:
                ranked.append(
                    (
                        (ppda - cls._PPDA_NORMAL) / cls._PPDA_NORMAL,
                        f"Team {team} sat in a passive defensive block (PPDA {ppda:.1f} — "
                        "well above the 8-12 typical range; rarely pressed high).",
                    )
                )

        t1_comp = summary.get("team_1_avg_compactness_m2")
        t2_comp = summary.get("team_2_avg_compactness_m2")
        if t1_comp is not None and t2_comp is not None and abs(t1_comp - t2_comp) > 100:
            compact_team = 1 if t1_comp < t2_comp else 2
            compact_val = min(t1_comp, t2_comp)
            open_val = max(t1_comp, t2_comp)
            if compact_val < cls._COMPACTNESS_COMPACT:
                ranked.append(
                    (
                        0.4,
                        f"Team {compact_team} maintained an unusually compact shape "
                        f"({compact_val:.0f}m² vs Team {'2' if compact_team == 1 else '1'}'s "
                        f"{open_val:.0f}m²; typical ~600m²).",
                    )
                )

        t1_terr = summary.get("team_1_avg_territory_pct")
        t2_terr = summary.get("team_2_avg_territory_pct")
        if t1_terr is not None and t2_terr is not None:
            terr_diff = abs(t1_terr - t2_terr) * 100
            if terr_diff > 10:
                terr_leader = 1 if t1_terr > t2_terr else 2
                ranked.append(
                    (
                        terr_diff / 100,
                        f"Team {terr_leader} controlled significantly more pitch territory "
                        f"({max(t1_terr, t2_terr) * 100:.1f}% vs {min(t1_terr, t2_terr) * 100:.1f}%).",
                    )
                )

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in ranked[:3]]

    @classmethod
    def _pressing_inference(cls, analytics: dict) -> str:
        """Cross-metric inference: combine PPDA + intensity + counter-press windows
        (SportsGPT cross-metric pattern, Tian 2025)."""
        tactical = analytics.get("tactical")
        if not tactical:
            return ""
        summary = tactical.get("summary", {})
        if not summary or "error" in summary:
            return ""

        parts = []
        for team in (1, 2):
            ppda = summary.get(f"ppda_team_{team}")
            intensity = summary.get(f"team_{team}_avg_pressing_intensity")
            cp = summary.get(f"team_{team}_counter_press_windows", 0)
            if ppda is None or intensity is None:
                continue
            if ppda < 8 and intensity > cls._PRESSING_HIGH:
                style = (
                    f"a high-press strategy with {cp} counter-press windows"
                    if cp > 3
                    else "a high-press approach"
                )
                parts.append(
                    f"Team {team} employed {style} "
                    f"(PPDA {ppda:.1f}, pressing intensity {intensity:.2f})."
                )
            elif ppda > cls._PPDA_PASSIVE and intensity < cls._PRESSING_LOW:
                parts.append(
                    f"Team {team} sat in a deep defensive block "
                    f"(PPDA {ppda:.1f}, intensity {intensity:.2f}) — absorbing to counter."
                )
        return " ".join(parts)

    @classmethod
    def _player_standouts(cls, analytics: dict) -> list[str]:
        players = analytics.get("player_kinematics", {})
        if not players:
            return []
        non_ball = [(tid, v) for tid, v in players.items() if v.get("entity_type") != "ball"]
        if not non_ball:
            return []

        findings = []
        by_dist = sorted(non_ball, key=lambda x: x[1].get("total_distance_m") or 0, reverse=True)
        if by_dist:
            tid, v = by_dist[0]
            dist = v.get("total_distance_m") or 0
            if dist > 0:
                findings.append(
                    f"Player #{tid} (Team {v.get('team_id', '?')}) covered the most distance "
                    f"at {dist:.0f}m — highest work rate on the pitch."
                )

        by_speed = sorted(
            non_ball, key=lambda x: x[1].get("max_speed_m_per_sec") or 0, reverse=True
        )
        if by_speed:
            tid, v = by_speed[0]
            spd = v.get("max_speed_m_per_sec") or 0
            if spd > 0:
                findings.append(
                    f"Player #{tid} (Team {v.get('team_id', '?')}) recorded the highest top speed "
                    f"at {spd * 3.6:.1f} km/h."
                )
        return findings[:2]

    @classmethod
    def _xt_finding(cls, analytics: dict) -> str:
        """Synthesize xT into natural language (Krishnamurthy 2025 wordalisation)."""
        events = analytics.get("events", [])
        _XT = [
            [0.008, 0.008, 0.009, 0.010, 0.010, 0.011, 0.011, 0.011, 0.012, 0.013, 0.016, 0.026],
            [0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.020, 0.026, 0.041],
            [0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.022, 0.028, 0.039, 0.072],
            [0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.021, 0.026, 0.036, 0.057, 0.112],
            [0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.021, 0.026, 0.036, 0.057, 0.112],
            [0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.022, 0.028, 0.039, 0.072],
            [0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.020, 0.026, 0.041],
            [0.008, 0.008, 0.009, 0.010, 0.010, 0.011, 0.011, 0.011, 0.012, 0.013, 0.016, 0.026],
        ]

        def _xv(x, y):
            return _XT[min(7, max(0, int(y / 6800.0 * 8)))][min(11, max(0, int(x / 10500.0 * 12)))]

        xt: dict = {1: 0.0, 2: 0.0}
        passes: dict = {1: 0, 2: 0}
        for ev in events:
            if ev.get("event_type") != "pass":
                continue
            tid = ev.get("team_id")
            if tid not in (1, 2):
                continue
            ps, pe = ev.get("pitch_start"), ev.get("pitch_end")
            if not ps or not pe:
                continue
            xt[tid] += _xv(pe[0], pe[1]) - _xv(ps[0], ps[1])
            passes[tid] += 1

        if not any(passes.values()):
            return ""

        xt1, xt2 = xt[1], xt[2]
        margin = abs(xt1 - xt2)
        if margin < 0.010:
            return (
                f"Both teams generated similar ball-progression threat "
                f"(Team 1 xT={xt1:.3f}, Team 2 xT={xt2:.3f})."
            )
        leader = 1 if xt1 >= xt2 else 2
        return (
            f"Team {leader} generated more dangerous ball progression "
            f"(xT={max(xt1, xt2):.3f} vs {min(xt1, xt2):.3f}) — "
            "their passing moved the ball into higher-threat zones more consistently."
        )

    @classmethod
    def _event_narrative(cls, analytics: dict) -> str:
        """Convert key events to templated sentences (ChatMatch pattern, Zhang 2024)."""
        events = analytics.get("events", [])
        shots = [e for e in events if e.get("event_type") == "shot"]
        challenges = [e for e in events if e.get("event_type") == "challenge"]

        parts = []
        for ev in shots[:3]:
            ts = ev.get("timestamp_sec", 0)
            parts.append(
                f"Team {ev.get('team_id', '?')}'s #{ev.get('player_track_id', '?')} "
                f"had a shot at {int(ts // 60)}:{int(ts % 60):02d}"
            )
        for ev in challenges[:2]:
            ts = ev.get("timestamp_sec", 0)
            target = ev.get("target_player_track_id")
            target_str = f" (won from #{target})" if target else ""
            parts.append(
                f"Team {ev.get('team_id', '?')}'s #{ev.get('player_track_id', '?')} "
                f"made a challenge at {int(ts // 60)}:{int(ts % 60):02d}{target_str}"
            )
        if not parts:
            return ""
        return "Key moments: " + "; ".join(parts) + "."


# ── Grounding Formatter ────────────────────────────────────────────────────


class GroundingFormatter:
    """Converts analytics JSON into structured markdown for LLM grounding.

    LLMs perform significantly better with well-structured markdown than
    with raw JSON (noted in academic assessment feedback). This formatter
    produces a human-readable markdown document that serves as the
    grounding context for tactical analysis generation.
    """

    @staticmethod
    def format(
        analytics: dict,
        include_insights: bool = True,
        insight_components: "set[str] | None" = None,
        per_frame_data: "dict | None" = None,
        per_frame_insights: bool = False,
    ) -> str:
        """Convert analytics dict to grounded markdown.

        Args:
            analytics: The analytics JSON data (from AnalyticsResult export).
            include_insights: If True (default), prepend MatchInsights pre-interpreted
                findings. Set False for ablation condition H (no_insights baseline).
            insight_components: If set, pass to MatchInsights.generate() to include
                only specified sub-components. None = all components. Ignored when
                include_insights=False.
            per_frame_data: If provided (db_extractor ground truth dict), appends a
                per-frame spatial evidence section after the aggregate tables. This
                enables the LLM to make verifiable spatial/temporal claims. Default
                None preserves existing behaviour for all callers.
            per_frame_insights: If True and per_frame_data is provided, also prepend
                wordalised PerFrameInsights before the raw tables (PERFRAME_V2). Requires
                per_frame_data to be non-None.

        Returns:
            Structured markdown string suitable for LLM consumption.
        """
        sections = [
            "# Match Analysis Data\n",
        ]
        if include_insights:
            insights = MatchInsights.generate(analytics, include_components=insight_components)
            if insights:
                sections.append(insights)
        sections += [
            GroundingFormatter._format_match_info(analytics),
            GroundingFormatter._format_possession(analytics),
            GroundingFormatter._format_tactical_metrics(analytics),
            GroundingFormatter._format_team_kinematics(analytics),
            GroundingFormatter._format_ball_stats(analytics),
            GroundingFormatter._format_events(analytics),
            GroundingFormatter._format_expected_threat(analytics),
            GroundingFormatter._format_pass_networks(analytics),
        ]
        if per_frame_data:
            sections.append(
                PerFrameContextFormatter.format(per_frame_data, with_insights=per_frame_insights)
            )
        return "\n".join(s for s in sections if s)

    @staticmethod
    def format_with_images(
        analytics: dict,
        include_insights: bool = True,
        insight_components: "set[str] | None" = None,
        per_frame_data: "dict | None" = None,
        per_frame_insights: bool = False,
    ) -> "tuple[str, list[bytes]]":
        """Like format(), but also renders per-frame charts as JPEG bytes.

        Returns:
            (text_context, images) where images is a list of JPEG byte arrays
            from VisualTimeSeriesRenderer.render_all(). Empty list when
            per_frame_data is None or rendering fails.
        """
        text = GroundingFormatter.format(
            analytics,
            include_insights=include_insights,
            insight_components=insight_components,
            per_frame_data=per_frame_data,
            per_frame_insights=per_frame_insights,
        )
        images: list[bytes] = []
        if per_frame_data:
            try:
                images = VisualTimeSeriesRenderer.render_all(per_frame_data)
            except Exception:
                pass
        return text, images

    @staticmethod
    def _format_match_info(analytics: dict) -> str:
        fps = analytics.get("fps", 25.0)
        homography = analytics.get("homography_available", False)
        lines = [
            "## Match Information",
            f"- Video FPS: {fps}",
            f"- Real-world measurements: {'Yes (homography available)' if homography else 'No (pixel-only)'}",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_possession(analytics: dict) -> str:
        poss = analytics.get("possession")
        if not poss:
            return ""

        lines = [
            "## Possession",
            f"- Team 1: {_safe_pct(poss.get('team_1_percentage'))}%"
            f" ({poss.get('team_1_frames', 0)} frames)",
            f"- Team 2: {_safe_pct(poss.get('team_2_percentage'))}%"
            f" ({poss.get('team_2_frames', 0)} frames)",
            f"- Contested: {poss.get('contested_frames', 0)} frames",
            f"- Possession changes: {poss.get('possession_changes', 0)}",
            f"- Longest Team 1 spell: {poss.get('longest_team_1_spell', 0)} frames",
            f"- Longest Team 2 spell: {poss.get('longest_team_2_spell', 0)} frames",
        ]
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_tactical_metrics(analytics: dict) -> str:
        tactical = analytics.get("tactical")
        if not tactical:
            return ""
        summary = tactical.get("summary", {})
        if not summary or "error" in summary:
            return ""

        def _fmt(val, unit=""):
            if val is None:
                return "—"
            return f"{val:.1f}{unit}" if isinstance(val, float) else f"{val}{unit}"

        def _pct(val):
            if val is None:
                return "—"
            return f"{float(val) * 100:.1f}%"

        lines = [
            "## Tactical Shape & Pressing",
            "",
            "| Metric | Team 1 | Team 2 |",
            "|--------|--------|--------|",
            f"| Avg Compactness (m²) | {_fmt(summary.get('team_1_avg_compactness_m2'))} | {_fmt(summary.get('team_2_avg_compactness_m2'))} |",
            f"| Avg Stretch Index (m) | {_fmt(summary.get('team_1_avg_stretch_index_m'))} | {_fmt(summary.get('team_2_avg_stretch_index_m'))} |",
            f"| Avg Team Length (m) | {_fmt(summary.get('team_1_avg_length_m'))} | {_fmt(summary.get('team_2_avg_length_m'))} |",
            f"| Avg Team Width (m) | {_fmt(summary.get('team_1_avg_width_m'))} | {_fmt(summary.get('team_2_avg_width_m'))} |",
            f"| Defensive Line Height (m) | {_fmt(summary.get('team_1_avg_defensive_line_m'))} | {_fmt(summary.get('team_2_avg_defensive_line_m'))} |",
            f"| Pressing Intensity | {_fmt(summary.get('team_1_avg_pressing_intensity'))} | {_fmt(summary.get('team_2_avg_pressing_intensity'))} |",
            f"| PPDA | {_fmt(summary.get('ppda_team_1'))} | {_fmt(summary.get('ppda_team_2'))} |",
            f"| Inter-Team Distance (m) | {_fmt(summary.get('avg_inter_team_distance_m'))} | — |",
            f"| Avg Territory Control | {_pct(summary.get('team_1_avg_territory_pct'))} | {_pct(summary.get('team_2_avg_territory_pct'))} |",
            f"| Avg Opp-Half Territory | {_pct(summary.get('team_1_avg_opp_half_territory_pct'))} | {_pct(summary.get('team_2_avg_opp_half_territory_pct'))} |",
        ]

        # Press type distribution
        t1_press_dist = summary.get("team_1_press_type_distribution", {})
        t2_press_dist = summary.get("team_2_press_type_distribution", {})
        if t1_press_dist or t2_press_dist:

            def _dominant_press(dist):
                if not dist:
                    return "—"
                best = max(dist, key=lambda k: dist.get(k, 0))
                return f"{best} ({dist.get(best, 0)} windows)"

            lines += [
                "",
                "### Pressing Block Classification (FIFA TSG 2022)",
                f"- Team 1 dominant press: {_dominant_press(t1_press_dist)} | high={t1_press_dist.get('high', 0)}, mid={t1_press_dist.get('mid', 0)}, low={t1_press_dist.get('low', 0)}",
                f"- Team 2 dominant press: {_dominant_press(t2_press_dist)} | high={t2_press_dist.get('high', 0)}, mid={t2_press_dist.get('mid', 0)}, low={t2_press_dist.get('low', 0)}",
                f"- Team 1 counter-press windows: {summary.get('team_1_counter_press_windows', 0)}",
                f"- Team 2 counter-press windows: {summary.get('team_2_counter_press_windows', 0)}",
            ]

        # Phase-of-play breakdown (Rein & Memmert 2016)
        phase_lines = []
        for team in (1, 2):
            for phase, label in [
                ("ip", "In possession"),
                ("oop", "Out of possession"),
                ("dat", "Transition attack"),
                ("adt", "Transition defense"),
            ]:
                n = summary.get(f"team_{team}_{phase}_window_count")
                if n is None or n == 0:
                    continue
                comp = summary.get(f"team_{team}_{phase}_compactness_m2")
                press = summary.get(f"team_{team}_{phase}_pressing_intensity")
                terr = summary.get(f"team_{team}_{phase}_territory_pct")
                phase_lines.append(
                    f"- Team {team} / {label} ({n} windows): "
                    f"compactness={_fmt(comp)}m², pressing={_fmt(press)}, territory={_pct(terr)}"
                )

        if phase_lines:
            lines += ["", "### Phase-of-Play Breakdown (Rein & Memmert 2016)"] + phase_lines

        # Interpretive notes
        t1_comp = summary.get("team_1_avg_compactness_m2")
        t2_comp = summary.get("team_2_avg_compactness_m2")
        t1_press = summary.get("team_1_avg_pressing_intensity")
        t2_press = summary.get("team_2_avg_pressing_intensity")
        ppda_1 = summary.get("ppda_team_1")
        ppda_2 = summary.get("ppda_team_2")
        t1_terr = summary.get("team_1_avg_territory_pct")
        t2_terr = summary.get("team_2_avg_territory_pct")

        notes = []
        if t1_comp is not None and t2_comp is not None:
            more_compact = "Team 1" if t1_comp < t2_comp else "Team 2"
            notes.append(
                f"{more_compact} maintained a more compact shape ({min(t1_comp, t2_comp):.0f}m² vs {max(t1_comp, t2_comp):.0f}m²)"
            )
        if t1_press is not None and t2_press is not None:
            higher_press = "Team 1" if t1_press > t2_press else "Team 2"
            notes.append(
                f"{higher_press} showed higher pressing intensity ({max(t1_press, t2_press):.2f} vs {min(t1_press, t2_press):.2f})"
            )
        if ppda_1 is not None and ppda_2 is not None:
            more_agg = "Team 1" if ppda_1 < ppda_2 else "Team 2"
            notes.append(
                f"{more_agg} pressed more aggressively (PPDA {min(ppda_1, ppda_2):.1f} vs {max(ppda_1, ppda_2):.1f})"
            )
        if t1_terr is not None and t2_terr is not None:
            terr_leader = "Team 1" if t1_terr > t2_terr else "Team 2"
            notes.append(
                f"{terr_leader} controlled more pitch territory ({max(t1_terr, t2_terr) * 100:.1f}% vs {min(t1_terr, t2_terr) * 100:.1f}%)"
            )

        if notes:
            lines.append("")
            lines.append("**Key tactical observations:**")
            for note in notes:
                lines.append(f"- {note}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_team_kinematics(analytics: dict) -> str:
        players = analytics.get("player_kinematics", {})
        if not players:
            return ""

        # Group by team
        team_1 = []
        team_2 = []
        unassigned = []

        for track_id, stats in players.items():
            tid = stats.get("team_id")
            entry = {
                "track_id": track_id,
                "entity_type": stats.get("entity_type", "player"),
                "distance_m": stats.get("total_distance_m"),
                "distance_px": stats.get("total_distance_px", 0),
                "avg_speed": stats.get("avg_speed_m_per_sec"),
                "max_speed": stats.get("max_speed_m_per_sec"),
                "sprint_count": stats.get("sprint_count"),
                "high_intensity_distance_m": stats.get("high_intensity_distance_m"),
                "speed_zones": stats.get("distance_by_speed_zone_m"),
            }
            if tid == 1:
                team_1.append(entry)
            elif tid == 2:
                team_2.append(entry)
            else:
                unassigned.append(entry)

        lines = ["## Player Performance"]

        for team_label, team_data in [("Team 1", team_1), ("Team 2", team_2)]:
            if not team_data:
                continue
            # Sort by distance (descending)
            team_data.sort(
                key=lambda x: x["distance_m"] if x["distance_m"] is not None else 0,
                reverse=True,
            )

            lines.append(f"\n### {team_label}")
            lines.append(
                "| Player | Role | Distance | Avg Speed | Max Speed | Sprints | HI Distance |"
            )
            lines.append(
                "|--------|------|----------|-----------|-----------|---------|-------------|"
            )

            for p in team_data:
                role = "GK" if p["entity_type"] == "goalkeeper" else "Player"
                dist = (
                    f"{p['distance_m']:.0f}m"
                    if p["distance_m"] is not None
                    else f"{p['distance_px']:.0f}px"
                )
                avg = f"{p['avg_speed'] * 3.6:.1f} km/h" if p["avg_speed"] is not None else "—"
                mx = f"{p['max_speed'] * 3.6:.1f} km/h" if p["max_speed"] is not None else "—"
                sprints = (
                    str(p.get("sprint_count", "—")) if p.get("sprint_count") is not None else "—"
                )
                hi_dist = (
                    f"{p['high_intensity_distance_m']:.0f}m"
                    if p.get("high_intensity_distance_m") is not None
                    else "—"
                )
                lines.append(
                    f"| #{p['track_id']} | {role} | {dist} | {avg} | {mx} | {sprints} | {hi_dist} |"
                )

            # Speed zone summary for the team
            team_zones: dict = {}
            for p in team_data:
                for zone, zone_dist in (p.get("speed_zones") or {}).items():
                    team_zones[zone] = team_zones.get(zone, 0.0) + zone_dist
            if team_zones:
                lines.append(
                    f"\n**{team_label} speed zone totals (m):** "
                    + ", ".join(
                        f"{z.replace('_', ' ').title()}: {team_zones.get(z, 0):.0f}m"
                        for z in (
                            "walking",
                            "jogging",
                            "running",
                            "high_speed_running",
                            "sprinting",
                        )
                        if z in team_zones
                    )
                )

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_ball_stats(analytics: dict) -> str:
        ball = analytics.get("ball_kinematics")
        if not ball:
            return ""

        lines = ["## Ball Statistics"]

        dist_m = ball.get("total_distance_m")
        if dist_m is not None:
            lines.append(f"- Total distance: {dist_m:.0f}m")
        else:
            lines.append(f"- Total distance: {ball.get('total_distance_px', 0):.0f}px")

        avg = ball.get("avg_speed_m_per_sec")
        if avg is not None:
            lines.append(f"- Average speed: {avg * 3.6:.1f} km/h")

        mx = ball.get("max_speed_m_per_sec")
        if mx is not None:
            lines.append(f"- Max speed: {mx * 3.6:.1f} km/h")

        bp = analytics.get("ball_path", {})
        dc = bp.get("direction_changes", 0)
        lines.append(f"- Direction changes: {dc}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_events(analytics: dict) -> str:
        events = analytics.get("events", [])
        if not events:
            return ""

        lines = [
            "## Events Timeline",
            "",
        ]

        for ev in events:
            ts = ev.get("timestamp_sec", 0)
            mins = int(ts // 60)
            secs = int(ts % 60)
            time_str = f"{mins}:{secs:02d}"

            etype = ev.get("event_type", "unknown")
            team = f"Team {ev.get('team_id', '?')}"
            player = f"#{ev.get('player_track_id', '?')}"
            conf = ev.get("confidence")
            conf_str = f" [confidence: {conf:.2f}]" if conf is not None else ""

            if etype == "pass":
                target = f"#{ev.get('target_player_track_id', '?')}"
                success = "✓" if ev.get("success") else "✗"
                lines.append(
                    f"- {time_str} — **Pass** {success}: {player} → {target} ({team}){conf_str}"
                )
            elif etype == "shot":
                lines.append(f"- {time_str} — **Shot**: {player} ({team}){conf_str}")
            elif etype == "challenge":
                target = ev.get("target_player_track_id")
                target_str = f" won from #{target}" if target else ""
                lines.append(
                    f"- {time_str} — **Challenge**: {player} ({team}){target_str}{conf_str}"
                )
            elif etype == "play":
                # ML-detected open-play moment (no team/player identity)
                lines.append(f"- {time_str} — **Open Play** (ML-detected){conf_str}")
            elif etype == "throwin":
                # ML-detected throw-in (no team/player identity)
                lines.append(f"- {time_str} — **Throw-in** (ML-detected){conf_str}")
            else:
                lines.append(
                    f"- {time_str} — **{etype.capitalize()}**: {player} ({team}){conf_str}"
                )

        # Summary counts
        counts: dict[str, dict] = {}
        prog_counts: dict[int, int] = {1: 0, 2: 0}
        total_passes: dict[int, int] = {1: 0, 2: 0}
        for ev in events:
            etype = ev.get("event_type", "unknown")
            tid = ev.get("team_id", 0)
            if etype not in counts:
                counts[etype] = {}
            counts[etype][tid] = counts[etype].get(tid, 0) + 1
            if etype == "pass" and tid in (1, 2):
                total_passes[tid] += 1
                if ev.get("is_progressive"):
                    prog_counts[tid] += 1

        lines.append("")
        lines.append("### Event Summary")
        for etype, team_counts in sorted(counts.items()):
            parts = [
                f"Team {t}: {c}"
                for t, c in sorted(
                    ((k, v) for k, v in team_counts.items() if k is not None),
                    key=lambda x: x[0],
                )
            ]
            lines.append(f"- {etype.capitalize()}: {', '.join(parts)}")

        # Progressive passes (A3)
        if any(prog_counts.values()):
            lines.append("")
            lines.append("### Progressive Passes (≥10m toward opponent goal)")
            for tid in (1, 2):
                total = total_passes.get(tid, 0)
                prog = prog_counts.get(tid, 0)
                pct = f" ({prog / total * 100:.1f}% of passes)" if total > 0 else ""
                lines.append(f"- Team {tid}: {prog} progressive passes{pct}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_expected_threat(analytics: dict) -> str:
        """Compute and format Expected Threat (xT) from pass events (Shaw & Sudarshan 2020)."""
        events = analytics.get("events", [])
        if not events:
            return ""

        # Compute xT inline using the pre-computed 12×8 lookup matrix
        _XT = [
            [0.008, 0.008, 0.009, 0.010, 0.010, 0.011, 0.011, 0.011, 0.012, 0.013, 0.016, 0.026],
            [0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.020, 0.026, 0.041],
            [0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.022, 0.028, 0.039, 0.072],
            [0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.021, 0.026, 0.036, 0.057, 0.112],
            [0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.021, 0.026, 0.036, 0.057, 0.112],
            [0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.022, 0.028, 0.039, 0.072],
            [0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.020, 0.026, 0.041],
            [0.008, 0.008, 0.009, 0.010, 0.010, 0.011, 0.011, 0.011, 0.012, 0.013, 0.016, 0.026],
        ]

        def _xt_val(x_cm, y_cm):
            col = min(11, max(0, int(x_cm / 10500.0 * 12)))
            row = min(7, max(0, int(y_cm / 6800.0 * 8)))
            return _XT[row][col]

        team_xt: dict = {1: 0.0, 2: 0.0}
        team_prog_xt: dict = {1: 0.0, 2: 0.0}
        team_passes: dict = {1: 0, 2: 0}

        for ev in events:
            etype = ev.get("event_type", "")
            if etype != "pass":
                continue
            tid = ev.get("team_id")
            if tid not in (1, 2):
                continue
            ps = ev.get("pitch_start")
            pe = ev.get("pitch_end")
            if not ps or not pe:
                continue

            xt_gain = _xt_val(pe[0], pe[1]) - _xt_val(ps[0], ps[1])
            team_xt[tid] += xt_gain
            team_passes[tid] += 1
            if ev.get("is_progressive"):
                team_prog_xt[tid] += xt_gain

        if not any(team_passes.values()):
            return ""

        lines = [
            "## Expected Threat (xT) — Ball Progression Value (Shaw & Sudarshan 2020)",
            "",
            "| Metric | Team 1 | Team 2 |",
            "|--------|--------|--------|",
        ]
        for label, key in [
            ("Total xT added", "total"),
            ("xT per pass", "per"),
            ("Progressive pass xT", "prog"),
        ]:
            if key == "total":
                v1 = f"{team_xt[1]:.3f}"
                v2 = f"{team_xt[2]:.3f}"
            elif key == "per":
                v1 = f"{team_xt[1] / team_passes[1]:.4f}" if team_passes[1] else "—"
                v2 = f"{team_xt[2] / team_passes[2]:.4f}" if team_passes[2] else "—"
            else:
                v1 = f"{team_prog_xt[1]:.3f}"
                v2 = f"{team_prog_xt[2]:.3f}"
            lines.append(f"| {label} | {v1} | {v2} |")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_pass_networks(analytics: dict) -> str:
        ig1 = analytics.get("interaction_graph_team1")
        ig2 = analytics.get("interaction_graph_team2")

        if not ig1 and not ig2:
            return ""

        lines = ["## Pass Networks"]

        for team_label, ig in [("Team 1", ig1), ("Team 2", ig2)]:
            if not ig:
                continue

            edges = ig.get("edges", [])
            if not edges:
                continue

            lines.append(f"\n### {team_label}")

            # Sort edges by weight (most passes first)
            sorted_edges = sorted(edges, key=lambda e: e.get("weight", 0), reverse=True)
            top_edges = sorted_edges[:10]  # Top 10 connections

            for edge in top_edges:
                src = f"#{edge.get('source', '?')}"
                tgt = f"#{edge.get('target', '?')}"
                w = edge.get("weight", 0)
                if isinstance(w, float):
                    lines.append(f"- {src} ↔ {tgt}: {w:.1f} interactions")
                else:
                    lines.append(f"- {src} ↔ {tgt}: {w} interactions")

        return "\n".join(lines) + "\n"


def digit_space_format(value: float, precision: int = 1) -> str:
    """Format a number with spaces between digits to prevent BPE fragmentation.

    Converts e.g. 534.2 → '5 3 4 , 2' per Gruver et al. (2023) and
    Schumacher et al. (2026) §3.2. Decimal point is represented as ' , '.

    Args:
        value: Numeric value to format.
        precision: Decimal places (default 1). Use 0 for integers.

    Returns:
        Digit-spaced string representation.
    """
    formatted = f"{value:.{precision}f}"
    parts = formatted.split(".")
    integer_part = " ".join(parts[0].lstrip("-"))
    if parts[0].startswith("-"):
        integer_part = "- " + integer_part
    if precision == 0 or len(parts) == 1:
        return integer_part
    decimal_part = " ".join(parts[1])
    return f"{integer_part} , {decimal_part}"


# ── Per-Frame Context Formatter ───────────────────────────────────────────────


class PerFrameContextFormatter:
    """Condenses per-frame DB ground truth into a prompt-friendly spatial evidence section.

    Takes the output of db_extractor.extract_analysis_data() (a dict containing
    frame_metrics, formations, events_db) and produces a '## Per-Frame Spatial Evidence'
    markdown section suitable for appending to GroundingFormatter output.

    Token budget: targets ~2,500 tokens (~10,000 chars). Trims greedily if exceeded.
    """

    _MAX_CHARS = 10_000  # ~2,500 tokens
    _FPS = 25.0

    @classmethod
    def format(
        cls, db_ground_truth: dict, with_insights: bool = False, digit_space: bool = False
    ) -> str:
        """Format per-frame data into a condensed markdown section.

        Args:
            db_ground_truth: Dict produced by db_extractor (keys: frame_metrics,
                formations, events_db, analytics, per_frame).
            with_insights: If True, prepend PerFrameInsights wordalised findings
                before the raw evidence tables (PERFRAME_V2 condition).
            digit_space: If True, numeric values in time-series tables are
                formatted with digit-space encoding (Gruver et al. 2023) to
                prevent BPE fragmentation. Default False preserves existing format.

        Returns:
            Markdown string with six sub-sections. Empty string if no data.
        """
        fm = db_ground_truth.get("frame_metrics", {})
        if not fm:
            return ""

        fps = db_ground_truth.get("analytics", {}).get("fps") or cls._FPS

        sections = []
        if with_insights:
            insights_text = PerFrameInsights.format(db_ground_truth)
            if insights_text:
                sections.append(insights_text)

        sections += [
            "## Per-Frame Spatial Evidence\n"
            "_Derived from per-frame tracking records. Use this data to ground spatial "
            "and temporal claims that cannot be confirmed from aggregate statistics._\n",
            cls._zone_occupancy(fm),
            cls._centroid_progression(fm, fps, digit_space=digit_space),
            cls._possession_phases(fm, fps),
            cls._compactness_series(fm, fps, digit_space=digit_space),
            cls._formation_estimates(db_ground_truth),
            cls._event_context(db_ground_truth, fm),
        ]

        result = "\n".join(s for s in sections if s)

        # Token budget guard: trim greedily from the end if too large
        if len(result) > cls._MAX_CHARS:
            result = result[: cls._MAX_CHARS]
            result = (
                result[: result.rfind("\n")]
                + "\n\n_[Per-frame context truncated to stay within token budget]_"
            )

        return result

    @classmethod
    def _zone_occupancy(cls, fm: dict) -> str:
        occ = fm.get("pitch_zone_occupancy", {})
        if not occ:
            return ""
        t1 = occ.get("team_1", {})
        t2 = occ.get("team_2", {})
        lines = [
            "### Zone Occupancy (all frames)",
            "| Team | Defensive Third | Middle Third | Attacking Third |",
            "|------|----------------|-------------|-----------------|",
        ]
        for label, d in [("Team 1", t1), ("Team 2", t2)]:
            def_ = d.get("defensive", 0) * 100
            mid = d.get("middle", 0) * 100
            att = d.get("attacking", 0) * 100
            lines.append(f"| {label} | {def_:.1f}% | {mid:.1f}% | {att:.1f}% |")
        lines.append(
            "\n_Interpretation: >50% defensive third = deep block; "
            "<10% attacking third = limited forward presence._"
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def _centroid_progression(cls, fm: dict, fps: float, digit_space: bool = False) -> str:
        centroids = fm.get("team_centroids", [])
        distances = fm.get("inter_team_distance_m", [])
        if not centroids:
            return ""

        dist_by_frame = {d["frame"]: d.get("distance_m") for d in distances} if distances else {}

        total = len(centroids)
        step = max(1, total // 10)
        sampled = centroids[::step][:10]

        def _fmt(v: float) -> str:
            return digit_space_format(v, 1) if digit_space else f"{v:.1f}"

        lines = [
            "### Team Centroid Progression (sampled every ~3s)",
            "| Time (s) | T1 Centroid (x,y) m | T2 Centroid (x,y) m | Inter-Team Dist |",
            "|---------|---------------------|---------------------|-----------------|",
        ]
        for entry in sampled:
            frame = entry["frame"]
            t_sec = frame / fps
            t1 = entry.get("team_1")
            t2 = entry.get("team_2")
            t1_str = f"({_fmt(t1[0])}, {_fmt(t1[1])})" if t1 else "—"
            t2_str = f"({_fmt(t2[0])}, {_fmt(t2[1])})" if t2 else "—"
            dist = dist_by_frame.get(frame)
            dist_str = f"{_fmt(dist)}m" if dist is not None else "—"
            lines.append(f"| {_fmt(t_sec)}s | {t1_str} | {t2_str} | {dist_str} |")
        lines.append(
            "\n_x=0 = own goal line, x=105m = opponent goal line (approx). y=0/68m = touchlines._"
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def _possession_phases(cls, fm: dict, fps: float) -> str:
        seq = fm.get("possession_sequence", [])
        if len(seq) < 2:
            return ""

        centroids = fm.get("team_centroids", [])
        centroid_by_frame = {e["frame"]: e for e in centroids}

        phases = []
        for i in range(len(seq) - 1):
            start_frame = seq[i]["frame"]
            end_frame = seq[i + 1]["frame"]
            duration = end_frame - start_frame
            if duration < 25:  # < 1 second at 25 FPS — too short to be meaningful
                continue
            team = seq[i]["team"]
            phases.append(
                {
                    "team": team,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration_frames": duration,
                    "duration_s": duration / fps,
                }
            )

        if not phases:
            return "### Possession Phases\n_No possession phases ≥1 second detected — high-turnover passage of play._\n"

        phases.sort(key=lambda p: p["duration_frames"], reverse=True)
        top_phases = phases[:8]
        top_phases.sort(key=lambda p: p["start_frame"])

        lines = [
            f"### Possession Phases (≥1s, {len(phases)} qualifying phases)",
            "| Team | Start | End | Duration | T1 Centroid Start | T2 Centroid Start |",
            "|------|-------|-----|----------|-------------------|-------------------|",
        ]
        for ph in top_phases:
            t_start = ph["start_frame"] / fps
            t_end = ph["end_frame"] / fps
            c_start = centroid_by_frame.get(ph["start_frame"], {})
            t1_s = c_start.get("team_1")
            t2_s = c_start.get("team_2")
            t1_str = f"({t1_s[0]:.1f}, {t1_s[1]:.1f})" if t1_s else "—"
            t2_str = f"({t2_s[0]:.1f}, {t2_s[1]:.1f})" if t2_s else "—"
            lines.append(
                f"| Team {ph['team']} | {t_start:.1f}s | {t_end:.1f}s | "
                f"{ph['duration_s']:.1f}s | {t1_str} | {t2_str} |"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def _compactness_series(cls, fm: dict, fps: float, digit_space: bool = False) -> str:
        comp = fm.get("compactness_m2", [])
        if not comp:
            return ""

        step = max(1, len(comp) // 10)
        sampled = comp[::step][:10]

        def _fmt_int(v: float) -> str:
            return digit_space_format(v, 0) if digit_space else f"{v:.0f}"

        def _fmt1(v: float) -> str:
            return digit_space_format(v, 1) if digit_space else f"{v:.1f}"

        lines = [
            "### Compactness Time Series (sampled every ~3s)",
            "| Time (s) | T1 Compactness (m²) | T2 Compactness (m²) |",
            "|---------|--------------------|--------------------|",
        ]
        for entry in sampled:
            frame = entry["frame"]
            t_sec = frame / fps
            t1_c = entry.get("team_1_m2")
            t2_c = entry.get("team_2_m2")
            t1_str = _fmt_int(t1_c) if t1_c is not None else "—"
            t2_str = _fmt_int(t2_c) if t2_c is not None else "—"
            lines.append(f"| {_fmt1(t_sec)}s | {t1_str} | {t2_str} |")

        t1_vals = [e["team_1_m2"] for e in comp if e.get("team_1_m2") is not None]
        t2_vals = [e["team_2_m2"] for e in comp if e.get("team_2_m2") is not None]
        if t1_vals and t2_vals:
            lines.append(
                f"\n_T1 compactness range: {min(t1_vals):.0f}–{max(t1_vals):.0f}m² "
                f"(mean {sum(t1_vals) / len(t1_vals):.0f}m²). "
                f"T2: {min(t2_vals):.0f}–{max(t2_vals):.0f}m² "
                f"(mean {sum(t2_vals) / len(t2_vals):.0f}m²). "
                "Typical compact block ~400m², open play ~800m²._"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def _formation_estimates(cls, db_ground_truth: dict) -> str:
        formations = db_ground_truth.get("formations", {})
        if not formations:
            return ""

        lines = ["### Formation Estimates (k-means on time-averaged positions)"]

        for team_key, label in [("team_1", "Team 1"), ("team_2", "Team 2")]:
            t = formations.get(team_key, {})
            if not t:
                continue
            formation = t.get("formation", "unknown")
            conf = t.get("confidence", 0)
            conf_note = (
                "low confidence — treat as plausible characterisation"
                if conf < 0.3
                else "moderate confidence"
            )
            lines.append(f"\n**{label}**: {formation} (confidence {conf:.2f} — {conf_note})")
            avg_pos = t.get("avg_positions", {})
            if avg_pos:
                lines.append("Player average positions (x=pitch length, y=pitch width, metres):")
                pos_parts = [
                    f"#{pid}: ({xy[0]:.1f}, {xy[1]:.1f})" for pid, xy in list(avg_pos.items())[:8]
                ]
                lines.append("  " + " | ".join(pos_parts))

        temporal = formations.get("temporal", [])
        if temporal:
            lines.append(
                f"\n**Temporal formation windows** ({len(temporal)} windows of ~{150 / 25:.0f}s each):"
            )
            for i, win in enumerate(temporal[:4]):
                # Key names vary between db_extractor versions
                t1_f = win.get("team_1") or win.get("team_1_formation") or "—"
                t2_f = win.get("team_2") or win.get("team_2_formation") or "—"
                start_frame = win.get("window_start") or win.get("start_frame", i * 150)
                start_s = start_frame / 25.0
                lines.append(f"  Window {i + 1} ({start_s:.0f}s): T1={t1_f} T2={t2_f}")

        return "\n".join(lines) + "\n"

    @classmethod
    def _event_context(cls, db_ground_truth: dict, fm: dict) -> str:
        events = db_ground_truth.get("events_db", [])
        if not events:
            return ""

        centroids = fm.get("team_centroids", [])
        centroid_by_frame = {e["frame"]: e for e in centroids}

        # Priority: shots > challenges > passes
        priority = {"shot": 0, "challenge": 1, "pass": 2}
        sorted_events = sorted(
            events,
            key=lambda e: (priority.get(e.get("type", ""), 99), -e.get("confidence", 0)),
        )
        top = [e for e in sorted_events if e.get("startX") is not None][:5]

        if not top:
            return ""

        lines = ["### Event Spatial Context (top 5 by type priority)"]
        for ev in top:
            frame = ev.get("frameNumber", 0)
            ts = ev.get("timestamp", frame / 25.0)
            etype = ev.get("type", "unknown")
            player = ev.get("playerId", "?")
            target = ev.get("targetPlayerId")
            sx = (ev.get("startX") or 0) / 100.0  # pixel/cm → m
            sy = (ev.get("startY") or 0) / 100.0
            raw_ex = ev.get("endX")
            raw_ey = ev.get("endY")
            ex = raw_ex / 100.0 if raw_ex is not None else sx
            ey = raw_ey / 100.0 if raw_ey is not None else sy

            centroid = centroid_by_frame.get(frame, {})
            t1_c = centroid.get("team_1")
            t2_c = centroid.get("team_2")
            centroid_str = ""
            if t1_c and t2_c:
                centroid_str = f" | T1 centroid ({t1_c[0]:.1f}, {t1_c[1]:.1f}), T2 centroid ({t2_c[0]:.1f}, {t2_c[1]:.1f})"

            if target:
                lines.append(
                    f"- {ts:.1f}s — **{etype.capitalize()}** #{player}→#{target}: "
                    f"start ({sx:.1f}, {sy:.1f})m → end ({ex:.1f}, {ey:.1f})m{centroid_str}"
                )
            else:
                lines.append(
                    f"- {ts:.1f}s — **{etype.capitalize()}** #{player}: "
                    f"position ({sx:.1f}, {sy:.1f})m{centroid_str}"
                )

        return "\n".join(lines) + "\n"


# ── Per-Frame Insights (Wordalisation Layer) ───────────────────────────────────


class PerFrameInsights:
    """Wordalised per-frame tactical insights (Sumpter Wordalisation approach).

    Converts per-frame time series data into analyst prose *before* LLM consumption.
    Following Context Engineering principles: 'without context, data doesn't mean
    anything'. Each sub-finding reads like a football analyst's notes, not a
    statistical summary. Uses football vocabulary, not stats jargon.

    Produces ~400 tokens of domain-contextualised narrative for PERFRAME_V2.
    """

    _FPS = 25.0

    @classmethod
    def format(cls, db_ground_truth: dict) -> str:
        """Generate wordalised per-frame insights section."""
        fm = db_ground_truth.get("frame_metrics", {})
        analytics = db_ground_truth.get("analytics", {})
        if not fm:
            return ""

        fps = analytics.get("fps") or cls._FPS
        findings = [
            cls._centroid_trend(fm, fps),
            cls._compactness_dynamics(fm, fps),
            cls._phase_density(fm, fps),
            cls._zone_paradox(fm, analytics),
            cls._formation_stability(db_ground_truth),
            cls._event_clustering(db_ground_truth, fps),
        ]
        valid = [f for f in findings if f]
        if not valid:
            return ""

        lines = [
            "## Per-Frame Tactical Insights",
            "_Pre-interpreted from tracking records using Context Engineering principles "
            "(Sumpter 2025). Read these findings first, then cross-reference the raw "
            "evidence tables below._",
            "",
        ]
        for i, finding in enumerate(valid, 1):
            lines.append(f"**F{i}.** {finding}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def _centroid_trend(cls, fm: dict, fps: float) -> str:
        centroids = fm.get("team_centroids", [])
        if len(centroids) < 4:
            return ""
        first = centroids[0]
        last = centroids[-1]
        duration = (last["frame"] - first["frame"]) / fps

        for team, label in [("team_2", "Team 2"), ("team_1", "Team 1")]:
            start = first.get(team)
            end = last.get(team)
            if not start or not end:
                continue
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = (dx**2 + dy**2) ** 0.5
            rate = dist / duration if duration > 0 else 0
            if abs(dx) > 5:
                direction = "pushed upfield" if dx > 0 else "dropped deeper"
            elif abs(dy) > 5:
                direction = (
                    "shifted to the right channel" if dy < 0 else "shifted to the left channel"
                )
            else:
                direction = "held their spatial position"
            note = f"{label} {direction} by {dist:.1f}m over the clip (drift rate {rate:.2f} m/s)"
            if abs(dx) > 8:
                note += (
                    " — the progressive forward shift is consistent with a sustained press"
                    if dx > 0
                    else " — the rearward drift suggests retreat under sustained pressure"
                )
            return note
        return ""

    @classmethod
    def _compactness_dynamics(cls, fm: dict, fps: float) -> str:
        comp = fm.get("compactness_m2", [])
        if len(comp) < 4:
            return ""
        t1_series = [
            (e["frame"] / fps, e["team_1_m2"]) for e in comp if e.get("team_1_m2") is not None
        ]
        if len(t1_series) < 4:
            return ""
        vals = [v for _, v in t1_series]
        mean_v = sum(vals) / len(vals)
        std_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
        cv = std_v / mean_v if mean_v > 0 else 0
        anomalies = [(t, v) for t, v in t1_series if v > mean_v + 2 * std_v]
        n = len(t1_series)
        times = [t for t, _ in t1_series]
        t_mean = sum(times) / n
        num = sum((t - t_mean) * (v - mean_v) for t, v in t1_series)
        den = sum((t - t_mean) ** 2 for t in times)
        slope = num / den if den > 0 else 0
        shape = "compact" if mean_v < 500 else ("moderate" if mean_v < 900 else "expansive")
        finding = f"Team 1's defensive block averaged {mean_v:.0f}m² ({shape}), coefficient of variation {cv:.2f}"
        if anomalies:
            t_a, v_a = anomalies[0]
            finding += (
                f" — shape fractured at ~{t_a:.0f}s ({v_a:.0f}m², >{mean_v + 2 * std_v:.0f}m² threshold),"
                " a hallmark of a defensive unit stretched by quick ball circulation"
            )
        if slope > 50:
            finding += "; the block steadily expanded as the passage progressed"
        elif slope < -50:
            finding += (
                "; the block progressively tightened — disciplined shape maintenance under pressure"
            )
        return finding

    @classmethod
    def _phase_density(cls, fm: dict, fps: float) -> str:
        seq = fm.get("possession_sequence", [])
        if len(seq) < 2:
            return ""
        durations = [(seq[i + 1]["frame"] - seq[i]["frame"]) / fps for i in range(len(seq) - 1)]
        n = len(durations)
        mean_d = sum(durations) / n
        sustained = [d for d in durations if d >= 1.0]
        longest = max(durations)
        finding = f"{n} possession transitions, mean interval {mean_d:.1f}s"
        if sustained:
            finding += f" — {len(sustained)} sustained phases (≥1s), longest {longest:.1f}s."
            if longest > 8:
                finding += (
                    " The extended build-up phase is consistent with a side patiently"
                    " circulating the ball to pull the defensive block apart layer by layer."
                )
            elif longest > 4:
                finding += " Moderate build-up cycles suggest deliberate shape manipulation."
            else:
                finding += " Short phase lengths indicate a high-turnover, scrappy passage."
        else:
            finding += " — all transitions ≤1s, a rapid high-turnover exchange with no side establishing control."
        return finding

    @classmethod
    def _zone_paradox(cls, fm: dict, analytics: dict) -> str:
        occ = fm.get("pitch_zone_occupancy", {})
        poss_data = analytics.get("possession", {})
        if not occ or not poss_data:
            return ""
        t1_poss = poss_data.get("team_1_percentage", 50)
        t2_poss = poss_data.get("team_2_percentage", 50)
        t2_occ = occ.get("team_2", {})
        t2_def = t2_occ.get("defensive", 0) * 100
        t2_att = t2_occ.get("attacking", 0) * 100
        t1_occ = occ.get("team_1", {})
        t1_def = t1_occ.get("defensive", 0) * 100

        if t2_poss > 60 and t2_def > 40:
            return (
                f"Zone occupancy paradox: despite {t2_poss:.0f}% possession, Team 2 stationed "
                f"{t2_def:.1f}% of their player-frames in their own defensive third — "
                "a deep-sitting possession side recycling laterally rather than committing numbers forward."
            )
        if t2_poss > 60 and t2_att < 10:
            return (
                f"Team 2's {t2_poss:.0f}% possession is striking against only {t2_att:.1f}% "
                "of player-frames in the attacking third — possession-dominant but not penetrating high."
            )
        if t1_poss < 30 and t1_def > 40:
            return (
                f"Team 1 held only {t1_poss:.0f}% possession while spending {t1_def:.1f}% of "
                "player-frames in their own defensive third — a textbook low-block defensive posture."
            )
        return ""

    @classmethod
    def _formation_stability(cls, db_ground_truth: dict) -> str:
        formations = db_ground_truth.get("formations", {})
        temporal = formations.get("temporal", [])
        if len(temporal) < 2:
            return ""
        t1_labels = {w.get("team_1") or w.get("team_1_formation") for w in temporal} - {None, ""}
        t2_labels = {w.get("team_2") or w.get("team_2_formation") for w in temporal} - {None, ""}
        n1, n2 = len(t1_labels), len(t2_labels)
        avg = (n1 + n2) / 2
        if avg <= 1.5:
            stability, note = "rigid", "a fixed tactical shape throughout"
        elif avg <= 2.5:
            stability, note = "semi-fluid", "minor positional adjustments between phases"
        else:
            stability, note = (
                "highly fluid",
                "a shape adapting to in/out-of-possession phases rather than a rigid blueprint",
            )
        window_str = " → ".join(
            f"T1:{w.get('team_1') or w.get('team_1_formation', '?')}"
            f"/T2:{w.get('team_2') or w.get('team_2_formation', '?')}"
            for w in temporal[:4]
        )
        return (
            f"Formation stability: {len(temporal)} temporal windows yield {n1} distinct T1 shapes "
            f"and {n2} T2 shapes — {stability} ({note}). Sequence: [{window_str}]."
        )

    @classmethod
    def _event_clustering(cls, db_ground_truth: dict, fps: float) -> str:
        events = db_ground_truth.get("events_db", [])
        events_pos = [e for e in events if e.get("startX") is not None]
        if not events_pos:
            return (
                f"{len(events)} events detected, no spatial coordinates available."
                if events
                else ""
            )
        xs = [e["startX"] / 100.0 for e in events_pos]
        ys = [e["startY"] / 100.0 for e in events_pos]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)
        windows: dict[int, int] = {}
        for ev in events_pos:
            t = ev.get("timestamp", ev.get("frameNumber", 0) / fps)
            windows[int(t // 5)] = windows.get(int(t // 5), 0) + 1
        peak_bucket = max(windows, key=lambda b: windows[b])
        peak_count = windows[peak_bucket]
        zone = "own half" if x_mean < 52.5 else "opposition half"
        side = (
            "left flank"
            if y_mean < 22.7
            else "right flank"
            if y_mean > 45.3
            else "central corridor"
        )
        finding = (
            f"{len(events_pos)} spatially-located events, activity centred in the "
            f"{zone}/{side} ({x_mean:.1f}m, {y_mean:.1f}m). "
            f"Peak intensity in the {peak_bucket * 5:.0f}–{peak_bucket * 5 + 5:.0f}s window "
            f"({peak_count} events"
        )
        if x_spread < 20 and y_spread < 15:
            finding += ", tightly clustered — a localised battle for a specific corridor)."
        else:
            finding += ", dispersed — a fluid, transitional contest across multiple zones)."
        return finding


class VisualTimeSeriesRenderer:
    """Renders per-frame time-series data as matplotlib charts (JPEG bytes).

    Implements the visual representation approach from Schumacher et al. (2026)
    'Prompting Underestimates LLM Capability for Time Series Classification'.
    Visual charts prevent BPE-level numeric misquoting (e.g. 1122.9m² instead
    of true mean 534m²) and reduce prompt instability across variants.

    All methods return None silently if the required data is absent, so callers
    can filter with [x for x in render_all(gt) if x].
    """

    _FPS = 25.0
    _PITCH_LEN = 105.0
    _PITCH_WID = 68.0

    @classmethod
    def render_all(cls, db_ground_truth: dict) -> "list[bytes]":
        """Render all available charts and return as JPEG byte arrays.

        Returns up to 4 charts: compactness, centroid trajectory, pressing
        dashboard, and a 2×2 combined overview. Charts are omitted (not
        added to list) when the underlying data is absent.

        Args:
            db_ground_truth: Dict from db_extractor.extract_analysis_data().

        Returns:
            List of JPEG byte arrays (0–4 items).
        """
        fm = db_ground_truth.get("frame_metrics", {})
        fps = db_ground_truth.get("analytics", {}).get("fps") or cls._FPS
        images = []
        for fn in (
            cls._render_compactness,
            cls._render_centroid_trajectory,
            cls._render_pressing_dashboard,
            cls._render_combined,
        ):
            try:
                img = fn(fm, fps)
                if img:
                    images.append(img)
            except Exception:
                pass
        return images

    @classmethod
    def render_named(cls, chart_name: str, fm: dict, fps: float) -> "bytes | None":
        """Render a single named chart and return JPEG bytes (or None).

        Public dispatcher over the private _render_* methods for use in
        per-analysis-type routing (FINDINGS_INFORMED condition).

        Args:
            chart_name: One of "compactness", "centroid", "pressing", "combined".
            fm: frame_metrics dict from db_extractor output.
            fps: Frames per second.

        Returns:
            JPEG bytes or None if data absent or chart_name unrecognised.
        """
        _dispatch = {
            "compactness": cls._render_compactness,
            "centroid": cls._render_centroid_trajectory,
            "pressing": cls._render_pressing_dashboard,
            "combined": cls._render_combined,
        }
        fn = _dispatch.get(chart_name)
        if fn is None:
            return None
        try:
            return fn(fm, fps)
        except Exception:
            return None

    @classmethod
    def _render_compactness(cls, fm: dict, fps: float) -> "bytes | None":
        """Dual-line compactness plot with mean/std bands and anomaly markers."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        comp = fm.get("compactness_m2", [])
        if not comp:
            return None

        t1_times = [e["frame"] / fps for e in comp if e.get("team_1_m2") is not None]
        t1_vals = [e["team_1_m2"] for e in comp if e.get("team_1_m2") is not None]
        t2_times = [e["frame"] / fps for e in comp if e.get("team_2_m2") is not None]
        t2_vals = [e["team_2_m2"] for e in comp if e.get("team_2_m2") is not None]

        if not t1_vals:
            return None

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t1_times, t1_vals, color="#1f77b4", linewidth=1.5, label="Team 1")
        if t2_vals:
            ax.plot(t2_times, t2_vals, color="#d62728", linewidth=1.5, label="Team 2")

        t1_arr = np.array(t1_vals)
        mean1, std1 = t1_arr.mean(), t1_arr.std()
        ax.axhline(mean1, color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhspan(mean1 - std1, mean1 + std1, alpha=0.10, color="#1f77b4")
        ax.annotate(
            f"T1 mean={mean1:.0f}m²",
            xy=(t1_times[-1], mean1),
            fontsize=7,
            color="#1f77b4",
            va="center",
        )

        # Mark anomalies (>2σ)
        thresh = mean1 + 2 * std1
        anom_t = [t for t, v in zip(t1_times, t1_vals) if v > thresh]
        anom_v = [v for v in t1_vals if v > thresh]
        if anom_t:
            ax.scatter(
                anom_t,
                anom_v,
                color="#ff7f0e",
                zorder=5,
                s=40,
                label=f"T1 anomaly (>{thresh:.0f}m²)",
            )

        if t2_vals:
            t2_arr = np.array(t2_vals)
            mean2 = t2_arr.mean()
            ax.axhline(mean2, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.annotate(
                f"T2 mean={mean2:.0f}m²",
                xy=(t2_times[-1], mean2),
                fontsize=7,
                color="#d62728",
                va="center",
            )

        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Compactness (m²)", fontsize=9)
        ax.set_title(
            "Team Compactness Over Time\n(convex hull area; lower = more compact shape)",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return cls._fig_to_jpeg(fig)

    @classmethod
    def _render_centroid_trajectory(cls, fm: dict, fps: float) -> "bytes | None":
        """2D pitch diagram — smoothed centroid paths, time-gradient colour, full pitch markings."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.collections as mcoll
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        centroids = fm.get("team_centroids", [])
        if len(centroids) < 4:
            return None

        t1_x = np.array([e["team_1"][0] for e in centroids if e.get("team_1")], dtype=float)
        t1_y = np.array([e["team_1"][1] for e in centroids if e.get("team_1")], dtype=float)
        t2_x = np.array([e["team_2"][0] for e in centroids if e.get("team_2")], dtype=float)
        t2_y = np.array([e["team_2"][1] for e in centroids if e.get("team_2")], dtype=float)

        if len(t1_x) == 0:
            return None

        def _smooth_sample(xs, ys, target=40):
            n = len(xs)
            hw = max(2, int(fps * 5) // 2)
            sx = np.array([xs[max(0, i - hw) : min(n, i + hw + 1)].mean() for i in range(n)])
            sy = np.array([ys[max(0, i - hw) : min(n, i + hw + 1)].mean() for i in range(n)])
            step = max(1, n // target)
            idx = list(range(0, n, step))
            if idx[-1] != n - 1:
                idx.append(n - 1)
            return sx[idx], sy[idx]

        def _make_lc(xs, ys, cmap):
            pts = np.column_stack([xs, ys])[:, np.newaxis, :]
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            colors = np.linspace(0.3, 1.0, len(segs))
            lc = mcoll.LineCollection(
                segs, cmap=cmap, norm=plt.Normalize(0.3, 1.0), linewidth=2.2, alpha=0.9, zorder=5
            )
            lc.set_array(colors)
            return lc

        t1_sx, t1_sy = _smooth_sample(t1_x, t1_y)
        has_t2 = len(t2_x) > 1
        if has_t2:
            t2_sx, t2_sy = _smooth_sample(t2_x, t2_y)

        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.set_facecolor("#3d8b3d")

        # Pitch surface and outline
        ax.add_patch(
            mpatches.Rectangle(
                (0, 0),
                cls._PITCH_LEN,
                cls._PITCH_WID,
                linewidth=2,
                edgecolor="white",
                facecolor="#4caf50",
                zorder=1,
            )
        )
        # Halfway line
        ax.axvline(cls._PITCH_LEN / 2, color="white", linewidth=1.5, alpha=0.85, zorder=2)
        # Centre circle (r=9.15m) + spot
        ax.add_patch(
            mpatches.Circle(
                (cls._PITCH_LEN / 2, cls._PITCH_WID / 2),
                9.15,
                fill=False,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.85,
                zorder=2,
            )
        )
        ax.plot(cls._PITCH_LEN / 2, cls._PITCH_WID / 2, "wo", markersize=4, zorder=3)
        # Penalty boxes (16.5m deep × 40.32m wide)
        pb_w, pb_d = 40.32, 16.5
        pb_y = (cls._PITCH_WID - pb_w) / 2
        for px in (0, cls._PITCH_LEN - pb_d):
            ax.add_patch(
                mpatches.Rectangle(
                    (px, pb_y),
                    pb_d,
                    pb_w,
                    linewidth=1.4,
                    edgecolor="white",
                    facecolor="none",
                    alpha=0.85,
                    zorder=2,
                )
            )
        # Goal areas (5.5m deep × 18.32m wide)
        ga_w, ga_d = 18.32, 5.5
        ga_y = (cls._PITCH_WID - ga_w) / 2
        for gx in (0, cls._PITCH_LEN - ga_d):
            ax.add_patch(
                mpatches.Rectangle(
                    (gx, ga_y),
                    ga_d,
                    ga_w,
                    linewidth=1.1,
                    edgecolor="white",
                    facecolor="none",
                    alpha=0.7,
                    zorder=2,
                )
            )
        # Penalty spots
        for px in (11.0, cls._PITCH_LEN - 11.0):
            ax.plot(px, cls._PITCH_WID / 2, "wo", markersize=4, zorder=3)
        # Goals (7.32m wide, shown as thick white line at each end)
        goal_w = 7.32
        gy = (cls._PITCH_WID - goal_w) / 2
        ax.plot([0, 0], [gy, gy + goal_w], color="white", linewidth=5, zorder=3)
        ax.plot(
            [cls._PITCH_LEN, cls._PITCH_LEN],
            [gy, gy + goal_w],
            color="white",
            linewidth=5,
            zorder=3,
        )

        # Attack-direction labels
        mid_y = cls._PITCH_WID / 2
        ax.annotate(
            "",
            xy=(10, mid_y - 6),
            xytext=(2, mid_y - 6),
            arrowprops=dict(arrowstyle="->", color="#90caf9", lw=1.8),
            zorder=4,
        )
        ax.text(
            6,
            mid_y - 8.5,
            "Team 1 →",
            fontsize=8,
            color="#90caf9",
            ha="center",
            va="top",
            fontweight="bold",
        )
        ax.annotate(
            "",
            xy=(cls._PITCH_LEN - 10, mid_y + 6),
            xytext=(cls._PITCH_LEN - 2, mid_y + 6),
            arrowprops=dict(arrowstyle="->", color="#ef9a9a", lw=1.8),
            zorder=4,
        )
        ax.text(
            cls._PITCH_LEN - 6,
            mid_y + 8.5,
            "← Team 2",
            fontsize=8,
            color="#ef9a9a",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        # Team trajectories
        lc1 = _make_lc(t1_sx, t1_sy, plt.cm.Blues)
        ax.add_collection(lc1)
        if has_t2:
            lc2 = _make_lc(t2_sx, t2_sy, plt.cm.Reds)
            ax.add_collection(lc2)

        # Start / end markers with offset labels
        def _endpoint(x, y, label, col):
            ax.scatter(x, y, s=120, color=col, edgecolors="white", linewidths=1.6, zorder=7)
            ax.annotate(
                label,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=col,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.85),
            )

        _endpoint(t1_sx[0], t1_sy[0], "T1 start", "#1565c0")
        _endpoint(t1_sx[-1], t1_sy[-1], "T1 end", "#1565c0")
        if has_t2:
            _endpoint(t2_sx[0], t2_sy[0], "T2 start", "#b71c1c")
            _endpoint(t2_sx[-1], t2_sy[-1], "T2 end", "#b71c1c")

        # Mean-position stars
        t1_mx, t1_my = float(t1_x.mean()), float(t1_y.mean())
        ax.scatter(
            t1_mx,
            t1_my,
            s=200,
            color="#1565c0",
            marker="*",
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
        )
        ax.annotate(
            f"T1 avg ({t1_mx:.1f}, {t1_my:.1f})",
            (t1_mx, t1_my),
            xytext=(5, -13),
            textcoords="offset points",
            fontsize=7.5,
            color="#1565c0",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#1565c0", alpha=0.85),
        )
        if has_t2:
            t2_mx, t2_my = float(t2_x.mean()), float(t2_y.mean())
            ax.scatter(
                t2_mx,
                t2_my,
                s=200,
                color="#b71c1c",
                marker="*",
                edgecolors="white",
                linewidths=0.8,
                zorder=8,
            )
            ax.annotate(
                f"T2 avg ({t2_mx:.1f}, {t2_my:.1f})",
                (t2_mx, t2_my),
                xytext=(5, -13),
                textcoords="offset points",
                fontsize=7.5,
                color="#b71c1c",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white", edgecolor="#b71c1c", alpha=0.85
                ),
            )

        # Colorbars
        sm1 = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(0.3, 1.0))
        sm1.set_array([])
        cb1 = fig.colorbar(sm1, ax=ax, fraction=0.018, pad=0.01)
        cb1.set_label("Team 1 (early→late)", fontsize=7.5, color="#1565c0")
        cb1.set_ticks([0.3, 0.65, 1.0])
        cb1.set_ticklabels(["early", "mid", "late"])
        cb1.ax.tick_params(labelsize=7)
        if has_t2:
            sm2 = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(0.3, 1.0))
            sm2.set_array([])
            cb2 = fig.colorbar(sm2, ax=ax, fraction=0.018, pad=0.04)
            cb2.set_label("Team 2 (early→late)", fontsize=7.5, color="#b71c1c")
            cb2.set_ticks([0.3, 0.65, 1.0])
            cb2.set_ticklabels(["early", "mid", "late"])
            cb2.ax.tick_params(labelsize=7)

        ax.set_xlim(-4, cls._PITCH_LEN + 4)
        ax.set_ylim(-4, cls._PITCH_WID + 4)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Pitch length (m)", fontsize=9)
        ax.set_ylabel("Pitch width (m)", fontsize=9)
        ax.set_title(
            "Team Centroid Trajectories\n"
            "(colour darkens over time  ·  ★ = mean position  ·  ● = start)",
            fontsize=11,
            pad=6,
        )
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        return cls._fig_to_jpeg(fig)

    @classmethod
    def _render_pressing_dashboard(cls, fm: dict, fps: float) -> "bytes | None":
        """Inter-team distance + compactness overlay with pressing phase shading."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        distances = fm.get("inter_team_distance_m", [])
        comp = fm.get("compactness_m2", [])
        if not distances:
            return None

        valid_d = [
            (d["frame"] / fps, d["distance_m"])
            for d in distances
            if d.get("distance_m") is not None
        ]
        if not valid_d:
            return None
        d_times, d_vals = zip(*valid_d)
        d_times, d_vals = list(d_times), list(d_vals)

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(d_times, d_vals, color="#2ca02c", linewidth=1.5, label="Inter-team distance (m)")
        d_arr = np.array(d_vals)
        pressing_thresh = d_arr.mean() - 0.5 * d_arr.std()
        ax1.axhline(pressing_thresh, color="#2ca02c", linestyle=":", linewidth=0.8, alpha=0.7)
        ax1.annotate(
            f"Press thresh={pressing_thresh:.1f}m",
            xy=(d_times[0], pressing_thresh),
            fontsize=7,
            color="#2ca02c",
        )

        # Shade pressing windows
        in_press = False
        p_start = None
        for t, v in zip(d_times, d_vals):
            if v < pressing_thresh and not in_press:
                in_press = True
                p_start = t
            elif v >= pressing_thresh and in_press:
                ax1.axvspan(p_start, t, alpha=0.15, color="#2ca02c")
                in_press = False
        if in_press and p_start is not None:
            ax1.axvspan(p_start, d_times[-1], alpha=0.15, color="#2ca02c")

        ax1.set_xlabel("Time (s)", fontsize=9)
        ax1.set_ylabel("Inter-team distance (m)", fontsize=9, color="#2ca02c")
        ax1.tick_params(axis="y", labelcolor="#2ca02c")

        if comp:
            ax2 = ax1.twinx()
            t1_times = [e["frame"] / fps for e in comp if e.get("team_1_m2") is not None]
            t1_vals = [e["team_1_m2"] for e in comp if e.get("team_1_m2") is not None]
            if t1_vals:
                ax2.plot(
                    t1_times,
                    t1_vals,
                    color="#1f77b4",
                    linewidth=1.2,
                    alpha=0.6,
                    linestyle="--",
                    label="T1 compactness (m²)",
                )
                ax2.set_ylabel("T1 Compactness (m²)", fontsize=9, color="#1f77b4")
                ax2.tick_params(axis="y", labelcolor="#1f77b4")

        ax1.set_title(
            "Pressing Dashboard\n(green shading = pressing phases: dist < mean−0.5σ)", fontsize=10
        )
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        return cls._fig_to_jpeg(fig)

    @classmethod
    def _render_combined(cls, fm: dict, fps: float) -> "bytes | None":
        """2×2 subplot: compactness, centroid x-position, inter-team distance, zone occupancy."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return None

        comp = fm.get("compactness_m2", [])
        centroids = fm.get("team_centroids", [])
        distances = fm.get("inter_team_distance_m", [])
        occ = fm.get("pitch_zone_occupancy", {})

        if not comp and not centroids:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Per-Frame Tactical Overview", fontsize=12)

        # 1. Compactness
        ax = axes[0, 0]
        t1_vals = [e["team_1_m2"] for e in comp if e.get("team_1_m2") is not None]
        t1_t = [e["frame"] / fps for e in comp if e.get("team_1_m2") is not None]
        t2_vals = [e["team_2_m2"] for e in comp if e.get("team_2_m2") is not None]
        t2_t = [e["frame"] / fps for e in comp if e.get("team_2_m2") is not None]
        if t1_vals:
            ax.plot(t1_t, t1_vals, color="#1f77b4", linewidth=1.2, label="T1")
            ax.axhline(np.mean(t1_vals), color="#1f77b4", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.annotate(
                f"mean={np.mean(t1_vals):.0f}",
                xy=(t1_t[-1], np.mean(t1_vals)),
                fontsize=6,
                color="#1f77b4",
            )
        if t2_vals:
            ax.plot(t2_t, t2_vals, color="#d62728", linewidth=1.2, label="T2")
            ax.axhline(np.mean(t2_vals), color="#d62728", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_title("Compactness (m²)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 2. Centroid x-position (territorial control)
        ax = axes[0, 1]
        if centroids:
            t1_cx = [(e["frame"] / fps, e["team_1"][0]) for e in centroids if e.get("team_1")]
            t2_cx = [(e["frame"] / fps, e["team_2"][0]) for e in centroids if e.get("team_2")]
            if t1_cx:
                ax.plot(
                    [x[0] for x in t1_cx],
                    [x[1] for x in t1_cx],
                    color="#1f77b4",
                    linewidth=1.2,
                    label="T1 x",
                )
            if t2_cx:
                ax.plot(
                    [x[0] for x in t2_cx],
                    [x[1] for x in t2_cx],
                    color="#d62728",
                    linewidth=1.2,
                    label="T2 x",
                )
            ax.axhline(52.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.annotate("halfway", xy=(0, 52.5), fontsize=6, color="gray")
        ax.set_title("Centroid x-Position (m)\n0=own goal, 105=opp", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 3. Inter-team distance
        ax = axes[1, 0]
        if distances:
            valid_dist = [
                (d["frame"] / fps, d["distance_m"])
                for d in distances
                if d.get("distance_m") is not None
            ]
            if valid_dist:
                d_t, d_v = zip(*valid_dist)
                d_t, d_v = list(d_t), list(d_v)
            else:
                d_t, d_v = [], []
        if distances and d_t:
            ax.plot(d_t, d_v, color="#2ca02c", linewidth=1.2)
            ax.axhline(np.mean(d_v), color="#2ca02c", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.annotate(
                f"mean={np.mean(d_v):.1f}m", xy=(d_t[-1], np.mean(d_v)), fontsize=6, color="#2ca02c"
            )
        ax.set_title("Inter-Team Distance (m)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.grid(True, alpha=0.3)

        # 4. Zone occupancy bar chart
        ax = axes[1, 1]
        t1_occ = occ.get("team_1", {})
        t2_occ = occ.get("team_2", {})
        if t1_occ or t2_occ:
            zones = ["Defensive\nThird", "Middle\nThird", "Attacking\nThird"]
            keys = ["defensive", "middle", "attacking"]
            x = np.arange(len(zones))
            w = 0.35
            t1_pct = [t1_occ.get(k, 0) * 100 for k in keys]
            t2_pct = [t2_occ.get(k, 0) * 100 for k in keys]
            ax.bar(x - w / 2, t1_pct, w, label="Team 1", color="#1f77b4", alpha=0.8)
            ax.bar(x + w / 2, t2_pct, w, label="Team 2", color="#d62728", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(zones, fontsize=7)
            ax.set_ylabel("% player-frames", fontsize=7)
            ax.legend(fontsize=7)
        ax.set_title("Zone Occupancy (%)", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return cls._fig_to_jpeg(fig)

    @staticmethod
    def _fig_to_jpeg(fig: "matplotlib.figure.Figure", dpi: int = 150) -> bytes:  # noqa: F821
        """Render a matplotlib figure to JPEG bytes in memory."""
        import io

        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        data = buf.read()
        plt.close(fig)
        return data


def augment_system_prompt_visual(base_prompt: str) -> str:
    """Append visual time-series usage guidance for vision-capable LLMs.

    Instructs the LLM to read trends and annotated statistics directly from
    the attached charts rather than computing from table values. This prevents
    the numeric misquoting problem (BPE fragmentation of decimal strings).

    Args:
        base_prompt: One of the SYSTEM_PROMPTS values.

    Returns:
        Augmented prompt string.
    """
    guidance = """

## Using Attached Time-Series Charts
The data context is accompanied by visual charts showing per-frame tactical metrics.
Use these charts as your primary evidence source for temporal and spatial claims:

- **Compactness Chart**: Read the annotated mean values (e.g. "T1 mean=534m²") directly
  from the chart. Do NOT compute averages from the text tables — the sampled table rows
  may not reflect the true mean. Orange scatter points mark statistical anomalies (>2σ).
- **Centroid Trajectory**: Interpret team spatial positions from the 2D pitch diagram.
  Circle = start position, triangle = end position. The path shows territorial drift.
- **Pressing Dashboard**: Green shaded regions indicate pressing phases (inter-team
  distance below threshold). Reference these windows when describing pressing sequences.
- **Combined Overview**: Use for cross-metric claims (e.g. compactness spike coinciding
  with high inter-team distance suggests a defensive shape break during a pressing phase).

When citing statistics from the charts, use the annotated values visible in the chart
(e.g. "mean compactness of 534m²") rather than values from the markdown tables below."""
    return base_prompt + guidance


def augment_system_prompt_visual_focused(base_prompt: str) -> str:
    """Append focused compactness-chart usage guidance for single-chart conditions.

    Used for the VISUAL_FOCUSED evaluation condition where only the compactness
    chart is attached. Prevents the LLM from expecting multiple charts.

    Args:
        base_prompt: One of the SYSTEM_PROMPTS values.

    Returns:
        Augmented prompt string.
    """
    guidance = """

## Using the Attached Compactness Chart
A single chart showing Team 1 and Team 2 compactness (convex hull area in m²) over time
is attached. Use it as your primary evidence source for defensive shape claims:

- Read the annotated mean value directly from the chart (dashed line + label).
  Do NOT compute averages from the table rows — sampled rows may be unrepresentative.
- Orange markers indicate anomalies (compactness >2σ above mean) — these often correspond
  to shape breaks during pressing triggers or possession transitions.
- Reference specific time windows (x-axis) when describing when shape changed.
- Lower values = more compact shape; typical compact block ~400m², open play ~800m²."""
    return base_prompt + guidance


def augment_system_prompt_findings_informed(
    base_prompt: str, analysis_type: str, chart_name: "str | list[str]"
) -> str:
    """System prompt for FINDINGS_INFORMED / FINDINGS_INFORMED_MC.

    Accepts either a single chart name (FINDINGS_INFORMED) or a list of chart
    names (FINDINGS_INFORMED_MC — minimum-sufficient chart set per analysis type).

    Args:
        base_prompt: One of the SYSTEM_PROMPTS values.
        analysis_type: e.g. "match_overview", "event_analysis".
        chart_name: Single chart name string, or list of chart names for multi-chart routing.

    Returns:
        Augmented prompt string.
    """
    _chart_info = {
        "compactness": (
            "compactness time-series",
            "Team 1 (blue) and Team 2 (red) compactness in m² over time. Annotated mean "
            "values appear as dashed lines with labels (e.g. 'T1 mean=534m²'). Orange "
            "scatter markers indicate anomalies (>2σ). Lower area = more compact shape. "
            "Read annotated mean values directly — do not compute from any text tables.",
        ),
        "centroid": (
            "team centroid trajectory",
            "2D pitch diagram (105×68m, full pitch markings) with Team 1 (Blues gradient) and "
            "Team 2 (Reds gradient) centroid paths — colour darkens from early (light) to late "
            "(dark) in the clip. ★ markers show each team's mean position with coordinates. "
            "Circle = start, arrow-head annotations = end. Attack-direction arrows show which "
            "team attacks which end. Ground territorial position, spatial dominance, and "
            "centroid drift claims from this diagram.",
        ),
        "pressing": (
            "pressing dashboard",
            "Two-panel chart: (top) inter-team distance over time — green shaded regions "
            "mark active pressing phases where distance falls below threshold; (bottom) "
            "compactness overlay. Reference specific shaded time windows for pressing claims.",
        ),
        "combined": (
            "combined 2×2 tactical overview",
            "Four panels: compactness time-series, centroid trajectory, pressing dashboard, "
            "and zone occupancy. Cross-reference panels for multi-metric claims.",
        ),
    }
    # Handle both single chart name and list of chart names
    names = [chart_name] if isinstance(chart_name, str) else list(chart_name)
    if len(names) == 1:
        chart_key = names[0]
        name, desc = _chart_info.get(chart_key, ("tactical chart", "tactical time-series data"))
        guidance = f"""

## Using the Attached {name.title()} Chart
A single {name} chart is attached as the visual evidence source for this analysis.
This chart was selected based on linear-probing evidence (Schumacher et al. 2026 §4):
the visual representation of {name} most reliably encodes the tactical patterns
central to {analysis_type.replace("_", " ")} claims.

**Chart contents**: {desc}

Cite values and patterns visible in the chart. Do not compute statistics from
any text tables — the chart annotations are the authoritative evidence."""
    else:
        chart_lines = []
        for i, cn in enumerate(names, 1):
            n, d = _chart_info.get(cn, ("tactical chart", "tactical time-series data"))
            chart_lines.append(f"**Chart {i} — {n.title()}**: {d}")
        charts_block = "\n".join(chart_lines)
        guidance = f"""

## Using the Attached Charts ({len(names)} charts)
{len(names)} charts are attached as the visual evidence source for this analysis.
This chart set was selected based on linear-probing evidence (Schumacher et al. 2026 §4)
as the minimum-sufficient set to encode the multi-modal claim space of
{analysis_type.replace("_", " ")} (temporal trigger + team shape + pitch location).

{charts_block}

Cross-reference the charts to ground claims. Do not compute statistics from
any text tables — the chart annotations are the authoritative evidence."""
    return base_prompt + guidance


def augment_system_prompt_perframe(base_prompt: str) -> str:
    """Append per-frame usage guidance to a standard system prompt.

    Instructs the LLM to use the '## Per-Frame Spatial Evidence' section
    for spatial and temporal claims, and to cite specific time windows.

    Args:
        base_prompt: One of the SYSTEM_PROMPTS values.

    Returns:
        Augmented prompt string.
    """
    guidance = """

## Using Per-Frame Spatial Evidence
The data context below includes a section titled '## Per-Frame Spatial Evidence' derived
from per-frame tracking records (player positions every frame, ball trajectory, zone
occupancy computed across all frames). Use it as follows:

- **Zone Occupancy**: Use defensive/middle/attacking third percentages to support or
  refute claims about team positioning ("deep defensive block", "high line", etc.)
- **Centroid Progression**: Reference team centroid positions at specific time points
  to describe territorial movement and phase transitions
- **Possession Phases**: Use phase durations and centroid positions to describe how
  teams moved the ball and their spatial positioning during sustained possession
- **Compactness Series**: Reference compactness variation over time to describe
  how defensive shape changed throughout the passage of play
- **Formation Estimates**: Cite the formation string and confidence score — use hedged
  language for low-confidence formations (conf < 0.30)
- **Event Context**: Reference specific event timestamps and spatial coordinates to
  anchor event-level claims

When making spatial claims, cite the per-frame evidence (e.g. "zone occupancy data
shows Team 1 spent 32.8% of player-frames in the defensive third"). When the per-frame
data contradicts aggregate statistics, note the discrepancy explicitly."""
    return base_prompt + guidance


def augment_system_prompt_perframe_v2(base_prompt: str) -> str:
    """Context Engineering augmentation for PERFRAME_V2.

    Implements the expert-persona + chain-of-thought approach described in
    Sumpter (2025) Context Engineering for Football. Structures the prompt so the
    LLM reasons within a football-analyst frame, not as a generic statistics
    summariser.

    The 6-step CoT mirrors how a club performance analyst works through a clip.
    """
    guidance = """

## Analyst Role and Analytical Framework

You are a **senior football performance analyst** reviewing automated tracking data from a computer vision pipeline. You specialise in interpreting centroid movement, convex-hull compactness metrics, k-means formation estimates, and zone occupancy distributions derived from broadcast video. You are aware that CV-derived tracking data contains noise — missed detections, player ID switches, homography drift — and you hedge claims where data quality warrants it (particularly for low-confidence formation estimates, conf < 0.30).

### 6-Step Chain-of-Thought Process

Work through this sequence before writing your response:

1. **Read the Per-Frame Tactical Insights** (if present — the F1–F6 findings): identify the headline narrative around territorial drift, shape fracture, phase density, zone paradox, formation stability, and event clustering.
2. **Cross-reference zone occupancy against possession percentage**: flag paradoxes — e.g. high possession paired with deep defensive zone occupancy suggests a deep-sitting, recycling possession style, not a dominant pressing team.
3. **Trace the compactness time series**: identify *when* and *how much* defensive shape changed — steady compression, sudden expansion, or statistically anomalous spikes. Anomalies at specific timestamps often coincide with possession transitions or pressing triggers.
4. **Examine possession phases**: distinguish patient build-up (long phases, gradual centroid shift) from direct or counter-attacking play (short phases, rapid centroid jumps). Reference the longest phase duration and the centroid position during it.
5. **Assess formation stability**: fluid shapes across temporal windows indicate in/out-of-possession role switching; rigid shapes suggest a fixed tactical blueprint. Use hedged language for low-confidence estimates.
6. **Synthesise into a coherent tactical read**: lead with the tactical interpretation, follow with supporting evidence. Weave numbers into sentences — do not bullet-list statistics.

### Output Style

Write as a **match analyst briefing the coaching staff**, not as a statistics engine. Use football vocabulary where the data supports it: 'deep defensive block', 'progressive press', 'half-space overload', 'recycling possession', 'transitional compact phase', 'pressing trigger'. Cite specific data points to anchor claims but embed them in prose. When per-frame data contradicts aggregate statistics, note the discrepancy explicitly."""
    return base_prompt + guidance


# ── Tactical Analyzer ──────────────────────────────────────────────────────


class TacticalAnalyzer:
    """Main interface for generating tactical analysis from pipeline data.

    Usage:
        analyzer = TacticalAnalyzer()  # Uses default provider from env
        commentary = await analyzer.analyze(analytics_json, "match_overview")
    """

    VALID_TYPES = set(SYSTEM_PROMPTS.keys())

    def __init__(self, provider: LLMProvider | None = None):
        """Initialize with an LLM provider.

        Args:
            provider: Specific LLMProvider instance. If None, auto-detects
                from environment variables.
        """
        self._provider = provider

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = get_provider()
        return self._provider

    def _normalize_type(self, analysis_type: str) -> str:
        type_aliases = {
            "match_summary": "match_overview",
            "tactical_analysis": "tactical_deep_dive",
        }
        normalized = type_aliases.get(analysis_type, analysis_type)
        if normalized not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown analysis type '{analysis_type}'. "
                f"Valid types: {', '.join(sorted(self.VALID_TYPES))}"
            )
        return normalized

    async def _prepare_images(self, video_path: str | None) -> list[bytes] | None:
        """Extract and encode keyframes from the annotated video for vision-augmented generation."""
        if not video_path:
            return None
        try:
            import asyncio

            from .vision import extract_keyframes, frames_to_jpeg_bytes

            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(None, lambda: extract_keyframes(video_path))
            if not frames:
                return None
            images = await loop.run_in_executor(None, lambda: frames_to_jpeg_bytes(frames))
            logger.info("Vision-augmented analysis: %d annotated frames prepared", len(images))
            return images or None
        except Exception as e:
            logger.warning("Frame extraction failed, falling back to text-only: %s", e)
            return None

    async def analyze(
        self,
        analytics_data: dict,
        analysis_type: str = "match_overview",
        video_path: str | None = None,
        include_insights: bool = True,
        system_prompt_override: str | None = None,
    ) -> dict:
        """Generate tactical analysis grounded in pipeline data.

        Args:
            analytics_data: Analytics JSON from the CV pipeline.
            analysis_type: One of: match_overview, tactical_deep_dive,
                event_analysis, player_spotlight.
            video_path: Optional path to the annotated video for vision-augmented
                generation (boosts grounding rate from ~61% to ~91%).
            include_insights: If False, skip MatchInsights pre-interpretation layer
                (used for ablation condition H: same data, no pre-interpretation).
            system_prompt_override: If provided, replaces the default system prompt
                (used for prompt ablation conditions I/J: strip few-shot or metric defs).

        Returns:
            Dict with keys: content (str), grounding_data (dict), analysis_type (str)

        Raises:
            ValueError: If analysis_type is not recognized.
            RuntimeError: If no LLM provider is available.
        """
        normalized_type = self._normalize_type(analysis_type)

        # Format analytics as markdown grounding
        grounded_markdown = GroundingFormatter.format(
            analytics_data, include_insights=include_insights
        )
        system_prompt = system_prompt_override or SYSTEM_PROMPTS[normalized_type]

        logger.info(
            "Generating '%s' analysis (%d chars of grounding data)",
            normalized_type,
            len(grounded_markdown),
        )

        images = await self._prepare_images(video_path)
        provider = self._get_provider()
        content = await provider.generate(system_prompt, grounded_markdown, images=images)

        return {
            "content": content,
            "analysis_type": normalized_type,
            "grounding_data": {
                "formatted_length": len(grounded_markdown),
                "analysis_type": normalized_type,
                "provider": type(provider).__name__,
                "vision_augmented": images is not None,
            },
        }

    async def stream_analyze(
        self,
        analytics_data: dict,
        analysis_type: str = "match_overview",
        video_path: str | None = None,
        include_insights: bool = True,
        system_prompt_override: str | None = None,
    ):
        """Stream tactical analysis chunks as they arrive from the LLM.

        Same as analyze() but yields text progressively for streaming responses.
        Falls back to yielding the full response at once for non-streaming providers.
        """
        normalized_type = self._normalize_type(analysis_type)
        grounded_markdown = GroundingFormatter.format(
            analytics_data, include_insights=include_insights
        )
        system_prompt = system_prompt_override or SYSTEM_PROMPTS[normalized_type]

        logger.info(
            "Streaming '%s' analysis (%d chars of grounding data)",
            normalized_type,
            len(grounded_markdown),
        )

        images = await self._prepare_images(video_path)
        provider = self._get_provider()
        async for chunk in provider.stream_generate(
            system_prompt, grounded_markdown, images=images
        ):
            yield chunk

    @staticmethod
    def available_types() -> list[dict]:
        """Return available analysis types with descriptions."""
        return [
            {
                "id": "match_overview",
                "name": "Match Overview",
                "description": "3-4 paragraph tactical match summary",
            },
            {
                "id": "tactical_deep_dive",
                "name": "Tactical Deep Dive",
                "description": "Detailed formation, pressing, and space analysis",
            },
            {
                "id": "event_analysis",
                "name": "Event Analysis",
                "description": "Event-by-event tactical commentary",
            },
            {
                "id": "player_spotlight",
                "name": "Player Spotlight",
                "description": "Individual player performance analysis",
            },
        ]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _safe_pct(val: Any) -> str:
    """Format a percentage value safely."""
    if val is None:
        return "0.0"
    try:
        return f"{float(val):.1f}"
    except (TypeError, ValueError):
        return "0.0"
