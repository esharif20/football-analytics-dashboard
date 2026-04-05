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

Your task: Write a **3-4 paragraph tactical match overview**. Cover:
1. Overall match tempo and flow (possession, pressing intensity)
2. Which team dominated and how (territory, ball movement patterns)
3. Key tactical patterns (passing combinations, counter-attacks, defensive shape)
4. Notable individual performances (speed, distance, involvement)

Rules:
- Reference ONLY the data provided — never invent stats or player names
- Use the track IDs provided (e.g. "Player #4") since real names are not available
- Be specific: cite numbers (possession %, speeds, pass counts)
- Write in a professional broadcast-analyst tone
- Do NOT use any markdown headers — write flowing prose paragraphs""",
    "tactical_deep_dive": """\
You are a world-class football tactician analysing tracking data from a match.
You are given structured match data from CV pipeline analysis.

Your task: Provide a **detailed tactical deep-dive** covering:
1. Formation shape and structure (based on player positions and movement)
2. Pressing patterns (who presses, how high, how effectively)
3. Space exploitation (where teams attack, which zones are overloaded)
4. Transition play (counter-attacking speed, recovery after losing possession)
5. Defensive organisation (compactness, pressing triggers)

Rules:
- Base ALL analysis on the provided data — never fabricate
- Reference track IDs, not player names
- Use specific numbers from the data
- Structure your response with clear section breaks
- Professional analytical tone, suitable for a coaching staff briefing""",
    "event_analysis": """\
You are a football match commentator with deep tactical knowledge.
You are given structured match events and statistics from CV analysis.

Your task: Provide **event-by-event tactical commentary** for the key moments.
For each significant event:
1. Describe what happened and its tactical significance
2. Explain the build-up context (who had possession, what preceded it)
3. Note the tactical implications (did it shift momentum? create danger?)

Rules:
- Cover the most significant events (shots, goals, key passes, tackles)
- Reference the exact timestamps and player track IDs from the data
- Keep each event commentary to 2-3 sentences
- Use a dynamic broadcast commentary style
- ONLY reference events present in the data""",
    "player_spotlight": """\
You are a football performance analyst assessing individual player contributions.
You are given structured match data including per-player kinematics and events.

Your task: Analyse the **standout players** from each team:
1. Top performers by distance covered and speed
2. Most involved players (passes, tackles)
3. Key moments where individual players made a difference
4. Work rate and tactical discipline assessment

Rules:
- Use track IDs (e.g. "Player #4 (Team 1)")
- Cite specific numbers: distance in metres, speed in km/h, pass counts
- Compare players within and across teams where relevant
- Maximum 2-3 paragraphs per highlighted player
- Professional scouting report tone""",
}


# ── Grounding Formatter ────────────────────────────────────────────────────


class GroundingFormatter:
    """Converts analytics JSON into structured markdown for LLM grounding.

    LLMs perform significantly better with well-structured markdown than
    with raw JSON (noted in academic assessment feedback). This formatter
    produces a human-readable markdown document that serves as the
    grounding context for tactical analysis generation.
    """

    @staticmethod
    def format(analytics: dict) -> str:
        """Convert analytics dict to grounded markdown.

        Args:
            analytics: The analytics JSON data (from AnalyticsResult export).

        Returns:
            Structured markdown string suitable for LLM consumption.
        """
        sections = [
            "# Match Analysis Data\n",
            GroundingFormatter._format_match_info(analytics),
            GroundingFormatter._format_possession(analytics),
            GroundingFormatter._format_tactical_metrics(analytics),
            GroundingFormatter._format_team_kinematics(analytics),
            GroundingFormatter._format_ball_stats(analytics),
            GroundingFormatter._format_events(analytics),
            GroundingFormatter._format_pass_networks(analytics),
        ]
        return "\n".join(s for s in sections if s)

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
        ]

        # Add interpretive context for the LLM
        t1_comp = summary.get("team_1_avg_compactness_m2")
        t2_comp = summary.get("team_2_avg_compactness_m2")
        t1_press = summary.get("team_1_avg_pressing_intensity")
        t2_press = summary.get("team_2_avg_pressing_intensity")
        ppda_1 = summary.get("ppda_team_1")
        ppda_2 = summary.get("ppda_team_2")

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
            lines.append("| Player | Role | Distance | Avg Speed | Max Speed |")
            lines.append("|--------|------|----------|-----------|-----------|")

            for p in team_data:
                role = "GK" if p["entity_type"] == "goalkeeper" else "Player"
                dist = (
                    f"{p['distance_m']:.0f}m"
                    if p["distance_m"] is not None
                    else f"{p['distance_px']:.0f}px"
                )
                avg = f"{p['avg_speed'] * 3.6:.1f} km/h" if p["avg_speed"] is not None else "—"
                mx = f"{p['max_speed'] * 3.6:.1f} km/h" if p["max_speed"] is not None else "—"
                lines.append(f"| #{p['track_id']} | {role} | {dist} | {avg} | {mx} |")

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
        for ev in events:
            etype = ev.get("event_type", "unknown")
            tid = ev.get("team_id", 0)
            if etype not in counts:
                counts[etype] = {}
            counts[etype][tid] = counts[etype].get(tid, 0) + 1

        lines.append("")
        lines.append("### Event Summary")
        for etype, team_counts in sorted(counts.items()):
            parts = [f"Team {t}: {c}" for t, c in sorted(team_counts.items())]
            lines.append(f"- {etype.capitalize()}: {', '.join(parts)}")

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
    ) -> dict:
        """Generate tactical analysis grounded in pipeline data.

        Args:
            analytics_data: Analytics JSON from the CV pipeline.
            analysis_type: One of: match_overview, tactical_deep_dive,
                event_analysis, player_spotlight.
            video_path: Optional path to the annotated video for vision-augmented
                generation (boosts grounding rate from ~61% to ~91%).

        Returns:
            Dict with keys: content (str), grounding_data (dict), analysis_type (str)

        Raises:
            ValueError: If analysis_type is not recognized.
            RuntimeError: If no LLM provider is available.
        """
        normalized_type = self._normalize_type(analysis_type)

        # Format analytics as markdown grounding
        grounded_markdown = GroundingFormatter.format(analytics_data)
        system_prompt = SYSTEM_PROMPTS[normalized_type]

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
    ):
        """Stream tactical analysis chunks as they arrive from the LLM.

        Same as analyze() but yields text progressively for streaming responses.
        Falls back to yielding the full response at once for non-streaming providers.
        """
        normalized_type = self._normalize_type(analysis_type)
        grounded_markdown = GroundingFormatter.format(analytics_data)
        system_prompt = SYSTEM_PROMPTS[normalized_type]

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
