import asyncio
import json
from types import SimpleNamespace

import pytest
from services.llm_providers import StubProvider, get_provider
from services.tactical import GroundingFormatter, TacticalAnalyzer

# conftest.py adds the api root to sys.path — direct imports work
from api.routers.commentary import _load_analytics_data

# ── Shared fixture ────────────────────────────────────────────────────────


def make_analytics(**overrides) -> dict:
    base = {
        "fps": 25.0,
        "homography_available": True,
        "possession": {
            "total_frames": 500,
            "team_1_frames": 290,
            "team_2_frames": 180,
            "contested_frames": 30,
            "team_1_percentage": 58.0,
            "team_2_percentage": 36.0,
            "possession_changes": 22,
            "longest_team_1_spell": 120,
            "longest_team_2_spell": 80,
            "events": [],
        },
        "player_kinematics": {
            "1": {
                "track_id": 1,
                "entity_type": "player",
                "team_id": 1,
                "total_distance_m": 2500.0,
                "total_distance_px": 0.0,
                "avg_speed_m_per_sec": 3.2,
                "max_speed_m_per_sec": 7.8,
            },
            "2": {
                "track_id": 2,
                "entity_type": "goalkeeper",
                "team_id": 2,
                "total_distance_m": 800.0,
                "total_distance_px": 0.0,
                "avg_speed_m_per_sec": 1.5,
                "max_speed_m_per_sec": 5.0,
            },
        },
        "ball_kinematics": {
            "track_id": 0,
            "entity_type": "ball",
            "team_id": None,
            "total_distance_m": 4200.0,
            "total_distance_px": 0.0,
            "avg_speed_m_per_sec": 8.4,
            "max_speed_m_per_sec": 23.0,
        },
        "ball_path": {
            "positions": [],
            "pitch_positions": [],
            "total_distance_m": 4200.0,
            "avg_speed_m_per_sec": 8.4,
            "direction_changes": 47,
        },
        "events": [
            {
                "event_type": "pass",
                "frame_idx": 12,
                "timestamp_sec": 0.48,
                "team_id": 1,
                "player_track_id": 1,
                "target_player_track_id": 3,
                "confidence": 0.87,
                "success": True,
            },
            {
                "event_type": "shot",
                "frame_idx": 200,
                "timestamp_sec": 8.0,
                "team_id": 1,
                "player_track_id": 5,
                "confidence": 0.92,
                "success": None,
            },
            {
                "event_type": "challenge",
                "frame_idx": 350,
                "timestamp_sec": 14.0,
                "team_id": 2,
                "player_track_id": 8,
                "target_player_track_id": 1,
                "confidence": 0.74,
                "success": None,
            },
        ],
        "interaction_graph_team1": {
            "nodes": [{"id": 1}, {"id": 3}],
            "edges": [{"source": 1, "target": 3, "weight": 5.0}],
        },
        "interaction_graph_team2": None,
    }
    base.update(overrides)
    return base


# ── _load_analytics_data ──────────────────────────────────────────────────


def test_load_analytics_from_json_string():
    data = {"fps": 30, "possession": {"team_1_percentage": 60}}
    analysis = SimpleNamespace(analyticsDataUrl=json.dumps(data))
    loaded = asyncio.run(_load_analytics_data(analysis))
    assert loaded["fps"] == 30
    assert loaded["possession"]["team_1_percentage"] == 60


def test_load_analytics_from_file(tmp_path):
    data = {"fps": 24, "possession": {"team_2_percentage": 52}}
    p = tmp_path / "analytics.json"
    p.write_text(json.dumps(data))
    analysis = SimpleNamespace(analyticsDataUrl=str(p))
    loaded = asyncio.run(_load_analytics_data(analysis))
    assert loaded["fps"] == 24
    assert loaded["possession"]["team_2_percentage"] == 52


def test_load_analytics_no_url_raises():
    from fastapi import HTTPException

    analysis = SimpleNamespace(analyticsDataUrl=None)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_load_analytics_data(analysis))
    assert exc_info.value.status_code == 400


def test_load_analytics_missing_file_raises():
    from fastapi import HTTPException

    analysis = SimpleNamespace(analyticsDataUrl="/tmp/does_not_exist_analytics.json")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_load_analytics_data(analysis))
    assert exc_info.value.status_code == 400


# ── GroundingFormatter ────────────────────────────────────────────────────


def test_grounding_formatter_contains_sections():
    rendered = GroundingFormatter.format(make_analytics())
    assert "# Match Analysis Data" in rendered
    assert "## Possession" in rendered
    assert "## Player Performance" in rendered
    assert "## Ball Statistics" in rendered
    assert "## Events Timeline" in rendered
    assert "## Pass Networks" in rendered


def test_possession_percentages_rendered():
    rendered = GroundingFormatter.format(make_analytics())
    assert "58.0%" in rendered
    assert "36.0%" in rendered


def test_speed_converted_to_kmh():
    rendered = GroundingFormatter.format(make_analytics())
    # 7.8 m/s * 3.6 = 28.08 → "28.1 km/h"
    assert "28.1 km/h" in rendered


def test_goalkeeper_labelled_gk():
    rendered = GroundingFormatter.format(make_analytics())
    assert "GK" in rendered


def test_pass_event_formatted():
    rendered = GroundingFormatter.format(make_analytics())
    assert "**Pass**" in rendered
    assert "#1" in rendered and "#3" in rendered


def test_shot_event_formatted():
    rendered = GroundingFormatter.format(make_analytics())
    assert "**Shot**" in rendered


def test_challenge_event_formatted():
    rendered = GroundingFormatter.format(make_analytics())
    assert "**Challenge**" in rendered


def test_event_summary_counts():
    rendered = GroundingFormatter.format(make_analytics())
    assert "### Event Summary" in rendered
    assert "Pass" in rendered
    assert "Shot" in rendered


def test_pass_network_rendered():
    rendered = GroundingFormatter.format(make_analytics())
    assert "## Pass Networks" in rendered
    assert "#1 ↔ #3" in rendered


def test_empty_events_skips_section():
    rendered = GroundingFormatter.format(make_analytics(events=[]))
    assert "## Events Timeline" not in rendered


def test_no_possession_skips_section():
    rendered = GroundingFormatter.format(make_analytics(possession=None))
    assert "## Possession" not in rendered


def test_no_interaction_graphs_skips_section():
    rendered = GroundingFormatter.format(
        make_analytics(interaction_graph_team1=None, interaction_graph_team2=None)
    )
    assert "## Pass Networks" not in rendered


def test_pixel_only_label():
    rendered = GroundingFormatter.format(make_analytics(homography_available=False))
    assert "pixel-only" in rendered.lower()


def test_homography_available_label():
    rendered = GroundingFormatter.format(make_analytics(homography_available=True))
    assert "homography available" in rendered.lower()


def test_none_speed_renders_dash():
    data = make_analytics()
    data["player_kinematics"]["99"] = {
        "track_id": 99,
        "entity_type": "player",
        "team_id": 1,
        "total_distance_m": None,
        "total_distance_px": 1234.0,
        "avg_speed_m_per_sec": None,
        "max_speed_m_per_sec": None,
    }
    rendered = GroundingFormatter.format(data)
    assert "—" in rendered


def test_ball_distance_rendered():
    rendered = GroundingFormatter.format(make_analytics())
    assert "4200m" in rendered


def test_ball_direction_changes_rendered():
    rendered = GroundingFormatter.format(make_analytics())
    assert "47" in rendered


# ── TacticalAnalyzer ──────────────────────────────────────────────────────


def make_analyzer(stub_text="STUB"):
    return TacticalAnalyzer(provider=StubProvider(stub_text))


def test_analyze_returns_content():
    result = asyncio.run(make_analyzer("RESPONSE").analyze(make_analytics(), "match_overview"))
    assert result["content"] == "RESPONSE"


def test_analyze_returns_analysis_type():
    result = asyncio.run(make_analyzer().analyze(make_analytics(), "match_overview"))
    assert result["analysis_type"] == "match_overview"


def test_analyze_returns_grounding_data():
    result = asyncio.run(make_analyzer().analyze(make_analytics(), "match_overview"))
    gd = result["grounding_data"]
    assert gd["formatted_length"] > 0
    assert "provider" in gd
    assert gd["provider"] == "StubProvider"


def test_all_valid_types_accepted():
    for t in TacticalAnalyzer.VALID_TYPES:
        result = asyncio.run(make_analyzer().analyze(make_analytics(), t))
        assert result["analysis_type"] == t


def test_type_alias_match_summary():
    result = asyncio.run(make_analyzer().analyze(make_analytics(), "match_summary"))
    assert result["analysis_type"] == "match_overview"


def test_type_alias_tactical_analysis():
    result = asyncio.run(make_analyzer().analyze(make_analytics(), "tactical_analysis"))
    assert result["analysis_type"] == "tactical_deep_dive"


def test_invalid_type_raises():
    with pytest.raises(ValueError, match="Unknown analysis type"):
        asyncio.run(make_analyzer().analyze(make_analytics(), "not_a_real_type"))


def test_available_types_returns_four():
    types = TacticalAnalyzer.available_types()
    assert len(types) == 4
    ids = {t["id"] for t in types}
    assert ids == {"match_overview", "tactical_deep_dive", "event_analysis", "player_spotlight"}


# ── StubProvider ──────────────────────────────────────────────────────────


def test_stub_provider_is_available():
    assert StubProvider().is_available() is True


def test_stub_provider_returns_configured_text():
    result = asyncio.run(StubProvider("MY-RESPONSE").generate("sys", "user"))
    assert result == "MY-RESPONSE"


def test_stub_provider_accessible_via_factory():
    provider = get_provider("stub")
    assert isinstance(provider, StubProvider)


def test_get_provider_no_keys_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No LLM provider available"):
        get_provider("gemini")
