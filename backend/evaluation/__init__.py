"""Dissertation evaluation scripts for the football analytics pipeline.

Modules:
    _common          -- Shared loaders, table formatters, figure helpers
    tracking_quality -- Proxy tracking metrics (no GT needed)
    team_classification -- Team assignment accuracy (requires manual annotations)
    homography_error -- Reprojection error (requires manual landmark annotations)
    llm_grounding    -- LLM commentary grounding rate + format comparison
    vlm_comparison   -- Text-only vs text+vision grounding A/B test
"""
