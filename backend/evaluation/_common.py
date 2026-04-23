"""Shared utilities for dissertation evaluation scripts."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script usage
import matplotlib.pyplot as plt


# ── Data loaders ────────────────────────────────────────────────────────────


def load_analytics(path: str) -> dict:
    """Load pipeline analytics JSON.

    This is the same JSON produced by export_analytics_json() and consumed
    by GroundingFormatter.format() in services/tactical.py.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_db_ground_truth(path: str) -> dict:
    """Load per-frame DB ground-truth JSON (from db_extractor.py).

    Structure:
        {
            "analysis_id": int,
            "analytics": {...},          # same as load_analytics()
            "per_frame": {"frames": [...], "total_frames": int, "fps": float},
            "events_db": [{type, frame_number, timestamp, player_id, team_id,
                           start_x, start_y, end_x, end_y, success}, ...],
            "statistics": {...},
            "frame_metrics": {
                "inter_team_distance_m": [...],
                "ball_speed_m_per_s": [...],
                "team_1_centroid": [...],
                "team_2_centroid": [...],
                "compactness": {...},
                "zone_occupancy": {...},
            },
            "formations": {"team_1": {...}, "team_2": {...}},
        }
    """
    with open(path, "r") as f:
        return json.load(f)


def load_tracks(path: str) -> list[dict]:
    """Load per-frame tracks JSON (from export_tracks_json())."""
    with open(path, "r") as f:
        return json.load(f)


def load_homography_matrices(path: str) -> dict[int, list]:
    """Load per-frame homography matrices (from pipeline export).

    Returns dict: {frame_idx -> 3x3 matrix as nested list}
    """
    with open(path, "r") as f:
        raw = json.load(f)
    # Keys are strings in JSON — convert to int
    return {int(k): v for k, v in raw.items()}


# ── Output helpers ───────────────────────────────────────────────────────────


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, name: str, output_dir: str) -> None:
    """Save figure as both PDF (for LaTeX) and PNG (for preview)."""
    out = ensure_output_dir(output_dir)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def latex_table(
    headers: list[str],
    rows: list[list[Any]],
    caption: str,
    label: str = "",
) -> str:
    """Produce a LaTeX tabular environment string.

    Usage: paste directly into \\begin{table} ... \\end{table}.
    """
    n = len(headers)
    col_spec = " | ".join(["l"] + ["r"] * (n - 1))
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(str(v) for v in row) + " \\\\")
    lines += [
        "\\hline",
        "\\end{tabular}",
    ]
    if caption:
        label_str = f"\\label{{{label}}}" if label else ""
        lines.append(f"% \\caption{{{caption}}} {label_str}")
    return "\n".join(lines)


def save_latex_table(
    headers: list[str],
    rows: list[list[Any]],
    caption: str,
    name: str,
    output_dir: str,
    label: str = "",
) -> None:
    out = ensure_output_dir(output_dir)
    table_str = latex_table(headers, rows, caption, label)
    (out / f"{name}.tex").write_text(table_str)


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    """Container for evaluation script paths and options."""
    analytics_path: str = ""
    ground_truth_path: str = ""       # DB ground truth from db_extractor.py
    tracks_path: str = ""
    homography_path: str = ""
    video_path: str = ""
    annotations_path: str = ""
    output_dir: str = "eval_output"
    provider: str = "gemini"

    @classmethod
    def from_args(cls, args) -> "EvalConfig":
        return cls(**{
            k: v for k, v in vars(args).items()
            if k in cls.__dataclass_fields__ and v is not None
        })
