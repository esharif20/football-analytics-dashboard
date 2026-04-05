"""Tactical metric cross-validation against Metrica Sports professional tracking data.

Downloads Metrica Sports sample data (3 professional matches with sub-cm accuracy),
computes the same tactical metrics as our pipeline, and compares distributions.

Usage:
    python3 -m evaluation.tactical_validation \
        --our-analytics ../eval_output/phase12/10_analytics.json \
        --output ../eval_output/phase16/tactical/
        [--metrica-dir ../eval_output/phase16/metrica/]  # auto-downloads if not present
"""

import argparse
import json
import logging
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Add backend/ to sys.path for relative imports when run as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation._common import (
    ensure_output_dir,
    load_analytics,
    save_figure,
    save_latex_table,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Metrica Sports sample data URLs ──────────────────────────────────────────

_METRICA_BASE = (
    "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
)

_METRICA_FILES = {
    "game1_home": f"{_METRICA_BASE}/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "game1_away": f"{_METRICA_BASE}/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
}

# Pitch dimensions used by Metrica (standard FIFA)
_PITCH_LENGTH_M = 105.0
_PITCH_WIDTH_M = 68.0

# Minimum players for convex hull (scipy requires >= 3)
_MIN_HULL_PLAYERS = 3


# ── Download ──────────────────────────────────────────────────────────────────


def download_metrica_data(local_dir: str) -> dict[str, Path]:
    """Download Metrica Sports sample CSV files if not already present.

    Args:
        local_dir: Directory to save downloaded files.

    Returns:
        Dict mapping file key to local Path.
    """
    out = ensure_output_dir(local_dir)
    log_path = out / "metrica_download.log"
    local_paths: dict[str, Path] = {}
    log_lines: list[str] = []

    for key, url in _METRICA_FILES.items():
        dest = out / f"{key}.csv"
        local_paths[key] = dest

        if dest.exists():
            log_lines.append(f"SKIP {key}: already at {dest}")
            log.info("Skipping %s (already downloaded)", key)
            continue

        log.info("Downloading %s from %s ...", key, url)
        try:
            urllib.request.urlretrieve(url, dest)
            log_lines.append(f"OK {key}: {url} -> {dest}")
            log.info("Saved %s to %s", key, dest)
        except Exception as exc:
            log_lines.append(f"FAIL {key}: {exc}")
            log.error("Failed to download %s: %s", key, exc)
            raise RuntimeError(f"Could not download Metrica data: {exc}") from exc

    log_path.write_text("\n".join(log_lines) + "\n")
    return local_paths


# ── CSV parsing ───────────────────────────────────────────────────────────────


def parse_metrica_csv(
    home_path: str | Path,
    away_path: str | Path,
) -> list[dict]:
    """Parse Metrica Sports tracking CSV into our per-frame dict format.

    Metrica format:
        Row 0: player labels (blank, blank, PlayerXX, PlayerXX, ...)
        Row 1: column headers (Period, Frame, x, y, x, y, ...)
        Row 2+: data rows

    Coordinates are normalised [0,1]; we scale X by 105m and Y by 68m.

    Returns:
        List of frame dicts with keys:
            frameNumber, playerPositions (dict[str, {x, y, teamId}])
    """
    def _read_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if len(lines) < 3:
            raise ValueError(f"Unexpected Metrica CSV format in {path}")
        player_row = [c.strip() for c in lines[0].split(",")]
        header_row = [c.strip() for c in lines[1].split(",")]
        data_rows = [line.split(",") for line in lines[2:] if line.strip()]
        return player_row, header_row, data_rows

    def _build_player_columns(player_row: list[str], header_row: list[str]) -> list[tuple[str, int, int]]:
        """Return list of (player_id, x_col_idx, y_col_idx)."""
        players = []
        i = 0
        while i < len(player_row):
            pid = player_row[i]
            if pid and pid.lower() not in ("period", "frame", ""):
                # Expect x at i, y at i+1
                if i + 1 < len(header_row):
                    players.append((pid, i, i + 1))
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return players

    home_player_row, home_header, home_data = _read_csv(home_path)
    away_player_row, away_header, away_data = _read_csv(away_path)

    home_players = _build_player_columns(home_player_row, home_header)
    away_players = _build_player_columns(away_player_row, away_header)

    # Find Period (col 0) and Frame (col 1) in data rows
    # Build frame -> {player_id: (x_m, y_m)} lookup
    frame_lookup: dict[int, dict[str, dict]] = {}

    def _parse_team(data_rows: list[list[str]], player_cols: list[tuple[str, int, int]], team_id: int) -> None:
        for row in data_rows:
            if len(row) < 2:
                continue
            try:
                frame = int(row[1])
            except (ValueError, IndexError):
                continue
            if frame not in frame_lookup:
                frame_lookup[frame] = {}
            for pid, xi, yi in player_cols:
                try:
                    xv = row[xi].strip()
                    yv = row[yi].strip()
                    if not xv or not yv or xv.lower() == "nan" or yv.lower() == "nan":
                        continue
                    x_m = float(xv) * _PITCH_LENGTH_M
                    y_m = float(yv) * _PITCH_WIDTH_M
                    if math.isnan(x_m) or math.isnan(y_m):
                        continue
                    unique_id = f"t{team_id}_{pid}"
                    frame_lookup[frame][unique_id] = {
                        "x": x_m,
                        "y": y_m,
                        "teamId": team_id,
                        "pitchX": x_m,
                        "pitchY": y_m,
                    }
                except (ValueError, IndexError):
                    continue

    _parse_team(home_data, home_players, team_id=0)
    _parse_team(away_data, away_players, team_id=1)

    frames = []
    for frame_num in sorted(frame_lookup.keys()):
        frames.append({
            "frameNumber": frame_num,
            "playerPositions": frame_lookup[frame_num],
        })

    log.info("Parsed %d frames from Metrica CSVs", len(frames))
    return frames


# ── Tactical metric computation ───────────────────────────────────────────────


def _team_points(frame: dict, team_id: int) -> np.ndarray:
    """Return Nx2 array of (pitchX, pitchY) for the given team in this frame."""
    pts = []
    for pos in (frame.get("playerPositions") or {}).values():
        if pos.get("teamId") != team_id:
            continue
        px = pos.get("pitchX") if pos.get("pitchX") is not None else pos.get("x")
        py = pos.get("pitchY") if pos.get("pitchY") is not None else pos.get("y")
        if px is not None and py is not None:
            pts.append([float(px), float(py)])
    return np.array(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def _convex_hull_area(pts: np.ndarray) -> float | None:
    """Return convex hull area in m² (or None if too few points)."""
    if len(pts) < _MIN_HULL_PLAYERS:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return float(hull.volume)  # volume = area in 2D
    except Exception:
        return None


def _stretch_index(pts: np.ndarray) -> float | None:
    """Mean distance from centroid in metres."""
    if len(pts) < 2:
        return None
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    return float(dists.mean())


def _inter_team_distance(pts0: np.ndarray, pts1: np.ndarray) -> float | None:
    """Euclidean distance between team centroids in metres."""
    if len(pts0) < 1 or len(pts1) < 1:
        return None
    c0 = pts0.mean(axis=0)
    c1 = pts1.mean(axis=0)
    return float(np.linalg.norm(c0 - c1))


def _possession_fraction(frames: list[dict], team_id: int) -> float:
    """Fraction of frames where the given team has more players in opponent's half."""
    count = 0
    valid = 0
    half_line = _PITCH_LENGTH_M / 2.0
    for frame in frames:
        pp = frame.get("playerPositions") or {}
        team_pts = _team_points(frame, team_id)
        if len(team_pts) == 0:
            continue
        valid += 1
        # "opponent's half" for team 0 is x > half_line; for team 1 is x < half_line
        if team_id == 0:
            in_opp_half = np.sum(team_pts[:, 0] > half_line)
        else:
            in_opp_half = np.sum(team_pts[:, 0] < half_line)
        # Compare with the other team
        other_id = 1 - team_id
        other_pts = _team_points(frame, other_id)
        if team_id == 0:
            other_in_opp_half = np.sum(other_pts[:, 0] > half_line) if len(other_pts) > 0 else 0
        else:
            other_in_opp_half = np.sum(other_pts[:, 0] < half_line) if len(other_pts) > 0 else 0
        if in_opp_half > other_in_opp_half:
            count += 1
    return count / valid if valid > 0 else 0.0


def _defensive_line_height(pts: np.ndarray, attack_direction: str = "right") -> float | None:
    """Mean X of the 4 deepest defenders (lowest X for team attacking right → defending left).

    Args:
        pts: Player positions (pitchX, pitchY) for the defending team.
        attack_direction: 'right' means team attacks toward high X (standard).
    """
    if len(pts) < 4:
        return None
    # Sort by X ascending; deepest defenders have smallest X (closest to own goal)
    x_vals = pts[:, 0]
    sorted_x = np.sort(x_vals)
    return float(sorted_x[:4].mean())


def compute_tactical_from_metrica(frames: list[dict]) -> dict[str, Any]:
    """Compute our 5 tactical metrics from Metrica-format parsed frames.

    Metrics:
        compactness_t0/t1: Mean convex hull area in m²
        stretch_index_t0/t1: Mean stretch index in m
        inter_team_distance: Mean distance between team centroids in m
        possession_fraction_t0/t1: Fraction of frames with territorial advantage
        defensive_line_t0/t1: Mean Y of 4 deepest defenders in m

    Returns:
        Dict with per-metric lists (for distribution) and summary statistics.
    """
    compactness = {0: [], 1: []}
    stretch = {0: [], 1: []}
    inter_dists: list[float] = []
    def_lines = {0: [], 1: []}

    for frame in frames:
        for tid in (0, 1):
            pts = _team_points(frame, tid)
            a = _convex_hull_area(pts)
            if a is not None:
                compactness[tid].append(a)
            s = _stretch_index(pts)
            if s is not None:
                stretch[tid].append(s)
            dl = _defensive_line_height(pts)
            if dl is not None:
                def_lines[tid].append(dl)

        pts0 = _team_points(frame, 0)
        pts1 = _team_points(frame, 1)
        d = _inter_team_distance(pts0, pts1)
        if d is not None:
            inter_dists.append(d)

    poss0 = _possession_fraction(frames, 0)
    poss1 = _possession_fraction(frames, 1)

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "median": None, "p25": None, "p75": None, "std": None}
        arr = np.array(vals)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "std": float(arr.std()),
            "values": vals,
        }

    return {
        "compactness_t0": _stats(compactness[0]),
        "compactness_t1": _stats(compactness[1]),
        "stretch_index_t0": _stats(stretch[0]),
        "stretch_index_t1": _stats(stretch[1]),
        "inter_team_distance": _stats(inter_dists),
        "possession_fraction_t0": poss0,
        "possession_fraction_t1": poss1,
        "defensive_line_t0": _stats(def_lines[0]),
        "defensive_line_t1": _stats(def_lines[1]),
    }


# ── Comparison ────────────────────────────────────────────────────────────────


def compare_distributions(
    our_analytics: dict,
    metrica_metrics: dict,
) -> dict[str, Any]:
    """For each metric, compare our value to the Metrica P25-P75 range.

    Returns:
        Dict mapping metric_key -> {our_value, metrica_p25, metrica_median, metrica_p75, verdict}
    """
    tac = our_analytics.get("tactical", {})
    summary = tac.get("summary", {})

    our_values = {
        "compactness_t0": summary.get("team_1_avg_compactness_m2"),
        "compactness_t1": summary.get("team_2_avg_compactness_m2"),
        "stretch_index_t0": summary.get("team_1_avg_stretch_index_m"),
        "stretch_index_t1": summary.get("team_2_avg_stretch_index_m"),
        "inter_team_distance": summary.get("avg_inter_team_distance_m"),
        "defensive_line_t0": summary.get("team_1_avg_defensive_line_m"),
        "defensive_line_t1": summary.get("team_2_avg_defensive_line_m"),
    }

    results = {}
    for key, our_val in our_values.items():
        metrica = metrica_metrics.get(key, {})
        if isinstance(metrica, dict):
            p25 = metrica.get("p25")
            med = metrica.get("median")
            p75 = metrica.get("p75")
        else:
            p25 = med = p75 = None

        verdict = "N/A"
        if our_val is not None and p25 is not None and p75 is not None:
            if p25 <= our_val <= p75:
                verdict = "in range"
            elif our_val > p75:
                verdict = "above"
            else:
                verdict = "below"

        results[key] = {
            "our_value": our_val,
            "metrica_p25": p25,
            "metrica_median": med,
            "metrica_p75": p75,
            "verdict": verdict,
        }

    # Possession fractions are scalars
    for key in ("possession_fraction_t0", "possession_fraction_t1"):
        our_val = None
        poss = our_analytics.get("possession", {})
        if key == "possession_fraction_t0":
            pct = poss.get("team_1_percentage")
            our_val = pct / 100.0 if pct is not None else None
        else:
            pct = poss.get("team_2_percentage")
            our_val = pct / 100.0 if pct is not None else None

        metrica_val = metrica_metrics.get(key)
        # Scalar vs scalar comparison
        p25 = p75 = med = None
        if isinstance(metrica_val, (int, float)):
            med = metrica_val
        verdict = "N/A"
        if our_val is not None and med is not None:
            verdict = "in range" if abs(our_val - med) < 0.1 else ("above" if our_val > med else "below")

        results[key] = {
            "our_value": our_val,
            "metrica_p25": p25,
            "metrica_median": med,
            "metrica_p75": p75,
            "verdict": verdict,
        }

    return results


def build_comparison_table(comparison: dict, output_dir: str) -> None:
    """LaTeX table: metric, our value, Metrica P25-P75 range, verdict."""
    _METRIC_LABELS = {
        "compactness_t0": "Team 1 compactness (m²)",
        "compactness_t1": "Team 2 compactness (m²)",
        "stretch_index_t0": "Team 1 stretch index (m)",
        "stretch_index_t1": "Team 2 stretch index (m)",
        "inter_team_distance": "Inter-team distance (m)",
        "possession_fraction_t0": "Team 1 possession fraction",
        "possession_fraction_t1": "Team 2 possession fraction",
        "defensive_line_t0": "Team 1 defensive line (m)",
        "defensive_line_t1": "Team 2 defensive line (m)",
    }

    _VERDICT_TEX = {
        "in range": "\\checkmark in range",
        "above": "$\\uparrow$ above",
        "below": "$\\downarrow$ below",
        "N/A": "N/A",
    }

    rows = []
    for key, label in _METRIC_LABELS.items():
        c = comparison.get(key, {})
        ov = c.get("our_value")
        p25 = c.get("metrica_p25")
        p75 = c.get("metrica_p75")
        med = c.get("metrica_median")
        verdict = c.get("verdict", "N/A")

        our_str = f"{ov:.2f}" if ov is not None else "N/A"
        if p25 is not None and p75 is not None:
            range_str = f"{p25:.2f}--{p75:.2f} (med {med:.2f})" if med is not None else f"{p25:.2f}--{p75:.2f}"
        elif med is not None:
            range_str = f"{med:.2f}"
        else:
            range_str = "N/A"
        verdict_str = _VERDICT_TEX.get(verdict, verdict)
        rows.append([label, our_str, range_str, verdict_str])

    save_latex_table(
        headers=["Metric", "Our value", "Metrica P25--P75", "Verdict"],
        rows=rows,
        caption="Tactical metric comparison: our pipeline vs Metrica Sports professional tracking data",
        name="tactical_comparison",
        output_dir=output_dir,
        label="tab:tactical_comparison",
    )


def plot_distributions(
    our_analytics: dict,
    metrica_metrics: dict,
    output_dir: str,
) -> None:
    """One subplot per metric: histogram of Metrica values, vertical line for our value."""
    tac = our_analytics.get("tactical", {})
    summary = tac.get("summary", {})

    poss = our_analytics.get("possession", {})

    plot_specs = [
        ("compactness_t0", "Team 1 Compactness (m²)", summary.get("team_1_avg_compactness_m2")),
        ("compactness_t1", "Team 2 Compactness (m²)", summary.get("team_2_avg_compactness_m2")),
        ("stretch_index_t0", "Team 1 Stretch Index (m)", summary.get("team_1_avg_stretch_index_m")),
        ("stretch_index_t1", "Team 2 Stretch Index (m)", summary.get("team_2_avg_stretch_index_m")),
        ("inter_team_distance", "Inter-Team Distance (m)", summary.get("avg_inter_team_distance_m")),
    ]

    n = len(plot_specs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (key, label, our_val) in zip(axes, plot_specs):
        metrica = metrica_metrics.get(key, {})
        values = metrica.get("values", []) if isinstance(metrica, dict) else []

        if values:
            ax.hist(values, bins=30, color="steelblue", edgecolor="white", alpha=0.8, label="Metrica")
        else:
            ax.text(0.5, 0.5, "No Metrica data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)

        if our_val is not None:
            ax.axvline(our_val, color="red", linestyle="--", linewidth=2.0, label=f"Ours: {our_val:.1f}")

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    fig.suptitle("Tactical Metric Distributions: Metrica vs Our Pipeline", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "tactical_distributions", output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tactical metric cross-validation against Metrica Sports data"
    )
    parser.add_argument(
        "--our-analytics",
        required=True,
        help="Path to our pipeline analytics JSON (export_analytics_json output)",
    )
    parser.add_argument(
        "--output",
        default="../eval_output/phase16/tactical/",
        help="Output directory for LaTeX tables and figures",
    )
    parser.add_argument(
        "--metrica-dir",
        default=None,
        help="Local directory with Metrica CSV files (auto-downloads if not present)",
    )
    args = parser.parse_args()

    out = ensure_output_dir(args.output)

    # Resolve Metrica directory
    metrica_dir = args.metrica_dir or str(Path(args.output) / "metrica")

    print(f"Loading our analytics from: {args.our_analytics}")
    our_analytics = load_analytics(args.our_analytics)

    # Download Metrica data if needed
    print(f"Checking Metrica data in: {metrica_dir}")
    local_paths = download_metrica_data(metrica_dir)

    # Parse Metrica CSVs
    print("Parsing Metrica tracking data...")
    metrica_frames = parse_metrica_csv(
        local_paths["game1_home"],
        local_paths["game1_away"],
    )
    print(f"Parsed {len(metrica_frames)} Metrica frames")

    # Compute tactical metrics from Metrica
    print("Computing tactical metrics from Metrica data...")
    metrica_metrics = compute_tactical_from_metrica(metrica_frames)

    # Compare distributions
    print("Comparing distributions...")
    comparison = compare_distributions(our_analytics, metrica_metrics)

    # Write comparison JSON
    comparison_json_path = out / "tactical_comparison.json"
    # Strip large 'values' lists before writing comparison JSON (keep only stats)
    comparison_json_path.write_text(json.dumps(comparison, indent=2))
    print(f"Comparison JSON written: {comparison_json_path}")

    # LaTeX table
    build_comparison_table(comparison, args.output)
    print(f"LaTeX table written: {out / 'tactical_comparison.tex'}")

    # Figures
    plot_distributions(our_analytics, metrica_metrics, args.output)
    print(f"Distributions figure written: {out / 'tactical_distributions.pdf'}")

    # Print summary to stdout
    print("\n--- Tactical Comparison Summary ---")
    for key, result in comparison.items():
        ov = result.get("our_value")
        verdict = result.get("verdict", "N/A")
        p25 = result.get("metrica_p25")
        p75 = result.get("metrica_p75")
        range_str = f"[{p25:.2f}, {p75:.2f}]" if p25 is not None and p75 is not None else "N/A"
        ov_str = f"{ov:.3f}" if ov is not None else "N/A"
        print(f"  {key}: ours={ov_str}, Metrica range={range_str}, verdict={verdict}")


if __name__ == "__main__":
    main()
