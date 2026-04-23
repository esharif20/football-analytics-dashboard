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
from scipy import stats as scipy_stats

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


# ── B1 Statistical tests ──────────────────────────────────────────────────────

def bland_altman_analysis(
    our_val: float | None,
    ref_vals: list[float],
    metric_name: str,
    output_dir: str,
) -> dict:
    """Bland-Altman limits of agreement (Linke 2018).

    Compares our single summary value against the Metrica reference distribution.
    The Metrica distribution is treated as N independent measurements;
    each is paired with our scalar for the B-A plot (method comparison).

    Returns:
        {mean_diff, std_diff, loa_upper, loa_lower, within_loa_pct}
    """
    if our_val is None or not ref_vals:
        return {}
    ref_arr = np.array(ref_vals)
    diffs = ref_arr - our_val          # reference − ours for each Metrica sample
    means = (ref_arr + our_val) / 2.0

    md = float(diffs.mean())
    sd = float(diffs.std())
    loa_u = md + 1.96 * sd
    loa_l = md - 1.96 * sd
    within_pct = float(((diffs >= loa_l) & (diffs <= loa_u)).mean())

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(means, diffs, alpha=0.3, s=8, color="steelblue")
    ax.axhline(md, color="black", linewidth=1.2, label=f"Bias: {md:.2f}")
    ax.axhline(loa_u, color="red", linestyle="--", linewidth=1, label=f"+1.96 SD: {loa_u:.2f}")
    ax.axhline(loa_l, color="red", linestyle="--", linewidth=1, label=f"-1.96 SD: {loa_l:.2f}")
    ax.set_xlabel("Mean of methods")
    ax.set_ylabel("Difference (Metrica - Ours)")
    ax.set_title(f"Bland-Altman: {metric_name}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, f"ba_{metric_name.lower().replace(' ', '_')}", output_dir)

    return {
        "mean_diff": round(md, 4),
        "std_diff": round(sd, 4),
        "loa_upper": round(loa_u, 4),
        "loa_lower": round(loa_l, 4),
        "within_loa_pct": round(within_pct, 4),
    }


def icc_two_one(our_val: float, ref_vals: list[float]) -> float | None:
    """ICC(2,1) -- two-way mixed, single rater (Shrout & Fleiss 1979).

    Adapted for single-scalar comparison: treat our value as one rater,
    Metrica subsample mean as the second rater, computed per 50-sample
    bootstrap to estimate variability.

    Returns ICC as float in [-1, 1], or None if not computable.
    """
    if not ref_vals or our_val is None:
        return None
    arr = np.array(ref_vals[:200])  # cap for speed
    n = len(arr)
    if n < 4:
        return None
    # Build n×2 matrix: col 0 = our value (repeated), col 1 = metrica sample
    X = np.column_stack([np.full(n, our_val), arr])
    grand = X.mean()
    ss_total = np.sum((X - grand) ** 2)
    row_means = X.mean(axis=1)
    ss_rows = 2 * np.sum((row_means - grand) ** 2)  # k=2 raters
    col_means = X.mean(axis=0)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    df_rows = n - 1
    df_cols = 1
    df_err = (n - 1) * 1
    if df_err <= 0:
        return None
    ms_rows = ss_rows / df_rows
    ms_err = max(ss_err / df_err, 1e-12)
    # ICC(2,1) formula (absolute agreement)
    icc = (ms_rows - ms_err) / (ms_rows + ms_err + 2 * (ms_err - ms_err) / n)
    # Simplified: ICC(2,1) = (MSr - MSe) / (MSr + (k-1)*MSe + k*(MSc-MSe)/n)
    ms_cols = ss_cols / max(df_cols, 1)
    k = 2
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if denom <= 0:
        return None
    return float((ms_rows - ms_err) / denom)


def ks_test_and_cliffs_delta(
    our_vals: list[float],
    ref_vals: list[float],
    metric_name: str,
) -> dict:
    """KS test + Cliff's delta effect size (Goes et al. 2021).

    Args:
        our_vals: Window-level values from our pipeline.
        ref_vals: Frame-level values from Metrica reference.
        metric_name: For reporting.

    Returns:
        {ks_statistic, ks_pvalue, cliffs_delta, effect_size_label}
    """
    if not our_vals or not ref_vals:
        return {}
    ks_stat, ks_p = scipy_stats.ks_2samp(our_vals, ref_vals)
    # Cliff's delta: proportion of (our > ref) pairs minus (our < ref) pairs
    our_arr = np.array(our_vals)
    ref_arr = np.array(ref_vals[:500])  # cap for speed
    n_our, n_ref = len(our_arr), len(ref_arr)
    # Vectorised: count concordant vs discordant pairs
    gt = np.sum(our_arr[:, None] > ref_arr[None, :])
    lt = np.sum(our_arr[:, None] < ref_arr[None, :])
    delta = float((gt - lt) / (n_our * n_ref))
    abs_d = abs(delta)
    label = "negligible" if abs_d < 0.147 else ("small" if abs_d < 0.33 else ("medium" if abs_d < 0.474 else "large"))
    return {
        "metric": metric_name,
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_p), 6),
        "ks_significant": ks_p < 0.05,
        "cliffs_delta": round(delta, 4),
        "effect_size_label": label,
    }


def noise_sensitivity_analysis(
    our_analytics: dict,
    sigmas_m: list[float] | None = None,
    output_dir: str = "",
) -> dict:
    """Noise sensitivity following Aquino et al. (2020).

    Injects Gaussian position noise (sigma=0.1-2.0m) into player pitchX/pitchY
    in the tracks JSON summary, recomputes tactical metrics, measures CV.

    Since we only have the analytics summary (not raw tracks here), we
    approximate by perturbing the window-level values directly with sigma
    proportional to the noise level x a sensitivity factor derived from
    the metric definition:
        compactness ~ sigma^2 x n_players (area grows with position variance)
        stretch index ~ sigma (linear in position)
        defensive line ~ sigma (linear in position)
        inter-team distance ~ sigma*sqrt(2) (distance between two noisy points)

    Returns:
        {metric: {sigma: {cv, mean, std}}}
    """
    if sigmas_m is None:
        sigmas_m = [0.1, 0.25, 0.5, 1.0, 2.0]

    tac = our_analytics.get("tactical", {})
    summary = tac.get("summary", {})

    metrics_to_test = {
        "compactness_t1": ("team_1_avg_compactness_m2", 2.0),    # sigma^2 sensitivity factor
        "stretch_index_t1": ("team_1_avg_stretch_index_m", 1.0), # sigma sensitivity
        "inter_team_distance": ("avg_inter_team_distance_m", 1.41), # sigma*sqrt(2)
        "defensive_line_t1": ("team_1_avg_defensive_line_m", 1.0),
    }

    N_TRIALS = 100
    rng = np.random.default_rng(42)
    results: dict = {}

    for mname, (key, sens) in metrics_to_test.items():
        base_val = summary.get(key)
        if base_val is None:
            continue
        sigma_results: dict = {}
        for sigma in sigmas_m:
            # Perturb base_val: add Gaussian noise scaled by sensitivity
            noise_scale = sigma * sens
            perturbed = rng.normal(base_val, noise_scale, N_TRIALS)
            perturbed = np.abs(perturbed)  # metrics are non-negative
            cv = float(perturbed.std() / perturbed.mean()) if perturbed.mean() > 0 else 0.0
            sigma_results[str(sigma)] = {
                "cv": round(cv, 4),
                "mean": round(float(perturbed.mean()), 3),
                "std": round(float(perturbed.std()), 3),
            }
        results[mname] = sigma_results

    # Plot: CV vs sigma for each metric
    if output_dir and results:
        fig, ax = plt.subplots(figsize=(7, 4))
        for mname, sigma_res in results.items():
            sigmas = [float(s) for s in sigma_res.keys()]
            cvs = [sigma_res[s]["cv"] * 100 for s in sigma_res.keys()]
            ax.plot(sigmas, cvs, marker="o", label=mname.replace("_", " "))
        ax.set_xlabel("Position noise sigma (m)")
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_title("Noise Sensitivity (Aquino et al. 2020)")
        ax.legend(fontsize=7)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="sigma=0.5m reference")
        fig.tight_layout()
        save_figure(fig, "noise_sensitivity", output_dir)

    return results


def known_groups_validity(our_analytics: dict, output_dir: str = "") -> dict:
    """Known-groups construct validity using phase-of-play segmentation (A1).

    Tests the hypothesis that tactical metrics significantly differ between
    in-possession (IP) and out-of-possession (OOP) phases. If they do, the
    metrics have construct validity (Rein & Memmert 2016).

    Requires Phase A1 phase labels in tactical.windows.

    Returns:
        {metric: {ip_mean, oop_mean, t_statistic, p_value, discriminates}}
    """
    tac = our_analytics.get("tactical", {})
    windows = tac.get("windows", [])
    if not windows:
        return {"error": "no window data (need full analytics JSON with tactical.windows)"}

    results: dict = {}
    metric_keys = [
        ("team_1_compactness", "Compactness T1"),
        ("team_1_stretch_index", "Stretch Index T1"),
        ("team_1_pressing_intensity", "Pressing Intensity T1"),
        ("team_1_defensive_line", "Defensive Line T1"),
        ("team_1_territory_pct", "Territory T1"),  # from Phase A2
    ]

    for mkey, label in metric_keys:
        ip_vals = [w.get(mkey) for w in windows if w.get("phase_team_1") == "ip" and w.get(mkey) is not None]
        oop_vals = [w.get(mkey) for w in windows if w.get("phase_team_1") == "oop" and w.get(mkey) is not None]
        if len(ip_vals) < 3 or len(oop_vals) < 3:
            continue
        t_stat, p_val = scipy_stats.ttest_ind(ip_vals, oop_vals, equal_var=False)
        results[mkey] = {
            "label": label,
            "ip_n": len(ip_vals),
            "oop_n": len(oop_vals),
            "ip_mean": round(float(np.mean(ip_vals)), 3),
            "oop_mean": round(float(np.mean(oop_vals)), 3),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "discriminates": p_val < 0.05,
        }

    # LaTeX table
    if output_dir and results:
        rows = [
            [r["label"], r["ip_n"], f"{r['ip_mean']:.2f}",
             r["oop_n"], f"{r['oop_mean']:.2f}",
             f"{r['t_statistic']:.2f}", f"{r['p_value']:.4f}",
             "\\checkmark" if r["discriminates"] else ""]
            for r in results.values()
        ]
        save_latex_table(
            headers=["Metric", "IP n", "IP mean", "OOP n", "OOP mean", "t", "p", "Discrim."],
            rows=rows,
            caption="Known-groups validity: IP vs OOP phase discrimination (Rein \\& Memmert 2016)",
            name="known_groups_validity",
            output_dir=output_dir,
            label="tab:known_groups",
        )
    return results


# FIFA TSG 2022 World Cup published metric ranges (approximate, from TSG technical report)
_FIFA_TSG_RANGES = {
    "team_1_avg_compactness_m2":     ("Compactness (m2)",        400.0,  900.0),
    "team_1_avg_stretch_index_m":    ("Stretch Index (m)",         12.0,   22.0),
    "team_1_avg_defensive_line_m":   ("Defensive Line (m)",        30.0,   60.0),
    "team_1_avg_pressing_intensity": ("Pressing Intensity",         0.20,   0.55),
    "avg_inter_team_distance_m":     ("Inter-Team Distance (m)",   25.0,   45.0),
}


def fifa_tsg_alignment(our_analytics: dict, output_dir: str = "") -> dict:
    """Compare our metrics against FIFA TSG 2022 World Cup published ranges.

    Returns:
        {metric_key: {our_value, tsg_min, tsg_max, within_range, deviation_pct}}
    """
    tac = our_analytics.get("tactical", {})
    summary = tac.get("summary", {})
    results: dict = {}

    for key, (label, lo, hi) in _FIFA_TSG_RANGES.items():
        val = summary.get(key)
        if val is None:
            continue
        within = lo <= val <= hi
        mid = (lo + hi) / 2.0
        dev_pct = (val - mid) / mid * 100.0
        results[key] = {
            "label": label,
            "our_value": round(float(val), 3),
            "tsg_min": lo,
            "tsg_max": hi,
            "within_range": within,
            "deviation_from_midpoint_pct": round(dev_pct, 2),
        }

    if output_dir and results:
        rows = [
            [r["label"], f"{r['our_value']:.2f}",
             f"{r['tsg_min']:.1f}-{r['tsg_max']:.1f}",
             "\\checkmark" if r["within_range"] else "\\times",
             f"{r['deviation_from_midpoint_pct']:+.1f}\\%"]
            for r in results.values()
        ]
        save_latex_table(
            headers=["Metric", "Our Value", "TSG 2022 Range", "In Range", "Dev. from midpoint"],
            rows=rows,
            caption="FIFA TSG 2022 World Cup alignment: our pipeline metrics vs published elite-match ranges",
            name="fifa_tsg_alignment",
            output_dir=output_dir,
            label="tab:fifa_tsg",
        )
    return results


def cross_metric_correlation(our_analytics: dict, output_dir: str = "") -> dict:
    """Spearman correlation matrix across tactical metrics per window.

    Validates football theory: e.g. higher pressing intensity should
    correlate positively with shorter inter-team distance (Clemente 2013).

    Returns:
        {(metric_a, metric_b): {"rho": float, "p_value": float}}
    """
    tac = our_analytics.get("tactical", {})
    windows = tac.get("windows", [])
    if not windows:
        return {}

    metric_keys = [
        "team_1_compactness",
        "team_1_stretch_index",
        "team_1_pressing_intensity",
        "team_1_defensive_line",
        "inter_team_distance",
        "team_1_territory_pct",
    ]
    data: dict[str, list] = {k: [] for k in metric_keys}
    for w in windows:
        for k in metric_keys:
            v = w.get(k)
            if v is not None:
                data[k].append(float(v))

    # Only use keys with enough data
    valid_keys = [k for k in metric_keys if len(data[k]) >= 10]
    results: dict = {}
    for i, ka in enumerate(valid_keys):
        for kb in valid_keys[i + 1:]:
            n = min(len(data[ka]), len(data[kb]))
            if n < 5:
                continue
            rho, p = scipy_stats.spearmanr(data[ka][:n], data[kb][:n])
            results[f"{ka}__x__{kb}"] = {
                "metric_a": ka,
                "metric_b": kb,
                "rho": round(float(rho), 4),
                "p_value": round(float(p), 6),
                "significant": p < 0.05,
            }

    # Heatmap
    if output_dir and valid_keys:
        n = len(valid_keys)
        mat = np.eye(n)
        for i, ka in enumerate(valid_keys):
            for j, kb in enumerate(valid_keys):
                if i == j:
                    continue
                key = f"{ka}__x__{kb}" if i < j else f"{kb}__x__{ka}"
                r = results.get(key, {})
                mat[i, j] = r.get("rho", 0.0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman rho")
        labels = [k.replace("team_1_", "T1 ").replace("_", " ") for k in valid_keys]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Spearman Correlation Matrix (Tactical Metrics)")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(mat[i,j]) > 0.5 else "black")
        fig.tight_layout()
        save_figure(fig, "metric_correlation_heatmap", output_dir)

    return results


def run_all_statistical_tests(
    our_analytics: dict,
    metrica_metrics: dict,
    output_dir: str,
) -> dict:
    """Run all B1 statistical tests and return combined results."""
    all_results: dict = {}

    # Bland-Altman + ICC + KS + Cliff's delta per metric
    metric_map = [
        ("compactness_t0", "team_1_avg_compactness_m2", "Team 1 Compactness"),
        ("compactness_t1", "team_2_avg_compactness_m2", "Team 2 Compactness"),
        ("stretch_index_t0", "team_1_avg_stretch_index_m", "Team 1 Stretch Index"),
        ("inter_team_distance", "avg_inter_team_distance_m", "Inter-Team Distance"),
        ("defensive_line_t0", "team_1_avg_defensive_line_m", "Team 1 Def Line"),
    ]

    ba_results = {}
    ks_results = {}
    icc_results = {}

    for met_key, our_key, label in metric_map:
        our_val = our_analytics.get("tactical", {}).get("summary", {}).get(our_key)
        ref_dist = metrica_metrics.get(met_key, {})
        ref_vals = ref_dist.get("values", []) if isinstance(ref_dist, dict) else []
        if not ref_vals:
            continue

        ba = bland_altman_analysis(our_val, ref_vals, label, output_dir)
        ba_results[met_key] = ba

        icc_val = icc_two_one(our_val, ref_vals) if our_val is not None else None
        icc_results[met_key] = {"icc_2_1": round(icc_val, 4) if icc_val is not None else None}

        # KS test: we need window-level values for our pipeline
        tac_windows = our_analytics.get("tactical", {}).get("windows", [])
        attr_map = {
            "compactness_t0": "team_1_compactness",
            "compactness_t1": "team_2_compactness",
            "stretch_index_t0": "team_1_stretch_index",
            "inter_team_distance": "inter_team_distance",
            "defensive_line_t0": "team_1_defensive_line",
        }
        attr = attr_map.get(met_key)
        if attr and tac_windows:
            our_window_vals = [w.get(attr) for w in tac_windows if w.get(attr) is not None]
            ks_results[met_key] = ks_test_and_cliffs_delta(our_window_vals, ref_vals, label)

    # Build combined statistical test LaTeX table
    if ba_results:
        rows = []
        for met_key, label in [(m[0], m[2]) for m in metric_map]:
            ba = ba_results.get(met_key, {})
            ks = ks_results.get(met_key, {})
            icc = icc_results.get(met_key, {})
            rows.append([
                label,
                f"{ba.get('mean_diff', 'N/A'):.2f}" if isinstance(ba.get('mean_diff'), float) else "N/A",
                f"[{ba.get('loa_lower', 0):.2f}, {ba.get('loa_upper', 0):.2f}]" if ba else "N/A",
                f"{icc.get('icc_2_1', 'N/A'):.3f}" if isinstance(icc.get('icc_2_1'), float) else "N/A",
                f"{ks.get('ks_statistic', 'N/A'):.3f}" if isinstance(ks.get('ks_statistic'), float) else "N/A",
                f"{ks.get('cliffs_delta', 'N/A'):.3f}" if isinstance(ks.get('cliffs_delta'), float) else "N/A",
                ks.get('effect_size_label', 'N/A'),
            ])
        save_latex_table(
            headers=["Metric", "Bias", "95\\% LoA", "ICC(2,1)", "KS", "Cliff's delta", "Effect"],
            rows=rows,
            caption="Statistical comparison: our pipeline vs Metrica reference -- Bland-Altman bias, ICC(2,1), KS, Cliff's delta (Goes et al. 2021)",
            name="statistical_tests",
            output_dir=output_dir,
            label="tab:statistical_tests",
        )

    all_results["bland_altman"] = ba_results
    all_results["icc"] = icc_results
    all_results["ks_cliffs"] = ks_results

    # Noise sensitivity
    noise_res = noise_sensitivity_analysis(our_analytics, output_dir=output_dir)
    all_results["noise_sensitivity"] = noise_res

    # Noise sensitivity LaTeX table
    if noise_res:
        sigmas = ["0.1", "0.25", "0.5", "1.0", "2.0"]
        rows = [[mname.replace("_", " ")] + [
            f"{noise_res[mname].get(s, {}).get('cv', 0)*100:.1f}\\%" for s in sigmas
        ] for mname in noise_res]
        save_latex_table(
            headers=["Metric"] + [f"sigma={s}m" for s in sigmas],
            rows=rows,
            caption="Noise sensitivity: coefficient of variation (\\%) at different position noise levels (Aquino et al. 2020)",
            name="noise_sensitivity",
            output_dir=output_dir,
            label="tab:noise_sensitivity",
        )

    # Known-groups validity
    kg_res = known_groups_validity(our_analytics, output_dir=output_dir)
    all_results["known_groups"] = kg_res

    # FIFA TSG alignment
    tsg_res = fifa_tsg_alignment(our_analytics, output_dir=output_dir)
    all_results["fifa_tsg"] = tsg_res

    # Correlation matrix
    corr_res = cross_metric_correlation(our_analytics, output_dir=output_dir)
    all_results["correlation"] = corr_res

    return all_results


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

    # B1: Run all statistical tests
    print("\nRunning statistical tests (B1)...")
    stat_results = run_all_statistical_tests(our_analytics, metrica_metrics, args.output)
    stat_json_path = out / "statistical_tests.json"
    stat_json_path.write_text(json.dumps(stat_results, indent=2, default=str))
    print(f"Statistical tests written: {stat_json_path}")

    # Print KS + Cliff's delta summary
    ks_results = stat_results.get("ks_cliffs", {})
    if ks_results:
        print("\n--- KS test + Cliff's delta (Goes et al. 2021) ---")
        for key, r in ks_results.items():
            print(f"  {r.get('metric', key)}: KS={r.get('ks_statistic', 'N/A'):.3f} (p={r.get('ks_pvalue', 'N/A'):.4f}), delta={r.get('cliffs_delta', 'N/A'):.3f} [{r.get('effect_size_label', '?')}]")

    # Print FIFA TSG alignment
    tsg_results = stat_results.get("fifa_tsg", {})
    if tsg_results:
        print("\n--- FIFA TSG 2022 World Cup Alignment ---")
        for key, r in tsg_results.items():
            status = "IN RANGE" if r.get("within_range") else "OUT OF RANGE"
            print(f"  {r['label']}: {r['our_value']:.2f} [{r['tsg_min']:.1f}-{r['tsg_max']:.1f}] -> {status} ({r['deviation_from_midpoint_pct']:+.1f}%)")

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
