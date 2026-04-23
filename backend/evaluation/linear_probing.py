"""Linear probing study: do LLMs encode football temporal patterns beyond what prompting extracts?

Implements the Schumacher et al. (2026) §4 probing methodology for football-specific
time-series classification tasks. Tests whether a frozen LLM's hidden states contain
more discriminative information about tactical patterns than zero-shot prompting extracts.

Core hypothesis: LLMs internally represent pressing type, formation compactness, and
territorial patterns — but prompting is a poor extraction interface. Linear probes trained
on hidden states should significantly outperform zero-shot prompting on the same tasks,
confirming the representation gap that explains why visual charts and wordalisation help.

Reference:
    Schumacher, Nourbakhsh, Slavin, Rios (2026) — "Prompting Underestimates LLM Capability
    for Time Series Classification" (arXiv:2601.03464v2)

Usage (on RunPod with Llama-3.2-11B-Vision-Instruct):
    python3 -m backend.evaluation.linear_probing \\
        --ground-truth eval_output/dissertation/db_grounded/18_db_ground_truth.json \\
        --model-path /workspace/models/llama-3.2-11b-vision/ \\
        --modalities d v \\
        --output eval_output/dissertation/probing/

Requires:
    pip install transformers torch scikit-learn matplotlib numpy
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import ensure_output_dir, load_db_ground_truth, save_figure

logger = logging.getLogger(__name__)

# Per-sample auxiliary data that the basic (series, label) tuple can't carry.
# Keyed by (task_name, sample_index) so the chart renderer can look up 2D
# coordinates for the pitch-overlay modality without changing every function
# signature in the pipeline.
_AUX_DATA_CACHE: "dict[tuple[str, int], dict]" = {}

# ── Task definitions ──────────────────────────────────────────────────────────

CLASSIFICATION_TASKS = {
    "pressing_type": {
        "description": "Classify the pressing style from inter-team distance + compactness",
        "classes": ["high_press", "mid_block", "low_block"],
        "labels": {
            # FIFA TSG thresholds: high press = mean distance < 25m AND compactness < 600m²
            "high_press": "High Press (aggressive, compact)",
            "mid_block": "Mid-Block (moderate, organised)",
            "low_block": "Low Block (deep, expansive)",
        },
    },
    "compactness_trend": {
        "description": "Classify team shape from compactness time series",
        "classes": ["compact", "moderate", "expansive"],
        "labels": {
            "compact": "Compact (<500m²)",
            "moderate": "Moderate (500–900m²)",
            "expansive": "Expansive (>900m²)",
        },
    },
    "possession_phase": {
        "description": "Classify possession rhythm from phase duration distribution",
        "classes": ["sustained", "transitional", "chaotic"],
        "labels": {
            "sustained": "Sustained (mean phase >4s)",
            "transitional": "Transitional (1–4s mean)",
            "chaotic": "Chaotic (<1s mean)",
        },
    },
    "territorial_dominance": {
        "description": "Classify territorial position from centroid x-coordinates",
        "classes": ["pressing_high", "balanced", "retreating"],
        "labels": {
            "pressing_high": "Pressing High (centroid >60m)",
            "balanced": "Balanced (40–60m)",
            "retreating": "Retreating (<40m)",
        },
    },
}

# ── Data preparation ───────────────────────────────────────────────────────────


def _label_pressing_type(inter_team_distances: list[float], compactness_vals: list[float]) -> str:
    """Label pressing type from mean distance + mean compactness."""
    if not inter_team_distances:
        return "mid_block"
    mean_dist = float(np.mean(inter_team_distances))
    mean_compact = float(np.mean(compactness_vals)) if compactness_vals else 800.0
    if mean_dist < 25.0 and mean_compact < 600.0:
        return "high_press"
    elif mean_dist < 40.0:
        return "mid_block"
    return "low_block"


def _label_compactness(compactness_vals: list[float]) -> str:
    """Label formation compactness from mean convex hull area."""
    if not compactness_vals:
        return "moderate"
    mean = float(np.mean(compactness_vals))
    if mean < 500.0:
        return "compact"
    elif mean < 900.0:
        return "moderate"
    return "expansive"


def _label_possession_phase(
    phases: list[dict],
    thresholds: tuple[float, float] = (1.0, 4.0),
) -> str:
    """Label possession rhythm from phase duration list.

    Args:
        phases: List of phase dicts with 'duration_frames'.
        thresholds: (low, high) mean-duration thresholds in seconds.
            Bins: chaotic = mean < low, transitional = low <= mean < high,
            sustained = mean >= high.
            Defaults to FIFA TSG thresholds (1s, 4s). Pass dataset-percentile
            values from prepare_classification_data for balanced classes.
    """
    if not phases:
        return "chaotic"
    durations = [p.get("duration_frames", 0) / 25.0 for p in phases if p.get("duration_frames")]
    if not durations:
        return "chaotic"
    mean_dur = float(np.mean(durations))
    lo, hi = thresholds
    if mean_dur >= hi:
        return "sustained"
    elif mean_dur >= lo:
        return "transitional"
    return "chaotic"


def _label_territorial(centroids: list[dict]) -> str:
    """Label territorial dominance from team centroid x-coordinate mean."""
    if not centroids:
        return "balanced"
    xs = [c.get("x", 52.5) for c in centroids if c.get("x") is not None]
    if not xs:
        return "balanced"
    mean_x = float(np.mean(xs))
    if mean_x > 60.0:
        return "pressing_high"
    elif mean_x >= 40.0:
        return "balanced"
    return "retreating"


def _window_series(series: list, window_size: int, step: int) -> list[list]:
    """Split a time series into overlapping windows."""
    if len(series) < window_size:
        return [series] if series else []
    windows = []
    i = 0
    while i + window_size <= len(series):
        windows.append(series[i : i + window_size])
        i += step
    return windows


def prepare_classification_data(
    db_ground_truths: "list[dict]",
    window_size: int = 30,
    window_step: int = 6,
) -> "dict[str, tuple[list, list]]":
    """Extract windowed time-series samples and labels for each classification task.

    Args:
        db_ground_truths: List of db_extractor outputs (one per analysis clip).
        window_size: Frames per window (default 30 = ~1.2s at 25fps).
        window_step: Stride between windows (default 6 = 5 windows per 30-frame clip).

    Returns:
        {task_name: ([time_series_window_0, ...], [label_0, ...])}
        where each time_series_window is a list of floats or dicts.
    """
    task_data: dict = {task: ([], []) for task in CLASSIFICATION_TASKS}
    _AUX_DATA_CACHE.clear()
    # Possession phase collected separately for dataset-percentile bin thresholds
    _poss_entries: list[list[float]] = []    # per-window duration series
    _poss_mean_durs: list[float] = []        # per-window mean duration in seconds

    for gt in db_ground_truths:
        fm = gt.get("frame_metrics", {})
        analytics = gt.get("analytics", {})
        fps = analytics.get("fps", 25.0) or 25.0

        # Inter-team distance series
        # Structure: [{frame, distance_m}, ...]
        dist_series = [
            d["distance_m"]
            for d in fm.get("inter_team_distance_m", [])
            if d.get("distance_m") is not None
        ]

        # Team 1 compactness series
        # Structure: [{frame, team_1_m2, team_2_m2}, ...]
        compact_t1 = [
            c["team_1_m2"]
            for c in fm.get("compactness_m2", [])
            if c.get("team_1_m2") is not None
        ]

        # Team 1 centroid x-series
        # Structure: [{frame, team_1: [x, y], team_2: [x, y]}, ...]
        centroid_t1 = [
            c["team_1"][0]
            for c in fm.get("team_centroids", [])
            if isinstance(c.get("team_1"), (list, tuple)) and len(c["team_1"]) >= 1
        ]

        # Derive possession phases from possession_sequence transitions
        # Structure: [{frame, team}, ...] sorted by frame
        poss_seq = sorted(fm.get("possession_sequence", []), key=lambda x: x.get("frame", 0))
        total_frames = analytics.get("total_frames") or len(dist_series) or 750
        phases = []
        for i, entry in enumerate(poss_seq):
            start_f = entry.get("frame", 0)
            end_f = poss_seq[i + 1].get("frame", total_frames) if i + 1 < len(poss_seq) else total_frames
            phases.append({
                "team": entry.get("team"),
                "start_frame": start_f,
                "duration_frames": max(1, end_f - start_f),
            })

        # ── pressing_type: inter-team distance windows ─────────────────────────
        dist_windows = _window_series(dist_series, window_size, window_step)
        compact_windows = _window_series(compact_t1, window_size, window_step)
        n_press = min(len(dist_windows), len(compact_windows))
        for i in range(n_press):
            label = _label_pressing_type(dist_windows[i], compact_windows[i])
            task_data["pressing_type"][0].append(dist_windows[i])
            task_data["pressing_type"][1].append(label)

        # ── compactness_trend: T1 compactness windows ──────────────────────────
        for window in compact_windows:
            label = _label_compactness(window)
            task_data["compactness_trend"][0].append(window)
            task_data["compactness_trend"][1].append(label)

        # ── possession_phase: collect windows (labelled after all gts, percentile bins) ──
        if len(phases) >= 5:
            for i in range(len(phases) - 4):
                phase_window = phases[i : i + 5]
                durations = [p.get("duration_frames", 0) / fps for p in phase_window]
                _poss_entries.append(durations)
                _poss_mean_durs.append(float(np.mean(durations)))

        # ── territorial_dominance: centroid x windows ──────────────────────────
        # Also build the parallel 2D-centroid windows (both teams' x,y per frame)
        # and stash them in the aux cache so modality='o' can render a proper
        # pitch overlay instead of a 1D line chart.
        centroid_windows = _window_series(centroid_t1, window_size, window_step)
        centroids_2d_all = [
            {"t1": list(c["team_1"]), "t2": list(c.get("team_2") or [None, None])}
            for c in fm.get("team_centroids", [])
            if isinstance(c.get("team_1"), (list, tuple))
            and len(c["team_1"]) >= 2
        ]
        centroid_2d_windows = _window_series(centroids_2d_all, window_size, window_step)
        for i, window in enumerate(centroid_windows):
            label = _label_territorial([{"x": x} for x in window])
            sample_idx = len(task_data["territorial_dominance"][0])
            task_data["territorial_dominance"][0].append(window)
            task_data["territorial_dominance"][1].append(label)
            if i < len(centroid_2d_windows):
                _AUX_DATA_CACHE[("territorial_dominance", sample_idx)] = {
                    "centroids_2d": centroid_2d_windows[i],
                }

    # Label possession_phase using dataset-percentile thresholds for balanced classes.
    # This avoids the FIFA hard thresholds (1s/4s) producing 89% "chaotic" in this data.
    if _poss_mean_durs:
        arr = np.array(_poss_mean_durs)
        lo = float(np.percentile(arr, 33))
        hi = float(np.percentile(arr, 67))
        logger.info(
            "possession_phase percentile thresholds: chaotic<%.3fs, transitional<%.3fs, sustained>=%.3fs",
            lo, hi, hi,
        )
        for durations, mean_dur in zip(_poss_entries, _poss_mean_durs):
            if mean_dur >= hi:
                label = "sustained"
            elif mean_dur >= lo:
                label = "transitional"
            else:
                label = "chaotic"
            task_data["possession_phase"][0].append(durations)
            task_data["possession_phase"][1].append(label)

    # Convert labels to integer indices for sklearn
    result: dict = {}
    for task, (series_list, label_list) in task_data.items():
        if not series_list:
            logger.warning("Task %s: no samples generated", task)
            result[task] = (series_list, label_list)
            continue
        result[task] = (series_list, label_list)
        logger.info(
            "Task %s: %d samples, classes: %s",
            task,
            len(series_list),
            {c: label_list.count(c) for c in set(label_list)},
        )
    return result


# ── Time-series formatting ────────────────────────────────────────────────────


def format_time_series_for_llm(
    series: "list[float]",
    modality: "Literal['d', 'v', 'd+v']",
    task_name: str = "pressing_type",
    fps: float = 25.0,
) -> "str | tuple[str, bytes]":
    """Format a time-series window for LLM input.

    Args:
        series: List of float values (one per frame).
        modality: 'd' = digit-space text, 'v' = line plot image, 'd+v' = both.
        task_name: Classification task (for axis labelling).
        fps: Frames per second (for time axis).

    Returns:
        str for modality='d', (str, bytes) for modality='v' or 'd+v'.
    """
    times = [f"{i / fps:.2f}s" for i in range(len(series))]

    # Digit-space text format
    def _to_digit_space(v: float) -> str:
        s = f"{v:.1f}"
        parts = s.split(".")
        return " ".join(parts[0]) + " , " + " ".join(parts[1]) if "." in s else " ".join(s)

    text = (
        f"Time series ({task_name}, {len(series)} frames at {fps:.0f}fps):\n"
        + " ".join(_to_digit_space(v) for v in series)
    )

    if modality == "d":
        return text

    # Visual format
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        t_seconds = [i / fps for i in range(len(series))]
        ax.plot(t_seconds, series, color="#4f86c6", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        task_labels = {
            "pressing_type": "Inter-team distance (m)",
            "compactness_trend": "Compactness (m²)",
            "possession_phase": "Phase duration (s)",
            "territorial_dominance": "Centroid x (m)",
        }
        ax.set_ylabel(task_labels.get(task_name, "Value"))
        ax.set_title(f"Time Series — {task_name.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=100, bbox_inches="tight")
        plt.close(fig)
        img_bytes = buf.getvalue()
    except ImportError:
        img_bytes = b""

    if modality == "v":
        return (text, img_bytes)
    return (text, img_bytes)  # d+v: both text and image


# ── Hidden state extraction ───────────────────────────────────────────────────


def extract_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    images: "list[bytes | None] | None" = None,
    layer: "int | None" = -1,
    batch_size: int = 4,
) -> "np.ndarray":
    """Extract final-token hidden states from a loaded HuggingFace model.

    Args:
        model: Loaded HuggingFace model (output_hidden_states=True will be set).
        tokenizer: Corresponding tokenizer.
        prompts: List of text prompts (one per sample).
        images: Optional list of JPEG byte arrays (for vision models).
        layer: Layer index to extract from. -1 = last layer, None = all layers.
        batch_size: Samples per forward pass.

    Returns:
        Array of shape (n_samples, hidden_dim) for single layer, or
        (n_samples, n_layers, hidden_dim) if layer is None.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("torch required for hidden state extraction. pip install torch") from e

    device = next(model.parameters()).device
    all_states: list = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            batch_images = (
                (images[batch_start : batch_start + batch_size] if images else [None] * len(batch_prompts))
            )

            batch_states = []
            for prompt, img_bytes in zip(batch_prompts, batch_images):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)

                # If vision model and image provided
                if img_bytes and hasattr(model, "vision_tower"):
                    try:
                        from PIL import Image
                        import io
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        from transformers import AutoProcessor
                        # pixel_values handled via processor — skipped if not available
                        _ = pil_img  # image loaded but processor not available here
                    except Exception:
                        pass

                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )

                hidden = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim) per layer
                # Use last token (generation position)
                if layer is None:
                    # Stack all layers: (n_layers, hidden_dim)
                    states = torch.stack([h[0, -1, :] for h in hidden]).cpu().numpy()
                else:
                    states = hidden[layer][0, -1, :].cpu().numpy()
                batch_states.append(states)

            all_states.extend(batch_states)
            logger.info("Extracted hidden states: %d/%d", batch_start + len(batch_prompts), len(prompts))

    return np.array(all_states)


# ── Vision model (Qwen2-VL) hidden state extraction ──────────────────────────


_TASK_YLABEL = {
    "pressing_type": "Inter-team distance (m)",
    "compactness_trend": "Compactness (m²)",
    "possession_phase": "Phase duration (s)",
    "territorial_dominance": "Centroid x (m)",
}

# Natural per-task y-axis ranges for the `vfix` fixed-axis modality.
# These anchor the visual to absolute units so the encoder sees a
# consistent reference frame across samples.
_TASK_YRANGE = {
    "pressing_type":        (0.0, 80.0),    # inter-team distance (m)
    "compactness_trend":    (0.0, 3000.0),  # convex-hull area (m²)
    "possession_phase":     (0.0, 5.0),     # phase duration (s)
    "territorial_dominance":(0.0, 105.0),   # pitch length (m)
}


def _format_series_string(
    series: "list[float]",
    text_modality: str,
    task_name: str,
    fps: float = 25.0,
) -> str:
    """Return the data portion of a prompt for a text or text-composite modality.

    Shared by `_build_classification_prompt` (pure-text path) and the Qwen2-VL
    vision-composite branch (e.g. `p+v`, `pi+v`, `n+v`) so the same string
    format is used whether the chart is attached or not.

    text_modality: one of 'd' (digit-space), 'p' (plain float), 'pi' (rounded
    integer), 'b' (quintile bins), 'n' (natural language).
    """
    if text_modality == "d":
        def _ds(v: float) -> str:
            s = f"{v:.1f}"
            p = s.split(".")
            return " ".join(p[0]) + " , " + " ".join(p[1])
        return " ".join(_ds(v) for v in series)
    if text_modality == "pi":
        return " ".join(str(int(round(v))) for v in series)
    if text_modality == "b":
        arr = np.asarray(series, dtype=float)
        edges = np.quantile(arr, [0.2, 0.4, 0.6, 0.8])
        toks = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        return " ".join(toks[int(np.searchsorted(edges, v, side="right"))] for v in arr)
    if text_modality in ("n", "ne", "nc", "nec"):
        # nc  = narration clean (stats only, no class-name-correlated words)
        # nec = narration extended clean (tactical vocabulary minus label-leak phrases)
        arr = np.asarray(series, dtype=float)
        n = len(arr)
        duration_s = n / max(fps, 1e-6)
        mean_v, std_v = float(np.mean(arr)), float(np.std(arr))
        min_v, max_v = float(np.min(arr)), float(np.max(arr))
        slope = float(np.polyfit(np.arange(n), arr, 1)[0]) if n > 1 else 0.0
        # Leak-free numeric trend word: "up / down / flat" instead of label-adjacent synonyms
        trend_neutral = "increasing" if slope > 0.05 else ("decreasing" if slope < -0.05 else "flat")
        extended = text_modality == "ne"
        clean_extended = text_modality == "nec"
        cv = std_v / (abs(mean_v) + 1e-9)
        volatility = "highly variable" if cv > 0.3 else ("moderately varied" if cv > 0.1 else "steady")

        if text_modality in ("nc", "nec"):
            # De-leaked templates: describe raw numeric statistics only.
            # No words that overlap with class names or their thresholds.
            series_numbers = ", ".join(f"{v:.2f}" for v in arr)
            base = (f"Time-series of length {n} over {duration_s:.2f}s. "
                    f"Values: {series_numbers}. "
                    f"Statistics: mean={mean_v:.2f}, std={std_v:.2f}, "
                    f"min={min_v:.2f}, max={max_v:.2f}, "
                    f"linear slope={slope:+.3f} per step, "
                    f"coefficient-of-variation={cv:.2f}.")
            if clean_extended:
                # Only generic trend/variability info, no task-linked vocabulary
                base += (f" The sequence is {volatility} and the direction of change "
                         f"is {trend_neutral}.")
            return base

        # Legacy label-leaky templates (kept for backward-compatibility so old
        # cache files still decode). New runs should prefer `nc` / `nec`.
        trend = "rising" if slope > 0.05 else ("falling" if slope < -0.05 else "flat")
        if task_name == "pressing_type":
            base = (f"Over a {duration_s:.1f}s window, the inter-team distance ranged "
                    f"from {min_v:.1f}m to {max_v:.1f}m (mean {mean_v:.1f}m, std "
                    f"{std_v:.1f}m). The trend was {trend} (slope {slope:+.2f} m/frame).")
            if extended:
                press_hint = ("indicating high pressing intent — teams were tight"
                              if mean_v < 22 else "suggesting a mid block or lower defensive line")
                base += (f" Tactically the signal is {volatility} and {press_hint}. "
                         f"The opening distance was {arr[0]:.1f}m and closed to {arr[-1]:.1f}m.")
            return base
        if task_name == "compactness_trend":
            shape = "compacting" if slope < -0.5 else ("expanding" if slope > 0.5 else "stable")
            base = (f"Over a {duration_s:.1f}s window, Team 1's convex-hull area ranged "
                    f"from {min_v:.0f}m² to {max_v:.0f}m² (mean {mean_v:.0f}m²). "
                    f"The shape was {shape} ({slope:+.1f} m²/frame).")
            if extended:
                tactic = ("compact defensive block" if mean_v < 800
                          else "stretched attacking shape" if mean_v > 1600
                          else "moderate inter-player spacing")
                base += (f" The team exhibited a {tactic} with {volatility} spacing dynamics."
                         f" Initial area {arr[0]:.0f}m², final {arr[-1]:.0f}m².")
            return base
        if task_name == "possession_phase":
            base = (f"Over five consecutive possession phases, durations were "
                    f"{', '.join(f'{v:.1f}s' for v in arr)} (mean {mean_v:.1f}s, "
                    f"longest {max_v:.1f}s). Rhythm: trend {trend}.")
            if extended:
                rhythm = ("chaotic with rapid possession changes" if mean_v < 0.46
                          else "transitional — phases of moderate length" if mean_v < 0.62
                          else "sustained possession with extended phases")
                base += (f" The rhythm suggests {rhythm}. Shortest phase {min_v:.1f}s."
                         f" Volatility: {volatility}.")
            return base
        if task_name == "territorial_dominance":
            side = "attacking half" if mean_v > 52.5 else "own half"
            base = (f"Over a {duration_s:.1f}s window, Team 1's centroid x ranged from "
                    f"{min_v:.1f}m to {max_v:.1f}m (mean {mean_v:.1f}m; pitch midline "
                    f"is at 52.5m). The team spent the window primarily in the {side}. "
                    f"Trend: {trend} ({slope:+.2f} m/frame).")
            if extended:
                stance = ("pressing high up the pitch" if mean_v > 60
                          else "retreating deep" if mean_v < 45
                          else "holding a balanced line")
                base += (f" Tactically they were {stance}, with {volatility} positional "
                         f"behaviour. Start centroid {arr[0]:.1f}m, end {arr[-1]:.1f}m.")
            return base
        return (f"{n} values from {min_v:.1f} to {max_v:.1f}, mean {mean_v:.1f}, "
                f"trend {trend}.")
    if text_modality == "pt":
        # Patch-text (PatchTST / Time-LLM style): break window into K=8 patches,
        # serialise each patch's descriptive stats. No per-frame digits.
        arr = np.asarray(series, dtype=float)
        n = len(arr)
        K = 8
        patch_size = max(1, n // K)
        parts = []
        for k in range(K):
            a = k * patch_size
            b = min(n, (k + 1) * patch_size) if k < K - 1 else n
            if a >= b:
                continue
            seg = arr[a:b]
            mu, sd = float(np.mean(seg)), float(np.std(seg))
            lo, hi = float(np.min(seg)), float(np.max(seg))
            slope = float(np.polyfit(np.arange(len(seg)), seg, 1)[0]) if len(seg) > 1 else 0.0
            parts.append(
                f"P{k+1}: mu{mu:.1f} sd{sd:.1f} r[{lo:.1f},{hi:.1f}] s{slope:+.2f}"
            )
        return " | ".join(parts)
    if text_modality == "sf":
        # Statistical-features prefix + digit-space suffix. Implements
        # contextual-feature augmentation: FFT peak frequency + top-3 bins,
        # ACF at lags 1, 25 (~1s), 75 (~3s), trend slope, seasonal strength.
        arr = np.asarray(series, dtype=float)
        n = len(arr)
        # FFT (drop DC, take magnitude)
        fft = np.abs(np.fft.rfft(arr - np.mean(arr)))
        freqs = np.fft.rfftfreq(n, d=1.0 / max(fps, 1e-6))
        if len(fft) > 1:
            top_idx = np.argsort(fft[1:])[::-1][:3] + 1
            peak_f = float(freqs[top_idx[0]])
            top_bins = ", ".join(f"{float(freqs[i]):.2f}Hz" for i in top_idx)
        else:
            peak_f = 0.0
            top_bins = "—"
        # Autocorrelation via normalised np.correlate
        a_cent = arr - np.mean(arr)
        denom = float(np.sum(a_cent ** 2)) + 1e-9
        def _acf(lag: int) -> float:
            if lag >= n or lag <= 0:
                return 0.0
            return float(np.sum(a_cent[:-lag] * a_cent[lag:]) / denom)
        slope = float(np.polyfit(np.arange(n), arr, 1)[0]) if n > 1 else 0.0
        # Seasonal strength: power in strongest non-DC bin / total non-DC power
        if len(fft) > 1 and fft[1:].sum() > 0:
            seasonal = float(fft[top_idx[0]] / fft[1:].sum())
        else:
            seasonal = 0.0
        # Compact digit suffix (shortened to keep prompt bounded)
        def _ds(v: float) -> str:
            s = f"{v:.1f}"
            p = s.split(".")
            return " ".join(p[0]) + " , " + " ".join(p[1])
        digit_str = " ".join(_ds(v) for v in arr)
        return (f"Features: FFT peak {peak_f:.2f}Hz (top3: {top_bins}) | "
                f"ACF(1)={_acf(1):+.2f} ACF(25)={_acf(25):+.2f} ACF(75)={_acf(75):+.2f} | "
                f"slope {slope:+.3f} per frame | seasonal {seasonal:.2f}. "
                f"Digits: {digit_str}")
    # p (plain) and anything else → plain float serialisation
    return " ".join(f"{v:.1f}" for v in series)


def _render_chart_for_modality(
    series: "list[float]",
    task_name: str,
    modality: str,
    fps: float = 25.0,
    aux: "dict | None" = None,
) -> bytes:
    """Render the visual input for a single sample, dispatched by modality.

    Modalities:
      v       — single line chart of the 1-D series (default).
      d+v     — same chart; combined with digit-space text in message build.
      m       — multi-panel (raw · first-diff · rolling mean · histogram).
      o       — pitch overlay: plots the series on a 105×68m pitch backdrop
                with defensive/mid/attacking thirds shaded. Designed for
                territorial_dominance (where the series is centroid x), but
                safe to call for any task — the y-axis label changes.
    """
    import io as _io
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
    except Exception as e:
        logger.warning("matplotlib unavailable: %s", e)
        return b""

    try:
        n = len(series)
        if n == 0:
            return b""
        t = [i / fps for i in range(n)]

        if modality == "m":
            fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2))
            raw = np.asarray(series, dtype=float)
            axes[0, 0].plot(t, raw, color="#1e3a8a", linewidth=1.4)
            axes[0, 0].set_title("Raw", fontsize=9); axes[0, 0].grid(alpha=0.3)
            axes[0, 1].plot(t[1:], np.diff(raw), color="#b45309", linewidth=1.2)
            axes[0, 1].set_title("First difference", fontsize=9); axes[0, 1].grid(alpha=0.3)
            w = max(3, n // 8)
            roll = np.convolve(raw, np.ones(w) / w, mode="valid")
            axes[1, 0].plot(t[: len(roll)], roll, color="#047857", linewidth=1.4)
            axes[1, 0].set_title(f"Rolling mean (w={w})", fontsize=9); axes[1, 0].grid(alpha=0.3)
            axes[1, 1].hist(raw, bins=15, color="#be123c", edgecolor="white")
            axes[1, 1].set_title("Distribution", fontsize=9)
            for ax in axes.flat:
                ax.tick_params(labelsize=7)
            fig.suptitle(f"{task_name.replace('_',' ').title()}  — multi-view",
                         fontsize=10)
            fig.tight_layout()
        elif modality == "o":
            # Pitch backdrop (105×68m) with thirds shaded. Use 2D centroid data
            # from aux if available (both teams' full x,y trajectory); fall back
            # to 1D centroid_x at midline y if not.
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.add_patch(Rectangle((0, 0), 105, 68, fill=True,
                                   facecolor="#f0fdf4", edgecolor="#166534", linewidth=1.3))
            ax.add_patch(Rectangle((0, 0), 35, 68, fill=True, alpha=0.25,
                                   facecolor="#bae6fd", edgecolor="none"))
            ax.add_patch(Rectangle((70, 0), 35, 68, fill=True, alpha=0.25,
                                   facecolor="#fecaca", edgecolor="none"))
            ax.plot([52.5, 52.5], [0, 68], color="#166534", linewidth=0.9)
            ax.add_patch(Circle((52.5, 34), 9.15, fill=False, edgecolor="#166534", linewidth=0.9))
            ax.add_patch(Rectangle((0, 14), 16.5, 40, fill=False, edgecolor="#166534", linewidth=0.7))
            ax.add_patch(Rectangle((88.5, 14), 16.5, 40, fill=False, edgecolor="#166534", linewidth=0.7))

            centroids_2d = (aux or {}).get("centroids_2d")
            if centroids_2d:
                # Proper 2D pitch trajectory — both teams' full paths
                import matplotlib.cm as cm
                t1 = np.array([c["t1"] for c in centroids_2d if c.get("t1")], dtype=float)
                t2 = np.array([c["t2"] for c in centroids_2d
                               if c.get("t2") and c["t2"][0] is not None], dtype=float)
                if len(t1) > 0:
                    colours = cm.cividis(np.linspace(0.1, 0.95, len(t1)))
                    ax.plot(t1[:, 0], t1[:, 1], color="#1e3a8a", linewidth=1.4,
                            alpha=0.8, zorder=3, label="Team 1")
                    ax.scatter(t1[:, 0], t1[:, 1], c=colours, s=10, zorder=4,
                               edgecolors="white", linewidths=0.3)
                    ax.scatter([t1[0, 0]], [t1[0, 1]], s=80, facecolor="#22c55e",
                               edgecolor="black", linewidth=0.8, zorder=5, marker="o")
                    ax.scatter([t1[-1, 0]], [t1[-1, 1]], s=100, facecolor="#1e3a8a",
                               edgecolor="black", linewidth=0.8, zorder=5, marker=">")
                if len(t2) > 0:
                    ax.plot(t2[:, 0], t2[:, 1], color="#be123c", linewidth=1.4,
                            alpha=0.8, zorder=3, linestyle="--", label="Team 2")
                    ax.scatter([t2[0, 0]], [t2[0, 1]], s=80, facecolor="#fecaca",
                               edgecolor="black", linewidth=0.8, zorder=5, marker="o")
                    ax.scatter([t2[-1, 0]], [t2[-1, 1]], s=100, facecolor="#be123c",
                               edgecolor="black", linewidth=0.8, zorder=5, marker=">")
                ax.legend(loc="upper right", fontsize=8, frameon=False)
            else:
                # Fallback 1D rendering
                raw = np.asarray(series, dtype=float)
                import matplotlib.cm as cm
                colours = cm.cividis(np.linspace(0.1, 0.95, len(raw)))
                ax.scatter(raw, np.full_like(raw, 34.0), c=colours, s=12, zorder=3,
                           edgecolors="white", linewidths=0.4)
                ax.plot(raw, np.full_like(raw, 34.0), color="#1e3a8a", linewidth=1.0,
                        alpha=0.55, zorder=2)
                ax.scatter([raw[0]], [34], s=70, facecolor="#22c55e", edgecolor="black",
                           linewidth=0.8, zorder=4, marker="o", label="start")
                ax.scatter([raw[-1]], [34], s=90, facecolor="#e11d48", edgecolor="black",
                           linewidth=0.8, zorder=4, marker=">", label="end")
                ax.legend(loc="upper right", fontsize=7, frameon=False)

            ax.set_xlim(-2, 107); ax.set_ylim(-2, 70)
            ax.set_xticks([0, 35, 52.5, 70, 105])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_title(f"{task_name.replace('_',' ').title()} — 2D pitch overlay",
                         fontsize=10)
            fig.tight_layout()
        elif modality == "v2":
            # Clean minimalist chart — no title, legend, grid. Larger canvas.
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t, series, color="#000000", linewidth=2.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=11)
            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel(_TASK_YLABEL.get(task_name, "Value"), fontsize=12)
            fig.tight_layout()
        elif modality == "va":
            # Annotated line chart — line + magnitude labels overlaid. Helps
            # VLMs read magnitudes through their vision-text channel.
            fig, ax = plt.subplots(figsize=(8, 4))
            arr = np.asarray(series, dtype=float)
            ax.plot(t, arr, color="#1e3a8a", linewidth=2.0)
            mean_v = float(np.mean(arr)); std_v = float(np.std(arr))
            min_v = float(np.min(arr)); max_v = float(np.max(arr))
            mn_idx = int(np.argmin(arr)); mx_idx = int(np.argmax(arr))
            ax.axhline(mean_v, linestyle="--", color="#64748b", linewidth=1)
            ax.annotate(f"mean = {mean_v:.1f}", xy=(t[-1], mean_v),
                        xytext=(5, -6), textcoords="offset points",
                        fontsize=9, color="#334155")
            ax.annotate(f"max {max_v:.1f}", xy=(t[mx_idx], arr[mx_idx]),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=9, color="#be123c")
            ax.annotate(f"min {min_v:.1f}", xy=(t[mn_idx], arr[mn_idx]),
                        xytext=(4, -14), textcoords="offset points",
                        fontsize=9, color="#047857")
            ax.scatter([t[mx_idx], t[mn_idx]], [arr[mx_idx], arr[mn_idx]],
                       s=30, c=["#be123c", "#047857"], zorder=5)
            ax.set_xlabel("Time (s)"); ax.set_ylabel(_TASK_YLABEL.get(task_name, "Value"))
            ax.set_title(f"σ = {std_v:.1f}   range {min_v:.1f}–{max_v:.1f}", fontsize=9)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        elif modality == "sp":
            # Spectrogram (power spectral density heatmap over time).
            # For short series we use a simple STFT-like view.
            from numpy.fft import rfft
            arr = np.asarray(series, dtype=float) - float(np.mean(series))
            win = min(max(4, len(arr) // 4), 16)
            hop = max(1, win // 2)
            segs = []
            for start in range(0, len(arr) - win + 1, hop):
                seg = arr[start : start + win]
                mag = np.abs(rfft(seg * np.hanning(len(seg))))
                segs.append(mag)
            if len(segs) < 2:
                segs = [arr, arr]
            S = np.asarray(segs).T  # (freq_bins, time_frames)
            S_db = 20 * np.log10(S + 1e-9)
            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(S_db, aspect="auto", origin="lower", cmap="magma",
                           interpolation="bilinear")
            ax.set_xlabel("Time frame"); ax.set_ylabel("Frequency bin")
            ax.set_title(f"{task_name.replace('_',' ').title()} — spectrogram", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="dB")
            fig.tight_layout()
        elif modality == "vfix":
            # Fixed-axis line chart — y-axis pinned to task natural range so
            # the vision encoder sees absolute magnitudes in a consistent
            # reference frame across samples. Addresses the auto-scaled-axis
            # limitation of `v` for magnitude-sensitive tasks.
            y_lo, y_hi = _TASK_YRANGE.get(task_name, (float(np.min(series)),
                                                      float(np.max(series))))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t, series, color="#1e3a8a", linewidth=2.0)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(_TASK_YLABEL.get(task_name, "Value"))
            # Task-specific reference lines
            if task_name == "territorial_dominance":
                ax.axhline(52.5, linestyle="--", color="#64748b", linewidth=1,
                           alpha=0.7, label="midline")
                ax.axhspan(60, y_hi, alpha=0.15, color="#fecaca",
                           label="pressing_high zone")
                ax.axhspan(y_lo, 45, alpha=0.15, color="#bae6fd",
                           label="retreating zone")
                ax.legend(fontsize=7, loc="best", frameon=False)
            mean_val = float(np.mean(series))
            ax.axhline(mean_val, linestyle=":", color="#be123c", linewidth=1,
                       alpha=0.8)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        elif modality == "vh":
            # Horizontal value-heatmap strip. Simplest possible shape encoding.
            arr = np.asarray(series, dtype=float).reshape(1, -1)
            fig, ax = plt.subplots(figsize=(10, 1.8))
            im = ax.imshow(arr, aspect="auto", cmap="viridis",
                           vmin=float(np.min(series)), vmax=float(np.max(series)))
            ax.set_yticks([])
            ax.set_xlabel("Frame")
            ax.set_title(f"{task_name.replace('_',' ').title()} — value heatmap"
                         f"  (range {float(np.min(series)):.1f}–{float(np.max(series)):.1f})",
                         fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            fig.tight_layout()
        else:
            # v / d+v / default: single line chart (original style)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(t, series, color="#1e3a8a", linewidth=1.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(_TASK_YLABEL.get(task_name, "Value"))
            mean_val = float(np.mean(series))
            ax.axhline(mean_val, linestyle="--", color="gray", linewidth=1,
                       label=f"mean={mean_val:.1f}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        buf = _io.BytesIO()
        fig.savefig(buf, format="jpeg", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        logger.warning("Chart render failed (modality=%s): %s", modality, e)
        return b""


def extract_hidden_states_vision(
    model,
    processor,
    series_list: "list[list[float]]",
    task_name: str,
    modality: str = "v",
    layer: "int | None" = -1,
    batch_size: int = 2,
    fps: float = 25.0,
) -> "np.ndarray":
    """Extract hidden states from a Qwen2-VL vision-language model.

    For modality='v': renders each series as a JPEG chart, passes through
    the vision encoder, extracts last-token hidden states.
    For modality='d+v': combines digit-space text + chart image.

    Requires:
        pip install qwen-vl-utils
        Model: Qwen/Qwen2-VL-7B-Instruct (or similar)

    Returns:
        np.ndarray of shape (n_samples, hidden_dim)
    """
    try:
        import torch
        import io
        from PIL import Image
    except ImportError as e:
        raise ImportError("torch and Pillow required for vision extraction") from e

    device = next(model.parameters()).device
    all_states: list = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(series_list), batch_size):
            batch_series = series_list[batch_start : batch_start + batch_size]

            for offset, series in enumerate(batch_series):
                sample_idx = batch_start + offset
                aux = _AUX_DATA_CACHE.get((task_name, sample_idx))
                # For composite modalities 'text+visual', render the VISUAL part.
                visual_mod = modality.split("+", 1)[1] if "+" in modality else modality
                img_bytes = _render_chart_for_modality(series, task_name, visual_mod, fps, aux=aux)

                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB") if img_bytes else None

                # Build text prompt
                task = CLASSIFICATION_TASKS[task_name]
                classes = task["classes"]

                # Composite text+visual modality: shape '<text>+<visual>'.
                # text ∈ {d, p, pi, n, ne, b}, visual ∈ {v, va, sp, v2, vh}.
                # Legacy 'd+v' fits this scheme.
                if "+" in modality:
                    text_mod, _visual_mod = modality.split("+", 1)
                    series_str = _format_series_string(series, text_mod, task_name, fps)
                    if text_mod in ("n", "ne"):
                        text = (f"{series_str}\nChoose one: {', '.join(classes)}")
                    else:
                        text = (
                            f"Time series ({task_name}, {len(series)} frames): "
                            f"{series_str}\nChoose one: {', '.join(classes)}"
                        )
                else:
                    text = (
                        f"This chart shows {task_name.replace('_', ' ')} over {len(series)} frames. "
                        f"Choose one: {', '.join(classes)}"
                    )

                # Build message for Qwen2-VL processor
                content: list = []
                if pil_img is not None:
                    content.append({"type": "image", "image": pil_img})
                content.append({"type": "text", "text": text})
                messages = [{"role": "user", "content": content}]

                try:
                    formatted = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if pil_img is not None:
                        inputs = processor(
                            text=[formatted],
                            images=[pil_img],
                            return_tensors="pt",
                        ).to(device)
                    else:
                        inputs = processor(
                            text=[formatted],
                            return_tensors="pt",
                        ).to(device)

                    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                    hidden = outputs.hidden_states
                    if layer is None:
                        # Return all layers stacked: (n_layers, hidden_dim)
                        state = torch.stack([h[0, -1, :] for h in hidden]).cpu().numpy()
                    elif layer == -1:
                        state = hidden[-1][0, -1, :].cpu().numpy()
                    else:
                        state = hidden[layer][0, -1, :].cpu().numpy()
                    all_states.append(state)
                except Exception as e:
                    logger.error("Vision forward pass failed: %s", e)
                    # Fallback: zero vector
                    hidden_dim = model.config.hidden_size
                    if layer is None:
                        n_layers = getattr(model.config, "num_hidden_layers", 28) + 1
                        all_states.append(np.zeros((n_layers, hidden_dim), dtype=np.float32))
                    else:
                        all_states.append(np.zeros(hidden_dim, dtype=np.float32))

            logger.info(
                "Vision hidden states: %d/%d",
                batch_start + len(batch_series), len(series_list),
            )

    return np.array(all_states)


# ── Probe training ────────────────────────────────────────────────────────────


def _build_label_encoder(y_train: list, y_test: list) -> "tuple[dict, dict, list[str]]":
    """Build dict-based label encoding that avoids numpy string truncation.

    LabelEncoder internally converts labels to numpy arrays whose dtype is
    determined by the longest string.  For multi-character labels like
    'transitional' this can cause truncation to 'transit' when sklearn
    creates intermediate numpy arrays with a too-narrow dtype.  This helper
    uses plain Python dicts so string identities are never numpy-mediated.

    Returns:
        (class_to_idx, idx_to_class, sorted_classes)
    """
    # Coerce to plain Python str to handle np.str_ scalars
    all_classes = sorted(set([str(y) for y in y_train] + [str(y) for y in y_test]))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    return class_to_idx, idx_to_class, all_classes


def _bootstrap_f1(
    y_true_enc: "np.ndarray",
    y_pred_enc: "np.ndarray",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for macro F1 by resampling the test set.

    Used to quantify uncertainty on probe F1 for n~30 test samples.
    """
    from sklearn.metrics import f1_score
    rng = np.random.RandomState(seed)
    n = len(y_true_enc)
    scores: list = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(set(y_true_enc[idx].tolist())) < 2:
            continue
        scores.append(
            float(f1_score(y_true_enc[idx], y_pred_enc[idx], average="macro", zero_division=0))
        )
    if not scores:
        return {"mean": None, "ci_low": None, "ci_high": None, "n_boot": 0}
    arr = np.array(scores)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_low": round(float(np.percentile(arr, 2.5)), 4),
        "ci_high": round(float(np.percentile(arr, 97.5)), 4),
        "n_boot": len(arr),
    }


def train_linear_probe(
    X_train: "np.ndarray",
    y_train: "list[str]",
    X_test: "np.ndarray",
    y_test: "list[str]",
) -> dict:
    """Train logistic regression probe with cross-validated regularisation.

    Follows Schumacher et al. §3.3: LogisticRegression with 5-fold CV over C,
    max_iter=1000, class_weight='balanced' for imbalanced football labels.

    Returns:
        {f1_macro, accuracy, per_class_f1, best_C, n_train, n_test}
    """
    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError("scikit-learn required. pip install scikit-learn") from e

    # Dict-based encoding avoids numpy string truncation (e.g. "transitional" -> "transit")
    class_to_idx, idx_to_class, all_classes = _build_label_encoder(y_train, y_test)
    y_train_enc = np.array([class_to_idx[str(y)] for y in y_train])
    y_test_enc = np.array([class_to_idx[str(y)] for y in y_test])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegressionCV(
        Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
        cv=5,
        max_iter=1000,
        class_weight="balanced",
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train_enc)

    y_pred = clf.predict(X_test_scaled)
    f1_macro = float(f1_score(y_test_enc, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_test_enc, y_pred))

    per_class = {}
    class_f1s = f1_score(y_test_enc, y_pred, average=None, zero_division=0)
    for cls_idx, cls_name in enumerate(all_classes):
        if cls_idx < len(class_f1s):
            per_class[cls_name] = round(float(class_f1s[cls_idx]), 4)

    # Decode predictions back to class names for stability study
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pred_labels = [idx_to_class.get(int(p), str(p)) for p in y_pred]
    y_test_labels = [str(y) for y in y_test]

    # Bootstrap 95% CI on test F1 — quantifies uncertainty on small test sets
    ci = _bootstrap_f1(y_test_enc, y_pred)

    return {
        "f1_macro": round(f1_macro, 4),
        "accuracy": round(accuracy, 4),
        "per_class_f1": per_class,
        "best_C": float(clf.C_[0]) if hasattr(clf, "C_") else None,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "classes": all_classes,
        "predictions": pred_labels,
        "y_test": y_test_labels,
        "bootstrap": ci,
    }


def train_mlp_probe(
    X_train: "np.ndarray",
    y_train: "list[str]",
    X_test: "np.ndarray",
    y_test: "list[str]",
    hidden_size: int = 128,  # retained for API compat; grid overrides it
) -> dict:
    """2-layer MLP probe — non-linear upper bound on the linear probe.

    Uses GridSearchCV over (hidden_layer_sizes, alpha) to mirror the linear
    probe's LogisticRegressionCV procedure, so linear-vs-MLP is a fair
    comparison instead of an overfit-inflated MLP score.
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GridSearchCV
    except ImportError as e:
        raise ImportError("scikit-learn required") from e

    class_to_idx, _, all_classes = _build_label_encoder(y_train, y_test)
    y_train_enc = np.array([class_to_idx[str(y)] for y in y_train])
    y_test_enc = np.array([class_to_idx[str(y)] for y in y_test])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # CV fold count must not exceed the smallest class's member count.
    _, class_counts = np.unique(y_train_enc, return_counts=True)
    cv_folds = int(max(2, min(5, class_counts.min())))

    base_mlp = MLPClassifier(
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
    )
    param_grid = {
        "hidden_layer_sizes": [(16,), (32,), (64,), (128,)],
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    }
    search = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="f1_macro",
        refit=True,
        n_jobs=-1,
    )
    search.fit(X_train_scaled, y_train_enc)

    y_pred = search.predict(X_test_scaled)
    f1_macro = float(f1_score(y_test_enc, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_test_enc, y_pred))

    best_params = search.best_params_
    # Stringify hidden_layer_sizes tuple for JSON-serialisability.
    best_params_json = {
        "hidden_layer_sizes": list(best_params["hidden_layer_sizes"]),
        "alpha": float(best_params["alpha"]),
    }

    return {
        "f1_macro": round(f1_macro, 4),
        "accuracy": round(accuracy, 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "classes": all_classes,
        "bootstrap": _bootstrap_f1(y_test_enc, y_pred),
        "best_params": best_params_json,
        "cv_mean_f1": round(float(search.best_score_), 4),
        "cv_folds": cv_folds,
    }


# ── Prompting baseline ────────────────────────────────────────────────────────


# 10 meaning-preserving prompt variants for the stability study (Phase 7.3).
# Variants differ in phrasing / data framing / output instruction only — no semantic change.
_PROMPT_VARIANTS: list[tuple[str, str, str]] = [
    # (role_phrase, data_label, output_instruction)
    ("You are a football tactics analyst.",
     "Time series data:", "Respond with only the class name (e.g. '{cls0}')."),
    ("As a match analyst,",
     "Numeric sequence:", "Output exactly one of: {class_list}."),
    ("You are a football performance scientist.",
     "Sampled values:", "Reply with only the class name."),
    ("You are a football tactics analyst.",
     "Time series data:", "Output exactly one of: {class_list}."),
    ("As a football data scientist,",
     "Time series data:", "Respond with only the class name (e.g. '{cls0}')."),
    ("You are a match analyst.",
     "Numeric sequence:", "Reply with the single class name: {class_list}."),
    ("You are a football tactics analyst.",
     "Sampled values:", "<class>{cls0}</class>-style: respond with one class name only."),
    ("As a performance analyst,",
     "Time series data:", "Respond with only the class name (e.g. '{cls0}')."),
    ("You are a football tactics analyst.",
     "Time series data:\n",
     "Choose the class. Respond with only the class name."),
    ("You are a football tactics analyst.",
     "Data:\n", "Respond with only the class name (e.g. '{cls0}')."),
]


def _build_classification_prompt(
    series: "list[float]",
    task_name: str,
    modality: str,
    fps: float = 25.0,
    variant_id: int = 0,
) -> str:
    """Build a zero-shot classification prompt for the time series.

    Args:
        series: Window of float values.
        task_name: One of CLASSIFICATION_TASKS.
        modality: 'd' (digit-space) or 'v' (plain numeric).
        fps: Frames per second.
        variant_id: 0–9 for prompt stability study; 0 = canonical prompt.
    """
    task = CLASSIFICATION_TASKS[task_name]
    classes = task["classes"]
    class_labels = "\n".join(
        f"  [{i+1}] {cls} — {task['labels'][cls]}"
        for i, cls in enumerate(classes)
    )
    class_list = ", ".join(classes)

    # Resolve composite modality to its text component (e.g. 'ne+v' -> 'ne').
    text_mod = modality.split("+", 1)[0] if "+" in modality else modality
    # Delegate to the shared formatter so every branch (d/p/pi/b/n/ne) uses
    # the same text as the vision-composite path.
    series_str = _format_series_string(series, text_mod, task_name, fps)

    variant_id = variant_id % len(_PROMPT_VARIANTS)
    role, data_label, output_instr = _PROMPT_VARIANTS[variant_id]
    output_instr = output_instr.replace("{cls0}", classes[0]).replace(
        "{class_list}", class_list
    )

    if variant_id == 0:
        # Canonical prompt — preserves original exact wording
        return (
            f"You are a football tactics analyst. Classify the following {len(series)}-frame "
            f"time series at {fps:.0f} fps.\n\n"
            f"Task: {task['description']}\n\n"
            f"Time series data:\n{series_str}\n\n"
            f"Choose exactly one class:\n{class_labels}\n\n"
            f"Respond with only the class name (e.g. '{classes[0]}')."
        )

    return (
        f"{role} Classify the following {len(series)}-frame time series at {fps:.0f} fps.\n\n"
        f"Task: {task['description']}\n\n"
        f"{data_label}\n{series_str}\n\n"
        f"Choose exactly one class:\n{class_labels}\n\n"
        f"{output_instr}"
    )


async def run_prompting_baseline(
    model,
    tokenizer,
    series_list: "list[list[float]]",
    labels: "list[str]",
    task_name: str,
    modality: str,
    fps: float = 25.0,
    variant_id: int = 0,
    cot: bool = False,
) -> dict:
    """Zero-shot prompting baseline: generate class prediction, parse, compute F1.

    Args:
        variant_id: Prompt variant index 0–9 (0 = canonical). Used in stability study.

    Returns:
        {f1_macro, accuracy, per_class_f1, n_samples, parse_failure_rate, predictions}
    """
    try:
        import torch
        from sklearn.metrics import f1_score, accuracy_score
    except ImportError as e:
        raise ImportError("torch and scikit-learn required") from e

    classes = CLASSIFICATION_TASKS[task_name]["classes"]
    device = next(model.parameters()).device

    predictions: list[str] = []
    parse_failures = 0

    model.eval()
    with torch.no_grad():
        for series in series_list:
            prompt = _build_classification_prompt(series, task_name, modality, fps,
                                                   variant_id=variant_id)
            if cot:
                prompt = (prompt.rstrip() +
                          "\n\nLet's think step by step about the tactical "
                          "pattern, then output only the final class name.\n\n")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=120 if cot else 20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated = generated.strip().lower()

            matched = None
            for cls in classes:
                if cls.lower() in generated:
                    matched = cls
                    break
            if matched is None:
                parse_failures += 1
                matched = classes[0]  # fallback to first class
            predictions.append(matched)

    # Dict-based encoding — avoids numpy string truncation
    class_to_idx, _, all_classes = _build_label_encoder(labels, predictions)
    y_true = np.array([class_to_idx[str(l)] for l in labels])
    y_pred_enc = np.array([class_to_idx.get(str(p), 0) for p in predictions])

    f1_macro = float(f1_score(y_true, y_pred_enc, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred_enc))

    per_class = {}
    class_f1s = f1_score(y_true, y_pred_enc, average=None, zero_division=0)
    for i, cls in enumerate(all_classes):
        if i < len(class_f1s):
            per_class[cls] = round(float(class_f1s[i]), 4)

    return {
        "f1_macro": round(f1_macro, 4),
        "accuracy": round(accuracy, 4),
        "per_class_f1": per_class,
        "n_samples": len(labels),
        "parse_failure_rate": parse_failures / max(len(series_list), 1),
        "predictions": predictions,  # per-sample predictions for stability study
    }


# ── Layer-wise analysis ────────────────────────────────────────────────────────


def run_layer_wise_analysis(
    model,
    tokenizer,
    series_list: "list[list[float]]",
    labels: "list[str]",
    task_name: str,
    modalities: "list[str]",
    test_fraction: float = 0.2,
    layer_step: int = 4,
    processor=None,
    run_mlp: bool = True,
    cache_dir: "Path | None" = None,
) -> dict:
    """For each layer × modality, extract hidden states and train a probe.

    Args:
        layer_step: Sample every N-th layer (default 4) to reduce runtime.
                    For a 28-layer Qwen-7B this gives ~8 probe fits instead of 29.
        processor: Qwen2-VL processor, required for v / d+v modalities to use
                   the vision encoder. If None, all modalities fall back to
                   text-only extraction and v / d+v will produce identical
                   hidden states (known limitation; see dissertation §4.5).
        run_mlp: If True, also fit a 2-layer MLP probe per layer so the
                 non-linear upper bound can be compared with depth.

    Returns:
        {modality: {layer_idx: {"linear_f1": ..., "mlp_f1": ...}}}
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError("torch required") from e

    cfg = model.config
    _n_hidden = getattr(cfg, "num_hidden_layers", None) or getattr(
        getattr(cfg, "text_config", cfg), "num_hidden_layers", 28
    )
    n_layers = _n_hidden + 1  # +1 for embedding layer
    layers_to_probe = list(range(0, n_layers, layer_step))
    # Always include the last layer
    if (n_layers - 1) not in layers_to_probe:
        layers_to_probe.append(n_layers - 1)

    n_test = max(1, int(len(series_list) * test_fraction))
    n_train = len(series_list) - n_test

    results: dict = {}

    for mod in modalities:
        logger.info(
            "Layer-wise analysis: modality=%s, probing %d/%d layers (step=%d)",
            mod, len(layers_to_probe), n_layers, layer_step,
        )
        layer_results: dict = {}

        # Single forward pass per sample, cache all layer hidden states.
        # Modality branching mirrors the main probing loop: v / d+v use the
        # vision encoder; d uses text-only extraction.
        if (mod in ("v", "d+v", "m", "o", "va", "sp", "v2", "vh")
                or "+v" in mod or mod.endswith("+va") or mod.endswith("+sp")
                or mod.endswith("+v2") or mod.endswith("+vh")) and processor is not None:
            all_hidden = extract_hidden_states_vision(
                model, processor, series_list, task_name,
                modality=mod, layer=None,
            )
        else:
            prompts = [
                _build_classification_prompt(s, task_name, mod)
                for s in series_list
            ]
            all_hidden = extract_hidden_states(
                model, tokenizer, prompts, layer=None,
            )
        # all_hidden shape: (n_samples, n_layers, hidden_dim)

        # Cache hidden states + labels so downstream analyses (CKA, silhouette,
        # selectivity, t-SNE strip, modality synergy) can run locally from
        # disk without another GPU pass.
        if cache_dir is not None:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                safe_mod = mod.replace("+", "_")
                np.savez_compressed(
                    cache_dir / f"hidden_{task_name}_{safe_mod}.npz",
                    hidden=all_hidden.astype(np.float32),
                    labels=np.array(labels),
                    layers_probed=np.array(layers_to_probe),
                    n_train=n_train,
                    n_test=n_test,
                )
                logger.info(
                    "  Cached hidden states to %s (shape %s)",
                    cache_dir / f"hidden_{task_name}_{safe_mod}.npz",
                    all_hidden.shape,
                )
            except Exception as e:
                logger.warning("  Hidden-state cache write failed: %s", e)

        for layer_idx in layers_to_probe:
            try:
                hidden = all_hidden[:, layer_idx, :]
                probe = train_linear_probe(
                    hidden[:n_train], labels[:n_train],
                    hidden[n_train:], labels[n_train:],
                )
                cell: dict = {"linear_f1": probe["f1_macro"]}
                if run_mlp:
                    try:
                        mlp = train_mlp_probe(
                            hidden[:n_train], labels[:n_train],
                            hidden[n_train:], labels[n_train:],
                        )
                        cell["mlp_f1"] = mlp["f1_macro"]
                        if "best_params" in mlp:
                            cell["mlp_best_params"] = mlp["best_params"]
                    except Exception as e:
                        logger.warning("  MLP layer %d failed: %s", layer_idx, e)
                layer_results[layer_idx] = cell
                logger.info(
                    "  Layer %d/%d: Linear F1=%.3f%s",
                    layer_idx, n_layers - 1, probe["f1_macro"],
                    f" MLP F1={cell['mlp_f1']:.3f}" if "mlp_f1" in cell else "",
                )
            except Exception as e:
                logger.warning("  Layer %d failed: %s", layer_idx, e)
                layer_results[layer_idx] = None

        results[mod] = layer_results

    return results


# ── Random baseline ────────────────────────────────────────────────────────────


def run_random_baseline(
    model_path: str,
    series_list: "list[list[float]]",
    labels: "list[str]",
    task_name: str,
    modality: str = "d",
    test_fraction: float = 0.2,
    model_type: str = "causal",
) -> dict:
    """Probe with randomly initialised model weights.

    Establishes the 'feature structure alone' baseline (Schumacher Table 1: Random-Probe).
    If a probe trained on random weights achieves above-chance F1, it reflects
    tokenisation/positional structure rather than learned temporal representations.

    Loads the architecture config from model_path (no pretrained weights),
    initialises randomly, moves to GPU in fp16, extracts hidden states, then
    deletes the model to free VRAM.  Must be called BEFORE the pretrained model
    is loaded so the full VRAM budget is available.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        import torch
        import gc
    except ImportError as e:
        raise ImportError("transformers required") from e

    logger.info("Initialising random-weight model from config: %s ...", model_path)
    config = AutoConfig.from_pretrained(model_path)

    if model_type == "qwen2vl":
        # Qwen2VL is not registered with AutoModelForCausalLM. Instantiate the class
        # directly (constructor call produces random weights without loading from disk).
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            rng_model = Qwen2VLForConditionalGeneration(config)
            proc = AutoProcessor.from_pretrained(model_path)
            tokenizer = proc.tokenizer
        except Exception as e:
            # Further fallback: use Qwen2 text backbone with matching hidden config
            logger.warning("Qwen2VL direct init failed (%s), using Qwen2 text backbone for random baseline", e)
            from transformers import Qwen2Config, Qwen2ForCausalLM
            text_cfg = Qwen2Config(
                hidden_size=getattr(config, "hidden_size", 3584),
                num_hidden_layers=getattr(config, "num_hidden_layers", 28),
                num_attention_heads=getattr(config, "num_attention_heads", 28),
                num_key_value_heads=getattr(config, "num_key_value_heads", 4),
                intermediate_size=getattr(config, "intermediate_size", 18944),
                vocab_size=getattr(config, "vocab_size", 152064),
            )
            rng_model = Qwen2ForCausalLM(text_cfg)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        rng_model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Move to GPU in fp16 — random model runs standalone (pretrained not yet loaded)
    device = "cpu"
    if torch.cuda.is_available():
        try:
            rng_model = rng_model.to(torch.float16).cuda()
            device = "cuda"
            logger.info(
                "Random model on GPU (fp16, %.1f GB)",
                sum(p.numel() * 2 for p in rng_model.parameters()) / 1e9,
            )
        except RuntimeError as e:
            logger.warning("GPU OOM for random model, falling back to CPU: %s", e)
            rng_model = rng_model.cpu()

    rng_model.eval()
    prompts = [_build_classification_prompt(s, task_name, modality) for s in series_list]
    n_test = max(1, int(len(prompts) * test_fraction))
    n_train = len(prompts) - n_test

    hidden = extract_hidden_states(rng_model, tokenizer, prompts)
    probe = train_linear_probe(
        hidden[:n_train], labels[:n_train],
        hidden[n_train:], labels[n_train:],
    )

    # Free VRAM before caller loads the pretrained model
    del rng_model
    if device == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Random model deleted, VRAM freed")

    return {"random_probe_f1": probe["f1_macro"], **probe}


# ── Full study orchestrator ────────────────────────────────────────────────────


def run_probing_study(
    model,
    tokenizer,
    task_data: "dict[str, tuple[list, list]]",
    modalities: "list[str]",
    output_dir: Path,
    test_fraction: float = 0.2,
    run_layer_wise: bool = True,
    run_prompting: bool = True,
    use_cot: bool = False,
    layer_step: int = 4,
    processor=None,
) -> dict:
    """Run the full probing vs prompting study for all tasks and modalities.

    For each task × modality:
      1. Prompting baseline (zero-shot classification)
      2. Probe (hidden states at best layer → logistic regression)
      3. Layer-wise analysis (optional, expensive)

    Results table mirrors Schumacher et al. Table 1:

    | Task | Prompting F1 | Probe F1 (d) | Probe F1 (v) | Probe F1 (d+v) | Random-Probe F1 |

    Returns:
        Full nested results dict saved to output_dir/probing_results.json
    """
    out = ensure_output_dir(str(output_dir))
    all_results: dict = {}

    for task_name, (series_list, labels) in task_data.items():
        if not series_list:
            logger.warning("Skipping %s — no samples", task_name)
            continue

        logger.info("\n=== Task: %s (%d samples) ===", task_name, len(series_list))

        # Stratified shuffle so every class appears in BOTH train and test,
        # avoiding the deterministic-tail bug where a minority class can be
        # entirely absent from the test set and inflate macro-F1.
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
            import numpy as _np
            y_arr = _np.array([str(l) for l in labels])
            # Can only stratify when every class has ≥2 samples; otherwise fall back.
            _, counts = _np.unique(y_arr, return_counts=True)
            if counts.min() >= 2:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=test_fraction, random_state=42,
                )
                train_idx, test_idx = next(sss.split(_np.zeros(len(y_arr)), y_arr))
                order = list(train_idx) + list(test_idx)
                series_list = [series_list[i] for i in order]
                labels = [labels[i] for i in order]
                logger.info("  stratified split: n_test=%d, min class count in test=%d",
                            len(test_idx),
                            int(min((y_arr[test_idx] == c).sum() for c in _np.unique(y_arr))))
            else:
                logger.info("  min class count=%d (<2) — using deterministic tail split",
                            int(counts.min()))
        except Exception as e:
            logger.warning("  stratified split failed: %s — using tail split", e)

        n_test = max(1, int(len(series_list) * test_fraction))
        n_train = len(series_list) - n_test

        task_results: dict = {"n_samples": len(series_list), "n_train": n_train, "n_test": n_test}

        for mod in modalities:
            logger.info("  Modality: %s", mod)
            mod_results: dict = {}

            # ── Probing (best layer = last) ────────────────────────────────────
            prompts = [_build_classification_prompt(s, task_name, mod) for s in series_list]
            hidden = None
            try:
                if (mod in ("v", "d+v", "m", "o", "va", "sp", "v2", "vh")
                or "+v" in mod or mod.endswith("+va") or mod.endswith("+sp")
                or mod.endswith("+v2") or mod.endswith("+vh")) and processor is not None:
                    hidden = extract_hidden_states_vision(
                        model, processor, series_list, task_name, modality=mod,
                    )
                else:
                    hidden = extract_hidden_states(model, tokenizer, prompts)
                probe_result = train_linear_probe(
                    hidden[:n_train], labels[:n_train],
                    hidden[n_train:], labels[n_train:],
                )
                mod_results["probe"] = probe_result
                ci = probe_result.get("bootstrap", {})
                logger.info(
                    "    Probe F1=%.3f  95%%CI=[%.3f, %.3f]",
                    probe_result["f1_macro"],
                    ci.get("ci_low") or 0.0, ci.get("ci_high") or 0.0,
                )
            except Exception as e:
                logger.error("    Probe failed: %s", e)
                mod_results["probe"] = {"error": str(e)}

            # ── Permutation negative control (same hidden states, shuffled labels) ──
            if hidden is not None:
                try:
                    rng = np.random.RandomState(42)
                    shuffled = list(labels)
                    rng.shuffle(shuffled)
                    perm = train_linear_probe(
                        hidden[:n_train], shuffled[:n_train],
                        hidden[n_train:], shuffled[n_train:],
                    )
                    mod_results["permutation"] = {
                        "f1_macro": perm["f1_macro"],
                        "accuracy": perm["accuracy"],
                        "bootstrap": perm.get("bootstrap"),
                    }
                    logger.info(
                        "    Permutation F1=%.3f (expect ~chance; sanity check)",
                        perm["f1_macro"],
                    )
                except Exception as e:
                    logger.warning("    Permutation failed: %s", e)

            # ── MLP probe (non-linear upper bound) ──────────────────────────────
            if hidden is not None:
                try:
                    mlp = train_mlp_probe(
                        hidden[:n_train], labels[:n_train],
                        hidden[n_train:], labels[n_train:],
                    )
                    mod_results["mlp_probe"] = mlp
                    logger.info(
                        "    MLP probe F1=%.3f (linear upper bound)",
                        mlp["f1_macro"],
                    )
                except Exception as e:
                    logger.warning("    MLP probe failed: %s", e)

            # ── Prompting baseline ────────────────────────────────────────────
            if run_prompting:
                try:
                    prompting_result = asyncio.run(
                        run_prompting_baseline(
                            model, tokenizer,
                            series_list[:n_train + n_test],
                            labels[:n_train + n_test],
                            task_name, mod,
                            cot=use_cot,
                        )
                    )
                    mod_results["prompting"] = prompting_result
                    logger.info("    Prompting F1=%.3f", prompting_result["f1_macro"])
                except Exception as e:
                    logger.error("    Prompting failed: %s", e)
                    mod_results["prompting"] = {"error": str(e)}

            # ── Layer-wise (optional) ─────────────────────────────────────────
            if run_layer_wise:
                try:
                    lw = run_layer_wise_analysis(
                        model, tokenizer, series_list, labels, task_name, [mod],
                        layer_step=layer_step,
                        processor=processor,
                        cache_dir=out / "hidden_cache",
                    )
                    mod_results["layer_wise"] = lw.get(mod, {})
                except Exception as e:
                    logger.error("    Layer-wise failed: %s", e)

            task_results[mod] = mod_results

        all_results[task_name] = task_results

    # Save
    (out / "probing_results.json").write_text(json.dumps(all_results, indent=2))

    # Summary table
    _save_summary_table(all_results, out)

    # Layer-wise plots
    _plot_layer_wise(all_results, modalities, out)

    # Confusion matrices
    try:
        _plot_confusion_matrices(all_results, modalities, out)
    except Exception as e:
        logger.warning("Confusion matrix plot failed: %s", e)

    return all_results


def _plot_confusion_matrices(results: dict, modalities: list, out: Path) -> None:
    """Grid of confusion matrices (rows=tasks, cols=modalities).

    Each cell shows counts + colour-coded row-normalised ratios. Turns the
    single-number F1 into a diagnostic of *which* classes get confused.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError as e:
        logger.warning("matplotlib/sklearn not available: %s", e)
        return

    tasks = list(CLASSIFICATION_TASKS.keys())
    fig, axes = plt.subplots(
        len(tasks), len(modalities),
        figsize=(4 * len(modalities), 3.5 * len(tasks)),
        squeeze=False,
    )
    fig.suptitle(
        "Probe Confusion Matrices (row-normalised) — Qwen2-VL-7B linear probe\n"
        "Rows = true class, columns = predicted class. Numbers = counts.",
        fontsize=11, y=1.01,
    )

    for row, task in enumerate(tasks):
        classes = CLASSIFICATION_TASKS[task]["classes"]
        task_res = results.get(task, {})
        for col, mod in enumerate(modalities):
            ax = axes[row][col]
            probe = task_res.get(mod, {}).get("probe", {})
            preds = probe.get("predictions", [])
            y_true = probe.get("y_test", [])
            if not preds or not y_true:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                ax.set_axis_off()
                continue
            cm = confusion_matrix(y_true, preds, labels=classes)
            row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
            cm_norm = cm / row_sums
            ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            short = [c[:9] for c in classes]
            ax.set_xticklabels(short, fontsize=7, rotation=30, ha="right")
            ax.set_yticklabels(short, fontsize=7)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    colour = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                            color=colour, fontsize=8)
            f1 = probe.get("f1_macro", 0.0)
            ci = probe.get("bootstrap", {})
            ci_str = ""
            if ci.get("ci_low") is not None:
                ci_str = f"  [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"
            ax.set_title(
                f"{task.replace('_', ' ').title()}  ×  {mod}\n"
                f"F1={f1:.3f}{ci_str}",
                fontsize=8,
            )
            if col == 0:
                ax.set_ylabel("true", fontsize=8)
            if row == len(tasks) - 1:
                ax.set_xlabel("predicted", fontsize=8)

    plt.tight_layout()
    save_figure(fig, "confusion_matrices", out)
    logger.info("Confusion matrices saved to %s/confusion_matrices.{pdf,png}", out)


def _save_summary_table(results: dict, out: Path) -> None:
    """Write markdown tables: headline F1 grid + CI / permutation / MLP detail."""
    lines = [
        "# Linear Probing Study — Summary Tables",
        "",
        "## Headline: Prompting vs Probe F1 (all modalities)",
        "",
        "| Task | Prompting F1 | Probe F1 (d) | Probe F1 (v) | Probe F1 (d+v) |",
        "|---|---|---|---|---|",
    ]
    for task, task_res in results.items():
        row = [task.replace("_", " ").title()]
        for mod in ["d", "v", "d+v"]:
            probe_f1 = task_res.get(mod, {}).get("probe", {}).get("f1_macro")
            prompt_f1 = task_res.get(mod, {}).get("prompting", {}).get("f1_macro")
            if mod == "d":
                row.insert(1, f"{prompt_f1:.3f}" if isinstance(prompt_f1, float) else "—")
            row.append(f"{probe_f1:.3f}" if isinstance(probe_f1, float) else "—")
        row = row[:5]
        while len(row) < 5:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    # Confidence intervals
    lines += [
        "",
        "## Probe F1 with 95% bootstrap CIs",
        "",
        "| Task | Modality | F1 | 95% CI | n_test |",
        "|---|---|---|---|---|",
    ]
    for task, task_res in results.items():
        for mod in ["d", "v", "d+v"]:
            probe = task_res.get(mod, {}).get("probe", {})
            f1 = probe.get("f1_macro")
            ci = probe.get("bootstrap") or {}
            n_test = probe.get("n_test", "—")
            if f1 is None:
                continue
            ci_str = f"[{ci.get('ci_low','?'):.3f}, {ci.get('ci_high','?'):.3f}]" if ci.get("ci_low") is not None else "—"
            lines.append(f"| {task.replace('_',' ')} | {mod} | {f1:.3f} | {ci_str} | {n_test} |")

    # Permutation negative control
    lines += [
        "",
        "## Permutation Negative Control",
        "Probe trained on SHUFFLED labels. Should hit chance F1 — confirms the real probe isn't",
        "just memorising or finding spurious structure.",
        "",
        "| Task | Modality | Real F1 | Permutation F1 | Chance F1 |",
        "|---|---|---|---|---|",
    ]
    for task, task_res in results.items():
        n_classes = len(CLASSIFICATION_TASKS[task]["classes"])
        chance = 1.0 / n_classes
        for mod in ["d", "v", "d+v"]:
            probe = task_res.get(mod, {}).get("probe", {})
            perm = task_res.get(mod, {}).get("permutation", {})
            if not probe.get("f1_macro"):
                continue
            lines.append(
                f"| {task.replace('_',' ')} | {mod} | {probe['f1_macro']:.3f} | "
                f"{perm.get('f1_macro', '—'):.3f} | {chance:.3f} |"
            )

    # MLP upper bound
    lines += [
        "",
        "## Linear vs MLP Probe (non-linear upper bound)",
        "MLP ≈ Linear → representation is genuinely linearly accessible.",
        "MLP ≫ Linear → information exists but is non-linearly encoded (weakens extraction-gap framing).",
        "",
        "| Task | Modality | Linear F1 | MLP F1 | Δ |",
        "|---|---|---|---|---|",
    ]
    for task, task_res in results.items():
        for mod in ["d", "v", "d+v"]:
            probe = task_res.get(mod, {}).get("probe", {})
            mlp = task_res.get(mod, {}).get("mlp_probe", {})
            lin_f1 = probe.get("f1_macro")
            mlp_f1 = mlp.get("f1_macro")
            if lin_f1 is None or mlp_f1 is None:
                continue
            delta = mlp_f1 - lin_f1
            lines.append(
                f"| {task.replace('_',' ')} | {mod} | {lin_f1:.3f} | "
                f"{mlp_f1:.3f} | {delta:+.3f} |"
            )

    (out / "probing_summary.md").write_text("\n".join(lines))
    logger.info("Summary table written to %s", out / "probing_summary.md")


def _plot_layer_wise(results: dict, modalities: list[str], out: Path) -> None:
    """Line plot of probe F1 vs layer depth for each task × modality."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for task, task_res in results.items():
        has_lw = any(
            "layer_wise" in task_res.get(mod, {})
            for mod in modalities
        )
        if not has_lw:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        for mod in modalities:
            lw = task_res.get(mod, {}).get("layer_wise", {})
            if not lw:
                continue
            layers = sorted(k for k in lw if lw[k] is not None)
            def _f1(v):
                # Back-compat: old runs stored float, new runs store dict
                return v["linear_f1"] if isinstance(v, dict) else v
            f1s = [_f1(lw[k]) for k in layers]
            ax.plot(layers, f1s, label=f"Linear ({mod})", marker=".", markersize=4)
            mlp_f1s = [lw[k].get("mlp_f1") for k in layers
                       if isinstance(lw[k], dict)]
            if any(v is not None for v in mlp_f1s):
                ax.plot(layers, mlp_f1s, label=f"MLP ({mod})",
                        marker="x", markersize=4, linestyle="--", alpha=0.6)

        # Mark prompting baseline
        for mod in modalities:
            prompt_f1 = task_res.get(mod, {}).get("prompting", {}).get("f1_macro")
            if isinstance(prompt_f1, float):
                ax.axhline(prompt_f1, linestyle="--", linewidth=1,
                           label=f"Prompting ({mod}) = {prompt_f1:.3f}")

        ax.set_xlabel("Layer depth")
        ax.set_ylabel("Probe F1 (macro)")
        ax.set_title(f"Layer-Wise Probe F1 — {task.replace('_', ' ').title()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_figure(fig, f"layer_wise_{task}", str(out))


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path="backend/.env", override=True)
        load_dotenv(dotenv_path=".env", override=False)
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Linear probing study: LLM hidden state encoding of football temporal patterns"
    )
    parser.add_argument(
        "--ground-truth",
        default="eval_output/dissertation/db_grounded/18_db_ground_truth.json",
        help="Path(s) to db_extractor ground truth JSON (comma-separated for multiple clips)",
    )
    parser.add_argument(
        "--model-path",
        default="/workspace/models/llama-3.2-11b-vision/",
        help="Path to loaded HuggingFace model (Llama-3.2-11B-Vision-Instruct recommended)",
    )
    parser.add_argument(
        "--modalities", nargs="+",
        choices=["d", "v", "d+v", "p", "m", "o", "pi", "b", "n", "ne",
                 "nc", "nec",
                 "pt", "sf", "pt+v", "sf+v",
                 "p+v", "pi+v", "n+v", "b+v", "ne+v",
                 "nc+v", "nec+v",
                 "va", "sp", "v2", "vh", "vfix",
                 "p+va", "p+sp", "p+v2", "p+vh", "p+vfix",
                 "n+va", "n+sp", "ne+va", "ne+sp", "ne+vfix",
                 "nc+va", "nc+sp", "nec+va", "nec+sp"],
        default=["d"],
        help="Time-series modalities to test: d=digit-space text, v=image, d+v=both",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        choices=list(CLASSIFICATION_TASKS.keys()),
        default=list(CLASSIFICATION_TASKS.keys()),
        help="Classification tasks to run",
    )
    parser.add_argument("--output", default="eval_output/dissertation/probing/")
    parser.add_argument("--no-layer-wise", action="store_true",
                        help="Skip layer-wise analysis (faster)")
    parser.add_argument("--layer-step", type=int, default=4,
                        help="Sample every N-th layer in layer-wise analysis (default 4)")
    parser.add_argument("--no-prompting", action="store_true",
                        help="Skip prompting baseline (run probing only)")
    parser.add_argument("--cot", action="store_true",
                        help="Enable chain-of-thought prompting baseline")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=30,
                        help="Frames per time-series window")
    parser.add_argument("--window-step", type=int, default=6,
                        help="Stride between windows")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit (bitsandbytes) — halves VRAM usage")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (bitsandbytes) — quarter VRAM usage")
    parser.add_argument("--model-type", choices=["causal", "qwen2vl"], default="causal",
                        help="Model architecture: causal (text-only) or qwen2vl (vision-language)")
    parser.add_argument("--random-baseline", action="store_true",
                        help="Run random-weight probe baseline (Schumacher Table 1: Random-Probe)")
    args = parser.parse_args()

    # Load ground truth(s)
    gt_paths = [p.strip() for p in args.ground_truth.split(",")]
    db_ground_truths = [load_db_ground_truth(p) for p in gt_paths]
    logger.info("Loaded %d ground truth clip(s)", len(db_ground_truths))

    # Prepare classification data
    task_data = prepare_classification_data(
        db_ground_truths,
        window_size=args.window_size,
        window_step=args.window_step,
    )
    # Filter to requested tasks
    task_data = {k: v for k, v in task_data.items() if k in args.tasks}

    try:
        import torch
    except ImportError:
        logger.error("torch is required. pip install torch")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Random baseline (BEFORE pretrained model to maximise free VRAM) ─
    if args.random_baseline:
        logger.info("=== Random-weight probe baseline (runs before pretrained model) ===")
        random_results: dict = {}
        for task_name, (series_list, labels) in task_data.items():
            if not series_list:
                continue
            try:
                res = run_random_baseline(
                    args.model_path, series_list, labels, task_name,
                    modality=args.modalities[0],
                    model_type=args.model_type,
                )
                random_results[task_name] = res
                logger.info("  %s random-probe F1=%.3f", task_name, res["random_probe_f1"])
            except Exception as e:
                logger.error("  %s random-probe failed: %s", task_name, e)
                random_results[task_name] = {"error": str(e)}
        (output_dir / "random_baseline.json").write_text(json.dumps(random_results, indent=2))
        logger.info("Random baseline saved")

    # ── Step 2: Load pretrained model ───────────────────────────────────────────
    load_kwargs: dict = {"device_map": "auto"}
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16

    processor = None
    if args.model_type == "qwen2vl":
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            logger.error("Qwen2-VL requires transformers>=4.45. pip install transformers --upgrade")
            sys.exit(1)
        logger.info("Loading Qwen2-VL model from %s ...", args.model_path)
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
        tokenizer = processor.tokenizer
        logger.info("Qwen2-VL loaded (vision-language model)")
    else:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.error("transformers required. pip install transformers")
            sys.exit(1)
        logger.info("Loading causal LM from %s ...", args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)

    model.eval()
    logger.info(
        "Model loaded. Parameters: %d M",
        sum(p.numel() for p in model.parameters()) // 1_000_000,
    )

    # Run study
    run_probing_study(
        model,
        tokenizer,
        task_data,
        args.modalities,
        output_dir,
        test_fraction=args.test_fraction,
        run_layer_wise=not args.no_layer_wise,
        run_prompting=not args.no_prompting,
        use_cot=args.cot,
        layer_step=args.layer_step,
        processor=processor,
    )

    logger.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    main()
