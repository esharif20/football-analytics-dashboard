"""Micro ground truth evaluation: annotate 5 keyframes, measure position error.

Two-phase workflow:

Phase A — generate annotation template:
    python3 -m evaluation.micro_ground_truth --sample \
        --tracks ../eval_output/10_tracks.json \
        --video ../eval_output/10_annotated.mp4 \
        --n-frames 5 \
        --output ../eval_output/phase16/micro_gt/

Phase B — compute accuracy from completed annotations:
    python3 -m evaluation.micro_ground_truth --evaluate \
        --template ../eval_output/phase16/micro_gt/annotation_template.csv \
        --tracks ../eval_output/10_tracks.json \
        --output ../eval_output/phase16/micro_gt/
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import ensure_output_dir, load_tracks, save_figure, save_latex_table


# ── Phase A: Sample ──────────────────────────────────────────────────────────

def extract_keyframe_images(video_path: str, frame_indices: list[int], out_dir: Path) -> list[Path]:
    """Extract specific frame indices from video as JPEG files."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paths = []
    for fi in frame_indices:
        fi = min(fi, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        path = out_dir / f"frame_{fi:04d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    cap.release()
    return paths


def overlay_pipeline_detections(
    video_path: str,
    frame_idx: int,
    players: dict,
    out_path: Path,
) -> None:
    """Draw pipeline-detected player positions on a frame and save."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return

    team_colors = {0: (200, 200, 200), 1: (255, 80, 80), 2: (80, 80, 255)}
    for tid, p in players.items():
        cx, cy = int(p.get("x", 0)), int(p.get("y", 0))
        team_id = p.get("teamId", 0)
        color = team_colors.get(team_id, (200, 200, 200))
        cv2.circle(frame, (cx, cy), 12, color, 2)
        cv2.putText(frame, str(tid), (cx - 8, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(out_path), frame)


def generate_sample(
    tracks_path: str,
    video_path: str,
    n_frames: int,
    output_dir: str,
) -> str:
    """Phase A: extract keyframes and write annotation_template.csv.

    Returns path to the template CSV.
    """
    out = ensure_output_dir(output_dir)
    frames = load_tracks(tracks_path)
    total = len(frames)

    # Uniformly spaced keyframe indices (avoid first/last 5% which may be blank)
    margin = max(1, total // 20)
    indices = np.linspace(margin, total - margin - 1, n_frames, dtype=int).tolist()

    rows: list[dict] = []
    for fi in indices:
        frame = frames[fi]
        players = frame.get("playerPositions", {})
        if isinstance(players, list):
            players = {str(p.get("trackId", i)): p for i, p in enumerate(players)}

        # Extract and annotate frame images
        if Path(video_path).exists():
            raw_path = out / f"frame_{fi:04d}_raw.jpg"
            ann_path = out / f"frame_{fi:04d}_annotated.jpg"
            import cv2
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_img = cap.read()
            cap.release()
            if ok:
                cv2.imwrite(str(raw_path), frame_img)
                overlay_pipeline_detections(video_path, fi, players, ann_path)

        for tid, p in players.items():
            rows.append({
                "frame_idx": fi,
                "player_id": str(tid),
                "pipeline_x": round(float(p.get("x", 0)), 1),
                "pipeline_y": round(float(p.get("y", 0)), 1),
                "pipeline_team": p.get("teamId", -1),
                "gt_x": "",
                "gt_y": "",
                "team_gt": "",
                "notes": "",
            })

    template_path = out / "annotation_template.csv"
    fieldnames = ["frame_idx", "player_id", "pipeline_x", "pipeline_y",
                  "pipeline_team", "gt_x", "gt_y", "team_gt", "notes"]
    with open(template_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Template written: {template_path}")
    print(f"  {len(indices)} keyframes: {indices}")
    print(f"  {len(rows)} player rows to annotate")
    print()
    print("Instructions:")
    print("  Open frame_{idx}_raw.jpg alongside frame_{idx}_annotated.jpg")
    print("  For each player you can see, fill in gt_x, gt_y (foot-contact pixel)")
    print("  and team_gt (1 or 2). Leave blank if player not visible/identifiable.")
    return str(template_path)


# ── Phase B: Evaluate ────────────────────────────────────────────────────────

def _load_template(template_path: str) -> list[dict]:
    rows = []
    with open(template_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _hungarian_match(
    pipeline_pts: list[tuple],
    gt_pts: list[tuple],
    max_dist_px: float = 80.0,
) -> list[tuple[int, int, float]]:
    """Simple greedy matching (sufficient for ~22 players per frame).

    Returns list of (pipeline_idx, gt_idx, pixel_distance).
    """
    used_p, used_g = set(), set()
    pairs = []
    # Build distance matrix
    dists = []
    for pi, (px, py) in enumerate(pipeline_pts):
        for gi, (gx, gy) in enumerate(gt_pts):
            d = math.hypot(px - gx, py - gy)
            if d <= max_dist_px:
                dists.append((d, pi, gi))
    dists.sort()
    for d, pi, gi in dists:
        if pi not in used_p and gi not in used_g:
            pairs.append((pi, gi, d))
            used_p.add(pi)
            used_g.add(gi)
    return pairs


def compute_position_error(
    rows: list[dict],
) -> dict:
    """Match GT annotations to pipeline detections per frame.

    Returns position error statistics in pixels (pitch projection done
    separately if homography available).
    """
    errors_px: list[float] = []
    tp = fp = fn = 0

    by_frame: dict[int, dict] = {}
    for row in rows:
        fi = int(row["frame_idx"])
        if fi not in by_frame:
            by_frame[fi] = {"pipeline": [], "gt": []}
        px = float(row["pipeline_x"])
        py = float(row["pipeline_y"])
        by_frame[fi]["pipeline"].append((px, py))
        if row["gt_x"] and row["gt_y"]:
            by_frame[fi]["gt"].append((float(row["gt_x"]), float(row["gt_y"])))

    for fi, data in by_frame.items():
        pp = data["pipeline"]
        gt = data["gt"]
        if not gt:
            fp += len(pp)
            continue
        pairs = _hungarian_match(pp, gt)
        tp += len(pairs)
        fp += len(pp) - len(pairs)
        fn += len(gt) - len(pairs)
        for pi, gi, d in pairs:
            errors_px.append(d)

    if not errors_px:
        return {"n_matched": 0, "mean_error_px": None, "median_error_px": None,
                "p90_error_px": None, "precision": None, "recall": None, "f1": None}

    errors_arr = np.array(errors_px)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_matched": int(tp),
        "n_fp": int(fp),
        "n_fn": int(fn),
        "mean_error_px": float(np.mean(errors_arr)),
        "median_error_px": float(np.median(errors_arr)),
        "p90_error_px": float(np.percentile(errors_arr, 90)),
        "std_error_px": float(np.std(errors_arr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_team_accuracy(rows: list[dict]) -> dict:
    """Accuracy of pipeline team assignment vs human-annotated team_gt."""
    correct = total = 0
    team_stats: dict[str, dict] = {}
    for row in rows:
        if row.get("team_gt") is None or str(row.get("team_gt", "")).strip() == "" or not row.get("gt_x"):
            continue
        total += 1
        p_team = str(row.get("pipeline_team", ""))
        g_team = str(row["team_gt"])
        if p_team not in team_stats:
            team_stats[p_team] = {"correct": 0, "total": 0}
        team_stats[p_team]["total"] += 1
        if p_team == g_team:
            correct += 1
            team_stats[p_team]["correct"] += 1
    if total == 0:
        return {"accuracy": None, "n_correct": 0, "n_total": 0}
    return {
        "accuracy": correct / total,
        "n_correct": int(correct),
        "n_total": int(total),
        "per_team": {t: v["correct"] / v["total"] for t, v in team_stats.items() if v["total"] > 0},
    }


def plot_error_scatter(rows: list[dict], output_dir: str) -> None:
    """Scatter: pipeline vs GT positions, coloured by match status."""
    matched_pairs: list[tuple] = []
    unmatched_p: list[tuple] = []
    unmatched_g: list[tuple] = []

    by_frame: dict[int, dict] = {}
    for row in rows:
        fi = int(row["frame_idx"])
        if fi not in by_frame:
            by_frame[fi] = {"pipeline": [], "gt": []}
        by_frame[fi]["pipeline"].append((float(row["pipeline_x"]), float(row["pipeline_y"])))
        if row["gt_x"] and row["gt_y"]:
            by_frame[fi]["gt"].append((float(row["gt_x"]), float(row["gt_y"])))

    for data in by_frame.values():
        pp, gt = data["pipeline"], data["gt"]
        if gt:
            pairs = _hungarian_match(pp, gt)
            used_p = {p for p, _, _ in pairs}
            used_g = {g for _, g, _ in pairs}
            for pi, gi, _ in pairs:
                matched_pairs.append((pp[pi], gt[gi]))
            for i, p in enumerate(pp):
                if i not in used_p:
                    unmatched_p.append(p)
            for i, g in enumerate(gt):
                if i not in used_g:
                    unmatched_g.append(g)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (px, py), (gx, gy) in matched_pairs:
        ax.plot([px, gx], [py, gy], "gray", alpha=0.3, linewidth=0.8)
    if matched_pairs:
        px_vals = [p[0] for p, _ in matched_pairs]
        py_vals = [p[1] for p, _ in matched_pairs]
        gx_vals = [g[0] for _, g in matched_pairs]
        gy_vals = [g[1] for _, g in matched_pairs]
        ax.scatter(px_vals, py_vals, c="#2980b9", s=30, zorder=3, label="Pipeline")
        ax.scatter(gx_vals, gy_vals, c="#27ae60", s=30, marker="x", zorder=3, label="Ground truth")
    if unmatched_p:
        ax.scatter(*zip(*unmatched_p), c="#e74c3c", s=20, marker="o", alpha=0.5, label="FP (pipeline only)")
    if unmatched_g:
        ax.scatter(*zip(*unmatched_g), c="#f39c12", s=20, marker="x", alpha=0.5, label="FN (GT only)")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    ax.set_title("Micro Ground Truth: Pipeline vs Annotated Positions")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "position_error_scatter", output_dir)


def run_evaluate(template_path: str, output_dir: str) -> None:
    out = ensure_output_dir(output_dir)
    rows = _load_template(template_path)

    annotated = [r for r in rows if r.get("gt_x") and r.get("gt_y")]
    if not annotated:
        print("No annotations found in template (gt_x/gt_y columns are empty).")
        print("Fill in the template first, then re-run with --evaluate.")
        return

    print(f"Loaded {len(rows)} rows, {len(annotated)} with GT annotations.")

    pos = compute_position_error(rows)
    team = compute_team_accuracy(rows)

    # Save JSON
    results = {"position_error": pos, "team_accuracy": team}
    (out / "micro_gt_results.json").write_text(json.dumps(results, indent=2))

    # Position error LaTeX table
    if pos["mean_error_px"] is not None:
        save_latex_table(
            headers=["Metric", "Value"],
            rows=[
                ["Matched pairs (TP)", pos["n_matched"]],
                ["False positives (pipeline only)", pos["n_fp"]],
                ["False negatives (GT only)", pos["n_fn"]],
                ["Mean position error (px)", f"{pos['mean_error_px']:.1f}"],
                ["Median position error (px)", f"{pos['median_error_px']:.1f}"],
                ["P90 position error (px)", f"{pos['p90_error_px']:.1f}"],
                ["Detection Precision", f"{pos['precision']:.3f}"],
                ["Detection Recall", f"{pos['recall']:.3f}"],
                ["Detection F1", f"{pos['f1']:.3f}"],
            ],
            caption="Micro ground truth evaluation: position error and detection metrics.",
            name="position_error",
            output_dir=str(out),
            label="tab:position_error",
        )
        print(f"  Mean position error: {pos['mean_error_px']:.1f}px")
        print(f"  Detection P/R/F1: {pos['precision']:.3f} / {pos['recall']:.3f} / {pos['f1']:.3f}")

    # Team accuracy LaTeX table
    if team["accuracy"] is not None:
        save_latex_table(
            headers=["Metric", "Value"],
            rows=[
                ["Team accuracy", f"{team['accuracy']:.3f}"],
                ["Correct", team["n_correct"]],
                ["Total annotated", team["n_total"]],
            ],
            caption="Team classification accuracy on micro ground truth sample.",
            name="team_accuracy",
            output_dir=str(out),
            label="tab:team_accuracy",
        )
        print(f"  Team accuracy: {team['accuracy']:.1%} ({team['n_correct']}/{team['n_total']})")

    # Scatter plot
    plot_error_scatter(rows, str(out))

    print(f"\nOutputs saved to: {out}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Micro ground truth evaluation: 5-keyframe annotation for in-domain accuracy."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", action="store_true", help="Phase A: generate annotation template")
    mode.add_argument("--evaluate", action="store_true", help="Phase B: evaluate completed annotations")

    parser.add_argument("--tracks", help="Path to tracks JSON (Phase A)")
    parser.add_argument("--video", help="Path to video file for keyframe extraction (Phase A)")
    parser.add_argument("--n-frames", type=int, default=5, help="Number of keyframes to sample")
    parser.add_argument("--template", help="Path to completed annotation CSV (Phase B)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    if args.sample:
        if not args.tracks:
            parser.error("--tracks required for --sample")
        video = args.video or ""
        generate_sample(args.tracks, video, args.n_frames, args.output)
    else:
        if not args.template:
            parser.error("--template required for --evaluate")
        run_evaluate(args.template, args.output)


if __name__ == "__main__":
    main()
