"""Team classification accuracy evaluator.

Two modes:
  --sample   Stratified sampling of player detections → crop images + CSV template
  (default)  Compute accuracy from completed annotations

Usage:
    # Step 1: Generate samples for manual annotation
    python -m backend.evaluation.team_classification --sample \\
        --tracks tracks.json --video video.mp4 --n-samples 200 --output eval_output/team/

    # Step 2: Fill in gt_team_id in the generated team_gt.csv

    # Step 3: Compute accuracy
    python -m backend.evaluation.team_classification \\
        --tracks tracks.json --annotations eval_output/team/team_gt.csv \\
        --output eval_output/team/
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    EvalConfig,
    ensure_output_dir,
    load_tracks,
    save_figure,
    save_latex_table,
)


# ── Sampling ──────────────────────────────────────────────────────────────────


def sample_for_annotation(
    tracks: list[dict],
    video_path: str,
    output_dir: str,
    n_samples: int = 200,
    seed: int = 42,
) -> None:
    """Stratified sampling of player detections for manual annotation.

    Extracts crop images and generates a CSV template with blank gt_team_id.
    """
    import cv2

    random.seed(seed)
    out = Path(output_dir)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Collect all detections (frame_idx, track_id, predicted_team_id)
    detections = []
    for frame in tracks:
        for p in frame.get("playerPositions", []):
            detections.append({
                "frame_idx": frame.get("frameNumber", 0),
                "track_id": p.get("trackId", p.get("id", -1)),
                "predicted_team_id": p.get("teamId", -1),
                "pixel_x": p.get("pixelX", 0),
                "pixel_y": p.get("pixelY", 0),
            })

    # Stratified by team
    team_0 = [d for d in detections if d["predicted_team_id"] == 0]
    team_1 = [d for d in detections if d["predicted_team_id"] == 1]
    other = [d for d in detections if d["predicted_team_id"] not in (0, 1)]

    n_each = n_samples // 2
    sample_0 = random.sample(team_0, min(n_each, len(team_0)))
    sample_1 = random.sample(team_1, min(n_each, len(team_1)))
    sampled = sample_0 + sample_1 + random.sample(other, min(10, len(other)))
    random.shuffle(sampled)

    print(f"Sampled {len(sampled)} detections (team_0={len(sample_0)}, team_1={len(sample_1)})")

    # Extract crops from video
    cap = cv2.VideoCapture(video_path)
    frame_cache: dict[int, np.ndarray] = {}

    rows = []
    for det in sampled:
        fi = det["frame_idx"]
        if fi not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame_img = cap.read()
            if ok:
                frame_cache[fi] = frame_img
            else:
                continue

        frame_img = frame_cache[fi]
        # Crop a 80x160 region around pixel centre
        cx, cy = int(det["pixel_x"]), int(det["pixel_y"])
        x1, y1 = max(0, cx - 40), max(0, cy - 80)
        x2, y2 = min(frame_img.shape[1], cx + 40), min(frame_img.shape[0], cy + 80)
        crop = frame_img[y1:y2, x1:x2]

        crop_name = f"{fi:06d}_{det['track_id']}.jpg"
        cv2.imwrite(str(crops_dir / crop_name), crop)
        rows.append({
            "frame_idx": fi,
            "track_id": det["track_id"],
            "predicted_team_id": det["predicted_team_id"],
            "gt_team_id": "",   # To be filled in by annotator
            "crop_file": crop_name,
            "annotator_notes": "",
        })

    cap.release()

    # Write CSV template
    csv_path = out / "team_gt.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV template: {csv_path}")
    print(f"Crops: {crops_dir}/")
    print(f"\nInstructions: fill in 'gt_team_id' (0 or 1) for each row by inspecting the crop images.")
    print(f"Then run without --sample to compute accuracy.")


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_team_accuracy(annotations_csv: str, output_dir: str) -> dict:
    """Compute team classification accuracy from completed annotations."""
    rows = []
    with open(annotations_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row.get("gt_team_id", "").strip()
            pred = row.get("predicted_team_id", "").strip()
            if gt == "" or gt not in ("0", "1"):
                continue  # Skip unannotated rows
            rows.append({
                "predicted": int(pred) if pred in ("0", "1") else -1,
                "ground_truth": int(gt),
            })

    if not rows:
        print("No annotated rows found. Fill in gt_team_id in the CSV first.")
        return {}

    n = len(rows)
    correct = sum(1 for r in rows if r["predicted"] == r["ground_truth"])
    accuracy = correct / n

    # Per-team metrics
    results: dict[str, dict] = {}
    for team in (0, 1):
        tp = sum(1 for r in rows if r["ground_truth"] == team and r["predicted"] == team)
        fp = sum(1 for r in rows if r["ground_truth"] != team and r["predicted"] == team)
        fn = sum(1 for r in rows if r["ground_truth"] == team and r["predicted"] != team)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[f"team_{team}"] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    # Confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    for r in rows:
        if 0 <= r["ground_truth"] <= 1 and 0 <= r["predicted"] <= 1:
            cm[r["ground_truth"]][r["predicted"]] += 1

    out = ensure_output_dir(output_dir)

    # Summary table
    table_rows = [
        ["Overall accuracy", f"{accuracy:.1%}", n],
        ["Team 0 precision", f"{results['team_0']['precision']:.1%}", ""],
        ["Team 0 recall", f"{results['team_0']['recall']:.1%}", ""],
        ["Team 0 F1", f"{results['team_0']['f1']:.1%}", ""],
        ["Team 1 precision", f"{results['team_1']['precision']:.1%}", ""],
        ["Team 1 recall", f"{results['team_1']['recall']:.1%}", ""],
        ["Team 1 F1", f"{results['team_1']['f1']:.1%}", ""],
    ]
    save_latex_table(
        headers=["Metric", "Value", "Support"],
        rows=table_rows,
        caption="SigLIP+KMeans team classification accuracy",
        name="team_accuracy",
        output_dir=str(out),
        label="tab:team_classification",
    )

    print(f"\n=== Team Classification Accuracy ===")
    print(f"  Overall: {accuracy:.1%} ({correct}/{n})")
    for k, v in results.items():
        print(f"  {k}: P={v['precision']:.1%}  R={v['recall']:.1%}  F1={v['f1']:.1%}")

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Team 0", "Pred: Team 1"])
    ax.set_yticklabels(["True: Team 0", "True: Team 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title("Team Classification Confusion Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_figure(fig, "team_confusion_matrix", str(out))

    print(f"\nOutputs saved to: {out}/")
    return {"accuracy": accuracy, "per_team": results, "confusion_matrix": cm.tolist()}


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Team classification accuracy evaluator")
    parser.add_argument("--tracks", required=True, help="Path to *_tracks.json")
    parser.add_argument("--video", help="Path to source video (required for --sample)")
    parser.add_argument("--annotations", help="Path to completed team_gt.csv")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output", default="eval_output/team")
    parser.add_argument("--sample", action="store_true", help="Generate annotation samples")
    args = parser.parse_args()

    if args.sample:
        if not args.video:
            parser.error("--video is required with --sample")
        tracks = load_tracks(args.tracks)
        sample_for_annotation(tracks, args.video, args.output, n_samples=args.n_samples)
    else:
        if not args.annotations:
            parser.error("--annotations is required without --sample")
        evaluate_team_accuracy(args.annotations, args.output)


if __name__ == "__main__":
    main()
