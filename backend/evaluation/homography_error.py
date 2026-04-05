"""Homography reprojection error evaluator.

Measures the Euclidean distance (in metres) between:
  - pitch coordinates produced by the homography transform
  - known ground-truth coordinates for annotated pitch landmarks

Annotation format (CSV):
    frame_idx,pixel_x,pixel_y,pitch_x_cm,pitch_y_cm,landmark_name

Known pitch coordinates come from SoccerPitchConfiguration in
backend/pipeline/src/pitch/config.py (FIFA 105m × 68m standard).

Usage:
    python -m backend.evaluation.homography_error \\
        --homography path/to/homography.json \\
        --annotations path/to/homography_gt.csv \\
        --output eval_output/homography/
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import (
    ensure_output_dir,
    load_homography_matrices,
    save_figure,
    save_latex_table,
)


# ── Evaluation ────────────────────────────────────────────────────────────────


def transform_point(matrix: list[list[float]], px: float, py: float) -> tuple[float, float]:
    """Apply a 3x3 homography matrix to a pixel coordinate."""
    m = np.array(matrix, dtype=np.float64)
    pt = np.array([[[px, py]]], dtype=np.float64)
    out = cv2_perspective_transform(m, pt)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def cv2_perspective_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """perspectiveTransform without requiring OpenCV import in evaluation."""
    n = points.shape[1]
    flat = points.reshape(-1, 2)
    ones = np.ones((len(flat), 1), dtype=np.float64)
    homog = np.hstack([flat, ones])  # (N, 3)
    transformed = (matrix @ homog.T).T  # (N, 3)
    transformed /= transformed[:, 2:3]  # Normalize
    return transformed[:, :2].reshape(1, n, 2)


def evaluate_homography(
    homography_path: str,
    annotations_csv: str,
    output_dir: str,
) -> dict:
    """Compute reprojection error from annotated landmarks."""
    matrices = load_homography_matrices(homography_path)

    rows = []
    with open(annotations_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame_idx": int(row["frame_idx"]),
                "pixel_x": float(row["pixel_x"]),
                "pixel_y": float(row["pixel_y"]),
                "pitch_x_cm": float(row["pitch_x_cm"]),
                "pitch_y_cm": float(row["pitch_y_cm"]),
                "landmark_name": row.get("landmark_name", ""),
            })

    if not rows:
        print("No annotations found in CSV.")
        return {}

    errors_cm = []
    errors_m = []
    per_frame: dict[int, list[float]] = {}
    detailed_rows = []

    for row in rows:
        fi = row["frame_idx"]
        if fi not in matrices:
            print(f"  Warning: No homography matrix for frame {fi}, skipping")
            continue

        pred_x, pred_y = transform_point(matrices[fi], row["pixel_x"], row["pixel_y"])
        gt_x, gt_y = row["pitch_x_cm"], row["pitch_y_cm"]

        error_cm = float(np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2))
        error_m = error_cm / 100.0
        errors_cm.append(error_cm)
        errors_m.append(error_m)

        per_frame.setdefault(fi, []).append(error_m)
        detailed_rows.append({
            "frame": fi,
            "landmark": row["landmark_name"],
            "pixel": f"({row['pixel_x']:.0f}, {row['pixel_y']:.0f})",
            "pred_cm": f"({pred_x:.0f}, {pred_y:.0f})",
            "gt_cm": f"({gt_x:.0f}, {gt_y:.0f})",
            "error_m": error_m,
        })

    if not errors_m:
        print("No valid homography matrices matched to annotations.")
        return {}

    errors_arr = np.array(errors_m)
    out = ensure_output_dir(output_dir)

    # --- Summary table ---
    summary_rows = [
        ["N landmarks", len(errors_arr)],
        ["N frames", len(per_frame)],
        ["Mean error", f"{errors_arr.mean():.3f} m"],
        ["Median error", f"{np.median(errors_arr):.3f} m"],
        ["Std error", f"{errors_arr.std():.3f} m"],
        ["Min error", f"{errors_arr.min():.3f} m"],
        ["Max error", f"{errors_arr.max():.3f} m"],
        ["% within 1m", f"{(errors_arr < 1.0).mean():.1%}"],
        ["% within 2m", f"{(errors_arr < 2.0).mean():.1%}"],
    ]
    save_latex_table(
        headers=["Metric", "Value"],
        rows=summary_rows,
        caption="Homography reprojection error (pitch coordinate accuracy)",
        name="homography_error_summary",
        output_dir=str(out),
        label="tab:homography_error",
    )

    print("\n=== Homography Reprojection Error ===")
    for metric, value in summary_rows:
        print(f"  {metric}: {value}")

    # --- Per-landmark table ---
    landmark_rows = [
        [r["landmark"], r["frame"], r["pixel"], r["pred_cm"], r["gt_cm"], f"{r['error_m']:.3f} m"]
        for r in sorted(detailed_rows, key=lambda x: x["error_m"], reverse=True)
    ]
    save_latex_table(
        headers=["Landmark", "Frame", "Pixel (px)", "Predicted (cm)", "GT (cm)", "Error"],
        rows=landmark_rows,
        caption="Per-landmark reprojection errors",
        name="homography_error_per_landmark",
        output_dir=str(out),
        label="tab:homography_landmarks",
    )

    # --- Error histogram ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors_arr, bins=15, color="#4f86c6", edgecolor="white")
    ax.axvline(errors_arr.mean(), color="tomato", linestyle="--", linewidth=1.5, label=f"Mean {errors_arr.mean():.2f}m")
    ax.axvline(np.median(errors_arr), color="gold", linestyle="--", linewidth=1.5, label=f"Median {np.median(errors_arr):.2f}m")
    ax.set_xlabel("Reprojection error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Homography Reprojection Error Distribution")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "homography_error_histogram", str(out))

    # --- Scatter: predicted vs GT pitch positions ---
    pred_xs = []
    pred_ys = []
    gt_xs = []
    gt_ys = []
    for row in rows:
        fi = row["frame_idx"]
        if fi not in matrices:
            continue
        px, py = transform_point(matrices[fi], row["pixel_x"], row["pixel_y"])
        pred_xs.append(px / 100)  # Convert to m
        pred_ys.append(py / 100)
        gt_xs.append(row["pitch_x_cm"] / 100)
        gt_ys.append(row["pitch_y_cm"] / 100)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(gt_xs, gt_ys, c="tomato", s=60, zorder=5, label="Ground truth", marker="x", linewidths=2)
    ax2.scatter(pred_xs, pred_ys, c="#4f86c6", s=40, zorder=4, label="Predicted", alpha=0.8)
    for gx, gy, px, py in zip(gt_xs, gt_ys, pred_xs, pred_ys):
        ax2.plot([gx, px], [gy, py], "gray", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("Pitch X (m)")
    ax2.set_ylabel("Pitch Y (m)")
    ax2.set_title("Homography: Predicted vs Ground Truth Pitch Positions")
    ax2.legend()
    fig2.tight_layout()
    save_figure(fig2, "homography_scatter", str(out))

    print(f"\nOutputs saved to: {out}/")
    return {
        "n_landmarks": len(errors_arr),
        "mean_error_m": float(errors_arr.mean()),
        "median_error_m": float(np.median(errors_arr)),
        "max_error_m": float(errors_arr.max()),
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Homography reprojection error evaluator")
    parser.add_argument("--homography", required=True, help="Path to *_homography.json")
    parser.add_argument("--annotations", required=True, help="Path to homography_gt.csv")
    parser.add_argument("--output", default="eval_output/homography")
    args = parser.parse_args()

    evaluate_homography(args.homography, args.annotations, args.output)


if __name__ == "__main__":
    main()
