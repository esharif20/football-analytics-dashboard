"""SoccerNet-Tracking benchmark evaluation.

Downloads SoccerNet-Tracking sequences, formats tracker output as MOT Challenge
CSVs, and evaluates using trackeval (HOTA/MOTA/IDF1).

Usage — download data and prepare GT:
    python -m evaluation.soccernet_eval --download --soccernet-dir /path/to/soccernet/

Usage — run evaluation (after pipeline has produced MOT CSVs):
    python -m evaluation.soccernet_eval \
        --mot-dir /path/to/mot_csvs/ \
        --gt-dir /path/to/soccernet/tracking/test/ \
        --output eval_output/phase16/soccernet/

Prerequisites:
    pip install SoccerNet trackeval
    GPU instance with pipeline running to produce MOT CSVs (see pipeline export_mot_csv()).

Note on domain gap:
    Our pipeline is trained on DFL Bundesliga broadcast data. SoccerNet contains
    matches from multiple leagues with varying camera heights, zoom levels, and
    jersey styles. Cross-domain evaluation naturally shows reduced recall, which
    is discussed as a generalization limitation.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from ._common import ensure_output_dir, save_latex_table


# ── SoccerNet download ────────────────────────────────────────────────────────


def download_soccernet_gt(soccernet_dir: str, split: str = "test") -> str:
    """Download SoccerNet-Tracking ground truth annotations.

    Returns path to extracted GT directory.
    Requires: pip install SoccerNet
    """
    from SoccerNet.Downloader import SoccerNetDownloader

    os.makedirs(soccernet_dir, exist_ok=True)
    d = SoccerNetDownloader(LocalDirectory=soccernet_dir)
    print(f"Downloading SoccerNet tracking {split} split to {soccernet_dir} ...")
    print("WARNING: This is ~8.7GB. Consider using --n-sequences to limit download.")
    d.downloadDataTask(task="tracking", split=[split], source="HuggingFace")
    gt_dir = Path(soccernet_dir) / "tracking" / split
    print(f"GT downloaded to: {gt_dir}")
    return str(gt_dir)


# ── MOT CSV formatting ────────────────────────────────────────────────────────


def format_mot_for_trackeval(
    mot_csv_dir: str,
    gt_dir: str,
    tracker_name: str = "our_pipeline",
) -> str:
    """Reorganise MOT CSVs into the directory structure trackeval expects.

    trackeval expects:
      <tracker_name>/<sequence_name>/pedestrian/data/<sequence_name>.txt

    Args:
        mot_csv_dir: Directory with our MOT CSVs (one per video).
        gt_dir: SoccerNet GT directory (to find sequence names).
        tracker_name: Label used in output directory.

    Returns:
        Path to formatted tracker directory.
    """
    mot_dir = Path(mot_csv_dir)
    out_base = mot_dir / "trackeval_formatted" / tracker_name
    out_base.mkdir(parents=True, exist_ok=True)

    mot_files = list(mot_dir.glob("*_mot.csv"))
    if not mot_files:
        raise FileNotFoundError(f"No *_mot.csv files found in {mot_csv_dir}")

    for mot_file in mot_files:
        seq_name = mot_file.stem.replace("_mot", "")
        out_seq = out_base / seq_name / "pedestrian" / "data"
        out_seq.mkdir(parents=True, exist_ok=True)
        # trackeval expects tab-separated or space-separated with no header
        with open(mot_file) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
        with open(out_seq / f"{seq_name}.txt", "w") as f:
            for row in rows:
                f.write(",".join(row) + "\n")
        print(f"  Formatted {mot_file.name} → {out_seq}/{seq_name}.txt")

    return str(out_base.parent)


# ── Evaluation ────────────────────────────────────────────────────────────────


def run_trackeval(
    gt_dir: str,
    tracker_dir: str,
    tracker_name: str = "our_pipeline",
) -> dict:
    """Run trackeval and parse HOTA/MOTA/IDF1 results.

    Requires: pip install trackeval
    Returns: {sequence -> {HOTA, DetA, AssA, MOTA, IDF1}} dict.
    """
    try:
        import trackeval
    except ImportError:
        raise ImportError("pip install trackeval")

    # trackeval CLI
    result = subprocess.run(
        [
            sys.executable, "-m", "trackeval.scripts.run_mot_challenge",
            "--GT_FOLDER", gt_dir,
            "--TRACKERS_FOLDER", tracker_dir,
            "--TRACKERS_TO_EVAL", tracker_name,
            "--METRICS", "HOTA", "CLEAR", "Identity",
            "--SPLIT_TO_EVAL", "test",
            "--OUTPUT_SUMMARY", "True",
            "--OUTPUT_EMPTY_CLASSES", "False",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("trackeval stderr:", result.stderr[-2000:])
        raise RuntimeError("trackeval failed")

    # Parse summary output
    scores: dict = {}
    for line in result.stdout.splitlines():
        # Format: SEQUENCE  HOTA  DetA  AssA  DetRe  DetPr  AssRe  AssPr  LocA  ...
        parts = line.split()
        if len(parts) >= 5 and parts[0] not in ("Seq", "COMBINED"):
            try:
                scores[parts[0]] = {
                    "HOTA": float(parts[1]),
                    "DetA": float(parts[2]),
                    "AssA": float(parts[3]),
                    "MOTA": float(parts[4]) if len(parts) > 4 else None,
                    "IDF1": float(parts[5]) if len(parts) > 5 else None,
                }
            except (ValueError, IndexError):
                pass

    return scores


# ── Results table ─────────────────────────────────────────────────────────────


def build_results_table(scores: dict, output_dir: str) -> None:
    """Save HOTA/MOTA/IDF1 results as LaTeX table."""
    out = ensure_output_dir(output_dir)

    rows = []
    hota_vals, mota_vals, idf1_vals = [], [], []
    for seq, s in sorted(scores.items()):
        hota = s.get("HOTA", 0) or 0
        mota = s.get("MOTA", 0) or 0
        idf1 = s.get("IDF1", 0) or 0
        rows.append([seq, f"{hota:.1f}", f"{mota:.1f}", f"{idf1:.1f}"])
        hota_vals.append(hota)
        mota_vals.append(mota)
        idf1_vals.append(idf1)

    if hota_vals:
        n = len(hota_vals)
        avg = [
            "Average",
            f"{sum(hota_vals)/n:.1f}",
            f"{sum(mota_vals)/n:.1f}",
            f"{sum(idf1_vals)/n:.1f}",
        ]
        rows.append(avg)

    save_latex_table(
        headers=["Sequence", "HOTA", "MOTA", "IDF1"],
        rows=rows,
        caption="SoccerNet-Tracking benchmark results (cross-domain evaluation)",
        name="soccernet_results",
        output_dir=str(out),
        label="tab:soccernet_results",
    )

    # JSON for programmatic use
    (out / "soccernet_scores.json").write_text(json.dumps(scores, indent=2))
    print(f"\nResults written to {out}/")

    # Print summary
    if hota_vals:
        n = len(hota_vals)
        print(f"\n=== SoccerNet Benchmark Results ===")
        print(f"  Sequences evaluated: {n}")
        print(f"  Mean HOTA: {sum(hota_vals)/n:.1f}")
        print(f"  Mean MOTA: {sum(mota_vals)/n:.1f}")
        print(f"  Mean IDF1: {sum(idf1_vals)/n:.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SoccerNet-Tracking benchmark evaluation")
    parser.add_argument("--download", action="store_true", help="Download SoccerNet GT data")
    parser.add_argument("--soccernet-dir", default="/tmp/soccernet/",
                        help="Local directory for SoccerNet data")
    parser.add_argument("--mot-dir", help="Directory containing *_mot.csv files from pipeline")
    parser.add_argument("--gt-dir", help="SoccerNet GT directory (tracking/test/)")
    parser.add_argument("--output", default="eval_output/phase16/soccernet/")
    parser.add_argument("--tracker-name", default="our_pipeline")
    args = parser.parse_args()

    if args.download:
        download_soccernet_gt(args.soccernet_dir)
        return

    if not args.mot_dir or not args.gt_dir:
        parser.error("--mot-dir and --gt-dir required for evaluation")

    print("Formatting tracker output for trackeval...")
    tracker_dir = format_mot_for_trackeval(args.mot_dir, args.gt_dir, args.tracker_name)

    print("Running trackeval...")
    scores = run_trackeval(args.gt_dir, tracker_dir, args.tracker_name)

    if scores:
        build_results_table(scores, args.output)
    else:
        print("No scores parsed from trackeval output.")
        print("Check that sequence names in mot_dir match SoccerNet GT sequences.")


if __name__ == "__main__":
    main()
