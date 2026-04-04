"""claim_audit.py — Dissertation evaluation: per-claim audit across providers and videos.

Loads grounding artifact JSON files, builds a flat audit CSV, computes bootstrap
confidence intervals on grounding rate, runs McNemar's test for baseline vs phase15,
and emits LaTeX tables + a forest-plot PDF.

Usage (from backend/):
    python3 -m evaluation.claim_audit \\
        --phase15-dir ../eval_output/phase15/grounding/ \\
        --baseline-dir ../eval_output/grounding/ \\
        --output ../eval_output/phase15/audit/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import uuid
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import save_latex_table, save_figure, ensure_output_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

VERDICTS = {"verified", "refuted", "unverifiable", "plausible"}
AUDIT_CSV_HEADERS = [
    "claim_id", "video", "provider", "format", "analysis_type",
    "text", "claim_type", "referenced_metric", "referenced_value",
    "verdict", "actual_value", "explanation",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _infer_provider(filename: str) -> str:
    """Infer provider name from artifact filename prefix (e.g. 'openai_...' -> 'openai')."""
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    return parts[0] if len(parts) > 1 else "unknown"


def _parse_artifact(path: Path, video_id: str, provider: str) -> list[dict]:
    """Parse a single artifact JSON and return flat claim rows. Returns [] on error."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping %s: %s", path, exc)
        return []

    fmt = data.get("format", "unknown")
    analysis_type = data.get("analysis_type", "unknown")
    verification_results = data.get("verification_results", [])

    if not isinstance(verification_results, list):
        logger.warning("Skipping %s: 'verification_results' is not a list", path)
        return []

    rows = []
    for item in verification_results:
        if not isinstance(item, dict):
            continue
        verdict = item.get("verdict", "unverifiable")
        if verdict not in VERDICTS:
            logger.warning("Unknown verdict %r in %s — treating as unverifiable", verdict, path)
            verdict = "unverifiable"
        rows.append({
            "claim_id": str(uuid.uuid4()),
            "video": video_id,
            "provider": provider,
            "format": fmt,
            "analysis_type": analysis_type,
            "text": item.get("text", ""),
            "claim_type": item.get("claim_type", ""),
            "referenced_metric": item.get("referenced_metric", ""),
            "referenced_value": str(item.get("referenced_value", "")),
            "verdict": verdict,
            "actual_value": str(item.get("actual_value", "")),
            "explanation": item.get("explanation", ""),
        })
    return rows


def load_all_claims(grounding_dir: str, video_id: str = "unknown", provider: str = "unknown") -> list[dict]:
    """Walk grounding_dir/artifacts/*.json, parse each, return flat claim rows.

    Each row: claim_id (uuid), video, provider, format, analysis_type, text,
    claim_type, referenced_metric, referenced_value, verdict, actual_value, explanation.
    """
    artifacts_dir = Path(grounding_dir) / "artifacts"
    if not artifacts_dir.is_dir():
        logger.warning("Artifacts dir not found: %s", artifacts_dir)
        return []

    claims: list[dict] = []
    for path in sorted(artifacts_dir.glob("*.json")):
        inferred_provider = _infer_provider(path.name) if provider == "unknown" else provider
        claims.extend(_parse_artifact(path, video_id=video_id, provider=inferred_provider))

    logger.info("Loaded %d claims from %s", len(claims), artifacts_dir)
    return claims


def load_phase15_claims(phase15_dir: str) -> list[dict]:
    """Load claims from all videos under phase15_dir.

    Expected layout: phase15_dir/<video_id>/artifacts/*.json
    Provider is inferred from filename prefix.
    """
    base = Path(phase15_dir)
    all_claims: list[dict] = []
    if not base.is_dir():
        logger.warning("Phase15 dir not found: %s", base)
        return all_claims

    for video_dir in sorted(base.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        claims = load_all_claims(str(video_dir), video_id=video_id, provider="unknown")
        all_claims.extend(claims)

    return all_claims


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def build_audit_csv(claims: list[dict], output_path: str) -> None:
    """Write claims list to CSV at output_path."""
    ensure_output_dir(str(Path(output_path).parent))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_CSV_HEADERS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(claims)
    logger.info("Wrote audit CSV: %s (%d rows)", output_path, len(claims))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(
    verdicts: list[str],
    n_boot: int = 10000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for grounding rate (proportion of 'verified').

    Returns (mean, lower, upper).
    """
    if not verdicts:
        return 0.0, 0.0, 0.0

    arr = np.array([1 if v == "verified" else 0 for v in verdicts], dtype=float)
    n = len(arr)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean = float(arr.mean())
    return mean, lower, upper


def mcnemar_test(verdicts_a: list[str], verdicts_b: list[str]) -> dict:
    """Paired McNemar's chi-squared test comparing two sets of verdicts.

    Pairs by index (truncated to min length).
    Returns {chi2, p_value, n_pairs, n_b_not_a, n_a_not_b, significant_at_005}.
    """
    n = min(len(verdicts_a), len(verdicts_b))
    if n == 0:
        return {
            "chi2": float("nan"), "p_value": float("nan"),
            "n_pairs": 0, "n_b_not_a": 0, "n_a_not_b": 0,
            "significant_at_005": False,
        }

    a_correct = [1 if v == "verified" else 0 for v in verdicts_a[:n]]
    b_correct = [1 if v == "verified" else 0 for v in verdicts_b[:n]]

    # Discordant cells: b correct, a not (n_b_not_a) and a correct, b not (n_a_not_b)
    n_b_not_a = sum(1 for a, b in zip(a_correct, b_correct) if b == 1 and a == 0)
    n_a_not_b = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)

    discordant = n_b_not_a + n_a_not_b
    if discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        # McNemar with continuity correction (Edwards)
        chi2 = float((abs(n_b_not_a - n_a_not_b) - 1) ** 2 / discordant)
        # Chi-squared CDF approximation via regularised incomplete gamma (no scipy)
        p_value = _chi2_sf(chi2, df=1)

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "n_pairs": n,
        "n_b_not_a": n_b_not_a,
        "n_a_not_b": n_a_not_b,
        "significant_at_005": p_value < 0.05,
    }


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function of chi-squared distribution using numpy for df=1."""
    if x <= 0:
        return 1.0
    # For df=1: SF(x) = erfc(sqrt(x/2))
    return float(np.real(np.exp(np.log(np.maximum(
        1.0 - _erf(np.sqrt(x / 2.0) / np.sqrt(2.0) * np.sqrt(2.0)),
        1e-300,
    )))))


def _erf(x: float) -> float:
    """Numerical erf approximation (Abramowitz & Stegun 7.1.26)."""
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    result = 1.0 - poly * np.exp(-(x ** 2))
    return result if x >= 0 else -result


def cohens_kappa_simple(y1: list[str], y2: list[str]) -> float:
    """Cohen's kappa without sklearn. Pure Python."""
    n = min(len(y1), len(y2))
    if n == 0:
        return float("nan")

    categories = sorted(set(y1[:n]) | set(y2[:n]))
    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    # Confusion matrix
    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1[:n], y2[:n]):
        cm[cat_idx[a]][cat_idx[b]] += 1

    p_o = sum(cm[i][i] for i in range(k)) / n  # observed agreement
    p_e = sum(
        (sum(cm[i][j] for j in range(k)) / n) * (sum(cm[j][i] for j in range(k)) / n)
        for i in range(k)
    )  # expected agreement

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_forest(ci_data: list[dict], output_dir: str) -> None:
    """Forest plot of bootstrap CIs.

    ci_data: list of {label, mean, lower, upper}
    """
    if not ci_data:
        logger.warning("No CI data for forest plot — skipping")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(ci_data) * 0.55 + 1.5)))

    y_positions = list(range(len(ci_data)))
    labels = [d["label"] for d in ci_data]
    means = [d["mean"] for d in ci_data]
    lowers = [d["lower"] for d in ci_data]
    uppers = [d["upper"] for d in ci_data]

    xerr_lo = [m - lo for m, lo in zip(means, lowers)]
    xerr_hi = [hi - m for m, hi in zip(means, uppers)]

    ax.errorbar(
        means, y_positions,
        xerr=[xerr_lo, xerr_hi],
        fmt="o", color="steelblue", ecolor="steelblue",
        capsize=4, capthick=1.5, linewidth=1.5, markersize=6,
    )
    ax.axvline(x=0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Grounding Rate (95% Bootstrap CI)", fontsize=10)
    ax.set_title("Bootstrap Confidence Intervals by Video / Provider", fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    save_figure(fig, "bootstrap_ci_forest", output_dir)
    logger.info("Saved forest plot to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _group_verdicts(claims: list[dict], key_fn) -> dict[str, list[str]]:
    """Group verdict strings by a computed key."""
    groups: dict[str, list[str]] = {}
    for c in claims:
        k = key_fn(c)
        groups.setdefault(k, []).append(c["verdict"])
    return groups


def _print_summary(ci_data: list[dict]) -> None:
    header = f"{'Label':<40} {'Mean':>6} {'Lower':>7} {'Upper':>7}  N"
    print("\n" + header)
    print("-" * len(header))
    for d in ci_data:
        print(f"{d['label']:<40} {d['mean']:>6.3f} {d['lower']:>7.3f} {d['upper']:>7.3f}  {d.get('n','')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Claim audit for dissertation grounding evaluation")
    parser.add_argument("--phase15-dir", required=True, help="Root of phase15 grounding output")
    parser.add_argument("--baseline-dir", required=True, help="Phase14 baseline grounding dir (single artifacts/)")
    parser.add_argument("--output", required=True, help="Output directory for audit artefacts")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap iterations (default 10000)")
    args = parser.parse_args()

    output_dir = args.output
    ensure_output_dir(output_dir)

    # 1. Load all phase15 claims
    phase15_claims = load_phase15_claims(args.phase15_dir)
    logger.info("Total phase15 claims: %d", len(phase15_claims))

    # 2. Load baseline claims
    baseline_claims = load_all_claims(args.baseline_dir, video_id="baseline", provider="baseline")
    logger.info("Total baseline claims: %d", len(baseline_claims))

    all_claims = phase15_claims + baseline_claims

    # 3. Write audit CSV
    csv_path = str(Path(output_dir) / "claims_audit.csv")
    build_audit_csv(all_claims, csv_path)

    # 4. Bootstrap CIs per video+provider (phase15 only)
    groups = _group_verdicts(phase15_claims, lambda c: f"{c['video']} / {c['provider']}")
    ci_data: list[dict] = []
    for label, verdicts in sorted(groups.items()):
        mean, lower, upper = bootstrap_ci(verdicts, n_boot=args.n_boot)
        ci_data.append({"label": label, "mean": mean, "lower": lower, "upper": upper, "n": len(verdicts)})

    # Also add baseline CI
    if baseline_claims:
        base_verdicts = [c["verdict"] for c in baseline_claims]
        mean, lower, upper = bootstrap_ci(base_verdicts, n_boot=args.n_boot)
        ci_data.append({"label": "baseline (phase14)", "mean": mean, "lower": lower, "upper": upper, "n": len(base_verdicts)})

    # 5. McNemar: baseline vs phase15 video=10, provider=openai, format=markdown
    p15_target = [
        c["verdict"] for c in phase15_claims
        if c["video"] == "10" and c["provider"] == "openai" and c["format"] == "markdown"
    ]
    base_verdicts_all = [c["verdict"] for c in baseline_claims]

    mcn = mcnemar_test(base_verdicts_all, p15_target)
    logger.info("McNemar result: %s", mcn)

    # 6. Save LaTeX: bootstrap CI table
    ci_rows = [
        [d["label"], d["n"], f"{d['mean']:.3f}", f"{d['lower']:.3f}", f"{d['upper']:.3f}"]
        for d in ci_data
    ]
    save_latex_table(
        headers=["Group", "N", "Mean", "Lower 95\\%", "Upper 95\\%"],
        rows=ci_rows,
        caption="Bootstrap 95\\% CIs for grounding rate by video/provider",
        name="bootstrap_ci",
        output_dir=output_dir,
        label="tab:bootstrap_ci",
    )

    # 7. Save LaTeX: McNemar result
    mcn_rows = [
        ["baseline vs phase15 (video 10, openai, markdown)",
         mcn["n_pairs"], f"{mcn['chi2']:.4f}", f"{mcn['p_value']:.4f}",
         "Yes" if mcn["significant_at_005"] else "No"],
    ]
    save_latex_table(
        headers=["Comparison", "N pairs", "chi2", "p-value", "Sig. (p<0.05)"],
        rows=mcn_rows,
        caption="McNemar's test: baseline vs phase15 grounding (video 10, openai, markdown)",
        name="mcnemar_result",
        output_dir=output_dir,
        label="tab:mcnemar",
    )

    # 8. Forest plot
    plot_forest(ci_data, output_dir)

    # 9. Print summary
    _print_summary(ci_data)
    print(f"\nMcNemar (baseline vs phase15/video10/openai/markdown):")
    print(f"  chi2={mcn['chi2']}, p={mcn['p_value']}, n_pairs={mcn['n_pairs']}, significant={mcn['significant_at_005']}")
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
