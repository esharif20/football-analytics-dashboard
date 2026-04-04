"""Human annotation baseline for automated claim verifier validation.

Two phases:

  Phase A — stratified sampling + annotation template
    python3 -m evaluation.human_baseline --sample \\
        --claims-dir ../eval_output/phase15/grounding/10/artifacts/ \\
        --n-claims 50 \\
        --output ../eval_output/phase15/human/

  Phase B — compute inter-rater agreement (Cohen's kappa)
    python3 -m evaluation.human_baseline --evaluate \\
        --template ../eval_output/phase15/human/annotation_template.csv \\
        --output ../eval_output/phase15/human/
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from ._common import ensure_output_dir, save_figure, save_latex_table


# ── Constants ────────────────────────────────────────────────────────────────

STRATA_CAPS: dict[str, int] = {
    "numeric": 25,
    "qualitative": 12,
    "comparative": 8,
    "entity_reference": 5,
}

VERDICTS = ("verified", "refuted", "unverifiable")

CSV_FIELDS = [
    "claim_id",
    "format",
    "analysis_type",
    "claim_type",
    "text",
    "source_sentence",
    "analytics_context",
    "automated_verdict",
    "human_verdict",
    "notes",
]


# ── Cohen's kappa (pure Python, no sklearn) ──────────────────────────────────


def cohens_kappa(y1: list[str], y2: list[str]) -> float:
    """Cohen's kappa for two nominal rating sequences."""
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0:
        return 0.0
    counts: dict[tuple[str, str], int] = {(a, b): 0 for a in labels for b in labels}
    for a, b in zip(y1, y2):
        counts[(a, b)] += 1
    p_o = sum(counts[(l, l)] for l in labels) / n
    p_e = sum(
        (sum(counts[(l, l2)] for l2 in labels) / n)
        * (sum(counts[(l2, l)] for l2 in labels) / n)
        for l in labels
    )
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0


def interpret_kappa(kappa: float) -> str:
    if kappa < 0.2:
        return "slight"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost perfect"


# ── Phase A helpers ──────────────────────────────────────────────────────────


def _load_claims_from_dir(claims_dir: str) -> list[dict]:
    """Walk an artifacts directory and collect all verification_results entries."""
    claims: list[dict] = []
    claim_id = 0
    for path in sorted(Path(claims_dir).rglob("*.json")):
        try:
            with open(path) as f:
                artifact = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        fmt = artifact.get("format", "")
        analysis_type = artifact.get("analysis_type", "")
        for result in artifact.get("verification_results", []):
            claims.append(
                {
                    "claim_id": claim_id,
                    "format": fmt,
                    "analysis_type": analysis_type,
                    "claim_type": result.get("claim_type", ""),
                    "text": result.get("text", ""),
                    "source_sentence": result.get("source_sentence", ""),
                    "referenced_metric": result.get("referenced_metric", ""),
                    "actual_value": result.get("actual_value", ""),
                    "automated_verdict": result.get("verdict", ""),
                }
            )
            claim_id += 1
    return claims


def _stratified_sample(claims: list[dict], n_total: int, seed: int = 42) -> list[dict]:
    """Proportional stratified sample capped per claim_type."""
    random.seed(seed)

    # Group by claim_type
    by_type: dict[str, list[dict]] = {}
    for c in claims:
        ct = c.get("claim_type", "other")
        by_type.setdefault(ct, []).append(c)

    # Proportional allocation
    total_claims = len(claims)
    allocations: dict[str, int] = {}
    for ct, group in by_type.items():
        proportional = round(n_total * len(group) / total_claims) if total_claims else 0
        cap = STRATA_CAPS.get(ct, max(STRATA_CAPS.values()))
        allocations[ct] = min(proportional, cap, len(group))

    # Trim total to n_total if over-allocated (rare rounding edge)
    allocated_total = sum(allocations.values())
    if allocated_total > n_total:
        # Reduce largest strata first
        for ct in sorted(allocations, key=lambda k: -allocations[k]):
            excess = allocated_total - n_total
            if excess <= 0:
                break
            cut = min(excess, allocations[ct])
            allocations[ct] -= cut
            allocated_total -= cut

    sampled: list[dict] = []
    for ct, group in by_type.items():
        n = allocations.get(ct, 0)
        sampled.extend(random.sample(group, n))

    random.shuffle(sampled)
    return sampled


def sample_phase(claims_dir: str, n_claims: int, output_dir: str) -> None:
    """Phase A: load claims, stratified-sample, write annotation_template.csv."""
    claims = _load_claims_from_dir(claims_dir)
    if not claims:
        print(f"No claims found under: {claims_dir}")
        return

    print(f"Loaded {len(claims)} claims from {claims_dir}")

    sampled = _stratified_sample(claims, n_claims)

    # Report strata
    strata_counts: dict[str, int] = {}
    for c in sampled:
        strata_counts[c["claim_type"]] = strata_counts.get(c["claim_type"], 0) + 1
    print("Stratified sample:")
    for ct, n in sorted(strata_counts.items()):
        print(f"  {ct}: {n}")
    print(f"  Total: {len(sampled)}")

    out = ensure_output_dir(output_dir)
    csv_path = out / "annotation_template.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for c in sampled:
            analytics_context = (
                f"{c['referenced_metric']} = {c['actual_value']}"
                if c.get("referenced_metric")
                else ""
            )
            writer.writerow(
                {
                    "claim_id": c["claim_id"],
                    "format": c["format"],
                    "analysis_type": c["analysis_type"],
                    "claim_type": c["claim_type"],
                    "text": c["text"],
                    "source_sentence": c["source_sentence"],
                    "analytics_context": analytics_context,
                    "automated_verdict": c["automated_verdict"],
                    "human_verdict": "",
                    "notes": "",
                }
            )

    print(f"\nAnnotation template: {csv_path}")
    _print_annotator_instructions()


def _print_annotator_instructions() -> None:
    print(
        """
=== Annotator Instructions ===

For each row, fill in the 'human_verdict' column with one of:

  verified      — The claim is supported by the analytics data shown in
                  'analytics_context'. Numeric claims must be within
                  reasonable tolerance (e.g. ±1 for percentages).

  refuted       — The claim contradicts the analytics data. The stated
                  value or direction is demonstrably wrong.

  unverifiable  — The claim cannot be assessed from the data provided
                  (e.g. qualitative judgement with no numeric anchor,
                  or the referenced metric is absent).

You may add free-text comments in the 'notes' column.
Leave 'human_verdict' blank only if you cannot review the row at all;
blank rows are excluded from the agreement calculation.
"""
    )


# ── Phase B helpers ──────────────────────────────────────────────────────────


def _load_annotations(template_path: str) -> tuple[list[dict], int]:
    """Read CSV; return (annotated_rows, skipped_count)."""
    annotated: list[dict] = []
    skipped = 0
    with open(template_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hv = row.get("human_verdict", "").strip().lower()
            if not hv:
                skipped += 1
                continue
            annotated.append(
                {
                    "claim_type": row.get("claim_type", "").strip(),
                    "automated_verdict": row.get("automated_verdict", "").strip().lower(),
                    "human_verdict": hv,
                }
            )
    return annotated, skipped


def _build_confusion_matrix(
    automated: list[str], human: list[str]
) -> tuple[list[list[int]], list[str]]:
    """3x3 confusion matrix over VERDICTS labels."""
    labels = list(VERDICTS)
    idx = {v: i for i, v in enumerate(labels)}
    cm = [[0] * len(labels) for _ in range(len(labels))]
    for a, h in zip(automated, human):
        ai = idx.get(a)
        hi = idx.get(h)
        if ai is not None and hi is not None:
            cm[hi][ai] += 1  # rows = human (true), cols = automated (predicted)
    return cm, labels


def evaluate_phase(template_path: str, output_dir: str) -> None:
    """Phase B: compute Cohen's kappa and produce outputs."""
    rows, skipped = _load_annotations(template_path)
    if skipped:
        print(f"Skipped {skipped} rows with empty human_verdict.")
    if not rows:
        print("No annotated rows found. Fill in human_verdict in the CSV first.")
        return

    print(f"Evaluating {len(rows)} annotated claims.")

    automated_all = [r["automated_verdict"] for r in rows]
    human_all = [r["human_verdict"] for r in rows]

    overall_kappa = cohens_kappa(automated_all, human_all)
    n_agree = sum(1 for a, h in zip(automated_all, human_all) if a == h)
    overall_agreement = n_agree / len(rows) * 100

    print(f"\n=== Overall Agreement ===")
    print(f"  Cohen's kappa: {overall_kappa:.3f} ({interpret_kappa(overall_kappa)})")
    print(f"  Agreement %:   {overall_agreement:.1f}% ({n_agree}/{len(rows)})")

    # Per claim_type breakdown
    types: dict[str, list[dict]] = {}
    for r in rows:
        types.setdefault(r["claim_type"], []).append(r)

    table_rows: list[list] = [
        ["Overall", f"{overall_kappa:.3f}", f"{overall_agreement:.1f}\\%", len(rows)]
    ]
    print("\n=== Per Claim Type ===")
    for ct in sorted(types):
        group = types[ct]
        a_list = [r["automated_verdict"] for r in group]
        h_list = [r["human_verdict"] for r in group]
        kappa = cohens_kappa(a_list, h_list)
        agree_n = sum(1 for a, h in zip(a_list, h_list) if a == h)
        agree_pct = agree_n / len(group) * 100
        print(
            f"  {ct:<20s} kappa={kappa:.3f}  agreement={agree_pct:.1f}%  n={len(group)}"
        )
        table_rows.append(
            [ct, f"{kappa:.3f}", f"{agree_pct:.1f}\\%", len(group)]
        )

    out = ensure_output_dir(output_dir)

    # LaTeX table
    save_latex_table(
        headers=["Claim Type", "κ", "Agreement", "N"],
        rows=table_rows,
        caption="Human annotation baseline: Cohen's kappa and agreement rate per claim type",
        name="cohens_kappa",
        output_dir=str(out),
        label="tab:human_baseline_kappa",
    )
    print(f"\nLaTeX table: {out}/cohens_kappa.tex")

    # Confusion matrix heatmap
    cm, labels = _build_confusion_matrix(automated_all, human_all)
    _save_confusion_heatmap(cm, labels, out)
    print(f"Heatmap: {out}/agreement_heatmap.pdf")

    # Kappa interpretation legend
    print(
        "\n=== Kappa Interpretation ===\n"
        "  < 0.20  slight\n"
        "  0.20–0.40  fair\n"
        "  0.40–0.60  moderate\n"
        "  0.60–0.80  substantial\n"
        "  > 0.80  almost perfect"
    )


def _save_confusion_heatmap(
    cm: list[list[int]], labels: list[str], out: Path
) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Automated verdict")
    ax.set_ylabel("Human verdict")
    ax.set_title("Agreement Confusion Matrix\n(rows=human, cols=automated)")

    max_val = max(cell for row in cm for cell in row) or 1
    for i in range(n):
        for j in range(n):
            val = cm[i][j]
            color = "white" if val > max_val * 0.6 else "black"
            ax.text(j, i, str(val), ha="center", va="center", fontsize=11, color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    save_figure(fig, "agreement_heatmap", str(out))


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human annotation baseline for automated claim verifier"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sample",
        action="store_true",
        help="Phase A: stratified-sample claims and generate annotation template CSV",
    )
    mode.add_argument(
        "--evaluate",
        action="store_true",
        help="Phase B: compute Cohen's kappa from completed annotation CSV",
    )

    parser.add_argument(
        "--claims-dir",
        help="Path to artifacts/ directory containing grounding JSON files (--sample)",
    )
    parser.add_argument(
        "--n-claims",
        type=int,
        default=50,
        help="Number of claims to sample (default: 50)",
    )
    parser.add_argument(
        "--template",
        help="Path to completed annotation_template.csv (--evaluate)",
    )
    parser.add_argument(
        "--output",
        default="eval_output/human",
        help="Output directory for generated files",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.sample:
        if not args.claims_dir:
            parser.error("--claims-dir is required with --sample")
        sample_phase(args.claims_dir, args.n_claims, args.output)
    else:
        if not args.template:
            parser.error("--template is required with --evaluate")
        evaluate_phase(args.template, args.output)


if __name__ == "__main__":
    main()
