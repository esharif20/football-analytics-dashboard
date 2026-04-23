"""Literature comparison — evaluates this project's methodology against academic papers.

Fetches abstracts from arXiv and Semantic Scholar, then performs gap analysis
comparing each paper's evaluation approach to the existing implementation.

Usage:
    python -m backend.evaluation.literature_comparison \\
        --output eval_output/literature/

    # Also fetch live abstracts (requires internet)
    python -m backend.evaluation.literature_comparison \\
        --fetch-abstracts \\
        --output eval_output/literature/
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Paper registry ────────────────────────────────────────────────────────────

PAPERS = {
    "tacticai": {
        "title": "TacticAI: An AI assistant for football tactics",
        "authors": "Wang et al. (DeepMind, 2024)",
        "arxiv_id": "2310.10553",
        "semantic_scholar_id": None,
        "venue": "Nature Communications",
        "methodology": {
            "evaluation_type": "Expert panel, blind A/B test",
            "n_experts": "20 professional coaches and analysts",
            "dimensions": [
                "Realism (1-5 Likert)",
                "Retrieval accuracy",
                "Preference ranking (forced-choice A/B)",
                "AI detection accuracy (blind test)",
            ],
            "key_finding": "AI suggestions preferred in 90% of corner kick scenarios",
            "metrics": ["Likert mean/SD", "Preference rate", "Cohen's kappa", "AI detection accuracy"],
            "blind_test": True,
            "forced_choice": True,
        },
        "our_implementation": "expert_validation.py",
        "our_gaps": [
            "No forced-choice A/B preference ranking (Likert only)",
            "No 'realism' or 'actionability' Likert dimensions",
            "Fewer annotators (template supports any N, no minimum enforced)",
            "No stratified expertise analysis (coach vs analyst vs researcher)",
        ],
        "gap_severity": "partial",
    },

    "factscore": {
        "title": "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
        "authors": "Min et al. (2023)",
        "arxiv_id": "2305.14251",
        "semantic_scholar_id": None,
        "venue": "EMNLP 2023",
        "methodology": {
            "evaluation_type": "Atomic claim decomposition + retrieval-augmented verification",
            "n_experts": None,
            "dimensions": [
                "Atomic claim extraction (LLM-based)",
                "Retrieval of supporting evidence",
                "Binary claim verification (supported/not supported)",
                "FActScore = fraction of supported atomic claims",
            ],
            "key_finding": "GPT-4 FActScore ~73% on Wikipedia biography generation",
            "metrics": ["FActScore (precision)", "Intrinsic hallucination rate", "Extrinsic hallucination rate"],
            "blind_test": False,
            "forced_choice": False,
        },
        "our_implementation": "llm_grounding.py (compute_factscore_breakdown)",
        "our_gaps": [
            "Claim extraction uses LLM-as-judge (not dedicated fact extraction model)",
            "No retrieval step — verification done via direct analytics lookup",
            "Qualitative claims use rule-based plausibility rather than retrieval",
        ],
        "gap_severity": "minor",
    },

    "vmage": {
        "title": "V-MAGE: Video Multi-Agent Generalist Evaluation",
        "authors": "V-MAGE authors (2024)",
        "arxiv_id": "2406.01862",
        "semantic_scholar_id": None,
        "venue": "arXiv 2024",
        "methodology": {
            "evaluation_type": "Multi-choice QA benchmark with Elo-based model ranking",
            "n_experts": None,
            "dimensions": [
                "Spatial reasoning (player positions, formations)",
                "Temporal reasoning (event sequencing, what happened before/after)",
                "Action prediction",
                "Elo rating across models (no manual score normalization needed)",
            ],
            "key_finding": "Current VLMs struggle with temporal reasoning across frames",
            "metrics": ["QA accuracy", "Elo rating", "Spatial vs non-spatial breakdown"],
            "blind_test": False,
            "forced_choice": False,
        },
        "our_implementation": "chat_qa_benchmark.py + vlm_comparison.py:classify_spatial_claims()",
        "our_gaps": [
            "No temporal reasoning QA pairs (what happened before/after a given frame?)",
            "No Elo-based model ranking (use fixed accuracy metrics instead)",
            "QA pairs auto-generated from analytics, not human-curated",
            "No action prediction questions",
        ],
        "gap_severity": "partial",
    },

    "soccernet": {
        "title": "SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos",
        "authors": "Cioppa et al. (2022)",
        "arxiv_id": "2210.02365",
        "semantic_scholar_id": None,
        "venue": "CVPR Workshop 2022",
        "methodology": {
            "evaluation_type": "Tracking benchmark against ground truth annotations",
            "n_experts": None,
            "dimensions": [
                "HOTA (Higher Order Tracking Accuracy) — decomposes into DetA + AssA",
                "MOTA (Multiple Object Tracking Accuracy)",
                "IDF1 (ID F1 Score)",
                "Action spotting mAP",
            ],
            "key_finding": "HOTA is the most informative single metric for tracking quality",
            "metrics": ["HOTA", "MOTA", "IDF1", "DetA", "AssA"],
            "blind_test": False,
            "forced_choice": False,
        },
        "our_implementation": "soccernet_eval.py",
        "our_gaps": [
            "Cross-domain gap: trained on Bundesliga, evaluated on SoccerNet (multi-league)",
            "No action spotting mAP evaluation implemented",
        ],
        "gap_severity": "minor",
    },

    "capture": {
        "title": "CAPTURE: Evaluating Spatial Understanding in Vision-Language Models",
        "authors": "CAPTURE benchmark authors (2024)",
        "arxiv_id": None,
        "semantic_scholar_id": None,
        "venue": "arXiv 2024",
        "methodology": {
            "evaluation_type": "Spatial hallucination rate measurement in VLMs",
            "n_experts": None,
            "dimensions": [
                "Spatial claim detection (position/zone/territory references)",
                "Grounding rate for spatial vs non-spatial claims",
                "Per-model spatial error rate",
            ],
            "key_finding": "GPT-4o has 14.75% spatial hallucination rate on sports images",
            "metrics": ["Spatial grounding rate", "Non-spatial grounding rate", "Spatial error rate"],
            "blind_test": False,
            "forced_choice": False,
        },
        "our_implementation": "vlm_comparison.py:classify_spatial_claims()",
        "our_gaps": [],  # Fully implemented
        "gap_severity": "implemented",
    },

    "ragas": {
        "title": "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
        "authors": "Es et al. (2023)",
        "arxiv_id": "2309.15217",
        "semantic_scholar_id": None,
        "venue": "EACL 2024",
        "methodology": {
            "evaluation_type": "RAG pipeline evaluation — faithfulness, relevancy, precision, recall",
            "n_experts": None,
            "dimensions": [
                "Faithfulness: are claims supported by retrieved context?",
                "Answer Relevancy: does answer address the question?",
                "Context Precision: is retrieved context relevant?",
                "Context Recall: is all necessary context retrieved?",
            ],
            "key_finding": "Faithfulness is most correlated with human judgment of RAG quality",
            "metrics": ["Faithfulness score", "Answer relevancy", "Context precision/recall"],
            "blind_test": False,
            "forced_choice": False,
        },
        "our_implementation": "chat_qa_benchmark.py",
        "our_gaps": [
            "Context Precision/Recall not measured (only faithfulness via QA accuracy)",
            "Answer Relevancy not measured (only correctness)",
            "No retrieval step — grounded markdown is provided as full context",
        ],
        "gap_severity": "partial",
    },
}


# ── Abstract fetcher ──────────────────────────────────────────────────────────


def fetch_arxiv_abstract(arxiv_id: str, timeout: int = 10) -> str:
    """Fetch abstract from arXiv API for a given paper ID."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
        # Extract abstract between <summary> tags
        start = content.find("<summary>")
        end = content.find("</summary>")
        if start != -1 and end != -1:
            abstract = content[start + 9:end].strip()
            # Clean whitespace
            return " ".join(abstract.split())
        return "[Abstract not found in arXiv response]"
    except Exception as exc:
        return f"[Error fetching abstract: {exc}]"


def fetch_semantic_scholar_abstract(paper_id: str, timeout: int = 10) -> str:
    """Fetch abstract from Semantic Scholar API."""
    encoded = urllib.parse.quote(paper_id)
    url = f"https://api.semanticscholar.org/graph/v1/paper/{encoded}?fields=abstract,year,citationCount"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "football-analytics-eval/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data.get("abstract", "[Abstract not available]")
    except Exception as exc:
        return f"[Error: {exc}]"


def fetch_abstracts(papers: dict, timeout: int = 10) -> dict[str, str]:
    """Fetch abstracts for all papers with known arXiv IDs."""
    abstracts: dict[str, str] = {}
    for key, paper in papers.items():
        arxiv_id = paper.get("arxiv_id")
        if arxiv_id:
            print(f"  Fetching abstract for '{key}' (arXiv:{arxiv_id})...", end=" ", flush=True)
            abstract = fetch_arxiv_abstract(arxiv_id, timeout)
            abstracts[key] = abstract
            preview = abstract[:80] + "..." if len(abstract) > 80 else abstract
            print(f"OK ({len(abstract)} chars): {preview}")
        else:
            abstracts[key] = "[No arXiv ID available]"
    return abstracts


# ── Gap analysis ──────────────────────────────────────────────────────────────

_SEVERITY_ORDER = {"implemented": 0, "minor": 1, "partial": 2, "significant": 3}
_SEVERITY_BADGES = {
    "implemented": "✅ Implemented",
    "minor":       "🟡 Minor gaps",
    "partial":     "🟠 Partial",
    "significant": "🔴 Significant gaps",
}


def build_gap_analysis(papers: dict) -> list[dict]:
    """Build structured gap analysis entries sorted by severity."""
    gaps = []
    for key, paper in papers.items():
        entry = {
            "paper_key": key,
            "title": paper["title"],
            "authors": paper["authors"],
            "venue": paper["venue"],
            "arxiv_id": paper.get("arxiv_id"),
            "our_implementation": paper["our_implementation"],
            "gap_severity": paper["gap_severity"],
            "gaps": paper["our_gaps"],
            "methodology_summary": _summarize_methodology(paper["methodology"]),
        }
        gaps.append(entry)
    return sorted(gaps, key=lambda g: _SEVERITY_ORDER.get(g["gap_severity"], 99))


def _summarize_methodology(m: dict) -> str:
    """One-line summary of a paper's evaluation methodology."""
    parts = [m["evaluation_type"]]
    if m.get("n_experts"):
        parts.append(f"N={m['n_experts']}")
    if m.get("blind_test"):
        parts.append("blind test")
    if m.get("forced_choice"):
        parts.append("forced-choice A/B")
    metrics = m.get("metrics", [])[:3]
    if metrics:
        parts.append("metrics: " + ", ".join(metrics))
    return " | ".join(parts)


# ── Output formatters ─────────────────────────────────────────────────────────


def format_markdown_report(gaps: list[dict], abstracts: dict | None = None) -> str:
    """Produce a markdown gap analysis report."""
    lines = [
        "# LLM/VLM Tactical Analysis Evaluation — Literature Gap Analysis",
        "",
        "Comparison of this project's evaluation methodology against key academic papers.",
        "",
        "| Paper | Venue | Our Implementation | Status | Key Gaps |",
        "|-------|-------|-------------------|--------|----------|",
    ]
    for g in gaps:
        badge = _SEVERITY_BADGES.get(g["gap_severity"], g["gap_severity"])
        arxiv = f"[arXiv:{g['arxiv_id']}](https://arxiv.org/abs/{g['arxiv_id']})" if g.get("arxiv_id") else "—"
        gaps_str = "<br>".join(f"• {gap}" for gap in g["gaps"]) if g["gaps"] else "None"
        title = g["title"][:60] + "..." if len(g["title"]) > 60 else g["title"]
        lines.append(
            f"| **{title}** ({g['authors']}) {arxiv} | {g['venue']} | "
            f"`{g['our_implementation']}` | {badge} | {gaps_str} |"
        )

    lines += [
        "",
        "## Detailed Gap Analysis",
        "",
    ]

    for g in gaps:
        badge = _SEVERITY_BADGES.get(g["gap_severity"], g["gap_severity"])
        lines += [
            f"### {g['title']}",
            f"**Authors**: {g['authors']} | **Venue**: {g['venue']}",
            f"**Status**: {badge}",
            f"**Our implementation**: `{g['our_implementation']}`",
            f"**Methodology**: {g['methodology_summary']}",
        ]

        if abstracts and g["paper_key"] in abstracts:
            abstract = abstracts[g["paper_key"]]
            if not abstract.startswith("["):
                lines += [f"\n> **Abstract**: {abstract[:400]}...", ""]

        if g["gaps"]:
            lines += ["**Gaps identified:**"]
            lines += [f"- {gap}" for gap in g["gaps"]]
        else:
            lines += ["**No gaps** — fully implemented."]

        lines.append("")

    lines += [
        "## Priority Recommendations",
        "",
        "1. **Add forced-choice A/B ranking** to `expert_validation.py` (TacticAI gap) — highest impact",
        "2. **Add temporal QA pairs** to `chat_qa_benchmark.py` (V-MAGE gap) — tests cross-frame reasoning",
        "3. **Add 'realism' and 'actionability' Likert dimensions** to expert validation",
        "4. **Add Elo-based model ranking** for cross-provider VLM comparison (V-MAGE)",
        "5. **Context Precision/Recall** from RAGAS — measures how much context the model actually uses",
        "",
    ]

    return "\n".join(lines)


def format_json_report(gaps: list[dict], abstracts: dict | None = None) -> dict:
    """Produce a JSON-serialisable gap report."""
    for g in gaps:
        g["abstract_preview"] = (abstracts or {}).get(g["paper_key"], "")[:300]
    return {"papers": gaps, "summary": {
        "total": len(gaps),
        "by_severity": {sev: sum(1 for g in gaps if g["gap_severity"] == sev) for sev in _SEVERITY_ORDER},
    }}


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Literature gap analysis: compare evaluation methodology to academic papers"
    )
    parser.add_argument("--output", default="eval_output/literature")
    parser.add_argument(
        "--fetch-abstracts",
        action="store_true",
        help="Fetch live abstracts from arXiv API (requires internet)",
    )
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Literature Gap Analysis ===")
    print(f"Comparing {len(PAPERS)} papers against existing evaluation scripts\n")

    abstracts: dict[str, str] | None = None
    if args.fetch_abstracts:
        print("Fetching abstracts from arXiv...")
        abstracts = fetch_abstracts(PAPERS, timeout=args.timeout)
        (out / "abstracts.json").write_text(json.dumps(abstracts, indent=2))
        print(f"Saved abstracts to {out / 'abstracts.json'}")

    gaps = build_gap_analysis(PAPERS)

    # Markdown report
    md = format_markdown_report(gaps, abstracts)
    md_path = out / "gap_analysis.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown gap analysis: {md_path}")

    # JSON report
    report = format_json_report(gaps, abstracts)
    json_path = out / "gap_analysis.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"JSON gap analysis:      {json_path}")

    # Console summary
    print("\nGap analysis summary:")
    print("-" * 60)
    for g in gaps:
        badge = _SEVERITY_BADGES.get(g["gap_severity"], g["gap_severity"])
        print(f"  {badge:<25} {g['authors'][:40]}")
        for gap in g["gaps"][:2]:
            print(f"    → {gap[:70]}")

    print(f"\nOutputs saved to: {out}/")


if __name__ == "__main__":
    main()
