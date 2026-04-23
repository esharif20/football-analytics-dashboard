"""Unified evaluation runner — orchestrates all LLM/VLM eval modules.

Runs grounding, QA benchmark, and (optionally) VLM comparison in sequence,
then produces a consolidated HTML report via report_builder.

Usage:
    # Core grounding + QA only
    python -m backend.evaluation.unified_runner \\
        --analytics eval_output/10_analytics.json \\
        --provider openai \\
        --output eval_output/unified/

    # Include VLM comparison (requires video + tracks)
    python -m backend.evaluation.unified_runner \\
        --analytics eval_output/10_analytics.json \\
        --tracks eval_output/10_tracks.json \\
        --video path/to/video.mp4 \\
        --provider openai \\
        --output eval_output/unified/

    # Run subset only
    python -m backend.evaluation.unified_runner \\
        --analytics eval_output/10_analytics.json \\
        --only grounding,qa \\
        --provider openai \\
        --output eval_output/unified/
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import EvalConfig, ensure_output_dir, load_analytics

VALID_EVALS = {"grounding", "qa", "vlm", "vision_grounding", "ablation_data", "judge"}


# ── Grounding wrapper ─────────────────────────────────────────────────────────


async def run_grounding(
    analytics: dict,
    analytics_path: str,
    provider: str,
    output_dir: str,
) -> dict:
    """Run LLM grounding evaluation for all formats x analysis types."""
    from .llm_grounding import run_async as _run

    print("\n[1/3] Running LLM grounding evaluation...")
    config = EvalConfig(
        analytics_path=analytics_path,
        provider=provider,
        output_dir=str(Path(output_dir) / "grounding"),
    )
    try:
        result = await _run(config)
        result["_status"] = "ok"
        _print_grounding_summary(result)
    except Exception as exc:
        print(f"  ERROR in grounding eval: {exc}")
        result = {"_status": "error", "_error": str(exc), "provider_scores": {}}
    return result


def _print_grounding_summary(result: dict) -> None:
    """Print a compact summary of grounding results to stdout."""
    provider_scores = result.get("provider_scores", {})
    for provider, fmt_scores in provider_scores.items():
        print(f"\n  Provider: {provider}")
        for fmt_name, type_scores in fmt_scores.items():
            rates = [
                s.get("grounding_rate", 0)
                for s in type_scores.values()
                if isinstance(s, dict)
            ]
            avg = sum(rates) / len(rates) if rates else 0.0
            print(f"    {fmt_name:<12} avg grounding rate: {avg:.1%}")


# ── QA benchmark wrapper ──────────────────────────────────────────────────────


async def run_qa(
    analytics: dict,
    provider: str,
    output_dir: str,
) -> dict:
    """Run RAGAS-style chat QA benchmark."""
    from .chat_qa_benchmark import run_benchmark

    print("\n[2/3] Running Chat QA benchmark (RAGAS + SQuAD 2.0)...")
    qa_dir = str(Path(output_dir) / "qa")
    try:
        result = await run_benchmark(analytics, provider, qa_dir)
        result["_status"] = "ok"
        _print_qa_summary(result)
    except Exception as exc:
        print(f"  ERROR in QA benchmark: {exc}")
        result = {"_status": "error", "_error": str(exc)}
    return result


def _print_qa_summary(result: dict) -> None:
    """Print a compact QA result summary."""
    overall = result.get("overall_accuracy", 0)
    unans = result.get("unanswerable_detection", {})
    print(f"  Overall accuracy: {overall:.1%}")
    print(f"  Unanswerable F1:  {unans.get('f1', 0):.1%}")
    by_cat = result.get("by_category", {})
    for cat, stats in by_cat.items():
        print(f"    {cat:<15} {stats.get('accuracy', 0):.1%}  ({stats.get('correct', 0)}/{stats.get('n', 0)})")


# ── VLM comparison wrapper ────────────────────────────────────────────────────


async def run_vlm(
    analytics_path: str,
    tracks_path: str,
    video_path: str,
    provider: str,
    output_dir: str,
    n_keyframes: int = 5,
) -> dict:
    """Run VLM text-only vs text+vision comparison."""
    from .vlm_comparison import run_async as _run

    print("\n[3/3] Running VLM comparison (text-only vs text+vision)...")
    vlm_dir = str(Path(output_dir) / "vlm")
    try:
        result = await _run(
            analytics_path=analytics_path,
            tracks_path=tracks_path,
            video_path=video_path,
            output_dir=vlm_dir,
            n_keyframes=n_keyframes,
            provider=provider,
        )
        result["_status"] = "ok"
        _print_vlm_summary(result)
    except Exception as exc:
        print(f"  ERROR in VLM comparison: {exc}")
        result = {"_status": "error", "_error": str(exc)}
    return result


def _print_vlm_summary(result: dict) -> None:
    """Print a compact VLM result summary."""
    for key, cond in result.items():
        if key.startswith("_") or not isinstance(cond, dict):
            continue
        cond_name = cond.get("condition", key)
        gr = cond.get("grounding_rate", 0)
        hr = cond.get("hallucination_rate", 0)
        print(f"  {cond_name:<30} grounding={gr:.1%}  hallucination={hr:.1%}")


# ── Vision grounding wrapper ──────────────────────────────────────────────────


async def run_vision_grounding(
    analytics_path: str,
    output_dir: str,
    video_path: str | None,
) -> dict:
    """Run SigLIP vision-text cosine similarity grounding."""
    print("\n[4] Vision-Text Grounding (SigLIP cosine similarity)")
    if not video_path:
        print("  Skipped — no --video provided")
        return {"_status": "skipped", "_reason": "no video provided"}
    try:
        from .vision_grounding import VisionTextGrounder, load_claims_from_artifacts
        from backend.api.services.vision import extract_keyframes

        frames = extract_keyframes(video_path, n_frames=8)
        if not frames:
            return {"_status": "error", "_reason": "could not extract frames from video"}

        claims = load_claims_from_artifacts(output_dir)
        if not claims:
            return {"_status": "skipped", "_reason": "no grounding artifacts found — run grounding first"}

        grounder = VisionTextGrounder()
        grounded = grounder.ground_claims(frames, claims)
        out = ensure_output_dir(output_dir)
        grounder.save_outputs(grounded, frames, claims, str(out / "vision_grounding"))

        rate = sum(1 for g in grounded if g.get("visually_grounded")) / max(len(grounded), 1)
        print(f"  Visual grounding rate: {rate:.1%}  ({len(grounded)} claims, {len(frames)} frames)")
        return {
            "visual_grounding_rate": rate,
            "n_claims": len(grounded),
            "n_frames": len(frames),
            "per_claim": grounded,
        }
    except Exception as exc:
        import traceback
        print(f"  Vision grounding failed: {exc}")
        traceback.print_exc()
        return {"_status": "error", "_reason": str(exc)}


# ── Unified run ───────────────────────────────────────────────────────────────


async def run_all(
    analytics_path: str,
    provider: str,
    output_dir: str,
    only: set[str] | None = None,
    tracks_path: str | None = None,
    video_path: str | None = None,
    n_keyframes: int = 5,
) -> dict:
    """Orchestrate all evaluations and return unified results dict."""
    evals_to_run = only if only else VALID_EVALS
    ensure_output_dir(output_dir)
    analytics = load_analytics(analytics_path)

    print(f"\n{'='*60}")
    print(f"Unified LLM/VLM Tactical Analysis Evaluation")
    print(f"Analytics: {analytics_path}")
    print(f"Provider:  {provider}")
    print(f"Output:    {output_dir}")
    print(f"Modules:   {', '.join(sorted(evals_to_run))}")
    print(f"{'='*60}")

    results: dict = {
        "_meta": {
            "analytics_path": analytics_path,
            "provider": provider,
            "output_dir": output_dir,
            "evals_run": sorted(evals_to_run),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    }

    if "grounding" in evals_to_run:
        results["grounding"] = await run_grounding(
            analytics, analytics_path, provider, output_dir
        )

    if "qa" in evals_to_run:
        results["qa"] = await run_qa(analytics, provider, output_dir)

    if "vlm" in evals_to_run:
        if tracks_path and video_path:
            results["vlm"] = await run_vlm(
                analytics_path, tracks_path, video_path, provider, output_dir, n_keyframes
            )
        else:
            print("\n[3/3] Skipping VLM comparison (--tracks and --video required)")
            results["vlm"] = {"_status": "skipped", "_reason": "no tracks/video provided"}

    if "vision_grounding" in evals_to_run:
        results["vision_grounding"] = await run_vision_grounding(
            analytics_path, output_dir, video_path
        )

    if "ablation_data" in evals_to_run:
        print("\n[5] Data Ablation Study")
        try:
            from .ablation_data import run_ablation
            ablation_dir = str(Path(output_dir) / "ablation_data")
            results["ablation_data"] = await run_ablation(
                analytics_path, provider, ablation_dir, video_path=video_path, skip_qa=True,
            )
        except Exception as exc:
            print(f"  Ablation study failed: {exc}")
            results["ablation_data"] = {"_status": "error", "_reason": str(exc)}

    if "judge" in evals_to_run:
        print("\n[6] LLM-as-a-Judge (G-Eval + Pairwise)")
        try:
            from .llm_judge import run_judge_study
            judge_dir = str(Path(output_dir) / "judge")
            result = await run_judge_study(
                analytics=analytics,
                judge_provider_name=provider,
                output_dir=judge_dir,
                conditions=["A", "F", "H", "I", "J"],
                n_runs=3,
                pairwise_pairs=[("A", "H"), ("A", "I"), ("A", "F"), ("A", "J")],
            )
            result["_status"] = "ok"
            results["judge"] = result
        except Exception as exc:
            print(f"  LLM judge failed: {exc}")
            results["judge"] = {"_status": "error", "_reason": str(exc)}

    # Save unified results
    out = ensure_output_dir(output_dir)
    results_path = out / "unified_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nUnified results saved to: {results_path}")

    # Build HTML report
    try:
        from .report_builder import build_report
        report_path = str(out / "report.html")
        build_report(results, report_path)
        print(f"HTML report:  {report_path}")
    except Exception as exc:
        print(f"  Warning: report generation failed: {exc}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_only(only_str: str | None) -> set[str] | None:
    if not only_str:
        return None
    parts = {p.strip().lower() for p in only_str.split(",") if p.strip()}
    invalid = parts - VALID_EVALS
    if invalid:
        raise ValueError(f"Unknown eval modules: {invalid}. Valid: {VALID_EVALS}")
    return parts


async def run_across_matches(
    analytics_paths: list[str],
    provider: str,
    output_dir: str,
    only: set[str] | None = None,
    tracks_path: str | None = None,
    video_path: str | None = None,
    n_keyframes: int = 5,
) -> dict:
    """Run evaluations across multiple analytics files and aggregate results.

    Adds an ``aggregated`` key with mean ± std across matches for each metric.
    """
    print(f"\nRunning across {len(analytics_paths)} match files...")
    per_match: list[dict] = []
    for i, ap in enumerate(analytics_paths):
        print(f"\n{'─'*60}")
        print(f"Match {i+1}/{len(analytics_paths)}: {ap}")
        match_out = str(Path(output_dir) / Path(ap).stem)
        result = await run_all(
            analytics_path=ap,
            provider=provider,
            output_dir=match_out,
            only=only,
            tracks_path=tracks_path,
            video_path=video_path,
            n_keyframes=n_keyframes,
        )
        result["_analytics_path"] = ap
        per_match.append(result)

    # Aggregate grounding + QA rates across matches
    def _mean_std(values: list[float]) -> dict:
        import statistics
        if not values:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": round(mean, 4), "std": round(std, 4), "n": len(values)}

    grounding_rates: list[float] = []
    hallucination_rates: list[float] = []
    qa_accuracies: list[float] = []

    for r in per_match:
        gr = r.get("grounding", {})
        for fmt_scores in gr.get("provider_scores", {}).values():
            for type_scores in fmt_scores.get("markdown", {}).values() if isinstance(fmt_scores.get("markdown"), dict) else []:
                if isinstance(type_scores, dict):
                    grounding_rates.append(type_scores.get("grounding_rate", 0.0))
                    hallucination_rates.append(type_scores.get("hallucination_rate", 0.0))
        qa_accuracies.append(r.get("qa", {}).get("overall_accuracy", 0.0))

    aggregated = {
        "n_matches": len(per_match),
        "grounding_rate": _mean_std(grounding_rates),
        "hallucination_rate": _mean_std(hallucination_rates),
        "qa_accuracy": _mean_std(qa_accuracies),
    }

    summary = {
        "_meta": {
            "analytics_paths": analytics_paths,
            "provider": provider,
            "output_dir": output_dir,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
        "aggregated": aggregated,
        "per_match": per_match,
    }

    out = ensure_output_dir(output_dir)
    (out / "multi_match_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n{'='*60}")
    print(f"Multi-match aggregation ({len(per_match)} matches):")
    print(f"  Grounding rate: {aggregated['grounding_rate']['mean']:.1%} ± {aggregated['grounding_rate']['std']:.1%}")
    print(f"  Hallucination:  {aggregated['hallucination_rate']['mean']:.1%} ± {aggregated['hallucination_rate']['std']:.1%}")
    print(f"  QA accuracy:    {aggregated['qa_accuracy']['mean']:.1%} ± {aggregated['qa_accuracy']['std']:.1%}")
    return summary


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env", override=True)
    except ImportError:
        pass
    parser = argparse.ArgumentParser(
        description="Unified LLM/VLM evaluation runner — runs all eval modules and produces HTML report"
    )
    analytics_group = parser.add_mutually_exclusive_group(required=True)
    analytics_group.add_argument("--analytics", help="Path to single *_analytics.json")
    analytics_group.add_argument(
        "--analytics-dir",
        help="Directory containing *_analytics.json files — runs eval across all and aggregates",
    )
    parser.add_argument("--provider", default="openai", choices=["gemini", "openai", "huggingface", "claude", "groq", "all"])
    parser.add_argument("--output", default="eval_output/unified")
    parser.add_argument("--tracks", default=None, help="Path to *_tracks.json (for VLM)")
    parser.add_argument("--video", default=None, help="Path to video.mp4 (for VLM)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset: grounding,qa,vlm,vision_grounding,ablation_data,judge")
    parser.add_argument("--n-keyframes", type=int, default=5)
    args = parser.parse_args()

    try:
        only = _parse_only(args.only)
    except ValueError as e:
        parser.error(str(e))

    if args.analytics_dir:
        analytics_paths = sorted(Path(args.analytics_dir).glob("*_analytics.json"))
        if not analytics_paths:
            parser.error(f"No *_analytics.json files found in {args.analytics_dir}")
        asyncio.run(run_across_matches(
            analytics_paths=[str(p) for p in analytics_paths],
            provider=args.provider,
            output_dir=args.output,
            only=only,
            tracks_path=args.tracks,
            video_path=args.video,
            n_keyframes=args.n_keyframes,
        ))
    else:
        asyncio.run(run_all(
            analytics_path=args.analytics,
            provider=args.provider,
            output_dir=args.output,
            only=only,
            tracks_path=args.tracks,
            video_path=args.video,
            n_keyframes=args.n_keyframes,
        ))


if __name__ == "__main__":
    main()
