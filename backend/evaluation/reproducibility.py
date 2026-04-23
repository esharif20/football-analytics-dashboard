"""Reproducibility study: K runs with confidence intervals and cross-provider Kruskal-Wallis.

Runs the LLM grounding evaluator K times (default K=5) to measure:
  - Mean +/- 95% CI for grounding rate per analysis type
  - Coefficient of Variation (CV) to quantify run-to-run stability
  - Cross-provider Kruskal-Wallis test for statistical equivalence/difference

References:
    - Standard practice in NLP evaluation (Bouthillier et al. 2021)
    - Cross-provider comparison methodology

Usage:
    python3 -m evaluation.reproducibility \\
        --analytics path/to/analytics.json \\
        --provider openai \\
        --k 5 \\
        --output eval_output/reproducibility/
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

from ._common import ensure_output_dir, load_analytics, load_db_ground_truth, save_figure, save_latex_table
from .llm_grounding import _run_provider, _ANALYSIS_TYPES
import matplotlib.pyplot as plt


# ── Confidence intervals ──────────────────────────────────────────────────────


def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    if len(values) < 2:
        return (values[0] if values else 0.0, values[0] if values else 0.0)
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def coefficient_of_variation(values: list[float]) -> float:
    """CV = SD / mean x 100%."""
    if not values or np.mean(values) == 0:
        return 0.0
    return float(np.std(values) / np.mean(values) * 100)


# ── K-run study ───────────────────────────────────────────────────────────────


async def run_k_times(
    analytics: dict,
    provider_name: str,
    k: int,
    output_dir: str,
) -> dict:
    """Run grounding evaluator K times, collect grounding rates.

    Returns:
        {analysis_type: {run_rates: list, mean, std, cv, ci_lower, ci_upper}}
    """
    out = ensure_output_dir(output_dir)
    all_runs: list[dict] = []  # List of fmt_scores per run

    for run_idx in range(k):
        print(f"\n  === Run {run_idx + 1}/{k} ===")
        run_out = str(out / f"run_{run_idx + 1}")
        try:
            fmt_scores = await _run_provider(provider_name, analytics, run_out)
            all_runs.append(fmt_scores)
        except Exception as e:
            print(f"  WARNING: Run {run_idx + 1} failed: {e}")
            all_runs.append({})

    # Aggregate: grounding rate per analysis type across runs
    results: dict = {}
    for atype in _ANALYSIS_TYPES:
        rates = []
        for run in all_runs:
            rate = run.get("markdown", {}).get(atype, {}).get("grounding_rate")
            if rate is not None:
                rates.append(float(rate))

        if not rates:
            results[atype] = {"run_rates": [], "error": "no data"}
            continue

        mean = float(np.mean(rates))
        std = float(np.std(rates))
        cv = coefficient_of_variation(rates)
        ci_lo, ci_hi = bootstrap_ci(rates)

        results[atype] = {
            "run_rates": rates,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv_pct": round(cv, 2),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "n_runs": len(rates),
        }

    # Save
    (out / "k_run_results.json").write_text(json.dumps(
        {"provider": provider_name, "k": k, "results": results}, indent=2
    ))

    # LaTeX table
    rows = [
        [atype.replace("_", " ").title(),
         k, f"{results[atype].get('mean', 0)*100:.1f}\\%",
         f"{results[atype].get('std', 0)*100:.1f}\\%",
         f"{results[atype].get('cv_pct', 0):.1f}\\%",
         f"[{results[atype].get('ci_lower', 0)*100:.1f}\\%, {results[atype].get('ci_upper', 0)*100:.1f}\\%]"]
        for atype in _ANALYSIS_TYPES if atype in results and results[atype].get("run_rates")
    ]
    save_latex_table(
        headers=["Analysis Type", "K", "Mean", "SD", "CV", "95\\% CI"],
        rows=rows,
        caption=f"Reproducibility: grounding rate over K={k} runs -- {provider_name} (markdown format)",
        name=f"{provider_name}_k_run_stability",
        output_dir=output_dir,
        label=f"tab:reproducibility_{provider_name}",
    )

    # Plot: mean +/- CI per analysis type
    _plot_stability(results, provider_name, output_dir)

    return results


def _plot_stability(results: dict, provider_name: str, output_dir: str) -> None:
    """Bar chart with CI error bars per analysis type."""
    atypes = [at for at in _ANALYSIS_TYPES if results.get(at, {}).get("run_rates")]
    means = [results[at]["mean"] * 100 for at in atypes]
    ci_lo = [results[at]["mean"] * 100 - results[at]["ci_lower"] * 100 for at in atypes]
    ci_hi = [results[at]["ci_upper"] * 100 - results[at]["mean"] * 100 for at in atypes]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(atypes))
    bars = ax.bar(x, means, color="#4f86c6", edgecolor="white",
                  yerr=[ci_lo, ci_hi], capsize=5, error_kw={"linewidth": 1.5})
    ax.bar_label(bars, fmt="%.1f%%", padding=8, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([at.replace("_", " ").title() for at in atypes], rotation=12, ha="right")
    ax.set_ylabel("Grounding Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title(f"Reproducibility: Mean +/- 95% CI -- {provider_name}")
    fig.tight_layout()
    save_figure(fig, f"{provider_name}_stability_ci", output_dir)


# ── Cross-provider Kruskal-Wallis ─────────────────────────────────────────────


def kruskal_wallis_test(provider_results: dict[str, dict]) -> dict:
    """Kruskal-Wallis H test across providers (non-parametric ANOVA).

    Args:
        provider_results: {provider_name: {analysis_type: {run_rates: list}}}

    Returns:
        {analysis_type: {h_statistic, p_value, significant}}
    """
    from scipy import stats as scipy_stats

    results: dict = {}
    for atype in _ANALYSIS_TYPES:
        groups = []
        for pname, pres in provider_results.items():
            rates = pres.get(atype, {}).get("run_rates", [])
            if rates:
                groups.append(rates)

        if len(groups) < 2:
            continue
        if all(len(g) < 2 for g in groups):
            continue

        try:
            h, p = scipy_stats.kruskal(*groups)
            results[atype] = {
                "h_statistic": round(float(h), 4),
                "p_value": round(float(p), 6),
                "significant": p < 0.05,
                "n_providers": len(groups),
            }
        except Exception as e:
            results[atype] = {"error": str(e)}

    return results


async def run_cross_provider(analytics: dict, provider_names: list[str], k: int, output_dir: str) -> dict:
    """Run K-run study across multiple providers and compare with Kruskal-Wallis."""
    out = ensure_output_dir(output_dir)
    all_provider_results: dict[str, dict] = {}

    for pname in provider_names:
        print(f"\n=== Provider: {pname} ===")
        pout = str(out / pname)
        results = await run_k_times(analytics, pname, k, pout)
        all_provider_results[pname] = results

    # Cross-provider KW test
    kw_results = kruskal_wallis_test(all_provider_results)

    # Save
    (out / "cross_provider_kw.json").write_text(json.dumps(kw_results, indent=2))

    # LaTeX KW table
    if kw_results:
        kw_rows = [
            [atype.replace("_", " ").title(),
             f"{v.get('h_statistic', 'N/A'):.3f}" if isinstance(v.get('h_statistic'), float) else "N/A",
             f"{v.get('p_value', 'N/A'):.4f}" if isinstance(v.get('p_value'), float) else "N/A",
             "\\checkmark" if v.get("significant") else ""]
            for atype, v in kw_results.items()
        ]
        save_latex_table(
            headers=["Analysis Type", "H", "p-value", "Significant"],
            rows=kw_rows,
            caption="Kruskal-Wallis test: cross-provider grounding rate differences (non-parametric ANOVA)",
            name="kruskal_wallis_providers",
            output_dir=output_dir,
            label="tab:kruskal_wallis",
        )

    return {"provider_results": all_provider_results, "kruskal_wallis": kw_results}


# ── Prompt stability study ────────────────────────────────────────────────────
#
# Methodology follows Schumacher et al. (2026) §5.1: apply 10 meaning-preserving
# prompt transforms, run N generations per variant, measure Δ (max−min mean
# grounding rate) as a stability metric. Hypothesis: visual conditions show
# lower Δ than text-only conditions.


PROMPT_VARIANTS: list[tuple[str, str]] = [
    ("original",                 "Identity — no changes"),
    ("add_chain_of_thought",     "Prepend chain-of-thought directive"),
    ("shorten_instructions",     "Remove qualifying adverbs/adjectives"),
    ("formal_register",          "Academic tone: 'you' → 'the analyst'"),
    ("imperative_to_declarative","Prefix imperative sentences with 'You should'"),
    ("remove_metric_definitions","Strip lines containing metric definitions ('=')"),
    ("passive_voice",            "Convert active imperatives to passive voice"),
    ("reorder_sentences",        "Shuffle non-header sentence order within paragraphs"),
    ("combined_minimal",         "Keep only sentences with cite/claim/ground/verif keywords"),
    ("add_persona",              "Prepend expert football analyst persona"),
]


def _apply_variant(prompt: str, variant_id: str) -> str:
    """Apply a meaning-preserving transform to a system prompt string."""
    if variant_id == "original":
        return prompt

    if variant_id == "add_chain_of_thought":
        return (
            "Think step by step before generating your analysis. "
            "Reason through each claim before stating it.\n\n" + prompt
        )

    if variant_id == "shorten_instructions":
        qualifiers = [
            "carefully ", "thoroughly ", "very ", "quite ", "highly ",
            "always ", "ensure that ", "make sure to ", "be sure to ",
            "specific ", "comprehensive ", "detailed ", "particular ",
        ]
        result = prompt
        for q in qualifiers:
            result = result.replace(q, "")
        return result

    if variant_id == "formal_register":
        result = prompt
        replacements = [
            ("you should", "the analyst is required to"),
            ("You should", "The analyst is required to"),
            ("you must", "the analyst must"),
            ("You must", "The analyst must"),
            ("you are", "the analyst is"),
            ("You are", "The analyst is"),
            ("your ", "their "),
            ("Your ", "Their "),
        ]
        for old, new in replacements:
            result = result.replace(old, new)
        return result

    if variant_id == "imperative_to_declarative":
        lines = prompt.split("\n")
        result = []
        for line in lines:
            stripped = line.lstrip()
            # Only transform non-header, non-bullet, non-empty lines
            # starting with an uppercase letter followed by a lowercase letter
            if (
                re.match(r'^[A-Z][a-z]', stripped)
                and not re.match(r'^(#+|>|\*\*|-\s|\*\s)', stripped)
                and stripped
            ):
                new_line = line[:len(line) - len(stripped)] + "You should " + stripped[0].lower() + stripped[1:]
                result.append(new_line)
            else:
                result.append(line)
        return "\n".join(result)

    if variant_id == "remove_metric_definitions":
        lines = prompt.split("\n")
        return "\n".join(
            line for line in lines
            if "=" not in line or line.lstrip().startswith("#")
        )

    if variant_id == "passive_voice":
        imperatives = [
            ("Cite ", "cited"),
            ("List ", "listed"),
            ("Include ", "included"),
            ("Provide ", "provided"),
            ("Describe ", "described"),
            ("Analyse ", "analysed"),
            ("Analyze ", "analyzed"),
            ("Report ", "reported"),
            ("State ", "stated"),
            ("Note ", "noted"),
        ]
        lines = prompt.split("\n")
        result = []
        for line in lines:
            stripped = line.lstrip()
            transformed = False
            for imp, past in imperatives:
                if stripped.startswith(imp):
                    rest = stripped[len(imp):]
                    indent = line[: len(line) - len(stripped)]
                    # "Cite X." → "X should be cited."
                    rest_clean = rest.rstrip(".")
                    result.append(f"{indent}{rest_clean[0].upper()}{rest_clean[1:]} should be {past}.")
                    transformed = True
                    break
            if not transformed:
                result.append(line)
        return "\n".join(result)

    if variant_id == "reorder_sentences":
        rng = np.random.default_rng(42)
        paragraphs = prompt.split("\n\n")
        rebuilt = []
        for para in paragraphs:
            lines = para.split("\n")
            header_lines = [l for l in lines if l.lstrip().startswith("#") or not l.strip()]
            content_lines = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
            if len(content_lines) > 1:
                rng.shuffle(content_lines)
            rebuilt.append("\n".join(header_lines + content_lines))
        return "\n\n".join(rebuilt)

    if variant_id == "combined_minimal":
        keywords = {"cite", "claim", "ground", "verif", "fact", "evidence", "source"}
        lines = prompt.split("\n")
        return "\n".join(
            line for line in lines
            if any(k in line.lower() for k in keywords)
            or line.lstrip().startswith("#")
            or not line.strip()
        )

    if variant_id == "add_persona":
        persona = (
            "You are an expert football tactics analyst with 10 years of experience "
            "analysing professional football matches using GPS tracking data. "
            "You provide precise, evidence-based analysis citing specific metrics.\n\n"
        )
        return persona + prompt

    return prompt


async def run_prompt_stability_study(
    analytics: dict,
    db_ground_truth: dict,
    condition_names: list[str],
    provider,
    n_variants: int,
    n_generations: int,
    output_dir: "Path",
    analysis_type: str = "match_overview",
) -> dict:
    """Measure grounding rate stability across prompt variants.

    For each condition × variant (first n_variants of PROMPT_VARIANTS) × n_generations:
      - Apply variant transform to the condition's base system prompt
      - Run _run_single() to get grounding rate
      - Aggregate rates across generations

    Key output metric: Δ = max(variant_mean) − min(variant_mean) per condition.
    Expected: visual conditions show lower Δ than text-only conditions
    (cf. Schumacher et al. 2026 §5.1: visual-only Δ=0.060 vs text Δ=0.094).

    Args:
        analytics:       Analytics dict (from db_ground_truth["analytics"]).
        db_ground_truth: Full db_extractor output dict (for visual chart rendering).
        condition_names: List of condition IDs from perframe_commentary.ALL_CONDITIONS.
        provider:        LLM provider instance.
        n_variants:      How many PROMPT_VARIANTS to test (max 10).
        n_generations:   LLM calls per variant (stochastic sampling).
        output_dir:      Directory to write prompt_stability_results.json + plots.
        analysis_type:   Which analysis type to evaluate (default: match_overview).

    Returns:
        {condition_name: {
            "variants": {variant_id: {mean, std, cv_pct, rates, description}},
            "delta": float,  # max(mean) - min(mean), in [0, 1]
            "n_variants": int,
            "n_generations": int,
            "analysis_type": str,
        }}
    """
    from .perframe_commentary import (
        build_baseline_context,
        build_visual_context,
        build_visual_focused_context,
        build_visual_multimodal_context,
        build_perframe_context,
        build_perframe_v2_context,
        build_digit_space_context,
        build_visual_system_prompt,
        build_visual_focused_system_prompt,
        build_perframe_system_prompt,
        build_perframe_v2_system_prompt,
        _run_single,
    )
    from services.tactical import SYSTEM_PROMPTS

    _context_builders: dict = {
        "BASELINE":          lambda: (build_baseline_context(analytics), None),
        "PERFRAME_V1":       lambda: (build_perframe_context(analytics, db_ground_truth), None),
        "PERFRAME_V2":       lambda: (build_perframe_v2_context(analytics, db_ground_truth), None),
        "DIGIT_SPACE":       lambda: (build_digit_space_context(analytics, db_ground_truth), None),
        "VISUAL":            lambda: build_visual_context(analytics, db_ground_truth),
        "VISUAL_FOCUSED":    lambda: build_visual_focused_context(analytics, db_ground_truth),
        "VISUAL_MULTIMODAL": lambda: build_visual_multimodal_context(analytics, db_ground_truth),
    }
    _base_prompt_fns: dict = {
        "BASELINE":          lambda at: SYSTEM_PROMPTS.get(at, ""),
        "PERFRAME_V1":       build_perframe_system_prompt,
        "PERFRAME_V2":       build_perframe_v2_system_prompt,
        "DIGIT_SPACE":       build_perframe_system_prompt,
        "VISUAL":            build_visual_system_prompt,
        "VISUAL_FOCUSED":    build_visual_focused_system_prompt,
        "VISUAL_MULTIMODAL": build_visual_system_prompt,
    }

    variants_to_run = PROMPT_VARIANTS[:n_variants]
    out = ensure_output_dir(str(output_dir))

    all_results: dict = {}

    for cname in condition_names:
        if cname not in _context_builders:
            print(f"  WARNING: Unknown condition '{cname}', skipping.")
            continue
        print(f"\n=== Prompt Stability: {cname} ===")
        ctx, imgs = _context_builders[cname]()
        base_prompt_fn = _base_prompt_fns[cname]

        condition_results: dict = {}
        for variant_id, variant_desc in variants_to_run:
            print(f"  Variant: {variant_id}")
            rates: list[float] = []
            base_prompt = base_prompt_fn(analysis_type)
            variant_prompt = _apply_variant(base_prompt, variant_id)

            for gen_idx in range(n_generations):
                try:
                    run_result = await _run_single(
                        analysis_type,
                        ctx,
                        variant_prompt,
                        analytics,
                        db_ground_truth,
                        provider,
                        images=imgs,
                    )
                    rates.append(run_result["grounding_rate"])
                    print(f"    gen {gen_idx + 1}/{n_generations}: {rates[-1] * 100:.1f}%")
                except Exception as e:
                    print(f"    WARNING gen {gen_idx + 1} failed: {e}")

            if rates:
                mean = float(np.mean(rates))
                std = float(np.std(rates))
                cv = coefficient_of_variation(rates)
                condition_results[variant_id] = {
                    "description": variant_desc,
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "cv_pct": round(cv, 2),
                    "rates": rates,
                    "n_generations": len(rates),
                }

        variant_means = [v["mean"] for v in condition_results.values() if "mean" in v]
        delta = (max(variant_means) - min(variant_means)) if len(variant_means) >= 2 else 0.0

        all_results[cname] = {
            "variants": condition_results,
            "delta": round(delta, 4),
            "n_variants": len(condition_results),
            "n_generations": n_generations,
            "analysis_type": analysis_type,
        }

    (out / "prompt_stability_results.json").write_text(json.dumps(all_results, indent=2))
    _plot_stability_comparison(all_results, str(out))

    return all_results


def _plot_stability_comparison(stability_results: dict, output_dir: str) -> None:
    """Two charts: Δ bar chart per condition + box plots per variant per condition."""
    out = Path(output_dir)
    conditions = list(stability_results.keys())
    if not conditions:
        return

    # Chart 1: Δ (max − min variant mean) per condition
    deltas = [stability_results[c].get("delta", 0) * 100 for c in conditions]
    colors = ["#4f86c6" if "VISUAL" in c else "#e07b4f" for c in conditions]

    fig1, ax1 = plt.subplots(figsize=(max(6, len(conditions) * 1.5), 4))
    bars = ax1.bar(range(len(conditions)), deltas, color=colors, edgecolor="white")
    ax1.bar_label(bars, fmt="%.1f pp", padding=4, fontsize=9)
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=9)
    ax1.set_ylabel("Δ Grounding Rate (pp)")
    ax1.set_title("Prompt Stability: Δ (max−min variant mean) per Condition\nLower = more stable")
    ax1.set_ylim(0, max(deltas + [5]) * 1.35)
    ax1.axhline(6.0, color="#4f86c6", linestyle="--", linewidth=1,
                label="Schumacher visual Δ benchmark (6pp)")
    ax1.axhline(9.4, color="#e07b4f", linestyle="--", linewidth=1,
                label="Schumacher text Δ benchmark (9.4pp)")
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    save_figure(fig1, "prompt_stability_delta", output_dir)

    # Chart 2: Box plots (one box per variant, one subplot per condition)
    n_cond = len(conditions)
    fig2, axes = plt.subplots(1, n_cond, figsize=(4 * n_cond, 5), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for ax, cname in zip(axes, conditions):
        cdata = stability_results[cname].get("variants", {})
        variant_ids = [v for v in cdata if cdata[v].get("rates")]
        rate_lists = [np.array(cdata[v]["rates"]) * 100 for v in variant_ids]

        if not rate_lists:
            ax.set_title(cname.replace("_", "\n"), fontsize=9)
            continue

        bp = ax.boxplot(rate_lists, labels=[v[:10] for v in variant_ids], patch_artist=True)
        fill_color = "#4f86c6" if "VISUAL" in cname else "#e07b4f"
        for patch in bp["boxes"]:
            patch.set_facecolor(fill_color)
            patch.set_alpha(0.7)

        ax.set_title(cname.replace("_", "\n"), fontsize=9)
        ax.set_ylabel("Grounding Rate (%)")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.set_ylim(0, 105)

        delta_val = stability_results[cname].get("delta", 0) * 100
        ax.text(0.98, 0.98, f"Δ={delta_val:.1f}pp",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"))

    fig2.suptitle("Grounding Rate Distribution Across Prompt Variants", fontsize=11)
    fig2.tight_layout()
    save_figure(fig2, "prompt_stability_boxplots", output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path="backend/.env", override=True)
        load_dotenv(dotenv_path=".env", override=False)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        description="Reproducibility study (K runs + KW cross-provider + prompt stability)"
    )
    parser.add_argument("--analytics", default=None,
                        help="Path to analytics JSON (required for K-run study)")
    parser.add_argument(
        "--ground-truth",
        default="eval_output/dissertation/db_grounded/18_db_ground_truth.json",
        help="Path to db_extractor ground truth JSON (for prompt stability study)",
    )
    parser.add_argument("--provider", default="gemini",
                        choices=["gemini", "openai", "huggingface", "claude", "groq", "all"])
    parser.add_argument("--k", type=int, default=5, help="Number of repeated runs (K-run study)")
    parser.add_argument("--output", default="eval_output/reproducibility")
    # Prompt stability flags
    parser.add_argument("--prompt-stability", action="store_true",
                        help="Run prompt stability study across prompt variants")
    parser.add_argument(
        "--conditions",
        default="BASELINE,VISUAL,VISUAL_FOCUSED",
        help="Comma-separated conditions for prompt stability study",
    )
    parser.add_argument("--n-variants", type=int, default=10,
                        help="Number of prompt variants to test (max 10, default 10)")
    parser.add_argument("--n-generations", type=int, default=20,
                        help="LLM generations per variant (default 20; use 3-5 for smoke test)")
    parser.add_argument("--analysis-type", default="match_overview",
                        choices=["match_overview", "tactical_deep_dive",
                                 "event_analysis", "player_spotlight"],
                        help="Analysis type to use in stability study")
    args = parser.parse_args()

    if args.prompt_stability:
        gt = load_db_ground_truth(args.ground_truth)
        analytics = gt.get("analytics", {})
        if not analytics:
            parser.error("No 'analytics' key in ground truth JSON.")

        from api.services.llm_providers import get_provider
        provider = get_provider(args.provider)
        condition_names = [c.strip().upper() for c in args.conditions.split(",")]
        n_variants = min(args.n_variants, len(PROMPT_VARIANTS))
        asyncio.run(
            run_prompt_stability_study(
                analytics,
                gt,
                condition_names,
                provider,
                n_variants,
                args.n_generations,
                Path(args.output),
                analysis_type=args.analysis_type,
            )
        )
    else:
        if not args.analytics:
            parser.error("--analytics is required for the K-run reproducibility study.")
        analytics = load_analytics(args.analytics)
        if args.provider == "all":
            providers = ["gemini", "openai"]
            asyncio.run(run_cross_provider(analytics, providers, args.k, args.output))
        else:
            asyncio.run(run_k_times(analytics, args.provider, args.k, args.output))


if __name__ == "__main__":
    main()
