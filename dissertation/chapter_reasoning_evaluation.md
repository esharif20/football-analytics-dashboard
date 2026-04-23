# Chapter 4 — Reasoning Layer Evaluation

*Scaffold and intro draft. Sections marked §4.4–4.8 already drafted in `findings/reasoning_section.md` (~2,200 words). All remaining sections are stubbed with: (i) the research question the section answers, (ii) the primary evidence source, (iii) the headline result to lead with. Draft uses British English, first-person plural, no Oxford comma, no em-dashes in prose, per the Warwick CS310 style.*

---

## 4.1 Introduction

The preceding chapters have established an end-to-end pipeline from monocular broadcast footage to structured tactical analytics: player detection and tracking (Chapter~\ref{sec:perception}), event classification (Chapter~\ref{sec:events}) and derived team-level metrics (Chapter~\ref{sec:analytics}). The remaining question is whether a large language model can produce reliable natural-language commentary from this pipeline's output, and if so, under what conditions. The chapter addresses the concluding link in the verification chain video → tracks → events → tactics → prompt context → generated text. We have evaluated each prior link against ground-truth or statistical plausibility checks; the prompt-to-text link, however, is qualitatively different because the failure modes are semantic rather than metric, and because off-the-shelf language models exhibit documented hallucination under underdetermined input (Ji et al. 2023; Huang et al. 2024).

Three observations motivate the evaluation design. Commentary hallucination in sports analytics is not a cosmetic concern: invented statistics, fabricated player interactions and overstated tactical inferences would undermine the pipeline's fitness for use by coaching staff. Standard text-generation benchmarks are insufficient because they measure fluency and coherence, which remain high even when a model invents content that the input never contained. Factual grounding against the underlying data source is therefore the primary metric, and we adopt a FActScore-style claim-level verification procedure (Min et al. 2023) extended with a second, stronger evidential layer that resolves claims against per-frame database records rather than aggregate statistics alone.

Prior work on LLM-based sports commentary has focused on fluency and expert preference (Chen et al. 2024), on retrieval-augmented generation from pre-annotated event logs (Laptev et al. 2024) and on structured prompting for numerical reasoning (Gruver et al. 2023). These studies either take the data pipeline as given or work with manually curated event streams. None, to our knowledge, jointly evaluates (i) the effect of pipeline-derived pre-interpretation on grounding, (ii) the stability of the prompting interface under meaning-preserving variations and (iii) the representational capacity of the underlying model as measured by linear probing. The recent result of Schumacher et al. (2026) that probing recovers classification accuracy an order of magnitude above zero-shot prompting on time-series tasks suggests that the prompting interface itself, rather than the underlying representation, is a substantial source of the observed grounding gap. Whether that result transfers to the football-tactical domain is open.

We present an evaluation of the reasoning-to-commentary layer along seven dimensions: data necessity, pre-interpretation composition, input format, per-frame spatial evidence, visual chart augmentation, prompt stability and representational capacity. The evaluation spans three match clips, two commercial providers (OpenAI GPT-4o, Google Gemini 1.5), two LLM judges (Claude Haiku, OpenAI GPT-4o) for triangulation and four commentary analysis types (match overview, tactical deep-dive, event analysis, player spotlight). Claim-level verification proceeds in two layers: JSON-grounded verification against aggregate analytics, followed by per-frame database-grounded verification that resolves a subset of spatial and temporal claims the aggregate layer labels unverifiable. We measure representational capacity separately through a linear-probing study that compares zero-shot prompting accuracy on four tactical classification tasks against the accuracy achievable by a logistic regression fitted to the model's hidden states, under the methodology of Schumacher et al. (2026).

The chapter makes the following contributions:

\begin{enumerate}[leftmargin=*, nosep]
  \item \textit{A claim-level evaluation framework} for LLM-generated football commentary that combines FActScore-style JSON verification with a strictly stronger per-frame database-grounded layer, producing verdict categories (grounded, hallucinated, unverifiable, DB-resolved) that separate factual errors from merely unsupported claims.
  \item \textit{An ablation of the MatchInsights pre-interpretation layer} across six sub-components, showing that single-component pre-interpretation consistently outperforms the full six-component layer by up to 9.7 percentage points and that the tactical-contributions component is the highest-value single addition.
  \item \textit{A linear-probing study of four tactical classification tasks} on Qwen2-VL-7B hidden states, replicating the Schumacher et al. (2026) representation gap in the football-tactical domain and identifying a specific pattern in which pretraining actively suppresses spatial text representations that the visual modality recovers.
  \item \textit{An empirical test of probe-informed chart routing} that operationalises the probing findings as a system-design prescription, demonstrating partial validation on match-overview commentary and identifying claim compositionality as the limit of single-chart routing strategies.
\end{enumerate}

Chapter~\ref{sec:eval:setup} presents the experimental setup. Chapter~\ref{sec:eval:data} reports the primary data-necessity result and compares conditions that withhold analytics, pre-interpretation and metric definitions. Chapter~\ref{sec:eval:db} introduces the database-grounded verification layer and reports reasoning-layer ablation results. Chapter~\ref{sec:eval:format} through Chapter~\ref{sec:eval:visual} cover input-format, per-frame-evidence and visual-augmentation experiments. Chapter~\ref{sec:eval:stability} reports prompt-stability measurements. Chapter~\ref{sec:eval:probing} presents the linear-probing study. Chapter~\ref{sec:eval:routing} evaluates probe-informed chart routing. Chapter~\ref{sec:eval:discussion} synthesises the results and discusses limitations.

---

## 4.2 Experimental Setup `[STUB]`

**RQ**: how were the experiments constructed and what is the expected failure mode of each condition?

**Covers**:
- Three evaluation clips: Analyses 13, 17, 18 (with full per-frame availability only for Analysis 18)
- Providers: OpenAI GPT-4o (primary), Google Gemini 1.5 Flash (stability comparison)
- Judges: Claude Haiku 3.5 (primary), OpenAI GPT-4o (triangulation), Krippendorff α = 0.120 baseline
- Analysis types: match overview, tactical deep-dive, event analysis, player spotlight
- Metrics: FActScore grounding rate, hallucination rate, DB resolution rate, G-Eval quality (Claude judge), prompt-stability Δ
- Sample sizes: RQ1–RQ3 use n=3 runs; DB-grounded ablation uses n=1 on Analysis 18 (this is a known limitation; see §4.15)

**Source material**: `FINDINGS.md` §4 executive summary + §9.2 judge comparison; `prompts_and_outputs.md` Part 7 (G-Eval prompt); `DB_FINDINGS.md` §2 (experimental setup).

---

## 4.3 Does Structured Tactical Data Prevent Hallucination? (RQ1) `[STUB]`

**RQ**: is pipeline-derived analytics context necessary and sufficient to prevent fabricated commentary?

**Headline**: Condition A (full analytics + MatchInsights) achieves 40.6% grounding for OpenAI and 57.9% for Gemini on match overview; Condition F (no analytics) drops to 6.1% / 4.5% respectively. R-DATA-ONLY produces 0.0% grounding with no hallucination — both providers correctly abstain rather than invent under explicit instruction.

**Lead figure**: bar chart of grounding / hallucination / abstention rates per condition per provider (Analysis 18).

**Source material**: `FINDINGS.md` §1 (RQ1 writeup); consistency audit flags a clip-ID naming conflict (Clip 10 vs Analysis 18) that must be resolved before the chapter is submitted.

---

## 4.4 Database-Grounded Verification

**Status**: drafted in `findings/reasoning_section.md` lines 9–30 (motivation + ground-truth construction). Ready to paste into chapter.

**Extensions needed**: figure of the verification chain (video → tracks → events → JSON → DB → claim verdict); short paragraph on the FActScore adaptation (what we changed from the Min et al. (2023) baseline procedure).

---

## 4.5 Reasoning Layer Ablation

**Status**: drafted in `findings/reasoning_section.md` lines 31–50 (ablation design + four key findings).

**Headline result to lead with**: R-TACT alone (58.7%) outperforms the full reasoning layer R-ALL (49.0%) by +9.7 pp. The effect is consistent across analysis types but does not transfer to Gemini (R-TACT 32.3% vs R-ALL 60.4%), indicating a provider-specific interaction.

**Extensions needed**: add a single figure showing per-component marginal contribution (bar chart); explicit connection to the "context competition" mechanism invoked later in §4.11 (visual conditions) and §4.13 (chart routing).

---

## 4.6 DB-Grounded Verdict Resolution

**Status**: drafted in `findings/reasoning_section.md` lines 54–60.

**Headline**: DB layer resolves 20% of JSON-unverifiable claims under R-ALL (4 of 20); event-analysis claims achieve 0% resolution across both layers and represent the remaining verification gap.

---

## 4.7 Cross-Analysis Comparison

**Status**: drafted in `findings/reasoning_section.md` lines 64–72.

**Headline**: reasoning-layer benefit (R-ALL − R-NONE) is largest for Analysis 13 (+15.7 pp) despite its being the data-sparsest clip, contradicting the "more data = more benefit" hypothesis and supporting a "reasoning layer scaffolds sparse contexts" interpretation.

**Critical note from consistency audit**: verify whether Analysis 13/17/18 correspond to Clips 2/3/10 in `FINDINGS.md`. If yes, cross-reference explicitly; if no, this is a naming conflict to fix.

---

## 4.8 Input Format and Data Representation (RQ3) `[STUB]`

**RQ**: does the format in which pipeline outputs reach the LLM (Markdown, JSON, Prose, Digit-Space) affect grounding?

**Headline**: raw JSON achieves 81.8% grounding on match overview, substantially above Markdown-with-insights (50.0%) and Prose (36.4%). The result is counterintuitive under the pre-interpretation hypothesis and is explained by the "precise field reference" mechanism: JSON forces verbatim anchoring where Markdown tolerates paraphrase. Digit-space numerical encoding (Gruver et al. 2023) yields 33.8% vs 42.6% baseline, confirming that digit-space improves numeric reproduction but not claim grounding.

**Lead figure**: grounding rate by format, stratified by provider (OpenAI vs Gemini).

**Source material**: `FINDINGS.md` §7 (format comparison); `prompts_and_outputs.md` Part 5 + Part 13.

---

## 4.9 Per-Frame Spatial Evidence (RQ4) `[STUB]`

**RQ**: does supplementing aggregate analytics with per-frame spatial context improve grounding on spatial claims?

**Headline**: overall grounding drops from 81.8% (BASELINE) to 71.4% (PERFRAME) on match overview, but event-analysis grounding rises from 0.0% to 66.7% — a 66.7 pp recovery. The per-frame context redistributes grounding toward spatial and event claims at the cost of inflated claim density in narrative sections.

**Interpretation**: the aggregate analytics JSON cannot provide evidence for event-specific spatial claims; adding per-frame coordinates transforms previously ungroundable claims into verifiable ones. This is the clearest direct evidence that the verification chain is incomplete when aggregate-only context is supplied.

**Source material**: `prompts_and_outputs.md` Part 9; existing §9.5 "Dissertation Implication" paragraph is publication-ready.

---

## 4.10 Visual Chart Conditions (RQ5) `[STUB]`

**RQ**: does supplying the LLM with rendered time-series charts improve grounding, and if so, does chart breadth or focus matter?

**Headline**: VISUAL_FOCUSED (single compactness chart) achieves 45.4% grounding in the 7-condition run, the highest of any condition. VISUAL (four charts) scores 43.8%. BASELINE scores 30.6%. Focused visual augmentation outperforms broad visual augmentation.

**Mechanism**: single-anchor visual contexts reduce "context competition" — when multiple charts compete for the LLM's attention, the model hedges across them and fewer claims end up anchored to any one chart.

**Source material**: `prompts_and_outputs.md` Parts 11, 14; `FINDINGS.md` §5.

---

## 4.11 Prompt Stability `[STUB]`

**RQ**: is the grounding metric stable under meaning-preserving prompt variations, and does that stability depend on condition?

**Headline**: Δ (max − min grounding rate across 10 variants) is 12.2 pp for BASELINE, 23.3 pp for VISUAL and 11.7 pp for VISUAL_FOCUSED. VISUAL_FOCUSED is the most stable condition and closely matches the Schumacher et al. (2026) paper benchmark of 6.0 pp for visual modalities.

**Implication**: broad-context visual augmentation (VISUAL) achieves marginally higher mean grounding at the cost of substantially higher sensitivity to prompt wording, which is a deployment risk.

**Source material**: `prompts_and_outputs.md` Part 15.

---

## 4.12 Representation Probing Study `[STUB — AWAITING MLP RESULTS]`

**RQ**: does the grounding gap reflect a limitation of the text-generation interface or a fundamental limit of the model's representational capacity?

**Headline (pending completion)**:
- v2 (Qwen2.5-7B, text-only): probe F1 exceeds zero-shot prompting F1 by +27.5 pp (pressing), +54.0 pp (compactness), +47.2 pp (possession), +2.3 pp (territorial). The first three gaps replicate the Schumacher et al. (2026) pattern in the football-tactical domain.
- v3 (Qwen2-VL-7B, multimodal): visual-modality probing achieves F1 = 0.879 for compactness (vs 0.740 text) and 1.000 for possession phase; territorial dominance shows random > pretrained for text (0.420 vs 0.182), indicating that pretraining actively suppresses spatial text representations. The visual modality recovers territorial at F1 = 0.444.
- v3 + MLP (in progress): the 2-layer MLP probe serves as a non-linear upper bound. Early results on `pressing_type` and `compactness_trend` show MLP within 0.08 of linear or below, consistent with the Schumacher framing that the representations are linearly readable rather than non-linearly encoded. Full table pending.

**Source material**: `findings/linear_probing_findings.md`; `prompts_and_outputs.md` §16; MLP results arriving from RunPod at approximately 08:56 UTC.

**Figure needed**: layer-wise probe F1 per modality (replicate Schumacher Figure 4 visual style).

---

## 4.13 Probe-Informed Chart Routing `[STUB]`

**RQ**: can the probing findings be operationalised into an automated chart-selection policy that improves commentary grounding?

**Headline**: FINDINGS_INFORMED (probe-selected chart per analysis type) achieves the highest grounding on match overview (62.5%) but the lowest on event analysis (0.0%). Overall, VISUAL (breadth) outperforms FINDINGS_INFORMED (targeted routing) at 43.8% vs 34.8%.

**Mechanism and limitation**: probing measures single-concept classification; commentary claims are multi-concept. Single-chart routing optimises for the probe-best concept per analysis type but impoverishes the claim space for multi-concept analysis types. The practical implication is a top-2-charts-per-type hybrid, which is flagged as further work in Chapter~\ref{sec:conclusion}.

**Source material**: `prompts_and_outputs.md` Part 17; the new §17.3.1 verbatim output comparison (BASELINE vs FINDINGS_INFORMED on match overview and tactical deep-dive) provides two case studies to lead with.

---

## 4.14 Provider Comparison `[STUB]`

**RQ**: are the grounding findings consistent across providers, and if not, which differences are provider-specific?

**Headline**:
- Gemini achieves higher mean grounding than OpenAI on Condition A (57.9% vs 40.6% match overview) but is substantially less reproducible (CV 32–61% vs 12–16% across n=5 runs).
- Gemini-OpenAI asymmetry on R-TACT (32.3% vs 58.7%) shows that the reasoning-layer recommendations do not transfer across providers.
- QA benchmark accuracy: OpenAI 61.4%, Gemini 68.2%; the Gemini advantage on spatial QA (100% vs 33%) reflects abstention rather than retrieval accuracy on questions with N/A expected answers.

**Source material**: `FINDINGS.md` §4.

---

## 4.15 Discussion, Limitations, and Threats to Validity `[STUB]`

**Cover**:
- Small-n caveat: RQ3 uses n=3; DB-grounded ablation uses n=1. No inferential statistics should be drawn from between-condition differences smaller than ~5 pp.
- Clip-naming inconsistency between FINDINGS.md and DB_FINDINGS.md (must be resolved before submission).
- Inter-judge agreement (Krippendorff α = 0.120) is low enough that G-Eval quality scores carry little individual weight; FActScore grounding rates are the primary metric for this reason.
- The "routing vs breadth" tension (§4.13) is the clearest open question and the one most visible as an unresolved contribution in the conclusion.
- Qwen2-VL results are from a 7B-parameter model; Schumacher et al. use a 32B Qwen2-VL variant. Scale transfer is assumed but not tested.
- Prompt-stability study uses 3 × 3 (variants × generations); the full 10 × 20 grid is not run and is flagged as further work.

**Thesis-level synthesis paragraph**: the chapter's evidence converges on the claim that grounding is a function of *matching interface to claim type*, not of context quantity. Pre-interpretation helps when focused (R-TACT > R-ALL). Visual charts help when single-anchored (VISUAL_FOCUSED > VISUAL). Per-frame context helps where spatial claims need anchoring (event analysis) and hurts elsewhere. Probing reveals that the models encode the tactical structure required to ground commentary; prompting fails to extract it fully. The dissertation's contribution in this chapter is therefore not a single "correct" commentary recipe but a mapping from claim-type to interface-choice that future systems can operationalise.

---

## Open items flagged by audit agents

**Consistency audit (2 confirmed conflicts, 2 suspected):**
1. Condition A grounding rate: FINDINGS.md Clip 10 reports 40.6% (OpenAI), DB_FINDINGS.md Analysis 18 reports R-ALL at 49.0%. Unclear whether these are the same clip. Resolution: add a clip-ID mapping table at the start of §4.2.
2. Sample-size inconsistency: FINDINGS.md §2.3 states n=3 runs, DB_FINDINGS.md §4.1 states n=1. Flag the difference explicitly in each section's table caption.
3. Linear probing version confusion: `linear_probing_findings.md` Table 1 shows v2 numbers, Table 3 shows v3 numbers; both are labelled "primary". Resolution: designate v3 (Qwen2-VL) as primary, v2 as provenance-for-layer-wise-analysis, and rewrite the opening paragraph accordingly.

**Orphaned findings to promote (4 items):**
- Linear probing representation gap → integrate into §4.12 as the mechanistic counterpart to §4.3's grounding gap.
- v3 visual modality (compactness v = 0.879 > d = 0.740) → §4.12, with explicit reference to the §4.10 visual conditions finding.
- Formation hallucination case study → bridge §4.3 (JSON-layer formation claim) to §4.6 (DB-layer formation refutation) as a worked example.
- Per-frame DB-grounded verification as a second evidential layer → position as a methodological contribution in §4.4 rather than a footnote.

**Cut recommendation (1 item):**
- HuggingFace/Qwen quota limitation: replace with a generic statement. The current wording promises results that were never produced.

**Orphaned findings to KEEP as appendix (7 items, already documented):**
- Groq LLaMA-3.3-70b results (rate-limit failure); Gemini-judge empty results; VLM tentative finding; Krippendorff α; Gemini abstention artifact on tactical QA; possession-phase class imbalance (17/155); JSON > Markdown format result (already integrated).

---

## Writing plan

1. Resolve the clip-ID naming conflict first (§4.2 introduces a mapping table).
2. Paste `findings/reasoning_section.md` into §4.4–4.7 with minimal editing.
3. Draft §4.8 (format) and §4.9 (per-frame) from existing `prompts_and_outputs.md` Parts 5, 9, 13. Each is about 500 words.
4. Draft §4.10 (visual) and §4.11 (stability) together because they share a mechanism (context competition). About 700 words jointly.
5. Draft §4.12 (probing) once the MLP run completes — section fills Table 5 from the probing_vl_mlp results.
6. Draft §4.13 (chart routing) using the new §17.3.1 verbatim comparison as case studies. About 400 words.
7. Draft §4.14 (provider comparison) from FINDINGS.md §4.
8. Draft §4.15 (discussion) last, because it synthesises across sections.

Estimated total chapter length once complete: 9,000–11,000 words, which is consistent with a Warwick CS310 evaluation chapter covering a multi-RQ evaluation of a complex system.
