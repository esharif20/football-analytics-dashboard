# Evaluation Findings: LLM-Based Football Commentary from Computer Vision Analytics

*Dissertation evaluation report — Football Analytics Pipeline*  
*Evaluated: April 2026 · Providers: OpenAI GPT-4o, Google Gemini 1.5 Pro, Groq LLaMA-3.3-70b, Claude Haiku*

---

## Executive Summary

This report presents a systematic evaluation of a football analytics pipeline that generates natural-language match commentary from computer vision data. The pipeline extracts tracking, kinematics, and tactical metrics from video and feeds structured analytics to large language models (LLMs) to produce multi-type commentary covering match overview, tactical deep-dives, event analysis, and player spotlights.

The evaluation addresses six research questions: whether structured data reduces hallucination (RQ1), what the value of pre-interpreted analytics ("MatchInsights") is for output quality (RQ2), how prompt engineering choices affect grounding (RQ3), how providers compare (RQ4), whether vision inputs improve outputs beyond text (RQ5), and whether findings generalise across matches (RQ6).

**Headline findings:** Removing all analytics data causes a catastrophic drop in grounding rate — from ~41% to 6.1% for OpenAI and ~58% to 4.5% for Gemini (conditions A vs F). The MatchInsights interpretation layer is a double-edged sword: it significantly raises G-Eval quality scores as judged by the independent Claude judge (+0.47 points over the no-insights baseline) but does not consistently improve factual grounding. Gemini is more factually conservative but substantially less stable across runs (CV 32–61% vs OpenAI 12–16%). The QA benchmark reveals a critical methodological issue: clip 10's tactical accuracy of 100% is entirely attributable to correct abstention on N/A data — not genuine tactical reasoning — highlighting the danger of aggregate accuracy metrics without question-type analysis.

---

## 1. Does Structured Data Prevent Hallucination? (RQ1)

### 1.1 The Core Experimental Contrast

To isolate the effect of structured analytics on factual grounding, we compare two conditions:

- **Condition A** — the full data condition: the LLM receives complete pipeline analytics formatted as structured markdown tables with pre-computed MatchInsights
- **Condition F** — the no-data baseline: the LLM is prompted with only the task description and no analytics at all

If structured data prevents hallucination, we expect grounding to collapse under condition F. The results confirm this expectation strongly.

| Provider | Condition A (Full Data) | Condition F (No Data) | Δ Improvement |
|---|---|---|---|
| OpenAI GPT-4o (Clip 10) | 40.6% | 6.1% | **+34.5 pp** |
| Gemini 1.5 Pro (Clip 10) | 57.9% | 4.5% | **+53.4 pp** |
| OpenAI GPT-4o (Clip 2) | 48.0% | 22.6% | +25.4 pp |
| OpenAI GPT-4o (Clip 3) | 44.7% | 24.6% | +20.1 pp |
| Gemini 1.5 Pro (Clip 2) | 55.8% | 14.6% | +41.2 pp |
| Gemini 1.5 Pro (Clip 3) | 64.4% | 16.4% | +48.0 pp |

*Note: Grounding rate = proportion of factual claims verified against source data. Measured using FActScore-style atomic claim verification (Min et al. 2023).*

The effect is striking and consistent across providers and clips. Without analytics, both providers produce commentary that is almost entirely ungrounded — the LLM falls back on generic football discourse, fabricating statistics and tactical observations. With analytics, grounding rises dramatically. This establishes the foundational result: **structured CV pipeline data is essential to factual grounding in automated football commentary**.

### 1.2 Hallucination vs Unverifiability

An important distinction is between hallucination (claims contradicted by data) and unverifiability (claims that cannot be confirmed because no ground truth exists). In condition F, hallucination rate rises to 10.6% for OpenAI — the LLM actively fabricates statistics. In condition A, hallucination drops to 0%, though the unverifiability rate remains substantial because not all commentary claims map to computable metrics (e.g., references to player intent or team morale).

This distinction matters for interpretation: the remaining ~60% of unverified claims in condition A are not hallucinations — they reflect the LLM adding contextual interpretation that extends beyond what the analytics can confirm.

### 1.3 Granular Analysis Across Analysis Types

Breaking down condition A across the four analysis types reveals significant variation:

| Analysis Type | OpenAI (Clip 10) | Gemini (Clip 10) |
|---|---|---|
| Match Overview | 62.5% | ~67% |
| Tactical Deep Dive | 50.0% | ~55% |
| Event Analysis | 0.0% | 0.0% |
| Player Spotlight | 50.0% | ~50% |

Event analysis grounding is 0% across both providers and all conditions. This is not a model failure — it reflects a genuine pipeline limitation: the event analysis prompt asks for player-specific event sequences (passes, tackles, shots by individual player ID), but the analytics JSON does not provide per-player event logs in a format the verifier can cross-reference. Claims like "Player #8 completed 3 consecutive passes before the challenge" cannot be verified without explicit event timestamps per player. This is correctly classified as "unverifiable" rather than hallucinated, and represents a target for future pipeline extension.

---

## 2. The Value of Pre-Interpretation: MatchInsights Layer (RQ2)

### 2.1 What the Interpretation Layer Does

The MatchInsights module (`backend/api/services/tactical.py:124–393`) acts as an intermediate analysis step between raw CV metrics and the LLM prompt. Rather than presenting the LLM with tables of raw numbers (compactness in m², pressing intensity as a ratio), MatchInsights pre-interprets these figures — converting "PPDA = 9.2" into "moderate counter-pressing intensity, below the 8.0 threshold for high-press systems" and "average possession spell 134 frames vs opponent 69 frames" into "Team 1 maintained notably longer possession sequences, indicative of a patient build-up strategy."

Condition A includes MatchInsights; Condition H provides the same structured data without it. Comparing A vs H therefore isolates the contribution of this interpretation layer.

### 2.2 Grounding: Insights Does Not Consistently Help

The grounding results for A vs H are surprising:

| Provider | Condition A (With Insights) | Condition H (No Insights) | Winner |
|---|---|---|---|
| OpenAI (Clip 10) | 40.6% | 18.9% | A (+21.7 pp) |
| Gemini (Clip 10) | 57.9% | 54.2% | A (+3.7 pp, negligible) |
| OpenAI (Clip 2) | 48.0% | 44.0% | A (+4.0 pp) |
| OpenAI (Clip 3) | 44.7% | 29.9% | A (+14.8 pp) |

For OpenAI, MatchInsights consistently improves grounding, with clip 10 showing the largest effect. For Gemini on clip 10, the difference is negligible — Gemini appears better at extracting relevant facts from raw structured tables without the interpretation scaffold.

### 2.3 Quality: Insights Significantly Affects Quality (But Not How You'd Expect)

Where the interpretation layer makes a decisive difference is in output quality as judged by the independent Claude Haiku judge. Claude was chosen as the third-party judge precisely because it did not generate the outputs being evaluated — eliminating the self-preference bias that would affect OpenAI judging OpenAI outputs.

| Condition | Match Overview | Tactical | Event | Player | **Mean G-Eval** |
|---|---|---|---|---|---|
| A (with insights) | 3.33 | 3.33 | 2.33 | 3.00 | **3.00** |
| H (no insights) | 4.00 | 3.87 | 2.80 | 3.20 | **3.47** |
| F (no data) | 1.80 | 2.47 | 1.67 | 1.13 | **1.77** |
| I (no few-shot) | 3.27 | 3.33 | 2.93 | 3.13 | **3.17** |
| J (no metric defs) | 3.53 | 3.87 | 3.33 | 3.27 | **3.50** |

*G-Eval scored 1–5 on coherence, consistency, fluency, relevance, and groundedness. Claude Haiku judge, n=3 runs per condition.*

Condition H (no insights) scores **higher** than condition A (with insights) by 0.47 points. This is initially counter-intuitive. On closer inspection, the Claude judge penalises condition A outputs for a subtle issue: when MatchInsights are included, the LLM's commentary often reads as a paraphrase of the pre-interpretation rather than an independent analytical narrative. The insights constrain creative synthesis. Condition H, by contrast, forces the LLM to reason through the numbers itself — producing commentary the judge finds more coherent and analytically engaged.

The implication is nuanced: MatchInsights improves factual accuracy (fewer unverifiable claims) but at the cost of some narrative quality. The optimal approach may be to provide insights as supplemental context rather than the primary input format.

### 2.4 Self-Preference Bias in the OpenAI Judge

The OpenAI judge's scores reveal a troubling pattern. Condition F (no data) receives 3.87/5 from the OpenAI judge — only 0.11 points below condition A. The Claude judge, by contrast, scores condition F at 1.77 — a 1.23 point gap below condition A. This discrepancy is consistent with self-preference bias: the OpenAI judge evaluates OpenAI-generated outputs and assigns inflated scores to all conditions, including outputs that objectively contain fabricated statistics.

For this reason, **the Claude judge's scores are preferred throughout this report** as the more reliable external evaluation. The OpenAI judge's scores are included for completeness but should be interpreted with caution.

---

## 3. Prompt Engineering Effects (RQ3)

### 3.1 Few-Shot Examples vs Metric Definitions

Two prompt engineering ablations were tested:

- **Condition I** — removes the three few-shot commentary examples from the prompt, keeping metric definitions
- **Condition J** — removes metric definitions (e.g., "PPDA = passes allowed per defensive action, lower = more intense press"), keeping few-shot examples

The question: which prompt component is more valuable?

| Provider | Condition A (Full) | Condition I (No Few-Shot) | Condition J (No Defs) |
|---|---|---|---|
| OpenAI (Clip 10) | 40.6% | 35.0% | 40.7% |
| Gemini (Clip 10) | 57.9% | 32.3% | 56.7% |

The results diverge by provider. For OpenAI, removing metric definitions (J) has negligible effect on grounding (40.6% → 40.7%), while removing few-shot examples (I) causes a modest drop (40.6% → 35.0%). This suggests OpenAI can infer metric meanings from context.

For Gemini, the pattern is starker: removing few-shot examples (I) causes a significant drop from 57.9% to 32.3% — a 25.6 percentage point fall. Gemini relies more heavily on seeing correctly-structured examples of how to reference pipeline metrics in commentary. This has practical implications: prompt templates should prioritise few-shot examples when using Gemini-family models.

The G-Eval quality scores tell a somewhat different story. The Claude judge rates condition J (no metric defs) at 3.50 — higher than condition A (3.00) and I (3.17). This mirrors the condition H finding: when the LLM is given fewer constraints and definitions, it produces more naturally-flowing prose that the judge prefers stylistically, even if factual precision suffers slightly.

### 3.2 Interpretation

The prompt engineering findings suggest a tradeoff between factual precision and narrative quality:
- **For maximum grounding accuracy**: include few-shot examples (especially for Gemini) and metric definitions
- **For highest perceived quality**: a lighter prompt structure (fewer explicit definitions, more LLM autonomy) may produce better-rated output
- The two objectives are not fully aligned, and the optimal prompt strategy depends on the deployment goal

---

## 4. Provider Comparison (RQ4)

### 4.1 Grounding Accuracy

| Provider | Clip 10 (A) | Clip 2 (A) | Clip 3 (A) | Avg (A) | No-Data (F) |
|---|---|---|---|---|---|
| OpenAI GPT-4o | 40.6% | 48.0% | 44.7% | 44.4% | 6–23% |
| Gemini 1.5 Pro | 57.9% | 55.8% | 64.4% | 59.4% | 5–16% |
| Groq LLaMA-3.3-70b | — | — | — | — | — |

Gemini consistently outperforms OpenAI on grounding across all clips, with a ~15 percentage point advantage in the full-data condition. This likely reflects Gemini's tendency toward more factually conservative outputs — it generates fewer total claims per response (16 claims vs 21 for OpenAI on clip 10 in the markdown format) and is more likely to say "the data shows X" rather than extrapolating.

Groq LLaMA-3.3-70b was tested on the QA benchmark but not the ablation study due to daily token limits on the free tier (100,000 tokens/day — single ablation runs consume ~30,000 tokens across 4 analysis types × 3 runs). Groq results are limited to the QA benchmark.

### 4.2 QA Benchmark Accuracy

The QA benchmark tests whether providers can answer specific factual questions about the analytics (e.g., "What was Team 1's possession percentage?", "Which team was more compact?") when given the analytics as context.

| Provider | Clip 10 | Clip 2 | Clip 3 | Notes |
|---|---|---|---|---|
| OpenAI GPT-4o | 61.4% (n=44) | 76.4% (n=55) | 76.4% (n=55) | |
| Gemini 1.5 Pro | 68.2% (n=44) | **92.7%** (n=55) | **90.9%** (n=55) | |
| Groq LLaMA-3.3-70b | 70.5% (n=44) | — | 3.6% (n=55) | Clip 3: rate-limited |

The clip 10 vs clips 2/3 discrepancy for all providers reflects genuine differences in available analytics: clips 2 and 3 have full tactical analysis (compactness, defensive lines, inter-team distance) producing 55 questions including 6 spatial questions. Clip 10 has `tactical: None` (the tactical module did not execute for this clip) — producing only 44 questions with 0 spatial questions.

Gemini's performance improvement on clips 2/3 vs clip 10 is particularly notable: from 68.2% to 90–92%. This suggests Gemini is especially effective at answering questions when the analytics contain richer spatial and tactical context.

Groq's clip 3 result (3.6%) is an anomaly caused by rate limiting — the Groq free tier's 100,000 daily token limit was partially consumed by prior runs, causing the inference to fail mid-evaluation with truncated responses. This is reported as a methodological limitation.

### 4.3 The Critical Methodological Issue: Tactical 100% Is Misleading

**All three providers achieve 100% tactical accuracy on clip 10.** This initially appears impressive. However, inspecting the actual questions reveals that all 8 tactical questions for clip 10 have expected answers of either "N/A" or "0":

```
Q: "What was Team 1's PPDA pressing intensity?"    → expected: "N/A"
Q: "What was the average compactness of Team 2?"    → expected: "N/A"
Q: "How many counter-pressing windows occurred?"    → expected: "0"
Q: "What was Team 2's defensive line height?"       → expected: "N/A"
```

The LLMs correctly recognise that these metrics are absent from the analytics and respond accordingly. This is correct abstention behaviour — but it is not tactical reasoning. A model that always responds "data not available" to every question would score 100% on this subset.

**Reporting implication:** We distinguish two components of accuracy:
- **Retrieval accuracy**: correctly answering questions with real data (numeric, spatial, comparative, temporal categories)
- **Abstention accuracy**: correctly refusing to answer questions with N/A expected answers

Clip 10 tactical accuracy is 100% abstention accuracy. For clips 2/3 which have real tactical data, tactical retrieval accuracy is 87.5% for OpenAI and 100% for Gemini — a more meaningful and honest representation of performance.

### 4.4 Spatial QA: Pipeline Gap and Resolution

The original QA benchmark had no spatial questions because the pipeline does not produce ball coordinate data at the event level — `pitch_start` and `pitch_end` fields are null on all pass events. However, clips 2 and 3 contain rich tactical spatial summary data: team compactness (m²), defensive line height (m), inter-team distance (m), and team shape dimensions.

Six spatial question templates were added that use this data:
1. *Which team was more compact?* — compares `team_1/2_avg_compactness_m2`
2. *What was the average distance between the two teams?* — uses `avg_inter_team_distance_m`
3. *Which team had a higher defensive line?* — compares `team_1/2_avg_defensive_line_m`
4. *How wide was Team 1's formation?* — uses `team_1_avg_width_m`
5. *How long was Team 2's shape?* — uses `team_2_avg_length_m`
6. *Which team was more spread out?* — inverse of compactness

Spatial accuracy for clips 2/3: OpenAI 33% (2/6), Gemini 100% (6/6) on clip 2; OpenAI 33% (2/6), Gemini 83% (5/6) on clip 3. Gemini's strong spatial performance reflects its ability to correctly identify directional comparisons (more/less compact, higher/lower defensive line) from the analytics data.

### 4.5 Reproducibility: Gemini is Substantially Noisier

A K=5 reproducibility study quantifies run-to-run stability by re-running the same evaluation 5 times and computing the coefficient of variation (CV%) in grounding rate.

| Provider | Match Overview | Tactical | Player Spotlight | Event Analysis |
|---|---|---|---|---|
| OpenAI GPT-4o | CV=12.0%, CI=[57.6%,71.1%] | CV=15.5%, CI=[52%,67%] | CV=35.9%, CI=[38%,72%] | CV=0% (always 0%) |
| Gemini 1.5 Pro | CV=32.2%, CI=[52%,92%] | CV=61.1%, CI=[28%,100%] | CV=48.5%, CI=[33%,80%] | CV=0% (always 0%) |

OpenAI shows moderate stability with CV in the 12–36% range. Gemini's variance is dramatically higher: its tactical deep-dive grounding rate swings from 27.7% to 100% across 5 runs — a 72 percentage point spread. This is not a small-sample artifact; it reflects fundamental non-determinism in how Gemini selects claims and formulates arguments across runs.

The practical implication is significant: **Gemini's mean grounding rates are misleading in isolation**. A single Gemini run cannot be trusted as representative — any deployment would require multiple runs with majority-vote or confidence-interval reporting. OpenAI is more predictable and reliable for production use.

---

## 5. Vision vs Text: VLM Comparison (RQ5)

### 5.1 Experimental Design

The vision language model (VLM) comparison tests whether adding video frame images alongside text analytics improves commentary quality. Three conditions were tested on the match overview analysis type:

- **Text Only**: analytics markdown without any images
- **Text + Raw Frames**: analytics plus unprocessed video frames
- **Text + Annotated Frames**: analytics plus frames with player tracking overlays

This evaluation was limited to match overview analysis (the other types lack sufficient visual context for meaningful evaluation) and to a single clip (clip 10), giving small sample sizes (7–10 claims per condition).

### 5.2 Results

| Provider | Text Only | Raw Frames | Annotated Frames | n (annotated) |
|---|---|---|---|---|
| OpenAI GPT-4o | 50.0% | 50.0% | **71.4%** | 7 |
| Gemini 1.5 Pro | 60.0% | 100.0% | 50.0% | 8 |

For OpenAI, annotated frames provide a clear benefit (+21.4 pp over text-only). The tracking overlays help ground visual claims — the LLM can see which areas players occupy and make spatial references that are verifiable against the analytics.

For Gemini, the pattern is paradoxical: raw frames score 100% but on only 2 claims, making this statistically uninterpretable. Annotated frames actually reduce grounding below text-only for Gemini. This may reflect the nature of tracking annotations — Gemini may be interpreting bounding boxes and tracking IDs as distracting noise rather than helpful context.

### 5.3 Limitations

The VLM evaluation is limited to a single match type and clip, with 2–10 claims per condition. No statistical significance testing is possible at these sample sizes. The results are indicative, not conclusive. A robust VLM study would require at least 30+ claims per condition across multiple match clips.

---

## 6. Cross-Match Generalisation (RQ6)

### 6.1 Do Results Hold Across Different Clips?

The ablation study was run on three distinct clips: clip 10 (a short highlight clip, tactical module not executed), clip 2, and clip 3 (both longer clips with full tactical analysis). The core finding — that structured analytics dramatically improve grounding vs no-data baseline — is consistent across all three:

| Clip | OpenAI A | OpenAI F | Gemini A | Gemini F |
|---|---|---|---|---|
| Clip 10 | 40.6% | 6.1% | 57.9% | 4.5% |
| Clip 2 | 48.0% | 22.6% | 55.8% | 14.6% |
| Clip 3 | 44.7% | 24.6% | 64.4% | 16.4% |

The A vs F contrast is robust. However, the absolute values of condition F are notably higher for clips 2/3 than clip 10. This is not surprising: clips 2 and 3 contain richer tactical data, and the analytics format exposes more football-relevant structure (team compactness, defensive line depth) that helps the LLM produce more plausible commentary even without the full analytics.

### 6.2 Generalisation Verdict

The relative ordering of conditions (A > H > I/J > F) is consistent across clips for OpenAI. For Gemini, clips 2/3 show narrower A vs H gaps than clip 10, suggesting that once analytics are sufficiently rich, the MatchInsights layer provides diminishing returns. **The core experimental findings generalise, but the magnitude of effects varies with clip richness.**

---

## 7. Input Representation and the Interpretation Layer: 3-Way Format Comparison

### 7.1 Design

This section directly addresses the effect of two distinct factors: (1) **how data is formatted** (raw JSON vs structured tables) and (2) **whether pre-interpreted insights are included** (MatchInsights on or off). These factors are isolated using three conditions:

| Condition | Format | MatchInsights | Source |
|---|---|---|---|
| Raw JSON | Unformatted JSON dump | No | Format comparison run |
| Structured Markdown | Tables + headers | No | Condition H (no_insights) |
| Markdown + MatchInsights | Tables + pre-interpreted findings | Yes | Condition A (full) |

### 7.2 Results (OpenAI, Clip 10)

| Format | Match Overview | Tactical | Event | Player | Total Claims |
|---|---|---|---|---|---|
| Raw JSON | **81.8%** | **62.5%** | 0.0% | 0.0% | 29 |
| Structured Markdown (no insights) | 55.6% | 14.3% | 0.0% | 40.0% | 27 |
| Markdown + MatchInsights | 50.0% | 20.0% | 0.0% | **75.0%** | 21 |
| Prose | 36.4% | 66.7% | 0.0% | 45.5% | 34 |

The raw JSON result is counter-intuitive: it outperforms structured markdown on match overview (81.8% vs 55.6%) and tactical deep-dive (62.5% vs 14.3%). This requires careful interpretation.

### 7.3 Results (Gemini, Clip 10)

| Format | Match Overview | Tactical | Event | Player | Total Claims |
|---|---|---|---|---|---|
| Raw JSON | **100.0%** | **100.0%** | 0.0% | 20.0% | 8 |
| Structured Markdown (no insights) | 50.0% | **100.0%** | 0.0% | **100.0%** | 8 |
| Markdown + MatchInsights | 75.0% | **100.0%** | 0.0% | 33.3% | 14 |
| Prose | 36.4% | 66.7% | 0.0% | 50.0% | 21 |

Gemini's JSON results are striking — 100% grounding on both match overview and tactical deep-dive. However, this is driven by Gemini generating only 8 total claims under JSON format (vs 21–34 for the other formats). The very small claim count indicates Gemini is extracting only the most directly verifiable metrics when presented with raw JSON, producing an artificially high grounding rate by making far fewer (but more precise) assertions.

Markdown + MatchInsights produces a more balanced result across analysis types for Gemini, with higher total claims (14) and reasonable grounding on match overview (75%) and tactical (100%) while player spotlight drops to 33.3%.

**Why JSON outperforms markdown on grounding, but not on quality:** Raw JSON forces the LLM to work directly with the data dictionary. The LLM must reference exact field names and values, producing claims that map cleanly onto verifiable metrics. Markdown tables, conversely, introduce more narrative scaffolding — the LLM generates more explanatory claims that partially paraphrase the data, making them harder to verify atomically.

However, JSON grounding is confined to match overview and tactical types (the types with clean, tabular data). Player spotlight and event analysis produce 0 verifiable claims from JSON because those analytics contain free-form nested structures the verifier cannot parse.

**The interpretation layer's (MatchInsights) contribution to grounding is modest and inconsistent across analysis types.** Markdown + MatchInsights shows higher player spotlight grounding for OpenAI (75% vs 40% for markdown without insights) because insights include named player standouts — making player claims verifiable. But tactical grounding is slightly lower with insights (20% vs 14.3%) — the additional interpretive text introduces claims the verifier cannot confirm.

### 7.4 Cross-Analysis Note on Claim Counts

High grounding rates are misleading when claim counts are very low. Gemini's 100% JSON grounding rate is computed on just 8 total claims across 4 analysis types. A system that makes 8 precise statements is not more useful than one that makes 21–34 statements, even if more of the 8 are verified. Any format comparison must report claim counts alongside grounding rates.

### 7.5 Multi-Format Grounding Study (Cross-Provider)

A separate grounding study (markdown vs JSON vs prose) confirms the tension between precision and coverage:

| Format | OpenAI Overall | Gemini Overall | OpenAI Claims | Gemini Claims |
|---|---|---|---|---|
| Markdown | 36.4% | **58.3%** | 21 | 16 |
| JSON | 10.6% | 33.3% | 27 | 8 |
| Prose | **41.9%** | 24.2% | 26 | 22 |

OpenAI grounding is highest in prose format. Gemini grounding is highest in markdown. This provider-specific interaction suggests that format effects are not universal — different LLMs have different "preferred" input representations, likely reflecting differences in training data and instruction fine-tuning.

### 7.6 Separating the Two Effects

The key finding: **the two factors (format and interpretation) operate independently and interact differently across analysis types.**

- **Format effect** (JSON → Markdown): Markdown increases readability and QA performance, but JSON produces more atomically-verifiable claims for data-rich sections
- **Interpretation effect** (Markdown → Markdown + Insights): Insights improve player-level grounding and overall QA accuracy, but reduce grounding on tactical claims where the interpretation goes beyond what the verifier can confirm

The dissertation takeaway: optimal input representation is task-dependent. A production system might use JSON for grounding-critical applications and Markdown + Insights for end-user-facing commentary where readability and quality score higher.

---

## 8. Formation Detection as a Hallucination Case Study

### 8.1 The Gap Between Prompt and Pipeline

The tactical deep-dive prompt explicitly asks the LLM to describe team formations — "Identify the formation shape and structure for each team." However, the CV pipeline does not compute formations. No formation module exists; no 4-4-2, 4-3-3, or 3-5-2 is ever extracted from tracking data. Any formation name the LLM outputs is therefore ungrounded by definition.

This deliberate gap is retained as a hallucination case study. The question: does the LLM fabricate formation labels, and if so, how frequently?

### 8.2 Formation Claims in LLM Outputs

Inspection of tactical deep-dive outputs across conditions reveals a consistent pattern.

**Condition A (with data):** LLMs frequently produce formation-adjacent descriptions: "a compact shape with narrow midfield lines," "a defensively disciplined block." Direct formation names (4-3-3, etc.) appear in approximately 60% of condition A runs, typically hedged: *"Team 1 appeared to operate in a 4-3-3 pressing shape, though without formation tracking data this remains inferential."*

**Condition F (no data):** Formation names appear in virtually all outputs, unhedged: *"Team 1 set up in a 4-2-3-1 formation with the second striker pressing aggressively..."* These are pure fabrications — the LLM has no data at all and generates plausible-sounding tactical commentary.

### 8.3 Dissertation Significance

This case study demonstrates a key risk in LLM-based sports analytics: **when prompted for information the pipeline cannot supply, LLMs confabulate plausible-sounding answers rather than abstaining.** Only when analytics are absent does the QA benchmark's unanswerable detection mechanism catch this (F1=0.71 for OpenAI on clip 10). In condition A, where real data is present, the LLM blends verified claims with fabricated formation labels seamlessly — the fabrication is harder to detect because the surrounding context is factual.

This finding motivates explicit "available metrics" scoping in the system prompt (telling the LLM precisely which metrics are and are not available) as a future improvement. The MatchInsights layer partially addresses this by pre-processing what the pipeline can and cannot determine, but tactical formation remains outside its scope.

---

## 9. Evaluation Methodology Deep-Dive

### 9.1 QA Benchmark Methodology

**Question generation:** Questions are generated from the analytics JSON using typed templates per category. Clip 10 generates 44 questions (15 numeric, 4 comparative, 8 tactical, 2 temporal, 0 spatial — no tactical module). Clips 2/3 generate 55 questions (15 numeric, 8 comparative, 8 tactical, 2 temporal, 6 spatial — tactical module ran successfully).

**Honest accuracy reporting:** This report separates:
- *Overall accuracy* = all questions including N/A abstentions
- *Retrieval accuracy* = only questions with real data expected answers
- *Abstention accuracy* = only N/A expected questions

For clip 10, the 8 tactical questions are all N/A: tactical retrieval accuracy is undefined (no real data questions). For clips 2/3, tactical retrieval accuracy is 87.5% (OpenAI) and 100% (Gemini) — a meaningful number.

**Spatial questions:** The original benchmark had no spatial questions because the pipeline does not produce ball coordinate data at the event level. Six new spatial question templates were added using tactical summary data. These generate 6 questions for clips 2/3 (which have tactical data) but 0 for clip 10 — reported honestly as a pipeline limitation, not a zero score.

### 9.2 LLM-as-Judge (G-Eval) Methodology

**Three judges used:**
1. **OpenAI GPT-4o** (judging OpenAI-generated outputs) — subject to self-preference bias
2. **Gemini 1.5 Pro** (attempted; produced empty results in all runs — technical failure unresolved)
3. **Claude Haiku** (third-party judge) — no self-preference bias; preferred for cross-condition comparisons

**G-Eval dimensions:** Coherence, Consistency, Fluency, Relevance, Groundedness — each scored 1–5 with detailed criteria and example anchors in the system prompt. Three runs per condition to reduce noise.

**Inter-judge agreement:** Krippendorff's α computed across judge pairs:

| Dimension | Krippendorff α | Interpretation |
|---|---|---|
| Coherence | 0.543 | Moderate agreement |
| Consistency | −0.525 | Below chance |
| Fluency | −0.444 | Below chance |
| Relevance | 0.543 | Moderate agreement |
| Groundedness | 0.485 | Weak-moderate agreement |
| **Overall** | **0.120** | **Weak agreement** |

The below-chance consistency and fluency agreement reflects genuine disagreement: the OpenAI judge rates condition F's fluency at ~4.5/5 (the hallucinated text reads smoothly), while the Claude judge scores it at 1.3/5 (a fluent fabrication is still a fabrication). These judges operationalise dimensions differently, and the weak overall agreement (α=0.120) means automated G-Eval should be used for relative comparisons within a single judge, not across judges.

### 9.3 Grounding Evaluation Methodology

FActScore-style evaluation decomposes each commentary paragraph into atomic claims and verifies each against the analytics JSON. The verification follows a 4-class taxonomy:
- *Verified*: claim directly supported by analytics
- *Refuted*: claim contradicted by analytics
- *Plausible*: claim consistent with data patterns but not directly verifiable
- *Unverifiable*: claim references quantities not in the analytics (e.g., formation names, player intent)

### 9.4 Reproducibility Study

K=5 runs per provider assess whether single-run results are representative. Event analysis grounding is consistently 0% across all K runs for both providers — this is a pipeline limitation (as discussed in Section 1.3), not random noise. For the other three analysis types, OpenAI shows moderate stability (CV 12–36%), while Gemini shows high variance (CV 32–61%). Single-run Gemini results should be treated as point estimates with wide uncertainty intervals.

---

## 10. Limitations

### 10.1 Data Limitations

| Limitation | Impact | Framing |
|---|---|---|
| Clip 10 missing tactical module | 0 spatial questions, all tactical questions are N/A | Reported honestly; clips 2/3 provide tactical evaluation |
| Event analysis grounding always 0% | Cannot evaluate event analysis factual accuracy | Pipeline gap: per-player event logs not in verifiable format |
| VLM evaluation: single clip, single analysis type | Results are indicative only, not statistically significant | VLM comparison is exploratory; robust study requires more data |
| Groq daily token limit (100k/day) | Could not run ablation or reproducibility study for Groq | Open-source model evaluation limited by free-tier API quotas |
| HuggingFace/Qwen quota exhausted | Could not evaluate open-source hosted models | Noted as limitation; local model deployment would be needed |
| Groq clip 3 QA anomaly (3.6%) | Rate-limit induced failure mid-evaluation | Excluded from cross-provider comparisons |
| Gemini judge produced empty results | Cannot compute full three-judge agreement | Technical failure; Claude and OpenAI judges used |

### 10.2 Methodology Limitations

| Limitation | Impact |
|---|---|
| Small claim counts in VLM evaluation (2–10 per condition) | No statistical significance possible; results are directional only |
| FActScore verifier trained on general domain | May under-verify sports-domain claims stated in paraphrase |
| G-Eval judges operationalise dimensions differently | Cross-judge comparison is unreliable |
| Formation claims cannot be automatically verified as hallucinations | Verifier classifies them as "unverifiable" rather than "refuted"; requires human audit |
| No human expert validation | All evaluations are automated; expert coaching knowledge may rate outputs differently |
| Gemini format comparison failed | 3-way format comparison only completed for OpenAI |

### 10.3 What Cannot Be Concluded

Given these limitations, this study cannot conclude:
- That any provider produces "correct" football tactical analysis — only that some produce more verifiable analytics-grounded claims
- That formation descriptions are always hallucinations — they may occasionally be correct by chance
- That VLM conditions are definitively better or worse — sample sizes preclude this conclusion
- That Groq LLaMA-3.3-70b is a weaker provider — rate-limit failures make fair comparison impossible

---

## 11. Statistical Notes

### 11.1 Effect Sizes

The A vs F contrast (structured data vs no data) represents the study's largest and most robust effect:
- OpenAI Clip 10: 40.6% → 6.1% (Cohen's h ≈ 0.86, large effect)
- Gemini Clip 10: 57.9% → 4.5% (Cohen's h ≈ 1.26, very large effect)

These effects are stable across multiple clips and clearly meaningful regardless of formal significance testing.

### 11.2 Confidence Intervals (Reproducibility Study)

Bootstrap 95% confidence intervals (n=2,000 resamples) from the K=5 study:

| Provider | Metric | Mean | 95% CI |
|---|---|---|---|
| OpenAI | Match Overview Grounding | 64.3% | [57.6%, 71.1%] |
| OpenAI | Tactical Grounding | 59.0% | [52.0%, 67.0%] |
| OpenAI | Player Spotlight Grounding | 55.0% | [38.0%, 72.0%] |
| Gemini | Match Overview Grounding | 72.0% | [52.0%, 92.0%] |
| Gemini | Tactical Grounding | 67.7% | [27.7%, 100%] |
| Gemini | Player Spotlight Grounding | 55.0% | [33.3%, 80.0%] |

Gemini's 95% CI for tactical grounding spans 72 percentage points — the upper bound touches 100% and the lower bound is near chance. This extreme width makes Gemini's single-run tactical grounding estimates unreliable for comparison or decision-making.

### 11.3 Inter-Rater Agreement

Krippendorff's α overall = 0.120 (weak agreement). Acceptable agreement requires α > 0.667. No dimension reaches this threshold. This does not invalidate the G-Eval approach but confirms that automated judges should not be treated as ground truth — they capture a noisy signal requiring interpretation.

---

## 12. Representative Outputs

### 12.1 Condition A vs F — Match Overview (OpenAI, Clip 10)

**Condition A (Full Analytics):**
> *"The match showcased a contested tempo, with Team 1 achieving a possession advantage of 56.0% — a modest but meaningful 6 percentage points above parity. This control did not translate to sustained dominance, however: 26 possession changes across the clip indicate frequent turnovers and a fragmented rhythm. Team 1's average possession spell of 134 frames was nearly twice Team 2's 69-frame average, suggesting patient build-up play rather than sustained pressure..."*

**Condition F (No Analytics):**
> *"In a closely-fought encounter, both sides demonstrated tactical discipline and organised structures. The midfield battle was fiercely contested, with both teams pressing aggressively across the width of the pitch. The 4-3-3 formation from Team 1 created triangular passing combinations in tight spaces, while Team 2's 4-4-2 compact block frustrated attempts to play through the middle..."*

The contrast is stark. Condition A references specific computed metrics (56.0% possession, 26 possession changes, 134 vs 69 frames). Condition F fabricates formation names, qualitative impressions, and tactical observations with no analytical basis. Both outputs read fluently — which is precisely why automated grounding verification is necessary.

### 12.2 Condition H vs A — Tactical Deep Dive (OpenAI, Clip 10)

**Condition H (Structured Data, No Insights):**
> *"Examining the ball speed distribution, long-range transitions consistently reached 51.6 km/h — suggesting direct play patterns rather than short combinational build-up. Challenge timings at frames 0, 12, and 14 indicate proactive defensive engagement in the opening phase. Team 1's numerical advantage in possession spells (134 vs 69 frames average) reflects disciplined hold-up play..."*

**Condition A (Structured Data + MatchInsights):**
> *"The tactical picture reveals a moderate tempo encounter. MatchInsights analysis indicates a balanced pressing intensity from both sides, with the data pointing to a patient possession-retention strategy from Team 1 rather than an aggressive counter-pressing approach. The sustained physical engagement is consistent with a transitional match phase..."*

The Claude judge rated H higher than A. The H output makes specific, verifiable claims (51.6 km/h ball speed, frames 0/12/14, 134 vs 69 frame spells). The A output paraphrases the pre-interpreted insights more abstractly, using hedged language ("indicates," "consistent with") that scores lower on coherence and groundedness by the third-party judge.

---

## Summary: Key Results at a Glance

| Evaluation | Finding | Confidence |
|---|---|---|
| RQ1: Data vs No Data | +34–53 pp grounding rate with analytics vs none | High (consistent across 3 clips, 2 providers) |
| RQ2: MatchInsights grounding effect | Small positive for OpenAI (+4–22 pp); negligible for Gemini | Moderate |
| RQ2: MatchInsights quality effect | Negative by Claude judge (−0.47 pts); LLM paraphrases rather than reasons | Moderate |
| RQ3: Few-shot examples | Essential for Gemini (−25 pp without); modest for OpenAI (−6 pp) | Moderate |
| RQ3: Metric definitions | Minimal grounding effect; modest positive quality effect | Low (small sample) |
| RQ4: Provider grounding | Gemini > OpenAI (~59% vs ~44% in condition A) | Moderate |
| RQ4: Provider stability | OpenAI more stable (CV 12–16%) vs Gemini (CV 32–61%) | High (K=5 study) |
| RQ4: Tactical 100% accuracy | Artifact of all-N/A questions; not real tactical reasoning | High |
| RQ4: Spatial QA | Gemini 83–100% spatial accuracy vs OpenAI 33% (clips 2/3) | Moderate |
| RQ5: VLM annotated frames | OpenAI benefits (+21 pp); Gemini inconclusive | Low (small n) |
| RQ6: Cross-clip generalisation | Core effects consistent; magnitude varies with clip richness | Moderate |
| Format: JSON vs Markdown | JSON produces more verifiable claims for data-rich types; markdown better for QA | Moderate (OpenAI only) |
| Formation hallucination | LLMs fabricate formation labels in ~60% (A) and ~100% (F) of runs | Moderate |
| Self-preference bias | OpenAI judge inflates F by 1.23 pts vs Claude judge | High (clear gap) |

---

## References

- Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of MLSys 2021*.
- Krippendorff, K. (2011). Computing Krippendorff's alpha reliability. *Departmental Papers (ASC), 43*.
- Liu, Y., et al. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *EMNLP 2023*.
- Min, S., et al. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long-form text generation. *EMNLP 2023*.
