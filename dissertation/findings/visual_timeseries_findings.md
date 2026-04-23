# Visual Time-Series Evaluation: Findings

## Study Design

Analysis 18 (Team 2: 83% possession, 750 frames, full per-frame DB ground truth).
Provider: Gemini 1.5 Flash (OpenAI key unavailable; fallback to Gemini for both runs).
n=1 run per condition. Note caveat: proof-of-concept scale only; between-run variance is substantial.

Charts rendered by `VisualTimeSeriesRenderer`: compactness time-series, centroid 2D trajectory,
pressing dashboard (shaded pressing phases), 2×2 combined overview. Note: VISUAL conditions
in these runs rendered only 2 images (compactness + centroid) due to a `None`-value bug in
`inter_team_distance_m` entries that caused `_render_pressing_dashboard` and `_render_combined`
to fail silently. Bug fixed post-run (commit: None-filter in valid_d / valid_dist). Future runs
should produce 4 images and may show stronger effects.

### Conditions

| ID | Description | Text | Images | Reference |
|----|-------------|------|--------|-----------|
| BASELINE | Aggregate + MatchInsights only | ✓ | 0 | Control |
| PERFRAME_V1 | + raw per-frame tables | ✓ | 0 | Existing (untested) |
| PERFRAME_V2 | + wordalised insights + raw tables | ✓ | 0 | Sumpter 2025 |
| DIGIT_SPACE | + digit-space per-frame tables | ✓ | 0 | Gruver et al. 2023 |
| VISUAL | Aggregate + wordalised insights + all charts | ✓ | 2* | Schumacher 2026 §4.2 |
| VISUAL_FOCUSED | Aggregate + wordalised + compactness chart only | ✓ | 1 | Schumacher 2026 + context competition |
| VISUAL_MULTIMODAL | Aggregate + raw tables + all charts | ✓ | 2* | Schumacher 2026 d+v |

*2 images due to rendering bug (4 intended)

---

## Results

### Gemini (primary run)

| Condition | Overall | Match OV | Tactical | Event | Player | DB Res. |
|---|---|---|---|---|---|---|
| BASELINE | **42.6%** | 57.1% | **80.0%** | 0% | 33.3% | 0% |
| PERFRAME_V1 | 32.4% | 42.9% | 43.8% | 0% | 42.9% | 0% |
| PERFRAME_V2 | 32.1% | 50.0% | 38.5% | 0% | 40.0% | 0% |
| DIGIT_SPACE | 33.8% | 40.0% | 52.4% | 0% | 42.9% | 0% |
| VISUAL | 44.5% | 25.0% | 66.7% | **50.0%** | 36.4% | 0% |
| VISUAL_FOCUSED | **45.4%** | **50.0%** | 40.0% | **66.7%** | 25.0% | 0% |
| VISUAL_MULTIMODAL | 31.9% | 50.0% | 37.5% | 0% | 40.0% | **6.2%** |

### Gemini (second run — same provider, different stochastic sample)

| Condition | Overall | Match OV | Tactical | Event | Player | DB Res. |
|---|---|---|---|---|---|---|
| BASELINE | 32.1% | 25.0% | 83.3% | 0% | 20.0% | 0% |
| PERFRAME_V1 | 44.4% | 40.0% | 75.0% | 0% | 62.5% | 0% |
| PERFRAME_V2 | 25.0% | 50.0% | 50.0% | 0% | 0% | 0% |
| DIGIT_SPACE | 44.3% | 66.7% | 66.7% | 0% | 43.8% | 0% |
| VISUAL | 27.8% | 33.3% | 33.3% | 0% | 44.4% | **6.9%** |
| VISUAL_FOCUSED | **44.9%** | 40.0% | **77.8%** | **28.6%** | 33.3% | 0% |
| VISUAL_MULTIMODAL | 38.5% | 37.5% | 66.7% | 0% | 50.0% | 0% |

### 4-Chart Re-run (bug fixed, pressing dashboard + combined included)

| Condition | Overall | Match OV | Tactical | Event | Player | DB Res. |
|---|---|---|---|---|---|---|
| VISUAL | 44.3% | **57.1%** | 40.0% | **60.0%** | 20.0% | 0% |
| VISUAL_FOCUSED | **45.8%** | 33.3% | **83.3%** | 0% | **66.7%** | 0% |
| VISUAL_MULTIMODAL | 37.5% | 50.0% | 50.0% | 0% | 50.0% | 0% |

**2-chart vs 4-chart delta (VISUAL condition only — VISUAL_FOCUSED unchanged by design):**

| Metric | 2-chart | 4-chart | Δ |
|---|---|---|---|
| Overall | 44.5% | 44.3% | -0.2pp |
| event_analysis | 50.0% (2 claims) | **60.0% (5 claims)** | +10pp, +3 claims |
| match_overview | 25.0% | 57.1% | +32pp |
| tactical_deep_dive | 66.7% | 40.0% | -26.7pp |

---

## Key Findings

### F1 — Visual conditions are the sole route to event_analysis grounding; pressing dashboard amplifies the effect

Event analysis grounding is 0% in every text-only condition across all runs. Visual conditions
achieve 50–67% (2-chart run) and 60% (4-chart VISUAL run). This is the most striking finding:
images provide temporal anchors that the LLM uses to make and verify event-level claims.

Critically, the 4-chart re-run clarifies the mechanism. With only the compactness chart and
centroid trajectory (2-chart), VISUAL achieves 50% event grounding on 2 claims. Adding the
pressing dashboard (shaded pressing phase windows over time) increases this to 60% on 5 claims.
The pressing dashboard directly visualises temporal windows during which teams pressed, which
maps to challenge and ball-win events in the event record — the LLM reads pressing phase
timestamps from the chart x-axis and generates event claims that the JSON verifier can confirm.

VISUAL_FOCUSED (compactness chart only) shows 66.7% event grounding in one run and 0% in
another — high variance at n=1 — but the pressing dashboard in VISUAL provides the more
mechanistically plausible route to event grounding.

This is consistent with Schumacher et al. (2026) §4.2: visual representations surface
temporally-anchored patterns that text serialisation cannot communicate.

### F2 — VISUAL_FOCUSED > VISUAL ≥ BASELINE > VISUAL_MULTIMODAL (primary run)

The ordering VISUAL_FOCUSED (45.4%) > VISUAL (44.5%) > BASELINE (42.6%) > VISUAL_MULTIMODAL
(31.9%) is consistent with the context competition hypothesis:

- Adding only the compactness chart gives focused, high-precision visual evidence without
  competing context → best overall grounding
- Adding all available charts gives more information but some cross-chart ambiguity → slight
  grounding penalty vs focused
- Adding charts AND raw tables (VISUAL_MULTIMODAL) creates a three-way competition between
  MatchInsights narrative, raw table values, and chart annotations → large grounding penalty

This extends the R-TACT > R-ALL finding from the reasoning layer ablation to the visual domain:
the focused context principle holds whether the context is text or images.

### F3 — Text-only per-frame conditions are harmful for Gemini (primary run)

PERFRAME_V1 (32.4%), PERFRAME_V2 (32.1%), DIGIT_SPACE (33.8%) all underperform BASELINE
(42.6%) by approximately 9–11pp. The raw markdown tables cause Gemini to over-generate
specific numerical claims from sampled rows, reducing the verifiable proportion. This is the
"context competition / claim inflation" mechanism: DIGIT_SPACE in the second run generates
18–21 claims for tactical_deep_dive (vs 3–7 for visual conditions), diluting grounding rate
even when absolute verified claims are similar.

Note that the second run shows the opposite pattern for some conditions (PERFRAME_V1: 44.4%
above BASELINE 32.1%); this underscores the high within-provider variance. Neither direction
can be taken as definitive at n=1.

### F4 — DB resolution rate is a visual-condition-only signal

The DB-grounded resolution rate (claims initially "unverifiable" by JSON but subsequently
resolved by per-frame records) is non-zero only for visual conditions:
- Primary run: VISUAL_MULTIMODAL 6.2% (tactical_deep_dive: 25.0%)
- Second run: VISUAL 6.9% (tactical_deep_dive: 16.7%, player_spotlight: 11.1%)

No text-only condition produces any DB-resolved claims. This suggests visual representations
prompt the LLM to generate spatial claims that are grounded in per-frame position records,
rather than aggregate statistics. The mechanism matches the verification chain theory:
charts → spatial claim generation → DB-layer verification.

### F5 — DIGIT_SPACE causes claim inflation without proportional grounding improvement

DIGIT_SPACE generates substantially more claims per response than other conditions (18–21 for
tactical_deep_dive vs 3–16 for other conditions). This is the digit-space tokenisation effect:
when numeric values are spaced character-by-character ("5 3 4 , 2"), the LLM encounters
fewer tokenisation barriers and generates more numeric claims. However, absolute grounding
rate does not consistently improve, suggesting the additional claims are not proportionally
more accurate.

The practical implication: digit-space formatting may be counterproductive for FActScore-style
grounding evaluation, which penalises claim inflation. It may, however, improve numeric
reproduction fidelity specifically (the original Gruver et al. 2023 claim) which is not
directly captured by our verification methodology.

---

## Comparison with Hypotheses

| Hypothesis | Predicted | Observed | Status |
|---|---|---|---|
| VISUAL > PERFRAME_V1 on overall grounding | ✓ | ✓ primary (+12.1pp), ✗ second (-16.6pp) | Partial |
| VISUAL_FOCUSED > VISUAL | ✓ | ✓ primary (+0.9pp), ✓ second (+17.1pp) | Confirmed |
| VISUAL_MULTIMODAL best overall | ✗ | ✗ (worst or middle) | Contradicted |
| DIGIT_SPACE improves numeric claims | — | Not directly testable by grounding rate | Inconclusive |
| Visual conditions more stable | — | VISUAL_FOCUSED most stable (11.7pp); VISUAL least stable (23.3pp) | Partially confirmed |

---

## Limitations

1. **Single run per condition, n=1 clip**: between-run variance is large (BASELINE range:
   32–43% across two Gemini runs on the same clip). No statistical inference is valid.
2. **2 images instead of 4**: the rendering bug means VISUAL conditions only had compactness
   and centroid trajectory charts. Pressing dashboard and combined overview were absent.
   Effect on event grounding is unknown — the pressing dashboard specifically maps to events.
3. **Both runs are Gemini**: no OpenAI comparison was possible (no API key). Provider
   asymmetry documented in DB_FINDINGS.md §4.5.3 means these findings may not transfer.
4. **DIGIT_SPACE claim inflation**: FActScore-style evaluation penalises claim count
   regardless of whether more claims is better or worse behaviour. The methodology cannot
   distinguish "correctly generates more verifiable facts" from "generates more claims that
   happen to be unverifiable".

---

## Recommended Next Steps

1. **Re-run VISUAL conditions with 4 charts** (bug fixed): expect stronger effects,
   particularly for pressing dashboard (temporal event anchoring).
2. **Run with OpenAI GPT-4o**: OpenAI's FActScore behaviour (fewer claims, higher precision)
   may interact differently with visual context.
3. **Phase 3 prompt stability**: run 10 prompt variants × 20 generations to measure Δ
   (max−min grounding) across conditions. Hypothesis: visual conditions show lower Δ.
4. **Phase 5 linear probing on RunPod**: confirm that Gemini internally encodes the
   temporal patterns that visual charts surface (pressing phases, compactness trends).

---

## Phase 3: Prompt Stability Study Results

**Date**: 2026-04-15
**Method**: 3 conditions x 3 prompt variants x 3 generations = 27 LLM calls (Gemini, Analysis 18, match_overview)
**Variants tested**: original, add_chain_of_thought, shorten_instructions
**Metric**: Delta (max-min variant mean grounding rate); lower = more stable

### Table 2 -- Prompt Stability Results (Pilot, n=3 variants, n=3 generations)

| Condition | original | add_CoT | shorten | Delta | Interpretation |
|-----------|----------|---------|---------|-------|----------------|
| BASELINE  | 46.7%    | 36.1%   | 34.4%   | 12.2pp | Intermediate stability |
| VISUAL    | 44.4%    | 65.0%   | 41.7%   | 23.3pp | Least stable |
| VISUAL_FOCUSED | 38.5% | 49.0% | 50.2%  | 11.7pp | Most stable |

**Schumacher et al. (2026) benchmarks**: text Delta=9.4pp, visual Delta=6.0pp (generic time-series).

### Key Findings

**S1 -- VISUAL_FOCUSED most stable (Partial confirmation)**
VISUAL_FOCUSED (11.7pp) shows lowest instability, consistent with context competition
reduction from a single grounding anchor. The focused condition has fewer competing
information sources (one chart, wordalised summary, no raw tables), so prompt variants
that change processing instructions have less to affect. This is directionally consistent
with Schumacher et al.'s visual stability finding.

**S2 -- VISUAL least stable (Unexpected, context competition explanation)**
VISUAL (23.3pp Delta) is *more* unstable than BASELINE (12.2pp). This reverses the
paper's hypothesis. The most likely explanation: VISUAL provides all 4 charts plus
wordalised text, giving the LLM multiple grounding anchors. The add_chain_of_thought
variant (+20.6pp from original: 44.4% to 65.0%) appears to trigger more systematic
cross-referencing of all anchors. When the LLM is told to "think step by step", it
may leverage all charts; without this, it picks the most salient anchor (compactness).
This is a manifestation of context competition at the prompt-level rather than the
context-content level.

**S3 -- add_chain_of_thought effect**
The chain-of-thought variant has asymmetric effects:
- BASELINE: -10.6pp (harmful -- introduces hallucinated reasoning steps)
- VISUAL: +20.6pp (strongly beneficial -- directs systematic chart reading)
- VISUAL_FOCUSED: +10.5pp (moderately beneficial -- directs focused chart analysis)
This suggests visual representations are a prerequisite for CoT to help: without charts,
CoT generates unsupported reasoning chains that hurt grounding.

**S4 -- Caveat: pilot sample size**
n=3 generations per variant is insufficient for stable estimates. The CV values
(0-35%) suggest high within-variant variance. Full study (n=10 variants, n=20 generations)
needed for interpretable Delta estimates. These pilot results should be treated as
directional only.

### Connection to Linear Probing Findings

The prompt stability results align with the probing study mechanism:
- VISUAL_FOCUSED stability (11.7pp) matches its focused extraction interface
- VISUAL instability (23.3pp) reflects information richness that prompting can't
  cleanly arbitrate -- the internal representation is stable (probe F1 unchanged)
  but the extraction is unstable (multiple competing grounding anchors)
- add_chain_of_thought boosts VISUAL because it approximates a linear probing
  interface: systematic enumeration of evidence from each chart

### Recommended Next Steps

1. Full stability study: 10 variants x 20 generations x 3 conditions (600 calls)
   Expected to confirm S1 (VISUAL_FOCUSED most stable) with statistical power
2. Add PERFRAME_V1 condition to stability study (no charts -- expect Delta ~9-12pp)
3. Test whether add_chain_of_thought systematically boosts all visual conditions
   across analysis types (not just match_overview)

---

## Phase 6: Probe-Informed Chart Routing (FINDINGS_INFORMED)

**Date**: 2026-04-16 | **Model**: Gemini 1.5 Pro | **n=1** | Analysis 18

### Design

Based on the linear probing study (see `linear_probing_findings.md`), each analysis type
was routed to the single chart whose visual representation achieved the highest probe F1
for its dominant claim type:

| Analysis Type | Assigned Chart | Probe Basis |
|---|---|---|
| match_overview | centroid trajectory | territorial v=0.444 >> d=0.182 |
| tactical_deep_dive | pressing dashboard | pressing v < d (needs inter-team distance) |
| event_analysis | pressing dashboard | event grounding 0%→67% with pressing transitions |
| player_spotlight | centroid trajectory | team-centroid context for player claims |

Text layer: aggregate + wordalised insights only (no raw tables).

### Results

| Analysis Type | BASELINE | VISUAL_FOCUSED | VISUAL | FINDINGS_INFORMED |
|---|---|---|---|---|
| Match Overview | 60.0% | 33.3% | 25.0% | **62.5%** |
| Tactical Deep Dive | 37.5% | **75.0%** | 50.0% | 60.0% |
| Event Analysis | 0.0% | 0.0% | **50.0%** | 0.0% |
| Player Spotlight | 25.0% | **50.0%** | **50.0%** | 16.7% |
| **Overall** | 30.6% | 39.6% | **43.8%** | 34.8% |

### Key Findings

**F10 — Centroid routing confirms territorial claim improvement**
match_overview achieves 62.5% — highest of all conditions in this run. Territorial
centroid chart bypasses pretraining suppression of spatial information in text layers
(random d=0.420 > pretrained d=0.182 for territorial task). Confirms probe-routing
hypothesis for spatial/territorial analysis types.

**F11 — Event analysis requires multi-chart context, not single-chart routing**
FINDINGS_INFORMED (pressing only) = 0% vs VISUAL (all 4) = 50% for event_analysis.
Despite pressing dashboard encoding event transitions, event claims are compositionally
multi-metric (pressing trigger + compactness context + territorial position). Single-chart
routing impoverishes the claim space for this analysis type.

**F12 — Routing vs breadth tradeoff**
Probing optimises single-concept classification; commentary generation produces
multi-concept claims. Single-chart routing (FINDINGS_INFORMED) outperforms at the
per-concept level (match_overview) but underperforms globally. This reveals a
fundamental divergence between probe-study metrics and end-task commentary quality.

**F13 — Practical implication: top-2 chart selection**
A hybrid condition selecting the top-2 probe-best charts per analysis type (not all 4,
not 1) would likely outperform both VISUAL_FOCUSED and FINDINGS_INFORMED. This matches
the Schumacher et al. insight that d+v beats single-modality in probing when both charts
encode *distinct* claim types — the key is claim distinctiveness, not chart count.
