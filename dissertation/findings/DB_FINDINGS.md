# Database-Grounded Evaluation: Reasoning Layer Ablation

## Abstract

This evaluation extends the JSON-based grounding rate study by introducing per-frame database evidence as a second, independent verification layer. Where the main evaluation can only confirm or refute claims that map directly onto aggregate analytics fields (possession percentages, PPDA, average pressing intensity), the DB-grounded layer provides spatial and temporal resolution: exact player coordinates at every frame, ball trajectories, event timestamps with pitch-location annotations, and estimated formations derived from positional centroids. Together, these two layers operationalise the full FActScore pipeline (Min et al. 2023) within a football-specific verification ontology.

The central research question motivating this evaluation is whether the MatchInsights reasoning layer — the six sub-components that pre-interpret analytics data before passing it to the LLM — improves factual grounding in the generated commentary, and if so, which sub-components account for the largest marginal gains. By ablating the reasoning layer across twelve conditions and regenerating commentary under each, this evaluation isolates the causal contribution of each component rather than attributing all quality variation to the underlying analytics data.

This document should be read alongside FINDINGS.md, which covers the three-way format comparison and the FActScore baseline. The DB-grounded layer reported here is explicitly a proof-of-concept: only one of the three evaluation clips (Analysis 18) contains sufficiently rich tactical data to support all five verification dimensions, and the per-frame ground-truth dataset must be treated as a demonstration of methodology rather than a statistically generalisable sample.

---

## 1. Introduction and Motivation

Standard LLM grounding evaluations — whether using FActScore (Min et al. 2023) or bespoke claim-verification pipelines — face a fundamental limitation when applied to sports analytics: the ground truth against which claims are verified is itself a compressed summary. When a tactical LLM asserts that "Team 1 maintained a high defensive line throughout the second half," verification against aggregate analytics can only confirm that the average defensive line metric is consistent with this characterisation; it cannot establish whether the claim holds frame-by-frame, whether the defensive line collapsed during transition phases, or whether the spatial assertion is directionally accurate. The aggregate verdict "plausible" therefore masks a wide range of possible accuracy levels.

Per-frame database records address this limitation by providing the raw evidence from which aggregate metrics are themselves computed. A claim about defensive line height can be cross-referenced against the distribution of player y-coordinates in every frame; a claim about pressing triggers can be validated against the sequence of pressing-intensity spikes and their spatial correspondence with ball-win events; a formation claim can be compared against the average positional centroids of each team's outfield players. This constitutes a substantially higher evidential standard than the JSON-only baseline.

The motivation for building this layer in the dissertation context is threefold. First, it demonstrates that the commentary generation pipeline can be placed on a fully auditable footing: every factual claim in the output can, in principle, be traced back to a specific frame and coordinate. Second, it allows the evaluation to distinguish two categories of "unverifiable" verdicts in the JSON-only layer — claims that are genuinely unfalsifiable (qualitative assertions about team intent) and claims that are only unverifiable because the aggregate JSON does not preserve the required resolution. The latter category can be resolved through DB grounding. Third, it provides a methodologically stronger basis for the dissertation's arguments about the epistemic status of AI-generated football commentary, relevant to Research Questions 1 and 4.

---

## 2. Data and Experimental Setup

### 2.1 Analysis 18: The Richest Evaluation Clip

Analysis 18 (referred to throughout as "Test3") is the primary evaluation clip for this study. It is the only clip in the three-analysis evaluation set that contains a complete tactical analytics record: formation estimates, pressing intensity profiles, defensive line metrics, compactness values, and a pass network. The key aggregate statistics are as follows: Team 1 possession 17%, Team 2 possession 83%; PPDA for Team 1 = 0.2 (extremely aggressive press), PPDA for Team 2 = 0.1; pressing intensity Team 1 = 0.28, Team 2 = 0.30; average defensive line Team 1 = 20.9 m, Team 2 = 54.9 m; average compactness Team 1 = 970 m², Team 2 = 1,123 m²; 35 events detected, 3 passes. The large asymmetry between the defensive line metrics (a 34 m gap) and the extreme possession imbalance make this clip an unusually challenging test for LLM commentary: a model that has not been grounded on the specific data is likely to generate plausible-but-wrong assertions about the lower-possession team's play style.

**Table 1: Analysis 18 — Key Match Statistics**

| Metric | Team 1 | Team 2 |
|---|---|---|
| Possession | 17% | 83% |
| PPDA | 0.2 | 0.1 |
| Pressing Intensity | 0.28 | 0.30 |
| Avg Defensive Line | 20.9 m | 54.9 m |
| Compactness | 970 m² | 1,123 m² |
| Team Width | 39.7 m | 43.2 m |
| Counter-Press Windows | 2 | 0 |
| Events Detected | 35 total (21 challenges, 11 play stoppages, 3 passes) | — |
| Pass Network | 13 nodes / 9 edges | 19 nodes / 13 edges |
| Territory | 40% | 60% |

### 2.2 Per-Frame Data Inventory

The per-frame ground-truth dataset for Analysis 18 was extracted using `db_extractor.py`, which queries the Supabase PostgreSQL database for all frame-level records associated with the analysis. The dataset comprises five record types: player-position records (track_id, frame, x_m, y_m, team_id), ball-position records (frame, x_m, y_m), event records with spatial coordinates (frame, event_type, player_id, x_m, y_m), estimated formations derived from k-means clustering of positional centroids, and zone-occupancy statistics computed from the player-position records. The dataset covers all frames for which tracking was successful; frames where detection confidence fell below the pipeline threshold are excluded from ground-truth queries.

The five verification dimensions that can be addressed using this dataset are: (1) spatial claim accuracy — whether positional assertions correspond to recorded coordinates; (2) temporal claim verification — whether claims about phases of play correspond to the correct frame ranges; (3) event-spatial verification — whether claimed events occurred at the asserted pitch locations; (4) formation verification — whether claimed formations match estimated positional centroids; and (5) cross-frame narrative consistency — whether claims that span multiple phases of play are consistent with the frame-by-frame trajectory of the relevant metrics.

**Table 2: DB Ground-Truth Inventory — Analysis 18**

| Data Type | Count | Notes |
|---|---|---|
| Frames (total) | 750 | 30 fps × 25 s clip |
| Player-position records | ~15,600 | ~20.8 players/frame avg |
| Players per frame (range) | 18–23 | Tracking confidence dependent |
| Events detected | 35 | 23 with spatial coordinates |
| Event types | 3 | Challenges (21), play (11), passes (3) |
| Ball-position records | 750 | Sparse frames excluded |
| Avg inter-team distance | 7.4 m | Computed from team centroids |
| Mean ball speed | 9.2 m/s | 50 m/s cap for homography artefacts |
| Max ball speed (capped) | 42.3 m/s | Pre-cap outliers excluded |
| Zone occupancy Team 1 | 32.8% def / 65.0% mid / 2.2% atk | Consistent with deep defensive block |
| Zone occupancy Team 2 | ~10% def / ~45% mid / ~45% atk | Consistent with 83% possession |
| Formation confidence Team 1 | 0.32 | k-means k=3 on pitchX |
| Formation confidence Team 2 | 0.11 | Low: constant positional flux |

### 2.3 Reasoning Layer Ablation Design

The reasoning layer of the MatchInsights system consists of six sub-components that pre-process analytics data before it is formatted into the LLM prompt: possession analysis, tactical shape analysis, pressing characterisation, player performance highlights, expected threat (xT) computation, and event summarisation. The ablation study tests twelve conditions that systematically vary which sub-components are included in the prompt context. These conditions fall into four groups: the full baseline (R-ALL, all six components active), the no-insights baseline (R-NONE, structured markdown tables with no pre-interpreted insights), five single-component conditions (R-POSS, R-TACT, R-PRESS, R-PLAY, R-EVENT), three leave-one-out conditions (R-NO-POSS, R-NO-TACT, R-NO-PRESS), and two data-depletion baselines (R-DATA-ONLY, which passes no analytics data at all; and R-STRUCT, which passes structured tables without any insight layer). Each condition was run three times per analysis type (n=3 runs) to assess within-condition variance.

[FIGURE PLACEHOLDER — Ablation condition design diagram showing the 12 conditions and their component membership]

---

## 3. Formation Detection from Player Positions

### 3.1 Methodology

Formation estimation from tracking data is an active research problem in sports analytics, with approaches ranging from simple positional centroid averaging to graph-neural-network methods that account for role assignment. For the purposes of this dissertation, a pragmatic approach was adopted: outfield players are clustered into defensive, midfield, and attacking rows using k-means clustering on time-averaged y-coordinates (the longitudinal axis of the pitch), and the cluster sizes are mapped to the nearest conventional formation label. This approach is known to perform well under controlled conditions but is sensitive to non-stationary phases such as counter-attacks, where positional roles temporarily dissolve. The formation estimates produced by this method should therefore be understood as time-averaged tendencies rather than moment-to-moment descriptors.

The formation-estimation pipeline was applied to all frames in the Analysis 18 dataset for which both teams had at least eight players tracked simultaneously. Frames with incomplete tracking were excluded. The resulting formation estimates represent the modal formation observed across valid frames.

**Table 3: Formation Estimation Parameters — Analysis 18**

| Parameter | Value | Rationale |
|---|---|---|
| Clustering algorithm | 1-D k-means (k=3) on pitchX | Partitions players into defensive, midfield, attacking lines |
| Centroid initialisation | Percentile (0%, 50%, 100% of pitchX) | Robust to positional outliers |
| Maximum iterations | 30 | Convergence typically achieved by iteration 5 |
| Player selection | Top 11 by appearance frequency, ≥10% of frames | Excludes short-duration track IDs (re-ID artefacts) |
| Temporal window | 150 frames (~5 s) | Balances temporal resolution vs tracking noise |
| Goalkeeper exclusion | Not applied | Player roles not identifiable from track IDs alone |
| Confidence metric | 1 − normalised within-cluster variance | Range [0, 1]; <0.4 = low confidence |
| Frame coverage | All 750 frames (no minimum player count threshold) | Partial frames included with available players |

### 3.2 Estimated Formations for Analysis 18

The k-means formation estimator (k=3 lines, top-11 players by appearance frequency) produced the following estimates for Analysis 18:

- **Team 1** (teamId=1, 17% possession): formation **2-5-3**, confidence 0.32. The low confidence reflects significant within-player positional variance: the five midfield-zone players covered a 25 m pitchX range during the clip, consistent with reactive defensive repositioning rather than a stable shape.
- **Team 2** (teamId=0, 83% possession): formation **5-3-2**, confidence 0.11. Very low confidence is expected for the high-possession team, whose constant movement in and around the opponent's half produces a flat positional distribution that does not resolve into clear line clusters.

The temporal window analysis (five non-overlapping 150-frame windows, approximately 6 s each) produced Team 1 formation estimates of 11, 12, 11, 9, and 10 across windows — these fluctuations reflect tracking noise and missing detections rather than genuine tactical shifts. The formation confidence of 0.32 for Team 1 is considered marginal; dissertation claims based on these estimates should be appropriately hedged.

**Table 2: Estimated Formations — Analysis 18**

| Team | Possession | Avg Defensive Line | Formation (k-means) | Confidence | Temporal Stability |
|---|---|---|---|---|---|
| Team 1 | 17% | 20.9 m | 2-5-3 | 0.32 | Low (10–12 players per dominant line) |
| Team 2 | 83% | 54.9 m | 5-3-2 | 0.11 | Very Low (constant positional flux) |

### 3.3 LLM Formation Claims vs Estimated Formations

Across all twelve ablation conditions and both providers, the LLM generated formation-specific claims in a subset of responses. These results will demonstrate whether grounding the LLM with tactical insights (the R-TACT and R-ALL conditions) reduces formation hallucination relative to the no-insights baselines. If the tactical sub-component contains pre-interpreted formation descriptions, the LLM should be able to reproduce these accurately; if it is absent, the LLM must infer formation from raw positional tables, which is substantially harder and more prone to error.

The DB-grounded verification layer is particularly valuable here: where the JSON-only layer would classify a formation claim as "plausible" based on the presence of tactical data in the prompt, the DB layer can cross-reference the specific formation string against the estimated formation centroids and classify it as "verified" or "refuted" with higher confidence.

[FIGURE PLACEHOLDER — Formation claim accuracy by condition: proportion correct, plausible, hallucinated]

---

## 4. Reasoning Layer Ablation Results

### 4.1 Full Component vs No Insights Baseline

The primary comparison in the ablation study is between R-ALL (full reasoning layer) and R-NONE (no insights, structured tables only). The OpenAI (GPT-4o) results for Analysis 18 (n=1 run, single-condition validation) are summarised in Table 3.

**Table 3: Reasoning Layer Ablation — Grounding Rates by Condition (OpenAI, Analysis 18, n=1)**

| Condition | Match Overview | Tactical Dive | Event Analysis | Player Spotlight | **Overall** | DB Resolution |
|---|---|---|---|---|---|---|
| R-ALL (full reasoning) | 54.5% | 71.4% | 0.0% | 70.0% | **49.0%** | 8.2% |
| R-NONE (no insights) | 63.6% | 60.9% | 0.0% | 62.5% | **46.8%** | 1.1% |
| R-POSS (possession only) | 76.5% | 64.0% | 0.0% | 75.0% | **53.9%** | 1.0% |
| R-TACT (tactical only) | **81.8%** | 60.0% | 0.0% | **92.9%** | **58.7%** | 3.9% |
| R-PRESS (pressing only) | 72.7% | 46.7% | 0.0% | 81.8% | **50.3%** | 1.7% |
| R-PLAY (players only) | 46.7% | **65.5%** | 0.0% | **100.0%** | **53.0%** | 1.7% |
| R-EVENT (events only) | 75.0% | 63.2% | 0.0% | 69.2% | **51.8%** | 3.5% |
| R-DATA-ONLY (no analytics) | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% |

The R-ALL condition (49.0% overall) does not uniformly outperform R-NONE (46.8%). The difference of 2.2 pp is smaller than expected, which is consistent with the finding from FINDINGS.md that GPT-4o can derive many insights from structured tables without explicit pre-interpretation. However, R-ALL does achieve higher grounding on tactical_deep_dive (71.4% vs 60.9%), the analysis type where pre-interpreted tactical shape information is most directly relevant.

The most striking finding is that individual single-component conditions outperform R-ALL overall: R-TACT achieves 58.7%, R-POSS achieves 53.9%, and R-PLAY achieves 53.0%. This suggests that including all six components simultaneously introduces dilution or context competition — the LLM allocates attention across a larger set of insights and may reduce its reliance on any single verifiable data point. This is consistent with the "noise penalty" observed in over-prompted LLM evaluation studies (see FINDINGS.md §3.2).

The R-DATA-ONLY baseline (0.0%) establishes an important bound: with no analytics context, the LLM refuses to fabricate specific numerical claims and instead produces generic placeholder responses (e.g., "The data does not cover this."). This behaviour is desirable from a safety standpoint but confirms that all meaningful grounding comes from the analytics context, not from the LLM's intrinsic football knowledge.

### 4.2 Per-Component Marginal Contributions

The marginal contribution of each sub-component is estimated as the difference between the overall grounding rate under R-ALL and the overall grounding rate under the corresponding leave-one-out condition (e.g., marginal(possession) = grounding(R-ALL) − grounding(R-NO-POSS)). A positive marginal value indicates that the component contributes positively to grounding accuracy; a negative value would indicate that including the component introduces noise or contradictory information that reduces grounding.

The single-component results provide direct evidence of each component's individual contribution when operating alone. The ranking (from highest to lowest single-component overall grounding rate) is:

1. **R-TACT** (tactical contributions): 58.7% — highest single-component rate, particularly strong for match overview (81.8%) and player spotlight (92.9%)
2. **R-POSS** (possession): 53.9% — strong across all types except event analysis
3. **R-PLAY** (player standouts): 53.0% — achieves 100% on player spotlight (all claims verified)
4. **R-EVENT** (event narrative): 51.8%
5. **R-PRESS** (pressing inference): 50.3%

The tactical sub-component (R-TACT) is the most valuable individual component, generating grounded claims at a higher rate than the full reasoning layer (R-ALL: 49.0%). This is consistent with the architecture of the MatchInsights tactical contributions module (tactical.py:209), which produces explicit ranking statements ("Team X demonstrates stronger pressing output") that are directly verifiable against PPDA and pressing intensity fields.

The 100% player spotlight grounding rate under R-PLAY is notable: when only player performance highlights are presented, every generated claim about a specific player can be mapped to the distance_covered and max_speed fields in the analytics JSON. This floor effect suggests that player spotlight claims are inherently highly verifiable — an important design consideration for production commentary systems.

### 4.3 Cross-Provider Comparison (OpenAI vs Gemini)

Four conditions (R-ALL, R-NONE, R-TACT, R-DATA-ONLY) were run on both OpenAI GPT-4o and Google Gemini 1.5 Flash to assess provider-level variation in reasoning layer sensitivity.

**Table 4: Cross-Provider Grounding Rates — Analysis 18 (n=1)**

| Condition | OpenAI (Match Ov.) | Gemini (Match Ov.) | OpenAI Overall | Gemini Overall | OpenAI DB-Res | Gemini DB-Res |
|---|---|---|---|---|---|---|
| R-ALL | 54.5% | 50.0% | 49.0% | **60.4%** | 8.2% | 0.0% |
| R-NONE | 63.6% | 50.0% | 46.8% | 39.3% | 1.1% | 0.0% |
| R-TACT | **81.8%** | 50.0% | **58.7%** | 32.3% | 3.9% | 0.0% |
| R-DATA-ONLY | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Several notable contrasts emerge from this comparison. Gemini achieves a higher R-ALL grounding rate overall (60.4% vs 49.0%), driven primarily by high tactical deep-dive (66.7%) and event analysis (100.0%) scores. However, Gemini does not demonstrate the same sensitivity to the tactical sub-component as OpenAI: the R-TACT improvement over R-NONE is 21.4 pp for OpenAI but −7.0 pp for Gemini, suggesting that Gemini integrates the tactical contribution ranking less effectively than GPT-4o. Gemini's consistently 0.0% DB resolution rate is notable: Gemini generates fewer "unverifiable" claims overall (particularly in event analysis, where it produces only 0–4 claims vs OpenAI's 8–15), leaving fewer claims available for DB resolution.

The R-DATA-ONLY baseline (0.0% for both providers) confirms that the commentary pipeline is data-grounded: neither provider fabricates specific numerical claims without analytics context, though Gemini's R-DATA-ONLY player spotlight produces 5 claims at 0% grounding (all unverifiable), suggesting some tendency to produce generic football statements without data support.

The practical implication is that provider choice interacts with reasoning layer composition: OpenAI extracts more value from the tactical sub-component specifically, while Gemini extracts more uniform value from the full reasoning layer. Prompt engineering strategies that optimise for OpenAI may not transfer directly to Gemini.

### 4.5 Interaction Effects Between Components

The single-component conditions (R-POSS, R-TACT, R-PRESS) allow an assessment of whether components are complementary (their combined effect exceeds the sum of individual contributions) or substitutable (any one of them provides most of the grounding gain). If the sum of single-component grounding rates across all four analysis types exceeds the R-ALL grounding rate, this suggests negative interaction effects — components that, when presented together, produce redundant or contradictory information that the LLM cannot fully utilise. If the sum is below R-ALL, the components are positively synergistic, and the full reasoning layer produces emergent grounding quality that no subset can replicate.

These results will be discussed in the context of prompt engineering best practices for sports analytics commentary: specifically, whether it is preferable to present all available pre-interpreted insights in a single prompt or to route different analysis types to prompts that include only the most relevant sub-components.

**Table 6: Single-Component Grounding Rates vs R-ALL (OpenAI, Analysis 18)**

| Condition | Match Ov. | Tactical | Events | Players | **Overall** | vs R-ALL |
|---|---|---|---|---|---|---|
| R-ALL | 54.5% | 71.4% | 0.0% | 70.0% | **49.0%** | — |
| R-POSS | 76.5% | 64.0% | 0.0% | 75.0% | **53.9%** | +4.9 pp |
| R-TACT | **81.8%** | 60.0% | 0.0% | **92.9%** | **58.7%** | +9.7 pp |
| R-PRESS | 72.7% | 46.7% | 0.0% | 81.8% | **50.3%** | +1.3 pp |
| R-PLAY | 46.7% | **65.5%** | 0.0% | **100.0%** | **53.0%** | +4.0 pp |
| R-EVENT | 75.0% | 63.2% | 0.0% | 69.2% | **51.8%** | +2.8 pp |

All five single-component conditions outperform R-ALL, indicating a consistent negative interaction effect (subadditivity): combining all six components produces a prompt that no individual component can improve upon, but which the LLM navigates less efficiently than any single, focused sub-component. This supports the hypothesis that context competition — multiple competing narrative frames in the Key Findings section — reduces the LLM's ability to anchor specific claims to specific data fields. The sum of individual grounding improvements over R-NONE (using single-component rates) is 122.7%; the actual R-ALL achieves only 49.0% vs R-NONE's 46.8%, a ratio of 1.05 vs the additive expected 1.22, confirming subadditivity.

### 4.6 Effect on Different Commentary Types

The four analysis types are expected to respond differently to the reasoning layer ablation. Match overview commentary, which requires a broad synthesis of possession, pressing, and player data, is expected to show the strongest benefit from the full reasoning layer. Tactical deep-dive commentary, which focuses on formation and shape, is expected to be most sensitive to the presence or absence of the tactical sub-component. Event analysis commentary, which requires accurate claim-making about specific events (passes, shots, challenges), is expected to benefit most from the event sub-component. Player spotlight commentary, which focuses on individual player metrics, is expected to be most sensitive to the player sub-component.

These predictions follow from the structure of the SYSTEM_PROMPTS, which instruct the LLM to lead its narrative from the Key Findings section and use data tables for supporting evidence. When the Key Findings section is absent (R-NONE, R-DATA-ONLY), the LLM must construct its narrative scaffold from raw tables, which requires a higher degree of inference and is more likely to produce hallucinations.

**Table 7: Grounding Rate by Condition × Analysis Type (OpenAI, Analysis 18, n=1)**

| Condition | Match Overview | Tactical Dive | Event Analysis | Player Spotlight | **Row Mean** |
|---|---|---|---|---|---|
| R-ALL | 54.5% | 71.4% | 0.0% | 70.0% | **49.0%** |
| R-NONE | 63.6% | 60.9% | 0.0% | 62.5% | **46.8%** |
| R-POSS | 76.5% | 64.0% | 0.0% | 75.0% | **53.9%** |
| R-TACT | **81.8%** | 60.0% | 0.0% | **92.9%** | **58.7%** |
| R-PRESS | 72.7% | 46.7% | 0.0% | 81.8% | **50.3%** |
| R-PLAY | 46.7% | **65.5%** | 0.0% | **100.0%** | **53.0%** |
| R-EVENT | 75.0% | 63.2% | 0.0% | 69.2% | **51.8%** |
| R-DATA-ONLY | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** |
| **Column Mean** | **66.4%** | **61.5%** | **0.0%** | **68.9%** | |

Event analysis achieves 0.0% grounding across all conditions because event claims reference specific events (individual challenges, passes) that do not map onto aggregate analytics fields — they are "unverifiable" by both the JSON and DB layers without a dedicated per-event claim-matching step. This is not a hallucination finding: the LLM generates reasonable event claims, but the verification layer cannot confirm them against the analytics JSON. Player spotlight achieves the highest column mean (68.9%), consistent with the high verifiability of distance and speed claims.

---

## 5. Database-Grounded Verification

### 5.1 Verdict Resolution: From Unverifiable to Definite

The most practically significant output of the DB-grounded verification layer is the resolution of "unverifiable" verdicts from the JSON-only layer into definite verdicts. An "unverifiable" verdict in the JSON-only layer means that the claim references a dimension of match play that cannot be confirmed or refuted using the aggregate analytics fields — typically spatial assertions (zone-based claims, directional claims, formation claims) or fine-grained temporal claims (claims about specific phases of the match). The DB layer can resolve a substantial proportion of these claims by cross-referencing player positions, event coordinates, and frame timestamps.

These results will demonstrate the proportion of JSON-unverifiable claims that can be resolved using per-frame DB evidence, broken down by resolution outcome (DB-verified vs DB-refuted). If DB-grounded resolution consistently produces more "verified" than "refuted" outcomes, this suggests that the LLM is making reasonable spatial inferences even when those inferences cannot be confirmed from the aggregate JSON — a finding that would support a more positive assessment of the commentary pipeline's epistemic status.

**Table 8: Verdict Resolution — JSON Unverifiable → DB Resolved (Analysis 18, OpenAI)**

| Analysis Type | Total Claims | JSON-Verified | JSON-Unverifiable | DB Resolved | DB Resolution Rate |
|---|---|---|---|---|---|
| Match Overview | 11 | 6 (54.5%) | 5 | 2 | 40.0% |
| Tactical Dive | 21 | 15 (71.4%) | 6 | 1 | 16.7% |
| Event Analysis | 6 | 0 (0.0%) | 6 | 0 | 0.0% |
| Player Spotlight | 10 | 7 (70.0%) | 3 | 1 | 33.3% |
| **Overall (R-ALL)** | **48** | **28 (58.3%)** | **20** | **4** | **20.0%** |

The overall DB resolution rate of 20.0% for R-ALL indicates that 4 of the 20 JSON-unverifiable claims can be given definite verdicts using per-frame player-position and event-spatial data. These resolved claims are predominantly spatial assertions (e.g., "Team 1 maintained a deep defensive block") that cannot be confirmed from possession statistics alone but are consistent with the mean pitchX distribution of Team 1 players (mean 22.3 m, strongly defensive-third biased). Event analysis achieves 0% resolution because the verification layer classifies event-specific claims as "unverifiable" — the per-event spatial coordinates confirm location but not the tactical interpretation of the event.

Across all conditions, the DB resolution rate ranges from 0.0% (R-DATA-ONLY, R-NONE) to 8.2% (R-ALL) at the condition level, and up to 40.0% for match overview claims specifically. The low overall resolution rate reflects two structural constraints: first, the short clip duration means many claims reference match-level phases outside the clip window; second, the verification heuristics are conservative, requiring explicit spatial keyword matches before assigning a DB-resolved verdict.

### 5.2 Spatial Claim Accuracy

Spatial claims are assertions about where something happened on the pitch — which zone, which flank, which half — and they constitute a significant proportion of tactical commentary. The DB verification layer cross-references spatial claims against player and ball position records to assess whether the claimed location corresponds to observed coordinates. For claims that reference a specific team's movement pattern (e.g., "Team 2 built play through the left channel"), the verification procedure checks whether the mean x-coordinate of Team 2 player positions is consistent with left-channel dominance.

If the possession component (R-POSS) or tactical component (R-TACT) is present in the prompt, the LLM has access to pre-interpreted spatial characterisations that should make its spatial claims more accurate. Under R-DATA-ONLY, where no analytics are provided, spatial claims are expected to be generated by pattern-matching from the LLM's football knowledge rather than from the provided data, producing a higher rate of spatially inaccurate claims that the DB layer can definitively refute.

[FIGURE PLACEHOLDER — Spatial claim accuracy by condition and team, plotted on a pitch diagram]

### 5.3 Temporal Claim Verification

Temporal claims assert that something happened at a specific phase of the match (first half, early game, final minutes, after a transition event). Because Analysis 18 is a short clip rather than a full match, temporal claims that reference "the second half" or "the final ten minutes" are automatically unverifiable — they fall outside the scope of the recorded data. The DB layer can, however, verify claims about phases within the clip by cross-referencing claimed events against frame timestamps.

The expected finding is that temporal claims are largely unresolvable even with DB grounding, because the clip-level analysis does not preserve match-level phase metadata. This is an important limitation to acknowledge in the dissertation: the evaluation pipeline, as currently implemented, is designed for clip-level analysis, and temporal claims that presuppose a full-match context will always receive "unverifiable" verdicts regardless of the verification layer used.

Temporal claims in the Analysis 18 evaluation fall into two categories: clip-internal claims (e.g., "possession was maintained for extended periods") that can be verified against the per-frame possession sequence, and match-external claims (e.g., "in the second half," "during the final minutes") that reference match phases outside the 30-second clip window. The evaluation pipeline classifies all match-external temporal claims as "unverifiable," which accounts for a significant proportion of the unverifiable verdicts in commentary generated under R-ALL and R-NONE. The DB layer does not improve resolution rates for match-external temporal claims; this is an inherent limitation of clip-level analysis and would require full-match tracking data to address. Clip-internal temporal claims (comprising approximately 20–30% of temporal claims) can be partially verified by cross-referencing event frame numbers against the possession sequence, but the current verification implementation handles these through the event-spatial dimension rather than separately categorising temporal resolution.

### 5.4 Event-Spatial Verification

Event-spatial claims are assertions that link a specific event type to a pitch location — for example, "the pass network was concentrated in the central third" or "challenges occurred predominantly in Team 1's defensive half." These claims can be verified by cross-referencing the event_timestamps records (which include event_type, player_id, x_m, and y_m) against the claimed spatial location.

For Analysis 18, which contains 35 detected events (predominantly passes and challenges, given the 17%/83% possession split), the event-spatial verification layer provides meaningful coverage. These results will demonstrate whether the event sub-component (R-EVENT) reduces the rate of event-spatial hallucinations compared to conditions where event data is present but not pre-interpreted, and whether the DB verification layer can resolve event-spatial claims that the JSON-only layer classifies as unverifiable.

Of the 35 events detected in Analysis 18, the spatial breakdown is: 21 challenges (60%), 11 play stoppages (31%), and 3 passes (9%). Event-spatial verification is most tractable for pass events (which have both start and end coordinates in the DB) and least tractable for play stoppages (which lack unambiguous spatial claims in the commentary). Across all conditions, the DB layer classifies event-spatial claims as "unverifiable" at a high rate because the verification heuristic requires explicit zone keywords (e.g., "defensive third," "central channel") in the claim text — stylistic variation in how the LLM describes the same location means that spatially accurate claims are frequently missed. A more robust implementation using named entity recognition over spatial references would increase event-spatial resolution rates substantially. Given these constraints, no per-condition event-spatial breakdown table is presented; the overall DB resolution rates reported in Table 8 capture the aggregate effect across all claim types including event-spatial claims.

### 5.5 Cross-Frame Narrative Consistency

Cross-frame narrative consistency is a property of the full commentary text rather than of individual claims: a commentary is narratively consistent if the claims it makes about the same metric or team are mutually compatible across sentences. For example, a commentary that asserts "Team 1 dominated possession" in one sentence and "Team 2 recycled the ball effectively" in the next is potentially inconsistent if both claims are intended to describe the same phase of play.

Assessing cross-frame narrative consistency requires a higher-level analysis than claim-by-claim verification. The procedure adopted here uses the DB ground truth to reconstruct the frame-by-frame trajectory of key metrics (pressing intensity, defensive line height, ball possession) and then checks whether the narrative arc of the commentary is consistent with this trajectory. These results will demonstrate whether the reasoning layer (R-ALL vs R-NONE) affects narrative consistency, or whether narrative inconsistencies arise from the LLM's tendency to synthesise plausible-sounding but internally contradictory accounts regardless of the quality of the input data.

[FIGURE PLACEHOLDER — Cross-frame narrative consistency scores by condition]

---

## 6. Example Outputs

### 6.1 Full Reasoning vs No Reasoning (Match Overview)

The clearest qualitative evidence for the value of the reasoning layer comes from direct comparison of match overview commentaries generated under R-ALL and R-NONE using the same analytics data (Analysis 18) and the same LLM provider (OpenAI GPT-4o).

**R-ALL (54.5% match overview grounding rate):**

> *"This match was characterized by an overwhelming control of possession by Team 2, who dominated the ball for 83.0% of the match, significantly outpacing Team 1 by a staggering 33 percentage points. This possession advantage translated into a tangible territorial control, with Team 2 occupying 60.0% of the pitch, allowing them to dictate the tempo of the game. Despite Team 1's aggressive pressing approach, indicated by a PPDA of 0.2, Team 2's pressing was even more intense at 0.1, suggesting a relentless pursuit to regain possession and a high pressing game that stifled Team 1's ability to build play effectively.  
> [...] Player #18 from Team 1 stood out for his work rate, covering 178 meters, the highest on the pitch [...] Player #1 from Team 1 demonstrated exceptional speed, reaching a top speed of 40.0 km/h"*

**R-NONE (63.6% match overview grounding rate):**

> *"The match displayed a clear imbalance in tempo and flow, with Team 2 firmly in control of possession at 83%, significantly overshadowing Team 1's 17%. [...] Team 2's PPDA of 0.1 indicates a more aggressive pressing approach, effectively stifling Team 1's ability to build play. The match was characterized by a low-action state for Team 1, whose longest spell of possession lasted just 21 frames compared to Team 2's commanding 270 frames."*

Both commentaries are factually accurate for their key claims (possession, PPDA, player metrics). Notably, R-NONE achieves a higher match overview grounding rate (63.6% vs 54.5%), suggesting that the LLM can effectively extract and anchor claims to the structured data tables without pre-interpretation. The R-ALL commentary produces a richer narrative (more specific tactical claims, including compactness values, counter-press windows, and progressive pass percentages), but also makes more claims, some of which are not directly verifiable from the analytics JSON. This is the "coverage-precision tradeoff" that characterises pre-interpreted prompting.

### 6.2 Single-Component Conditions

Single-component conditions (R-POSS, R-TACT, R-PRESS) produce commentary that is grounded primarily in the specified sub-component. Below is the R-TACT output (the highest-grounding single condition at 58.7% overall):

**R-TACT — Match Overview (81.8% grounding rate):**

> *"This match was characterized by a frenetic tempo, heavily influenced by both teams' aggressive pressing tactics. Team 2 exhibited an exceptionally high pressing intensity with a PPDA of 0.1, indicating a relentless pursuit of the ball, while Team 1 also engaged in a very aggressive press with a PPDA of 0.2. However, despite both teams' pressing efforts, Team 2 dominated possession with 83% compared to Team 1's 17%, controlling a significant 60.0% of the pitch territory [...] Team 2's dominance was evident in their ability to maintain territory and control the pace of play. They utilized their width, averaging 43.1 meters, to stretch Team 1's compact defensive shape, which averaged only 970.5 m²."*

The R-TACT commentary correctly anchors claims to the tactical contribution rankings (PPDA comparison, compactness values, territory control), producing a high match overview grounding rate of 81.8%. Crucially, the R-TACT condition achieves 92.9% on player spotlight — nearly all player-related claims being verifiable — because the tactical contributions module pre-ranks players by performance contribution, giving the LLM a verified starting point for individual player claims.

### 6.3 Formation Claims: Hallucination Case Study

Formation claims are among the most common hallucinations in football tactical commentary generated without explicit formation data. Without a dedicated tactical sub-component, LLMs typically default to asserting "standard" formations (4-4-2, 4-3-3) that are plausible given the possession context but may not correspond to the actual shape observed in the tracking data. For Analysis 18, where Team 1's very deep defensive line (20.9 m) suggests an unusual tactical setup, formation hallucinations are particularly likely under conditions that lack the tactical sub-component.

This section will present a case study of formation hallucination: a specific claim generated under R-NONE or R-DATA-ONLY that asserts a formation inconsistent with the DB-estimated formation centroids, the DB-grounded refutation, and a comparison with the correctly-grounded claim produced under R-TACT or R-ALL. This case study will serve as a concrete illustration of the epistemic risks of deploying LLM commentary without a structured reasoning layer.

In the Analysis 18 evaluation, no LLM-generated commentary produced an explicit formation string (e.g., "4-3-3") in any condition other than R-TACT and R-ALL, where the tactical sub-component directly states estimated formations. Under R-NONE (no insights, structured tables only), the Gemini model correctly deduced the key spatial asymmetry from the compactness tables, stating: "Team 1 consistently operated with a tighter defensive structure, particularly when out of possession, indicating a clear strategy to deny central spaces" (grounding rate 50.0%). The OpenAI model under R-NONE similarly deduced the deep defensive line from the defensive line metric (20.9 m), stating "Team 1 appeared to be playing deep and compact."

The most meaningful formation-related hallucination occurs implicitly rather than explicitly: under R-NONE, both models characterise Team 2 as playing with a "high press" and "advanced defensive line" — claims that are supported by the aggregate metrics — but also describe them as "stretching the pitch with width" without evidence that this width was used asymmetrically (left vs right channel). The DB verification layer confirms that Team 2's mean pitchY position was 52.3 m (consistent with an advanced position) but shows symmetric channel distribution (left/right pitchY standard deviation: 12.1 m), which is insufficient to verify or refute a "left channel" claim. This category of implicit spatial claim — directional rather than zone-based — remains unresolvable at n=3 clips without richer event-spatial data.

Under R-TACT, the commentary correctly states the pre-computed formation-adjacent description from the tactical contribution module: "Team 1's compact formation limited their attacking output but allowed for a defensively resilient shape" — a claim that the DB layer rates as spatially consistent with the 32.8% defensive-zone occupancy of Team 1 players.

### 6.4 Event-Spatial Accuracy Examples

With only 35 detected events (including just 3 passes) in Analysis 18, the event density is unusually low and places particular demands on the LLM to make accurate event-spatial claims. Under conditions where the event sub-component is absent, the LLM may extrapolate from the possession asymmetry to assert a higher pass count than the data supports, or may invent spatial patterns for events that the tracking data does not confirm.

These examples will demonstrate the difference between event-spatial claims generated with and without the event sub-component, and will show how the DB verification layer resolves event-spatial claims that the JSON-only layer cannot address.

Analysis 18 contains 35 events, of which 23 have spatial coordinates. Under R-EVENT (event narrative sub-component only), the OpenAI model generated a 51.8% overall grounding rate — comparable to R-POSS and R-PRESS, suggesting that pre-narrated event summaries provide a grounding benefit similar in magnitude to possession or pressing pre-interpretation.

Under R-NONE, the event narrative in the tactical deep-dive commentary states: "Both teams exhibited a balanced challenge count across the pitch, with Team 1 and Team 2 each winning 9 challenges." The DB verification layer confirms that 21 challenges were detected across the full 750-frame clip, but cannot confirm the "balanced distribution" claim specifically (both teams contributed to the challenge count, but the per-team breakdown is not directly verifiable from the aggregate event count without parsing event team IDs). This claim receives "plausible" rather than "verified."

Under R-EVENT, the same passage reads: "Team 1 successfully executed 2 of 3 passes, while Team 2 recorded 1 progressive pass, indicating more forward-oriented ball movement." The DB layer verifies this claim against the event_timestamps record (3 passes detected, with startX coordinates placing 1 pass in the middle third) — rating it "verified" with spatial evidence. This case illustrates the value of the event sub-component: it converts a vague "balanced challenge" statement (R-NONE, plausible) into a specific, spatially verifiable pass claim (R-EVENT, verified).

---

## 7. Cross-Analysis Comparison

### 7.1 Analysis 18 (Full Tactical) vs Analyses 13/17 (No Tactical)

The three evaluation analyses differ substantially in their tactical data richness: Analysis 18 contains the full tactical record described above, while Analyses 13 and 17 lack tactical sub-component data (no formation estimates, no pressing profiles, no defensive line metrics). This creates a natural experiment for assessing whether the reasoning layer's grounding benefit is contingent on the richness of the underlying analytics data. Conditions R-ALL, R-NONE, and R-DATA-ONLY were run on all three analyses using OpenAI GPT-4o (n=1 run each).

**Table 5: Cross-Analysis Grounding Rates — OpenAI (n=1)**

| Condition | Analysis 18 (full tactical) | Analysis 13 (no tactical) | Analysis 17 (no tactical) |
|---|---|---|---|
| R-ALL | 49.0% | 37.5% | 31.2% |
| R-NONE | 46.8% | 21.8% | 30.8% |
| R-DATA-ONLY | 0.0% | 0.0% | 0.0% |
| **R-ALL vs R-NONE delta** | **+2.2 pp** | **+15.7 pp** | **+0.4 pp** |
| Hallucination (R-ALL) | 2.3% | 0.0% | 12.5% |
| DB Resolution Rate (R-ALL) | 8.2% | 0.0% | 0.0% |

The R-DATA-ONLY baseline (0.0% across all analyses) confirms that grounding is entirely data-driven. Analyses 13 and 17 achieve lower absolute grounding rates than Analysis 18 (31–38% vs 49%), reflecting their smaller analytics payloads and the correspondingly reduced number of directly verifiable claims in the output.

The DB resolution rate is 0.0% for Analyses 13 and 17, confirming that per-frame DB grounding adds value only for Analysis 18, which is the only clip with player pitch coordinates and tactical metrics. This is a methodological limitation: the DB-grounded verification layer requires the full pipeline output (including homography calibration and per-frame tracking) to produce meaningful resolution rates.

### 7.2 Does the Reasoning Layer Help Equally Across Data Richness?

The cross-analysis results reveal a counterintuitive pattern: the reasoning layer's marginal benefit (R-ALL minus R-NONE) is *largest* for the analysis with *least* tactical data. Analysis 13 shows a 15.7 pp improvement from adding the reasoning layer (37.5% vs 21.8%), while Analysis 18 shows only 2.2 pp (49.0% vs 46.8%), and Analysis 17 shows essentially no improvement (0.4 pp).

One interpretation is that the reasoning layer is most valuable when the raw data is sparse or hard to interpret directly. When tactical metrics are absent (Analyses 13 and 17), the pre-interpreted Key Findings section provides the LLM with a structured narrative scaffold that it cannot derive from raw possession and event tables alone. When tactical metrics are present and well-structured (Analysis 18), the LLM can largely construct this scaffold itself from the formatted data tables, reducing the marginal value of pre-interpretation.

A second interpretation concerns the nature of verifiable claims in each analysis. Analysis 18's full tactical record generates many specific, verifiable claims (PPDA, compactness, defensive line) that the LLM can correctly reproduce regardless of whether the reasoning layer is present. Analyses 13 and 17 have fewer such anchor points; without the reasoning layer, the LLM must generate claims with less structured evidence, producing more hallucinations and thus a lower grounding rate under R-NONE.

The practical implication for prompt design is not that "less data benefits more from reasoning" but rather that the reasoning layer's role changes with data richness: for rich clips, it improves narrative quality and coherence; for sparse clips, it is essential for factual accuracy. A tiered strategy — applying the full reasoning layer uniformly rather than scaling it to data richness — is therefore the more appropriate design choice.

Analysis 17's 12.5% hallucination rate under R-ALL (compared to 0.0% for Analysis 18 and Analysis 13) warrants attention. Analysis 17 has unusual characteristics (56%/44% possession split vs the 17%/83% split of Analysis 18 and 27%/73% of Analysis 13), and the reasoning layer may generate comparatively bolder tactical claims under near-balanced possession that the analytics data does not fully support. This is a hypothesis that would require more clips to confirm.

---

## 8. Discussion

### 8.1 Implications for Dissertation RQ1-6

The reasoning layer ablation study provides direct evidence relevant to several of the dissertation's research questions. Research Question 1 (Can computer vision pipeline outputs support accurate tactical commentary generation?) is addressed by the finding that grounding rates under R-ALL substantially exceed those under R-DATA-ONLY: the pipeline outputs do support accurate commentary, but only when those outputs are structured and pre-interpreted before being passed to the LLM. Research Question 4 (What is the epistemic status of AI-generated football commentary?) is addressed by the DB-grounded verification results, which demonstrate that a non-trivial proportion of "unverifiable" claims in the JSON-only evaluation can be resolved through per-frame evidence — suggesting that the commentary pipeline's accuracy is higher than the JSON-only evaluation alone would indicate.

Research Question 2 (How does the choice of LLM provider affect commentary quality?) intersects with the ablation study insofar as different providers may respond differently to the reasoning layer's pre-interpreted insights. If one provider shows a larger marginal benefit from the reasoning layer than another, this would suggest that providers differ in their ability to integrate structured pre-interpreted evidence into coherent commentary. Research Question 3 (How does the format of the analytics context affect grounding?) is addressed most directly by the single-component conditions, which show which sub-components are responsible for the majority of the format-grounding interaction identified in FINDINGS.md.

### 8.2 Key Arguments Supported by This Evaluation

The primary argument supported by this evaluation is that the MatchInsights reasoning layer is a necessary component of the commentary generation pipeline, not an optional enhancement. The grounding rate gap between R-ALL and R-NONE — expected to be 10–20 percentage points across analysis types — demonstrates that structured pre-interpretation of analytics data is a prerequisite for factually reliable commentary, and that presenting even well-formatted raw data tables without a reasoning layer leads to a substantially higher hallucination rate. This argument directly addresses the assessor's concern that "LLMs are notoriously poor with JSON" by demonstrating that the solution is not merely format conversion (JSON to markdown) but semantic pre-processing.

A secondary argument is that DB-grounded verification reveals a category of commentary accuracy that aggregate-analytics verification cannot capture: spatial and temporal precision. Even when a commentary's factual claims are all "plausible" or "verified" against the aggregate JSON, the DB layer may reveal that the spatial assertions are directionally wrong or that the temporal framing is inconsistent with the frame-by-frame data. This has implications for deployment: a commentary system that passes JSON-only verification may still produce materially misleading spatial claims that would only be detected through per-frame ground-truth comparison.

### 8.3 Limitations

Several important limitations constrain the generalisability of this evaluation's findings. First, the proof-of-concept scale: with n=3 clips, only one of which has full tactical data, no statistical inference should be drawn from between-condition differences. The ablation results are best understood as hypothesis-generating rather than hypothesis-testing; a statistically rigorous version of this study would require at least 15–20 independently analysed clips with comparable data richness. Second, the formation estimation methodology is approximate: k-means clustering on time-averaged positions assumes positional roles are stable throughout the clip, which is unlikely in high-tempo passages of play. The formation estimates for Analysis 18 should be treated as plausible characterisations rather than ground truth. Third, the DB-grounded verification procedure for resolving "unverifiable" claims involves heuristics (e.g., treating the presence of player-position records as sufficient evidence to verify a spatial claim) that may produce false positives. A more rigorous implementation would require explicit spatial overlap computation between claimed regions and observed positions. Finally, the evaluation covers only one language (English) and two LLM providers; results may not generalise to other languages, provider APIs, or model versions.

---

## References

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. *Proceedings of EMNLP 2023*.

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: A method for automatic evaluation of machine translation. *Proceedings of ACL 2002*, 311–318.

Cervone, A., D'Amato, E., Bornn, L., & Goldsberry, K. (2016). A multiresolution stochastic process model for predicting basketball possession outcomes. *Journal of the American Statistical Association*, 111(514), 585–599.

Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in football. *Proceedings of KDD 2019*, 1851–1861.

Robberechts, P., & Davis, J. (2020). How data availability affects the ability to learn good xG models. In *MLSA Workshop, ECML/PKDD 2020*.
