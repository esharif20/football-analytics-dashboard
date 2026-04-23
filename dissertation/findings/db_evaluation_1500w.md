# Evaluation: Grounding, Reasoning, and Database-Grounded Verification

## Evaluation Chapter (~1,500 words)

---

### 4.1 System Architecture and Data Flow

The evaluation pipeline is inseparable from the system it evaluates. The dashboard follows a three-layer architecture (Figure 1): a computer vision perception layer that processes broadcast video frame-by-frame, an analytics computation layer that aggregates per-frame tracking records into tactical metrics, and a reasoning and generation layer that converts those metrics into LLM commentary. The perception layer runs a YOLOv8x detection cascade — separate models for players, ball, and pitch keypoints — followed by ByteTrack multi-object tracking and SigLIP-based team classification, culminating in homography projection that maps pixel coordinates to real-world pitch coordinates in metres.

The reasoning layer applies MatchInsights: a module that pre-interprets analytics into six named sub-components before LLM generation. R1 (possession analysis) synthesises spell-length distributions and ball-in-play statistics into a possession narrative; R2 (tactical contributions) ranks each team on PPDA and compactness; R3 (pressing characterisation) cross-references PPDA with pressing intensity into a multi-metric verdict; R4 (player standouts) identifies the top distance and speed player per team; R5 (expected threat) summarises xT zone contributions; R6 (event narrative) constructs templated pass and challenge sentences. Commentary is generated across four analysis types: match overview, tactical deep-dive, event analysis, and player spotlight. All intermediate outputs are stored in Supabase PostgreSQL — per-frame player positions in the tracks table, aggregated statistics in the statistics table — enabling the two-layer verification methodology.

### 4.2 Core Evaluation: Grounding Rate and the Data Effect

The primary metric is a FActScore-style grounding rate (Min et al., 2023): each generated commentary paragraph is decomposed into atomic claims by a secondary LLM call, and each claim is verified against the analytics JSON under one of four verdicts — verified, refuted, plausible, or unverifiable. Seven ablation conditions test the data effect across three clips and two providers. The effect is large: under condition A (full analytics context), OpenAI GPT-4o achieves 40.6–48.0% grounding and Gemini 1.5 Pro achieves 55.8–64.4%; under condition F (no analytics), grounding falls to 5–23% and the hallucination rate rises to 10.6–18%. Cohen's h = 0.86–1.26 — a very large practical difference. Without data, the LLM does not simply become less precise; it fabricates plausible-sounding statistics that disappear as an error class as soon as any structured context is provided.

A cross-analysis comparison adds a counterintuitive finding: the reasoning layer benefit (R-ALL minus R-NONE grounding rate) is largest for Analysis 13, the clip with the least tactical data (+15.7pp), and smallest for Analysis 18, the data-richest clip (+2.2pp). When raw data is sparse, the MatchInsights Key Findings section serves as an essential narrative scaffold; when the data is well-structured, the LLM constructs an equivalent scaffold from the tables alone.

### 4.3 LLM-as-Judge, Prompt Format, and the Quality–Grounding Tradeoff

Grounding rate captures factual precision but not narrative quality. A G-Eval protocol (Liu et al., 2023) was implemented: an external Claude Haiku judge scores each output on five dimensions — coherence, consistency, fluency, relevance, and groundedness — on a 1–5 Likert scale. Claude was selected as judge specifically to avoid self-preference bias: an OpenAI judge inflates condition F outputs by 1.23 points relative to Claude's scores, consistent with Zheng et al. (2023). Krippendorff's alpha between judges is 0.120, confirming that cross-judge comparisons are unreliable; within-judge comparisons are the only valid basis for condition-level inference.

The G-Eval radar chart (Figure 2) reveals a counterintuitive pattern: condition H (no pre-interpreted insights, raw data tables only) scores higher than condition A (full MatchInsights) by 0.47 points aggregate (3.47 vs 3.00). When the LLM receives pre-interpreted findings, it paraphrases them rather than reasoning independently through the numbers — producing commentary the judge finds less analytically engaged. Condition F collapses to 1.77/5; the judge correctly penalises fabricated outputs even when they read fluently (0.5/5 on groundedness).

Format ablation tests (conditions I and J) examine prompt scaffolding independently of data quantity. Removing few-shot examples reduces grounding by 2.3pp on average; removing metric definitions by 1.5pp — smaller than the data effect but consistent across clips, confirming that prompt structure contributes a secondary grounding benefit beyond the analytics context.

### 4.4 QA Benchmark and VLM Comparison

The QA benchmark provides a complementary accuracy signal: rather than evaluating narrative commentary, it tests whether providers can answer specific factual questions from the analytics context. Questions are generated deterministically across eight categories — numeric, comparative, tactical, temporal, spatial, entity, multi-hop, and unanswerable — yielding 44–55 questions per clip.

Gemini outperforms OpenAI: Gemini achieves 68–93% accuracy across clips (Figure 3), OpenAI achieves 61–76%. The spatial category is particularly discriminating — Gemini achieves 83–100% on questions like "Which team was more compact?", OpenAI only 33%, reflecting Gemini's tendency to compare values directionally rather than retrieve a single value without comparison. Multi-hop questions requiring two-step reasoning across analytics fields fail for both providers. A parallel visual language model (VLM) comparison — replacing the analytics context with direct video frames — achieves grounding rates of 12–37%: below condition H but above condition F. VLMs retrieve some visually salient statistics but cannot match structured data precision (Krishnamurthy, 2025), confirming that the analytics pipeline rather than raw visual inference is the necessary grounding layer.

### 4.5 Database-Grounded Verification and Reasoning Layer Ablation

The JSON-based evaluation verifies claims against aggregate statistics — but the most interesting commentary claims are spatial and temporal. This creates a *verification chain* (video → CV perception → per-frame DB records → aggregate analytics → MatchInsights → LLM → claims → two-layer verification) that extends FActScore to include a per-frame evidential layer (Lewis et al., 2020). The Supabase tracks table stores player pitchX/Y coordinates for all 750 frames of Analysis 18: approximately 15,600 player-position records, 35 detected events (23 with spatial coordinates), mean inter-team centroid distance 7.4 m, mean ball speed 9.2 m/s. Zone occupancy computed from these records shows Team 1 spending 32.8% of player-frames in the defensive third — confirming the deep-block characterisation at a frame-by-frame level invisible to aggregate verification.

Under R-ALL (full reasoning), 20 of 48 claims (41.7%) are initially unverifiable by JSON; the DB layer resolves 4 of these 20 (20.0%), with match overview claims achieving a 40% DB resolution rate. Event analysis claims achieve 0% across all conditions and both verification layers — not because the LLM hallucinates, but because per-event claims referencing specific track IDs and timestamps cannot be matched against aggregate JSON fields without a dedicated event-to-record step; this is a verification coverage gap, not a model failure.

The reasoning layer ablation reveals the most novel finding. Twelve conditions vary which of the six sub-components (R1–R6) are present while holding the data tables constant. The expected result — that full reasoning R-ALL (49.0%) substantially outperforms no reasoning R-NONE (46.8%) — does not hold. All five single-component conditions outperform R-ALL: R-TACT 58.7%, R-POSS 53.9%, R-PLAY 53.0%, R-EVENT 51.8%, R-PRESS 50.3%. This subadditivity effect occurs because six competing narrative frames diffuse the model's attention away from any single verifiable anchor (Lewis et al., 2020).

The contrast is tangible in generated text. Under R-TACT, the model produces: *"Team 2 exhibited an exceptionally high pressing intensity with a PPDA of 0.1, while Team 1 engaged in an aggressive press with a PPDA of 0.2."* Every number maps directly to a verifiable JSON field. Under R-NONE on the same clip, the model writes: *"Both teams exhibited a pressing intensity of 0.3; however, Team 2's PPDA of 0.1 indicates a more aggressive approach"* — fabricating the aggregate "0.3" value absent from the data. R-TACT dominates because the tactical contributions sub-component (`_tactical_contributions()`) generates explicit ranking statements that map directly to verifiable fields; the tactical deep-dive achieves 71.4% grounding under R-TACT and the player spotlight reaches 92.9%.

Formation detection illustrates the data effect spatially (Decroos et al., 2019). With full analytics, formation estimates from k-means clustering of player positions are reproduced at approximately 60% accuracy against manually coded ground-truth formations. Without analytics, the model defaults to generic formations — 4-4-2, 4-3-3 — regardless of actual player distributions, producing hallucination rates approaching 100%. The model's intrinsic football knowledge supplies a plausible formation that bears no relation to tracked positions. Even with data, confidence scores are low (Team 1: 0.32, Team 2: 0.11), reflecting the instability of k-means over 30-second clips.

### 4.6 Limitations

Evaluation scale constrains generalisation. n=3 clips and single-run ablation conditions are sufficient for hypothesis generation, not statistical testing; a rigorous study would require 15–20 clips (Bouthillier et al., 2021). The DB resolution heuristic uses keyword matching on spatial references — a conservative approach that likely underestimates true spatial accuracy by 30–50 percentage points relative to a named-entity recognition implementation. Provider asymmetry is prompt-specific: OpenAI's R-TACT advantage (+9.7pp over R-ALL) does not transfer to Gemini, whose R-TACT rate (32.3%) falls below its own R-NONE (39.3%), indicating that sub-component anchoring strategies require per-provider tuning. Clips are 30-second segments, making temporal claims about match phases unresolvable against aggregate statistics.

Notwithstanding these constraints, the subadditivity finding provides actionable guidance: deploying R-TACT in isolation rather than bundling all six sub-components produces more reliably grounded sports commentary. This is consistent with the broader principle that focused, high-precision context outperforms broad context in factual grounding tasks (Lewis et al., 2020; Min et al., 2023) and identifies a concrete prompt engineering strategy for systems combining structured analytics with LLM generation (Wang et al., 2024).

---

## References

Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of MLSys 2021*.

Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in football. *Proceedings of KDD 2019*, 1851–1861.

Krishnamurthy, G. (2025). ShotsGPT: Wordalisation of football data for language model grounding. *arXiv preprint*.

Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *Proceedings of EMNLP 2023*.

Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. *Proceedings of EMNLP 2023*.

Wang, S., Bhambhoria, R., Sheratt, D., & Xiong, Y. (2024). TacticAI: An AI assistant for football tactics. *Nature Communications, 15*, 1175.

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *arXiv preprint arXiv:2306.05685*.

---

## Figure References

- **Figure 1**: System architecture — three-layer data flow (Perception → Analytics → Reasoning)
- **Figure 2**: `dissertation/figures/judge/claude_geval_radar.pdf` — G-Eval quality dimensions by condition (Coherence, Consistency, Fluency, Relevance, Groundedness; conditions A, F, H, I, J; 1–5 Likert)
- **Figure 3**: `dissertation/figures/qa/gemini_qa_accuracy.pdf` / `openai_qa_accuracy.pdf` — QA accuracy by category and provider
- **Figure 4**: `dissertation/figures/db_reasoning_layer_heatmap.pdf` — Ablation condition × analysis type grounding rate heatmap
