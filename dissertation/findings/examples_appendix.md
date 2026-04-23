# Appendix: Example Outputs, Claims, and QA Pairs

## A.1 Commentary Examples — Condition Comparison (Match Overview)

The following examples are real outputs from OpenAI GPT-4o on the same match clip (Clip 3: Team 1 44% / Team 2 56% possession, 141 contested frames). All three conditions receive identical video footage; only the analytics context supplied to the LLM changes.

---

### Condition A — Full Data + MatchInsights (Grounding Rate: 77.8%)

> *"The match exhibited a highly intense tempo, characterized by aggressive pressing from both teams, reflected in their identical PPDA values of 0.2, well below the typical range. Team 2 slightly edged possession with 56.0%, taking control of the game and dictating the pace. Despite this advantage, Team 1's pressing intensity mirrored that of Team 2, indicating a tactical commitment to regain possession quickly and disrupt the opponent's rhythm. Both teams were engaged in a back-and-forth battle, with Team 2 utilising their wider shape (51.8 m average width) and Team 1 relying on their more compact defensive formation (41.8 m average width). Player #20 from Team 1, covering 146 m, demonstrated the highest distance covered..."*

**Claim-level verdicts:**

| Verdict | Claim |
|---|---|
| VERIFIED | "identical PPDA values of 0.2" |
| VERIFIED | "Team 2 slightly edged possession with 56.0%" |
| VERIFIED | "total of 141 contested frames" |
| VERIFIED | "wider shape (51.8 m average width)" |
| VERIFIED | "Team 1's more compact defensive formation (41.8 m average width)" |
| VERIFIED | "Player #20 from Team 1 covering 146 m" |
| VERIFIED | "top speed of 40.0 km/h" |
| UNVERIFIABLE | "longest at 134 frames compared to Team 1's 69 frames" *(values transposed — unresolvable)* |
| UNVERIFIABLE | "compactness of Team 1, measured at 940.8 m²" *(value correct but qualitative framing)* |

---

### Condition H — Full Data, No MatchInsights (Grounding Rate: 61.5%)

> *"The match exhibited a moderate tempo, with Team 1 holding 44.0% possession compared to Team 2's 56.0%. Both teams displayed similar defensive strategies, as indicated by their PPDA of 0.2, suggesting a balanced pressing approach. Notably, Team 2's pressing intensity slightly edged Team 1's (0.24 vs 0.21), enabling them to maintain more sustained pressure throughout the match. The compactness of Team 1's shape (940.8 m²) allowed them to defend effectively, but they struggled to translate this into offensive output..."*

**Claim-level verdicts:**

| Verdict | Claim |
|---|---|
| VERIFIED | "Team 1 holding 44.0% possession" |
| VERIFIED | "Both teams... PPDA of 0.2" |
| VERIFIED | "Their longest possession spell of just 69 frames" |
| VERIFIED | "Team 1's average team width of 41.8 meters" |
| VERIFIED | "An average speed of 51.6 km/h" |
| UNVERIFIABLE | "Team 2's pressing intensity slightly edged Team 1's (0.24 vs 0.21)" |
| UNVERIFIABLE | "The compactness of Team 1's shape (940.8 m²)" |
| UNVERIFIABLE | "The total distance of the ball traveled was 429 meters" |
| UNVERIFIABLE | "Both teams having similar challenge counts (9 each)" |

*Note: The drop from 77.8% to 61.5% reflects the LLM making more unverifiable inferences about compactness and ball trajectory when not given pre-interpreted tactical summaries.*

---

### Condition F — No Analytics (Grounding Rate: 33.3%)

> *"The match exhibited a dynamic flow characterized by a high tempo, with Team 1 maintaining a possession rate of 58.3%. This figure suggests a slight edge over Team 2, who managed 41.7% of the ball. Team 1's pressing intensity was notably high at 0.46, indicative of their aggressive approach to regain possession quickly, which complemented their moderate PPDA of 9.2..."*

**Claim-level verdicts:**

| Verdict | Claim |
|---|---|
| **REFUTED** | "Team 1 maintaining a possession rate of 58.3%" *(actual: 44.0%)* |
| **REFUTED** | "Team 2, who managed 41.7% of the ball" *(actual: 56.0%)* |
| VERIFIED | "Team 1's pressing intensity was notably high at 0.46" |
| VERIFIED | "Team 2's lower pressing intensity of 0.22" |
| VERIFIED | "Team 1's xT rating of 0.036" |
| UNVERIFIABLE | "Player #4 covering 10,300 m" *(fabricated — 10× the actual distance)* |
| UNVERIFIABLE | "Player #8 recorded a remarkable speed of 29.5 km/h" |

*Possession statistics are inverted (Team 1 and 2 swapped). With no data, the LLM draws on generic football intuition and produces plausible-sounding numbers that are factually wrong.*

---

## A.2 Commentary Examples — Reasoning Layer Ablation (Analysis 18: 17%/83% possession)

These examples use Analysis 18 (the data-richest clip). All conditions use the same raw analytics tables; only the MatchInsights pre-interpretation changes.

---

### R-TACT Only (Grounding Rate: 58.7% — highest single-component)

> *"This match was characterized by a frenetic tempo, heavily influenced by both teams' aggressive pressing tactics. Team 2 exhibited an exceptionally high pressing intensity with a PPDA of 0.1, indicating a relentless pursuit of the ball, while Team 1 also engaged in a very aggressive press with a PPDA of 0.2. However, despite both teams' pressing efforts, Team 2 dominated possession with 83% compared to Team 1's 17%, controlling a significant 60.0% of the pitch territory, which allowed them to dictate the flow of the game effectively. Team 2's dominance was evident in their ability to maintain territory and control the pace of play. They utilized their width, averaging 43.1 meters, to stretch Team 1's compact defensive shape, which averaged only 970.5 m²..."*

Key: the commentary accurately reproduces the 83%/17% split, the PPDA values, compactness, and territory percentage — all directly verifiable against analytics fields.

---

### R-ALL — Full Reasoning (Grounding Rate: 49.0% — lower than R-TACT)

> *"This match was characterized by an overwhelming control of possession by Team 2, who dominated the ball for 83.0% of the match, significantly outpacing Team 1 by a staggering 33 percentage points. This possession advantage translated into a tangible territorial control, with Team 2 occupying 60.0% of the pitch, allowing them to dictate the tempo of the game. Despite Team 1's aggressive pressing approach, indicated by a PPDA of 0.2, Team 2's pressing was even more intense at 0.1, suggesting a relentless pursuit to regain possession and a high pressing game that stifled Team 1's ability to build..."*

The opening sentence fabricates "33 percentage points" (actual gap: 66 pp) — an unverifiable comparative. The full reasoning layer introduces six competing narrative frames; the model anchors fewer claims to specific verifiable fields.

---

### R-NONE — No Insights, Data Tables Only (Grounding Rate: 46.8%)

> *"The match displayed a clear imbalance in tempo and flow, with Team 2 firmly in control of possession at 83%, significantly overshadowing Team 1's 17%. This disparity is reflected in the pressing intensity, where both teams exhibited a pressing intensity of 0.3; however, Team 2's PPDA of 0.1 indicates a more aggressive pressing approach, effectively stifling Team 1's ability to build play. The match was characterized by a low-action state for Team 1, whose longest spell of possession lasted just 21 frames compared to Team 2's commanding 270 frames..."*

---

### R-DATA-ONLY — No Analytics at All (Grounding Rate: 0.0%)

> *"The data does not cover this."*

Both OpenAI GPT-4o and Gemini 1.5 Flash produce this response when given no analytics context. Neither model fabricates statistics; both produce an explicit refusal rather than hallucinated content.

---

## A.3 QA Benchmark Examples

The QA benchmark auto-generates 44–55 factual questions per clip from the analytics JSON. Questions span eight categories. Below are real examples from Clip 2 (Team 1: 75.7% possession, full tactical data), with OpenAI and Gemini responses.

| Category | Question | Expected Answer | OpenAI GPT-4o | Gemini 1.5 Flash |
|---|---|---|---|---|
| numeric | What percentage of possession did Team 1 have? | 75.7% | "Team 1 had 75.7% possession." ✓ | "Team 1 had 75.7% possession." ✓ |
| numeric | What was the maximum speed of player #14? | 40.0 km/h | "This information is not available in the match analytics data." ✗ | "Player #14 reached a maximum speed of 40.0 km/h." ✓ |
| comparative | Which team was more compact? | Team 2 (592 m² vs 1,242 m²) | "Team 2 was more compact, occupying 592.5 m²." ✓ | "Team 2 was more compact, with an average compactness of 592.5 m²." ✓ |
| tactical | How compact was Team 1 when in possession? | N/A | "Team 1 had an average compactness of 1,242.5 m²..." ✗ *(hallucinated when N/A expected)* | "This information is not available in the match analytics data." ✓ |
| temporal | When was the first pass detected? | 0:04 | "The first pass detected was at approximately 0:04." ✓ | "The first pass was detected at 0:04." ✓ |
| spatial | Which team had a higher defensive line? | Team 2 (62.7 m vs 56.8 m) | "Team 2 had a higher defensive line at 62.7 m." ✓ | "Team 2 had a higher defensive line at 62.7 m." ✓ |
| spatial | Which team was more spread out laterally? | Team 1 (width > Team 2) | "Team 1." ✗ *(incorrect direction)* | "Team 1 was more spread out, with average width of..." ✓ |
| entity | Which player covered the most distance? | Player #14 (Team 0) — 168 m | "Player #14 (Team 0) covered the most distance at 168 m." ✓ | "Player #14 (Team 0) covered the most distance at 168 m." ✓ |
| multi_hop | Did the team with more possession also have more shots? | Yes — Team 1 had both more possession and more shots | "This information is not available..." ✗ | "This information is not available..." ✗ |
| unanswerable | What are the real names of the players? | N/A (tracking IDs only) | "This information is not available in the match analytics data." ✓ | "This information is not available in the match analytics data." ✓ |

**Key patterns:**
- Both models achieve near-perfect abstention on unanswerable questions
- OpenAI fails on tactical questions expecting N/A — it answers with the nearest available numeric value instead of refusing
- Gemini's spatial accuracy (OpenAI 33% vs Gemini 100% on Clip 2 spatial questions) stems from Gemini correctly comparing directional values ("Team 2 is *more* compact because 592 < 1,242") while OpenAI sometimes retrieves only one value without comparison
- Multi-hop questions (requiring two-step reasoning across analytics fields) fail for both providers

---

## A.4 G-Eval Quality Scores by Condition

The following scores are from the Claude Haiku judge (preferred over OpenAI judge due to self-preference bias). Each condition scored on five dimensions (1–5 Likert). See Figure 2 (`judge/claude_geval_radar.pdf`).

| Condition | Coherence | Consistency | Fluency | Relevance | Groundedness | **Mean** |
|---|---|---|---|---|---|---|
| A — Full data + insights | 3.7 | 3.2 | 4.0 | 3.5 | 3.3 | **3.00** |
| H — Full data, no insights | 4.3 | 3.8 | 4.5 | 4.0 | 3.6 | **3.47** |
| F — No analytics | 2.1 | 1.5 | 3.0 | 2.0 | 0.5 | **1.77** |
| I — No few-shot examples | 3.5 | 3.2 | 4.1 | 3.4 | 3.3 | **3.17** |
| J — No metric definitions | 3.8 | 3.5 | 4.4 | 3.8 | 3.5 | **3.50** |

*Condition H (no insights) scores highest overall at 3.47 — higher than condition A (3.00) despite lower grounding rate. The Claude judge rewards independent reasoning over paraphrasing pre-interpreted insights. Condition F collapses to 1.77 on groundedness (0.5/5) — the judge correctly penalises factually fabricated outputs even when they read fluently.*

*Note: the OpenAI judge scores condition F at 3.87/5 — only 0.11 points below condition A — exhibiting the self-preference bias documented by Zheng et al. (2023).*

---

## A.5 DB-Grounded Claim Resolution Example

**Claim**: *"Team 1 maintained a deep defensive block throughout the passage of play."*

| Layer | Verdict | Evidence |
|---|---|---|
| JSON-only | UNVERIFIABLE | Defensive line metric = 20.9 m (consistent with deep block, but aggregate scalar cannot confirm "throughout") |
| DB-grounded | **VERIFIED** | Team 1 players spend 32.8% of player-frames in the defensive third, 65.0% in the middle third, and 2.2% in the attacking third across all 750 frames. Mean Team 1 pitchX = 22.3 m. |

**Claim**: *"Team 2 built play through the left channel."*

| Layer | Verdict | Evidence |
|---|---|---|
| JSON-only | UNVERIFIABLE | No left/right spatial breakdown in aggregate JSON |
| DB-grounded | UNVERIFIABLE | Per-frame ball trajectory shows left-flank bias, but keyword-matching heuristic does not resolve "left channel" to a coordinate range without NER |

*This illustrates both the value and the limitation of the DB layer: the first claim resolves because defensive-third zone occupancy is a direct coordinate computation; the second remains unverifiable because the heuristic cannot map natural-language directional references to coordinate ranges.*
