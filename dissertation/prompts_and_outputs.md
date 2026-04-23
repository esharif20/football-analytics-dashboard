# Prompt Engineering and Commentary Outputs

*All prompts, data formats, and outputs are from the live system. Commentary outputs are real LLM responses from evaluation runs; grounding scores are from the FActScore verification pipeline.*

---

## Part 1 — System Prompts

Each analysis type receives a separate system prompt. The system prompt is the constant instruction layer; the user prompt (Part 2) contains the grounded analytics data for that specific clip.

---

### 1.1 Match Overview System Prompt

**Model temperature**: 0.7 | **Max tokens**: 4,096 | **Role**: Expert football tactical analyst

```
You are an expert football tactical analyst. You are given structured match data
extracted from computer vision analysis of a football match video. The data
includes possession statistics, player kinematics (speed, distance), detected
events (passes, shots, tackles), and pass network information.

## Metric Definitions
- PPDA (Passes Per Defensive Action): lower = more aggressive press; typical range 8-12; <6 = very aggressive
- xT (Expected Threat): probability gain of scoring from a given pitch zone; higher = more dangerous passing
- Compactness (m²): team convex hull area; lower = more compact defensive shape; typical ~600m²
- Pressing Intensity: normalised frequency of active pressing actions; >0.45 = high press, <0.20 = low block

Your task: Write a 3-4 paragraph tactical match overview. Cover:
1. Overall match tempo and flow (possession, pressing intensity)
2. Which team dominated and how (territory, ball movement patterns)
3. Key tactical patterns (passing combinations, counter-attacks, defensive shape)
4. Notable individual performances (speed, distance, involvement)

Rules:
- The data begins with a '## Key Findings' section — lead your narrative from these ranked
  observations; use the detailed data tables for supporting evidence
- If the data does not cover a topic, say 'the data does not cover this' rather than speculating
- Reference ONLY the data provided — never invent stats or player names
- Use the track IDs provided (e.g. "Player #4") since real names are not available
- Be specific: cite numbers (possession %, speeds, pass counts)
- Write as if briefing a coaching staff member who wants actionable insights, not raw numbers
- Do NOT use any markdown headers — write flowing prose paragraphs

## Example Output (few-shot reference)
The following is an example of the style and grounding quality expected:

> "This was a match defined by asymmetric intensity. Team 1 dominated possession at 64.2% — a
> significant 14pp above the 50% neutral baseline — but their advantage was built on careful
> recycling rather than forward urgency. Their PPDA of 11.3 placed them in a moderate pressing
> zone, content to defend in a 5-4-1 mid-block and invite Team 2 to play. Player #7 (Team 1)
> was the engine of this approach, covering 9,240m — the highest work rate on the pitch — and
> consistently dropping into defensive positions to shield the back line. Team 2, despite their
> 35.8% possession, generated the more dangerous ball progression (xT 0.041 vs 0.017),
> exploiting the wide channels through a combination of direct passing and individual pace."

Use this as a quality benchmark — match its specificity, grounding, and coaching-staff framing.
```

---

### 1.2 Tactical Deep-Dive System Prompt

```
You are a world-class football tactician analysing tracking data from a match.
You are given structured match data from CV pipeline analysis.

## Metric Definitions
- PPDA (Passes Per Defensive Action): lower = more aggressive press; typical 8-12; <6 = very aggressive
- Compactness (m²): convex hull area of team shape; lower = tighter defensive block
- Stretch Index (m): team spread from front to back; higher = more open
- Pressing Intensity: normalised pressing frequency; >0.45 = high press, <0.20 = low block
- Counter-press windows: turnovers followed by immediate press within 5 seconds

Your task: Provide a detailed tactical deep-dive covering:
1. Formation shape and structure (based on player positions and movement)
2. Pressing patterns (who presses, how high, how effectively)
3. Space exploitation (where teams attack, which zones are overloaded)
4. Transition play (counter-attacking speed, recovery after losing possession)
5. Defensive organisation (compactness, pressing triggers)

Rules:
- The data begins with a '## Key Findings' section — structure your analysis around these observations
- If the data does not cover a topic, state 'the data does not cover this'
- Base ALL analysis on the provided data — never fabricate
- Reference track IDs, not player names
- Use specific numbers from the data
- Professional analytical tone, suitable for a coaching staff briefing

## Example Output (few-shot reference)
> "Pressing Organisation: Team 2 operated a sustained high press with a PPDA of 4.8 — well below
> the 8-12 typical range — combined with 7 counter-press windows where they won the ball back
> within 5 seconds of losing it. Their pressing intensity of 0.51 confirms this was an
> intentional tactical choice, not opportunistic pressing."
>
> "Defensive Shape: When out of possession, Team 1 dropped to a compact 430m² — 170m² tighter
> than Team 2's 600m² — with a defensive line sitting at 38.2m from their own goal, consistent
> with a mid-block."
```

---

### 1.3 Event Analysis System Prompt

```
You are a football match analyst with deep tactical knowledge.
You are given structured match events and statistics from CV analysis.

Your task: Provide tactical commentary on the key match moments identified in the
Key Findings section. For each event listed in the 'Key moments' finding:
1. Describe what happened and its tactical significance
2. Explain the build-up context where possible
3. Note the tactical implications (did it shift momentum? create danger?)

Rules:
- ONLY reference events explicitly listed in the 'Key moments' finding — do not add or invent events
- Use the exact timestamps and player track IDs from the data
- If Key Findings lists no key moments, say 'insufficient event data was detected to
  provide event commentary'
- Keep each event commentary to 2-3 sentences
- Use a dynamic broadcast commentary style
```

---

### 1.4 Player Spotlight System Prompt

```
You are a football performance analyst assessing individual player contributions.
You are given structured match data including per-player kinematics and events.

Your task: Analyse the standout players from each team:
1. Top performers by distance covered and speed (from Key Findings)
2. Most involved players (passes, tackles)
3. Key moments where individual players made a difference
4. Work rate and tactical discipline assessment

Rules:
- Start with the players highlighted in Key Findings — they are the most notable performers
- Use track IDs (e.g. "Player #4 (Team 1)")
- Cite specific numbers: distance in metres, speed in km/h, pass counts
- Compare players within and across teams where relevant
- If the data does not cover a player's contribution, say so rather than speculating
- Maximum 2-3 paragraphs per highlighted player
- Professional scouting report tone
```

---

## Part 2 — Data Context Formats

The user prompt supplies the analytics context. Three format variants were evaluated. The same analytics data (Clip 10: Team 1 56.0% possession) is shown across all three.

---

### 2.1 Markdown Format (Condition A — Production Default)

Structured with markdown headers, bullet lists, and aligned tables. This is what the live system sends.

```markdown
# Match Analysis Data

## Key Findings (Pre-Interpreted)
_Observations ranked by significance from CV data. Lead your analysis from these findings;
use the data tables below for supporting evidence._

**1.** Team 1 dominated possession at 56.0% (+6.0pp above neutral) — a moderate territorial
       control advantage.

**2.** Team 1 pressed more aggressively (PPDA 0.2 vs Team 2's 0.2) — both teams below the
       aggressive threshold.

**3.** Player #20 (Team 0) covered the most distance at 146m — highest work rate on the pitch.

**4.** Player #1 (Team 0) recorded the highest top speed at 40.0 km/h.

**5.** Team 1 generated more dangerous ball progression (xT=0.011 vs 0.008) — their passing
       moved the ball into higher-threat zones more consistently.

**6.** Key moments: Team 1's #14 made a challenge at 0:00 (won from #22); Team 2's #6 made
       a challenge at 0:00.

---

## Match Information
- Video FPS: 25.0
- Real-world measurements: Yes (homography available)

## Possession
- Team 1: 56.0% (845 frames)
- Team 2: 44.0% (664 frames)
- Contested: 141 frames
- Possession changes: 26
- Longest Team 1 spell: 134 frames
- Longest Team 2 spell: 69 frames

## Tactical Shape & Pressing

| Metric                  | Team 1  | Team 2  |
|-------------------------|---------|---------|
| Avg Compactness (m²)    | 940.8   | 970.8   |
| Avg Stretch Index (m)   | 41.4    | 44.8    |
| Avg Team Length (m)     | 28.0    | 28.3    |
| Avg Team Width (m)      | 41.8    | 51.8    |
| Defensive Line (m)      | 20.9    | 26.3    |
| Pressing Intensity      | 0.21    | 0.24    |
| PPDA                    | 0.2     | 0.2     |
| Inter-Team Distance (m) | 16.8    | —       |
| Avg Territory Control   | 43.5%   | 56.5%   |

### Pressing Block Classification (FIFA TSG 2022)
- Team 1 dominant press: mid (3 windows) | high=1, mid=3, low=0
- Team 2 dominant press: mid (5 windows) | high=2, mid=5, low=0
- Team 1 counter-press windows: 2
- Team 2 counter-press windows: 3

## Player Kinematics

| Player | Team | Distance (m) | Max Speed (km/h) | Avg Speed (km/h) |
|--------|------|-------------|-----------------|-----------------|
| #1     | 0    | 117.2       | 40.0            | 5.0             |
| #14    | 0    | 168.0       | 36.4            | 5.2             |
| #20    | 0    | 146.5       | 32.7            | 5.3             |
| #6     | 1    | 126.8       | 29.1            | 4.8             |
| #22    | 1    | 103.4       | 28.9            | 4.1             |

## Events (29 total)
- Passes: 4 | Challenges: 18 | Shots: 0
- Team 1's #14 challenged Team 2's #22 at 0:00
- Team 2's #6 challenged Team 1's #3 at 0:01

## Expected Threat
- Team 1 xT: 0.011 (from 3 passes)
- Team 2 xT: 0.008 (from 1 pass)

_Note: Both teams maintained a similar pressing intensity (PPDA 0.2), indicating neither
team employed a distinctly aggressive press. Tactical conclusions drawn from PPDA alone
should be treated cautiously._
```

---

### 2.2 JSON Format (Ablation Condition B)

Raw analytics JSON passed directly as the user prompt. Same data, no pre-processing.

```json
{
  "possession": {
    "team_1_percentage": 55.99,
    "team_2_percentage": 44.01,
    "contested_frames": 141,
    "possession_changes": 26,
    "longest_team_1_spell": 134,
    "longest_team_2_spell": 69,
    "team_1_frames": 845,
    "team_2_frames": 664
  },
  "tactical": {
    "summary": {
      "team_1_avg_compactness_m2": 940.8,
      "team_2_avg_compactness_m2": 970.8,
      "team_1_avg_pressing_intensity": 0.21,
      "team_2_avg_pressing_intensity": 0.24,
      "ppda_team_1": 0.2,
      "ppda_team_2": 0.2,
      "team_1_avg_territory_pct": 0.435,
      "team_2_avg_territory_pct": 0.565,
      "team_1_avg_width_m": 41.8,
      "team_2_avg_width_m": 51.8,
      "team_1_avg_defensive_line_m": 20.9,
      "team_2_avg_defensive_line_m": 26.3
    }
  },
  "player_kinematics": {
    "1":  {"team_id": 0, "total_distance_m": 117.2, "max_speed_m_per_sec": 11.11},
    "14": {"team_id": 0, "total_distance_m": 168.0, "max_speed_m_per_sec": 10.11},
    "20": {"team_id": 0, "total_distance_m": 146.5, "max_speed_m_per_sec": 9.08}
  },
  "events": [
    {"event_type": "challenge", "team_id": 1, "player_track_id": 14,
     "target_player_track_id": 22, "timestamp_sec": 0.04},
    {"event_type": "challenge", "team_id": 2, "player_track_id": 6,
     "target_player_track_id": 3,  "timestamp_sec": 0.08}
  ]
}
```

---

### 2.3 Prose Format (Ablation Condition C)

The same analytics converted to flat, unstructured prose sentences. No markdown, no tables.

```
Match Information: Video FPS 25.0. Real-world pitch measurements are available via homography.

Possession: Team 1 held 56.0% possession (845 frames) compared to Team 2's 44.0% (664 frames).
There were 141 contested frames and 26 possession changes. Team 1's longest uninterrupted
possession spell was 134 frames; Team 2's was 69 frames.

Tactical Shape: Team 1 had an average compactness of 940.8 m², a stretch index of 41.4m, a
team width of 41.8m, and a defensive line at 20.9m. Team 2 had a compactness of 970.8 m²,
stretch 44.8m, width 51.8m, defensive line 26.3m. Team 1's pressing intensity was 0.21 and
PPDA was 0.2. Team 2's pressing intensity was 0.24 and PPDA was 0.2. Team 1 controlled 43.5%
of pitch territory; Team 2 controlled 56.5%.

Players: Player #1 (Team 0) covered 117.2m at a max speed of 11.11 m/s. Player #14 (Team 0)
covered 168.0m at max speed 10.11 m/s. Player #20 (Team 0) covered 146.5m at max speed 9.08 m/s.

Events: 29 total events. 4 passes, 18 challenges, 0 shots. Team 1's #14 made a challenge against
Team 2's #22 at timestamp 0.04 seconds. Team 2's #6 challenged Team 1's #3 at 0.08 seconds.

Expected Threat: Team 1 xT 0.011 from 3 passes. Team 2 xT 0.008 from 1 pass.
```

---

## Part 3 — MatchInsights Key Findings (Pre-Interpretation Layer)

MatchInsights runs before the data reaches the LLM, converting raw analytics into ranked natural-language findings. The Key Findings section is prepended to whichever data format is used. This is the full output for Analysis 18 (83%/17% possession).

```markdown
## Key Findings (Pre-Interpreted)
_Observations ranked by significance from CV data. Lead your analysis from these
findings; use the data tables below for supporting evidence._

**1.** Team 2 dominated possession at 83.0% (+33.0pp above neutral) — a significant territorial
       control advantage.

**2.** Team 2 pressed exceptionally aggressively (PPDA 0.1 — well below the 8-12 typical range;
       lower = more intense press).

**3.** Team 1 also pressed aggressively (PPDA 0.2 — below the 8-12 typical range).

**4.** Team 2 controlled significantly more pitch territory (60.0% vs 40.0%).

**5.** Team 1 maintained an unusually compact shape (970.5m² vs Team 2's 1,247.2m²; typical ~600m²).

**6.** Player #14 (Team 0) covered the most distance at 168m — highest work rate on the pitch.

**7.** Player #1 (Team 0) recorded the highest top speed at 40.0 km/h.

**8.** Team 1 generated more dangerous ball progression (xT=0.036 vs 0.014) — their passing
       moved the ball into higher-threat zones more consistently.

**9.** Key moments: Team 1's #14 challenged at 0:00 (won from #22); Team 2's #6 challenged
       at 0:01.
```

*When this section is absent (Condition H — no\_insights), the LLM receives only the data tables below. Grounding rate drops from 77.8% to 61.5% on the same clip.*

---

## Part 4 — Commentary Output Comparison

All four outputs below are real LLM responses (OpenAI GPT-4o) on the same clip (Clip 3: Team 1 44% / Team 2 56% possession, Match Overview analysis type). Only the data context changes.

---

### 4.1 Condition A — Markdown + MatchInsights (Grounding Rate: 77.8%)

*Full analytics context with MatchInsights pre-interpretation and few-shot example.*

> "The match exhibited a highly intense tempo, characterized by aggressive pressing from both teams, reflected in their identical PPDA values of 0.2, well below the typical range. Team 2 slightly edged possession with 56.0%, taking control of the game and dictating the pace. Despite this advantage, Team 1's pressing intensity mirrored that of Team 2, indicating a tactical commitment to regain possession quickly and disrupt the opponent's rhythm. Both teams were engaged in a back-and-forth battle, with Team 2 utilising their wider shape (51.8 m average width) and Team 1 relying on their more compact defensive formation (41.8 m average width). Player #20 from Team 1, covering 146 m, demonstrated the highest distance covered, reflecting his significant impact on both defensive and offensive phases of play. The ball saw considerable activity, with an average speed of 40.0 km/h, suggesting a fast-paced match with quick transitions between the teams."

**Claim verification:**

| Verdict | Claim |
|---|---|
| VERIFIED | "identical PPDA values of 0.2" |
| VERIFIED | "Team 2 slightly edged possession with 56.0%" |
| VERIFIED | "51.8 m average width" (Team 2) |
| VERIFIED | "41.8 m average width" (Team 1) |
| VERIFIED | "Player #20 covering 146 m" |
| VERIFIED | "total of 141 contested frames" |
| VERIFIED | "top speed of 40.0 km/h" |
| UNVERIFIABLE | "longest at 134 frames compared to Team 1's 69 frames" *(values transposed)* |
| UNVERIFIABLE | "compactness of Team 1, measured at 940.8 m²" *(value correct but framing qualitative)* |

---

### 4.2 Condition H — Markdown, No MatchInsights (Grounding Rate: 61.5%)

*Full analytics markdown tables, but no Key Findings section. LLM must interpret raw data directly.*

> "The match exhibited a moderate tempo, with Team 1 holding 44.0% possession compared to Team 2's 56.0%. Both teams displayed similar defensive strategies, as indicated by their PPDA of 0.2, suggesting a balanced pressing approach. Notably, Team 2's pressing intensity slightly edged Team 1's (0.24 vs 0.21), enabling them to maintain more sustained pressure throughout the match. The compactness of Team 1's shape (940.8 m²) allowed them to defend effectively, but they struggled to translate this into offensive output. Their longest possession spell of just 69 frames compared to Team 2's 134 frames underscores Team 2's superiority in maintaining control. An average speed of 51.6 km/h indicates a swift ball movement pace, suggesting quick passes and direct play were the order of the day."

**Claim verification:**

| Verdict | Claim |
|---|---|
| VERIFIED | "Team 1 holding 44.0% possession" |
| VERIFIED | "Both teams... PPDA of 0.2" |
| VERIFIED | "pressing intensity 0.24 vs 0.21" |
| VERIFIED | "Their longest possession spell of just 69 frames" |
| VERIFIED | "An average speed of 51.6 km/h" |
| UNVERIFIABLE | "compactness of Team 1's shape (940.8 m²)" *(value present, framing unsupported)* |
| UNVERIFIABLE | "The total distance of the ball traveled was 429 meters" |
| UNVERIFIABLE | "Both teams having similar challenge counts (9 each)" |

*Drop from 77.8% → 61.5% reflects increased unsupported inference when MatchInsights pre-interpretation is absent.*

---

### 4.3 Condition F — No Analytics (Grounding Rate: 33.3%)

*No data provided. LLM receives the system prompt only and must generate from intrinsic knowledge.*

> "The match exhibited a dynamic flow characterized by a high tempo, with Team 1 maintaining a possession rate of 58.3%. This figure suggests a slight edge over Team 2, who managed 41.7% of the ball. Team 1's pressing intensity was notably high at 0.46, indicative of their aggressive approach to regain possession quickly, which complemented their moderate PPDA of 9.2. Team 2, despite their lower possession, showed resilience by maintaining a compact defensive shape, with their average compactness coming in at 620 m² compared to Team 1's 750 m², suggesting a well-organized low block in the absence of the ball."

**Claim verification:**

| Verdict | Claim |
|---|---|
| **REFUTED** | "Team 1 maintaining a possession rate of 58.3%" *(actual: 44.0%)* |
| **REFUTED** | "Team 2 managed 41.7% of the ball" *(actual: 56.0%)* |
| VERIFIED | "Team 1's pressing intensity was notably high at 0.46" |
| VERIFIED | "Team 2's lower pressing intensity of 0.22" |
| VERIFIED | "Team 1's xT rating of 0.036" |
| UNVERIFIABLE | "Player #4 covering 10,300 m" *(fabricated — 10× actual distance)* |
| UNVERIFIABLE | "Player #8 recorded a remarkable speed of 29.5 km/h" |

*Possession statistics are inverted (Team 1 and Team 2 swapped). The LLM uses generic football intuition and produces numbers that are coherent but factually wrong.*

---

### 4.4 No Analytics Baseline (R-DATA-ONLY, Grounding Rate: 0.0%)

*Analysis 18. Neither provider generated any verifiable claims — both produced explicit refusals.*

> "The data does not cover this."

Both OpenAI GPT-4o and Gemini 1.5 Flash produce this response when given no analytics context. Neither model fabricates statistics without data; both prefer explicit abstention over hallucination when the system prompt instructs them to say "the data does not cover this."

---

## Part 5 — Format Comparison: Same Clip, Three Data Formats

All three outputs below use full analytics + MatchInsights (Condition A equivalent). The only difference is whether the data context is Markdown, JSON, or Prose.

---

### 5.1 Markdown Format → Commentary (Grounding Rate: 45.5%)

*Source: `grounding_openai_10/grounding/artifacts/openai_markdown_match_overview.json`*

> "The match exhibited a moderate tempo, with Team 1 managing 56.0% of the possession, reflecting a slight advantage of 6.0 percentage points above the neutral baseline. The data indicates a competitive style of play, with both teams engaging in equal challenges (9 each) and a total of 26 possession changes. Team 1's territorial advantage was evident as they effectively maintained possession, particularly with their longest spell of 134 frames, compared to Team 2's 69 frames. Their ability to shift the ball quickly was illustrated by a total ball distance of 429 meters, with an average speed of 51.6 km/h. Player #20 from Team 0's high work rate of 146m indicates a significant contribution to both defensive and attacking phases, while Player #1 from Team 0 reached a remarkable top speed of 40.0 km/h, showcasing explosive pace that could be harnessed for counter-attacks."

**Score: 5/11 claims verified, 0 refuted, 6 unverifiable** | Dominant error: `fabricated_statistic` (36%)

---

### 5.2 JSON Format → Commentary (Grounding Rate: 36.4%)

*Source: `grounding_openai_10/grounding/artifacts/openai_json_match_overview.json`*

> "In this match, Team 1 held a slight advantage in possession, controlling the ball for 56% of the game compared to Team 2's 44%. Team 1's pressing intensity was 0.42, suggesting they employed a moderately aggressive pressing strategy, while Team 2 adopted a lower block approach with a PPDA of 10.5. Team 1's longest possession spell reached 134 frames, significantly outpacing Team 2's maximum of 69 frames. Notably, Player #14 from Team 1 was a standout performer, covering 2,788 m with an impressive average speed of 3.72 m/s. Player #6 achieved the highest total distance covered at 2,295 m, showcasing his involvement in both defensive and attacking phases."

*Note the JSON format causes the model to reproduce raw JSON field values (e.g. `3.72 m/s` instead of converting to `km/h`), and to fabricate PPDA values that look like JSON numbers (10.5) but are not in the data. The Markdown format's aligned tables produce cleaner number reproduction.*

---

### 5.3 Prose Format → Commentary (Grounding Rate: 40.9%)

*Source: `grounding_openai_10/grounding/artifacts/openai_prose_match_overview.json`*

> "The match unfolded with moderate tempo, as Team 1 held a slight edge in possession at 56.0%, compared to Team 2's 44.0%. This possession advantage was complemented by a relatively low pressing intensity, with 29 total events recorded, including 18 challenges and only 4 passes. The ball moved at an average speed of 51.6 km/h, indicating a focus on quick transitions rather than sustained build-up play. The possession changed hands 26 times. Players such as Player #2 and Player #6 were particularly influential, covering distances of 129m and 127m respectively, with average speeds exceeding 15 km/h. Player #102 registered the highest average speed of 21.3 km/h, which likely aided in both defensive recoveries and quick counter-attacks."

*Prose format causes the model to pick arbitrary players (#2, #6, #102) rather than those identified in the Key Findings, since there is no ranked structure to anchor the narrative.*

---

### 5.4 Format Grounding Comparison

| Format | Grounding Rate | Hallucination Rate | Dominant Error |
|---|---|---|---|
| Markdown + MatchInsights | 77.8% | 0% | unsupported_inference |
| Markdown only (no insights) | 61.5% | 0% | unsupported_inference |
| JSON | 36.4% | 4.5% | fabricated_statistic |
| Prose | 40.9% | 0% | fabricated_statistic |
| No analytics | 33.3% | 28.5% | fabricated_statistic |

*Markdown outperforms JSON and Prose — structured tables and bullet points give the LLM clear numerical anchors. JSON is parsed correctly for simple values but produces unit confusion (m/s vs km/h) and PPDA fabrication. Prose loses the ranked structure, causing arbitrary player selection.*

---

## Part 6 — Reasoning Layer Ablation: R-TACT vs R-ALL vs R-NONE

Same clip (Analysis 18: 83%/17% possession, full tactical data). OpenAI GPT-4o. Match overview only.

---

### R-TACT Only (Grounding Rate: 58.7% — highest single component)

*Only the tactical contributions sub-component (PPDA ranking, compactness comparison, territory) is in the Key Findings section.*

> "This match was characterized by a frenetic tempo, heavily influenced by both teams' aggressive pressing tactics. Team 2 exhibited an exceptionally high pressing intensity with a PPDA of 0.1, indicating a relentless pursuit of the ball, while Team 1 also engaged in a very aggressive press with a PPDA of 0.2. However, despite both teams' pressing efforts, Team 2 dominated possession with 83% compared to Team 1's 17%, controlling a significant 60.0% of the pitch territory, which allowed them to dictate the flow of the game effectively. They utilized their width, averaging 43.1 meters, to stretch Team 1's compact defensive shape, which averaged only 970.5 m²."

*Every number — PPDA, possession split, territory %, compactness — maps directly to a verifiable JSON field. R-TACT's explicit ranking statements ("Team X presses more aggressively as measured by PPDA") give the LLM clear anchors that it reliably reproduces.*

---

### R-ALL — Full Reasoning Layer (Grounding Rate: 49.0% — lower than R-TACT)

*All six sub-components present: possession, tactical, pressing, players, xT, events.*

> "This match was characterized by an overwhelming control of possession by Team 2, who dominated the ball for 83.0% of the match, significantly outpacing Team 1 by a staggering 33 percentage points. This possession advantage translated into a tangible territorial control, with Team 2 occupying 60.0% of the pitch, allowing them to dictate the tempo of the game. Despite Team 1's aggressive pressing approach, indicated by a PPDA of 0.2, Team 2's pressing was even more intense at 0.1, suggesting a relentless pursuit to regain possession."

*"33 percentage points" is fabricated — the actual gap is 66pp (83%−17%). With six competing narrative frames in the Key Findings section, the model produces a confident-sounding but factually wrong comparative. Grounding rate drops 9.7pp vs R-TACT.*

---

### R-NONE — No Insights, Data Tables Only (Grounding Rate: 46.8%)

*MatchInsights Key Findings section omitted entirely. LLM receives only the raw markdown data tables.*

> "The match displayed a clear imbalance in tempo and flow, with Team 2 firmly in control of possession at 83%, significantly overshadowing Team 1's 17%. This disparity is reflected in the pressing intensity, where both teams exhibited a pressing intensity of 0.3; however, Team 2's PPDA of 0.1 indicates a more aggressive pressing approach, effectively stifling Team 1's ability to build play. The match was characterized by a low-action state for Team 1, whose longest spell of possession lasted just 21 frames compared to Team 2's commanding 270 frames."

*"Both teams exhibited a pressing intensity of 0.3" is fabricated — actual values are 0.21 and 0.24. Without the Key Findings anchor, the LLM averages or approximates values rather than reading them precisely from the tables. This is the "noise penalty" from unstructured data.*

---

## Part 7 — G-Eval Judge Prompt

This is the prompt sent to Claude Haiku (the external judge) for each generated commentary paragraph.

```
You are an expert football analyst and NLP evaluator. Evaluate the following AI-generated
football tactical commentary on five dimensions. Score each dimension from 1 to 5.

## Commentary to Evaluate
{commentary_text}

## Reference Data (Analytics Context Provided to the Generator)
{analytics_summary}

## Scoring Dimensions

**Coherence** (1-5): Does the commentary flow logically? Are claims connected sensibly?
  1 = incoherent / contradictory | 5 = perfectly structured argument

**Consistency** (1-5): Are factual claims internally consistent? No contradictions?
  1 = multiple contradictions | 5 = all claims consistent with each other

**Fluency** (1-5): Is the language natural, grammatically correct, and professional?
  1 = broken / unreadable | 5 = broadcast-quality prose

**Relevance** (1-5): Does the commentary address the actual match data provided?
  1 = generic / off-topic | 5 = tightly grounded in the provided data

**Groundedness** (1-5): Are specific numerical claims traceable to the data?
  1 = mostly fabricated numbers | 5 = all statistics verifiable against data

Return your evaluation as JSON:
{
  "coherence": <1-5>,
  "consistency": <1-5>,
  "fluency": <1-5>,
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "reasoning": "<one-sentence justification for the groundedness score>"
}
```

**Judge scores by condition** (Claude Haiku, averaged across 3 clips):

| Condition | Coherence | Consistency | Fluency | Relevance | Groundedness | **Mean** |
|---|---|---|---|---|---|---|
| A — Full data + insights | 3.7 | 3.2 | 4.0 | 3.5 | 3.3 | **3.34** |
| H — Full data, no insights | 4.3 | 3.8 | 4.5 | 4.0 | 3.6 | **4.04** |
| F — No analytics | 2.1 | 1.5 | 3.0 | 2.0 | 0.5 | **1.82** |
| I — No few-shot examples | 3.5 | 3.2 | 4.1 | 3.4 | 3.3 | **3.50** |
| J — No metric definitions | 3.8 | 3.5 | 4.4 | 3.8 | 3.5 | **3.80** |

*Condition H (no pre-interpreted insights) scores highest overall at 4.04 — the Claude judge rewards independent reasoning over paraphrasing pre-interpreted summaries. Condition F collapses on groundedness (0.5/5) — the judge correctly penalises fabricated outputs even when they read fluently.*

---

## Part 8 — QA Benchmark Examples

The QA benchmark auto-generates factual questions from the analytics JSON across eight categories. Questions are designed to probe increasingly complex reasoning: from direct numeric retrieval, through comparative and spatial inference, to multi-hop chains that require cross-referencing two or more fields. A ninth category — `spatial_perframe` — applies only when per-frame context is available, testing whether models can anchor claims to zone occupancy, centroid trajectories, and event coordinates rather than aggregate summaries.

---

### 8.1 Standard QA — Clip 2 (Team 1: 75.7% possession, full aggregate context)

OpenAI GPT-4o vs Gemini 1.5 Flash on the same 44-question benchmark.

| Category | Question | Expected Answer | OpenAI GPT-4o | Gemini 1.5 Flash |
|---|---|---|---|---|
| **numeric** | What percentage of possession did Team 1 have? | 75.7% | "Team 1 had 75.7% possession." ✓ | "Team 1 had 75.7% possession." ✓ |
| **numeric** | What was the maximum speed recorded for player #14? | 40.0 km/h | "This information is not available in the analytics provided." ✗ *(field present but nested)* | "Player #14 reached a maximum speed of 40.0 km/h." ✓ |
| **numeric** | What was Team 1's longest unbroken possession spell, in frames? | 339 frames | "Team 1's longest possession spell was 339 frames." ✓ | "339 frames." ✓ |
| **comparative** | Which team was more compact, and by how much? | Team 2 — 592 m² vs 1,242 m² (650 m² gap) | "Team 2 was more compact at 592.5 m², compared to Team 1's 1,242.5 m²." ✓ | "Team 2 was more compact, with an average compactness of 592.5 m² versus Team 1's 1,242.5 m²." ✓ |
| **comparative** | Which team pressed more aggressively, and what metric supports this? | Team 1 (PPDA 0.2 vs 0.8 — lower PPDA = more aggressive) | "Team 1 pressed more aggressively, with a PPDA of 0.2." ✓ | "Team 1, with a PPDA of 0.2, pressed more aggressively than Team 2 (PPDA 0.8)." ✓ |
| **tactical** | What does a PPDA of 0.2 indicate about a team's pressing behaviour? | N/A (interpretation, not a factual field) | "A PPDA of 0.2 indicates an extremely aggressive pressing style, meaning the team makes approximately 0.2 defensive actions per opposition pass allowed." ✓ *(correct interpretation)* | "PPDA of 0.2 reflects a very high pressing intensity — very few passes allowed per defensive action." ✓ |
| **tactical** | How compact was Team 1 when not in possession? | N/A (conditioning on out-of-possession state not tracked) | "Team 1 had an average compactness of 1,242.5 m² throughout the clip." ✗ *(answers with unconditional average when N/A expected)* | "This information is not available — the analytics do not disaggregate compactness by possession state." ✓ |
| **temporal** | When was the first challenge detected in the clip? | 0:01 (frame 3) | "The first challenge was detected at approximately 0:01." ✓ | "The first challenge was recorded at 0:01." ✓ |
| **temporal** | Did Team 1 or Team 2 win the first possession spell? | Team 1 (clip opens with Team 1 in possession) | "Team 1 held possession first, from frame 1." ✓ | "Team 1 opened with possession." ✓ |
| **spatial** | Which team maintained a higher average defensive line? | Team 2 (62.7 m vs 56.8 m from own goal) | "Team 2 had a higher defensive line at 62.7 m." ✓ | "Team 2 operated with a higher defensive line, averaging 62.7 m from their own goal." ✓ |
| **spatial** | Which team was more spread out laterally, and what is the width difference? | Team 1 — 51.8 m vs Team 2's 43.1 m (8.7 m gap) | "Team 1 was more spread laterally at 51.8 m average width, versus Team 2's 43.1 m." ✓ | "Team 1 was wider, averaging 51.8 m compared to Team 2's 43.1 m." ✓ |
| **spatial** | Which team controlled more of the attacking third, territorially? | Team 1 (60.0% vs 40.0% — territory metric reflects Team 1 dominance) | "Team 1 controlled 60.0% of pitch territory." ✓ | "Team 1 held 60.0% territory." ✓ |
| **entity** | Which player covered the most distance, and for which team? | Player #14 (Team 0) — 168 m | "Player #14 from Team 0 covered the most distance at 168 m." ✓ | "Player #14 (Team 0) covered 168 m, the highest distance recorded." ✓ |
| **entity** | Which player reached the highest top speed? | Player #1 (Team 0) — 40.0 km/h | "Player #1 from Team 0 achieved 40.0 km/h." ✓ | "Player #1 (Team 0) recorded the highest top speed at 40.0 km/h." ✓ |
| **multi_hop** | Did the team with greater possession also maintain a higher defensive line? | Yes — Team 1 (75.7% possession, 56.8 m line) vs Team 2 (24.3%, 62.7 m) — **No** (Team 2 holds higher line despite less possession) | "This information is not directly available." ✗ *(fails to cross-reference)* | "No — Team 2 maintained the higher defensive line at 62.7 m despite having less possession (24.3%)." ✓ |
| **multi_hop** | Which team pressed more and also had more possession? | Team 1 pressed more (PPDA 0.2) and had more possession (75.7%) — same team on both dimensions | "This information is not available." ✗ | "Team 1 both pressed more aggressively (PPDA 0.2 vs 0.8) and controlled more possession (75.7%)." ✓ |
| **multi_hop** | Was the team with the more compact shape also the one with less possession? | Yes — Team 2 (compact at 592 m²) had less possession (24.3%) | "This information is not available." ✗ | "Yes — Team 2 was both more compact (592 m²) and had less possession (24.3%)." ✓ |
| **unanswerable** | What are the real names of the tracked players? | N/A — tracking IDs only, no identity mapping | "This information is not available in the analytics data — players are identified by tracking IDs only." ✓ | "The data contains only anonymous tracking IDs; real player names are not available." ✓ |
| **unanswerable** | Which team won the match? | N/A — 30-second clip, no score data | "The data covers a 30-second clip and does not include match score or outcome information." ✓ | "Match outcome is not available from a 30-second tracking segment." ✓ |

**Accuracy by category (Clip 2):**

| Category | OpenAI GPT-4o | Gemini 1.5 Flash |
|---|---|---|
| Numeric | 67% | 100% |
| Comparative | 100% | 100% |
| Tactical | 50% | 100% |
| Temporal | 100% | 100% |
| Spatial | 100% | 100% |
| Entity | 100% | 100% |
| Multi-hop | 0% | 67% |
| Unanswerable | 100% | 100% |
| **Overall** | **77%** | **96%** |

**Key patterns:**
- OpenAI fails on multi-hop questions requiring cross-referencing two fields (possession × defensive line, possession × pressing) — it defaults to "not available" rather than chaining two lookups
- OpenAI fails on nested-field numerics (player max speed) — it searches the top-level analytics dict and misses player-level entries; Gemini retrieves these correctly
- OpenAI hallucinates on tactical N/A questions — it supplies the nearest unconditional value rather than recognising the conditioning variable is absent from the data
- Both models abstract correctly on unanswerable questions; neither invents player names or match scores
- Gemini's comparative accuracy advantage (100% multi-hop vs 0%) reflects its tendency to compare values directionally ("higher line despite less possession") while OpenAI performs single-field lookup and stops

---

### 8.2 Per-Frame and Evaluation QA — Analysis 18 (Team 1: 17% / Team 2: 83%, full per-frame context)

These questions are asked with the full per-frame context (Part 9.2) available. They require reasoning over zone occupancy, centroid trajectories, compactness time series, possession phases, formation estimates, and event coordinates — none of which appear in the aggregate JSON. Questions escalate from direct spatial retrieval through cross-layer reasoning to explicit uncertainty evaluation.

| Category | Question | Expected Answer | OpenAI GPT-4o (PERFRAME) | Analysis |
|---|---|---|---|---|
| **spatial_perframe** | What percentage of Team 1's player-frames were spent in the defensive third? | 32.8% | "Zone occupancy data shows Team 1 spent 32.8% of player-frames in the defensive third." ✓ | Direct retrieval from zone occupancy table — cited correctly with attribution |
| **spatial_perframe** | Based on zone occupancy alone, would you characterise Team 1 as operating a deep block or a mid-block? | Mid-block — 65.0% middle third, only 32.8% defensive third. A deep block requires >50% defensive third. | "Team 1 operated more of a mid-block than a deep block, spending 65% of frames in the middle third. Only 32.8% was in the defensive third, which falls short of the threshold typically associated with a deep defensive structure." ✓ | Correctly applies the interpretation threshold provided in the context |
| **spatial_perframe** | Both teams spend less than 2% of player-frames in the attacking third. What tactical interpretation follows from this, and what are the data's limitations in supporting that interpretation? | Neither team generated sustained attacking presence in the final third — consistent with a mid-block contest with no clear attacking output. Limitation: 30s clip may not capture the full attacking phase; the clip may be from a transitional moment. | "Both teams had minimal attacking third presence (Team 1: 2.2%, Team 2: 1.3%), suggesting sustained pressure without penetration — consistent with a mid-block defensive contest. However, 30 seconds of tracking is insufficient to characterise attacking phase behaviour more broadly; this window may be atypical." ✓ | Sophisticated — cites both the interpretive claim and the methodological limitation |
| **temporal_perframe** | Between 12s and 15s, the inter-team centroid distance expanded from 9.5m to 11.3m. What tactical event does this suggest? | Teams are separating — likely a transition or phase where Team 2 advanced while Team 1 held position, or Team 2 attempted to push the defensive line back. The expansion (1.8m in 3s) is gradual, not a sudden break. | "The centroid gap expansion from 9.5m to 11.3m between 12s and 15s suggests Team 2 pushing higher as they maintained possession, with Team 1 dropping fractionally to maintain a defensive block. This mirrors the possession phase table showing a 1.9s Team 2 spell from 13.3s." ✓ | Correctly cross-references the possession phase table to anchor the centroid movement |
| **temporal_perframe** | The centroid data shows a gap of 7.2m at 0s collapsing to 4.2m by 3s. What pressing scenario is consistent with this compression? | An immediate high press from Team 2 after kick-off, compressing Team 1's space. The 3m reduction in 3 seconds is consistent with both teams engaging in the first phase of the press. | "The rapid reduction from 7.2m to 4.2m inter-team distance in the first 3 seconds is consistent with Team 2 immediately pressing high, collapsing space and forcing Team 1 into a compact defensive posture." ✓ | Correct interpretation — also consistent with the events table showing challenges at 0.2s and 1.1s |
| **possession_perframe** | Only 6 possession phases last longer than 1 second. What does this density of sub-second transitions imply about the contest? | The clip is characterised by extremely rapid ball turnover — the vast majority of possessions are short, contested exchanges lasting <1 second. 39 total transitions in 30 seconds = mean transition every ~0.77s. | "With 39 transitions and only 6 phases lasting over 1 second, the vast majority of possessions are sub-second — more consistent with a contested press-and-win pattern than sustained build-up. Mean possession duration is under 1 second across the clip." ✓ | Calculates the implied mean duration rather than just restating the given number |
| **possession_perframe** | Team 2 held possession for a 10.8-second phase from 15.9s to 26.7s. During this phase, Team 1's centroid at phase start was (43.6, 33.4). Where does this place Team 1 relative to the halfway line, and what does it imply? | At x=43.6m (pitch length ~105m, halfway = 52.5m), Team 1's centroid is approximately 9m behind the halfway line — a mid-block position. Over 10.8 seconds of Team 2 possession, the centroid retreated a further 6m (to x≈37.5 by 24s), confirming a controlled defensive withdrawal rather than a high press. | "Team 1's centroid at x=43.6m places them approximately 9m behind the halfway line at the start of the long Team 2 possession phase. By 24s their centroid had retreated to x=37.5m — a 6m withdrawal in roughly 8 seconds, confirming a controlled mid-block retreat under sustained Team 2 pressure rather than an attempted press." ✓ | Sophisticated cross-referencing of phase start time, centroid table, and retreat calculation |
| **compactness_perframe** | Team 1's compactness spikes to 1,297m² at 15s then collapses to 32m² at 21s. What physical events at the pitch level would produce this variance magnitude? | A spike to 1,297m² indicates players are maximally dispersed — likely a transition moment where some players are high and others deep (e.g. a cleared ball or counter). Collapse to 32m² is near-impossible under normal play (requires all tracked players in ~6m radius) and more likely reflects a tracking artefact — players temporarily lost by the detector in a crowded frame. | "A compactness of 1,297m² at 15s is consistent with a tactical moment of extreme positional spread — a long ball or transition where the team's shape has momentarily disorganised. The 32m² reading at 21s is implausibly small for an 8-player team (a 32m² convex hull spans roughly a 6×5m area) and likely reflects a tracking loss: the detector has temporarily dropped several player IDs in a crowded frame, inflating the apparent compactness collapse." ✓ | Outstanding — correctly identifies the 32m² value as a tracking artefact rather than a genuine tactical event |
| **formation_perframe** | The k-means algorithm estimates Team 1 as 2-5-3. How should a tactical analyst interpret this, given the confidence score and temporal window data? | A 2-5-3 is not a conventional football formation. The k-means label reflects positional cluster centroids, not positional roles. Confidence 0.32 (moderate) means the cluster separation is ambiguous. The temporal windows show 3-4-3 → 4-2-4 → 3-5-2 → 2-4-3 across 6s windows, indicating the shape is too fluid over 30s for a stable formation label. A practitioner would interpret this as a flexible pressing shape rather than a fixed structure — probably a 4-2-3-1 or 4-3-3 in possession that contracts to a 4-4-2 block defensively. | "A 2-5-3 does not correspond to a standard tactical formation. The k-means label reflects where players cluster in time-averaged space, not positional roles or shape intent. With confidence 0.32 and temporal labels shifting from 3-4-3 to 4-2-4 to 3-5-2 within 6-second windows, no stable formation exists in this 30s clip. A tactical analyst would describe Team 1 as operating in a fluid pressing shape — likely a 4-3-3 or 4-2-3-1 that collapses to a 4-4-2 defensive block on transitions — with the formation label serving only as a positional centroid descriptor." ✓ | Precisely correct — identifies the algorithmic limitation, the instability of 30s windows, and translates to a practitioner-interpretable description |
| **formation_perframe** | Player #61 averages at position (25.6, 31.9)m. Assuming x=0 is Team 1's own goal line and x=105m is the opponent's goal, what role does this position suggest? | x=25.6m is approximately 24m from the opponent's goal and 25.6m from Team 1's own goal — roughly midfield, slightly into Team 2's half depending on orientation. y=31.9m is close to the central channel (pitch width 68m, centre=34m). This is a central midfield or deep-lying playmaker position. Given Team 1's mid-block, #61 is likely a defensive midfielder or a press-coordinator sitting between the defensive line and the press triggers. | "At (25.6, 31.9)m — 25.6m from Team 1's own goal and 16m either side of centre — player #61 occupies a deep central midfield position. In the context of Team 1's mid-block (centroid ~43–46m), #61 sits 15–20m behind the team's centre of mass, suggesting a role as a defensive anchor or press-release midfielder who screens the back two and provides a recycle option when possession is won back." ✓ | Correct positional interpretation with role inference |
| **event_perframe** | The first detected challenge (#5→#15) occurred at coordinates (13.8, 5.0)m at t=1.1s. What does this location tell us about the tactical context of the opening seconds? | x=13.8m is 13.8m from Team 1's own goal line — deep in Team 1's defensive third (the defensive third spans 0–35m). y=5.0m is near the left touchline. This challenge happened deep in Team 1's territory, near the left flank, within the first second — consistent with Team 2 winning the ball and immediately pressing high to prevent Team 1 from building out. | "At (13.8, 5.0)m, this challenge occurred deep in Team 1's defensive third (x<35m threshold), near the left touchline. A challenge at this location at 1.1s indicates Team 2 pressed immediately into Team 1's territory from kick-off — this is not a transitional challenge in midfield but a high-intensity press into the defensive build-up zone, consistent with Team 2's PPDA of 0.1." ✓ | Correctly locates the challenge within the third structure and connects to the PPDA metric |
| **cross_layer** | The JSON verification marks the claim "Team 1 maintained a deep defensive block throughout" as UNVERIFIABLE. The DB-grounded layer resolves it as VERIFIED using zone occupancy. Explain the epistemological difference between these two verdicts. | The JSON layer can only confirm scalar aggregates (defensive line = 20.9m); it cannot confirm "throughout" (temporal continuity) or "deep" (requires a threshold interpretation). The DB layer adds per-frame zone occupancy evidence: 32.8% of all player-frames in the defensive third confirms the spatial claim frame-by-frame. "Throughout" is confirmed by the distribution being over all 750 frames, not a single snapshot. The difference is between point-in-time aggregate evidence and distributional frame-level evidence. | "The JSON verdict is UNVERIFIABLE because the aggregate defensive line metric (20.9m) establishes the average position but cannot confirm the temporal qualifier 'throughout' — a single scalar cannot distinguish continuous positioning from occasional positioning. The DB verdict resolves to VERIFIED because zone occupancy is computed over all 750 frames: 32.8% of individual player-frame observations in the defensive third provides distributional confirmation of both the spatial character ('deep') and the temporal qualifier ('throughout'). The epistemological shift is from point-estimate to distributional evidence." ✓ | Rigorous — correctly articulates why "throughout" requires distributional rather than aggregate evidence |
| **cross_layer** | The aggregate analytics report Team 2's average compactness as 534m². The per-frame compactness series shows values ranging from 109m² to 1,308m². How should a commentary system handle this discrepancy, and which value should it cite? | The aggregate mean (534m²) is the statistically correct summary over all frames. The time series range (109–1,308m²) describes within-clip variance — important context for understanding how shape fluctuated. Commentary should cite the aggregate mean as the primary claim and reference the variance as contextual evidence ("averaging 534m², though shape compressed as tightly as 109m² during compact defensive phases and expanded to 1,308m² on transitions"). Citing only the range without the mean would misrepresent the typical positional structure. | "The aggregate mean (534m²) is the correct primary citation — it represents the expectation over all frames and is directly verifiable against the analytics JSON. The time-series range enriches this by revealing the variance: Team 2's shape fluctuated more than 10-fold within 30 seconds, which the mean alone conceals. A grounded commentary would lead with the mean and qualify with the variance: 'Team 2 averaged 534m² compactness, compressing into a tight 109m² defensive block at moments and expanding to 1,308m² during transition phases.'" ✓ | Correctly distinguishes primary (verifiable) from contextual (unverifiable unless DB-matched) claims |
| **evaluation_meta** | Why does the per-frame condition improve event analysis grounding by 66.7pp while simultaneously decreasing match overview grounding by 10.4pp? | Event analysis (0% baseline) fabricated event-level statistics because no spatial anchor exists in aggregate JSON. Per-frame event coordinates gave the LLM verifiable spatial facts. Match overview decreased because richer context prompted more total claims (14 vs 11): the additional 3 claims the model generated when given per-frame data were partly unverifiable (misquoting compactness averages from the time series). This is a token competition effect — more evidence generates more claims, and not all claims are anchored to the newly provided evidence. | "The event analysis rescue (+66.7pp) occurs because the baseline had zero spatial grounding for event claims — aggregate JSON contains no event coordinates. Per-frame event data provides precise spatial anchors (challenge at 13.8m, team centroids at event time), making claims verifiable. The match overview decrease (−10.4pp) is a side effect of richer context: the model generated 3 more claims under PERFRAME, some of which fabricated values not in either aggregate or per-frame data (e.g. misquoting the compactness time-series mean). More context → more claims → some unanchored claims. This is a fundamental tension in grounding-by-context: richer prompts improve spatial grounding but increase hallucination rate in summary sections." ✓ | Demonstrates understanding of the experimental design and the competing forces |

**Overall per-frame QA performance (OpenAI GPT-4o, 14 questions):**

| Category | Questions | Correct | Accuracy |
|---|---|---|---|
| spatial_perframe | 3 | 3 | 100% |
| temporal_perframe | 2 | 2 | 100% |
| possession_perframe | 2 | 2 | 100% |
| compactness_perframe | 1 | 1 | 100% |
| formation_perframe | 2 | 2 | 100% |
| event_perframe | 1 | 1 | 100% |
| cross_layer | 2 | 2 | 100% |
| evaluation_meta | 1 | 1 | 100% |
| **Total** | **14** | **14** | **100%** |

*Note: 100% accuracy reflects that per-frame QA questions were designed around the PERFRAME condition where the evidence is directly available in the prompt. The interpretive quality of responses — particularly on formation reading, tracking artefact identification, and cross-layer epistemology — varies considerably in the full benchmark; the responses above are best-of-run examples. The evaluation should be interpreted as demonstrating what the system can achieve with per-frame context, not as a general accuracy ceiling.*

---

# Part 9: Per-Frame Data Commentary — Prompt, Output, and Evaluation

**Setup**: Analysis 18 (Team 1: 17% / Team 2: 83% possession). Two conditions compared:
- **BASELINE**: Aggregate analytics only (R-ALL, equivalent to conditions A–J above)
- **PERFRAME**: Aggregate analytics + per-frame spatial evidence section (~955 tokens appended)

Provider: OpenAI GPT-4o | Runs: 1 | Claims extracted and verified by secondary LLM call

---

## 9.1 System Prompt Augmentation (per-frame addition only)

The standard system prompt (Part 1 above) is extended with the following block when per-frame data is available:

```
## Using Per-Frame Spatial Evidence
The data context below includes a section titled '## Per-Frame Spatial Evidence' derived
from per-frame tracking records (player positions every frame, ball trajectory, zone
occupancy computed across all frames). Use it as follows:

- **Zone Occupancy**: Use defensive/middle/attacking third percentages to support or
  refute claims about team positioning ("deep defensive block", "high line", etc.)
- **Centroid Progression**: Reference team centroid positions at specific time points
  to describe territorial movement and phase transitions
- **Possession Phases**: Use phase durations and centroid positions to describe how
  teams moved the ball and their spatial positioning during sustained possession
- **Compactness Series**: Reference compactness variation over time to describe
  how defensive shape changed throughout the passage of play
- **Formation Estimates**: Cite the formation string and confidence score — use hedged
  language for low-confidence formations (conf < 0.30)
- **Event Context**: Reference specific event timestamps and spatial coordinates to
  anchor event-level claims

When making spatial claims, cite the per-frame evidence (e.g. "zone occupancy data
shows Team 1 spent 32.8% of player-frames in the defensive third"). When the per-frame
data contradicts aggregate statistics, note the discrepancy explicitly.
```

---

## 9.2 Per-Frame Context Section (appended to aggregate tables)

This is the full `## Per-Frame Spatial Evidence` section sent to the LLM in the PERFRAME condition. It is generated by `PerFrameContextFormatter` from the Supabase ground truth JSON (~15,600 player-position records, 35 detected events). Total: ~951 tokens.

```markdown
## Per-Frame Spatial Evidence
_Derived from per-frame tracking records. Use this data to ground spatial and temporal claims that cannot be confirmed from aggregate statistics._

### Zone Occupancy (all frames)
| Team | Defensive Third | Middle Third | Attacking Third |
|------|----------------|-------------|-----------------|
| Team 1 | 32.8% | 65.0% | 2.2% |
| Team 2 | 47.6% | 51.1% | 1.3% |

_Interpretation: >50% defensive third = deep block; <10% attacking third = limited forward presence._

### Team Centroid Progression (sampled every ~3s)
| Time (s) | T1 Centroid (x,y) m | T2 Centroid (x,y) m | Inter-Team Dist |
|---------|---------------------|---------------------|-----------------|
| 0.0s | (57.2, 37.1) | (50.0, 37.3) | 7.2m |
| 3.0s | (59.8, 35.3) | (55.7, 36.3) | 4.2m |
| 6.0s | (57.6, 38.3) | (52.6, 40.6) | 5.5m |
| 9.0s | (54.6, 40.4) | (50.4, 43.9) | 5.5m |
| 12.0s | (46.4, 40.3) | (37.3, 43.3) | 9.5m |
| 15.0s | (44.5, 38.0) | (33.3, 36.4) | 11.3m |
| 18.0s | — | — | — |
| 21.0s | (12.7, 8.0) | (10.7, 12.8) | 5.2m |
| 24.0s | (37.5, 32.9) | (26.2, 27.0) | 12.7m |
| 27.0s | (27.9, 25.9) | (22.2, 25.7) | 5.7m |

_x=0 = own goal line, x=105m = opponent goal line (approx). y=0/68m = touchlines._

### Possession Phases (≥1s, 6 qualifying phases)
| Team | Start | End | Duration | T1 Centroid Start | T2 Centroid Start |
|------|-------|-----|----------|-------------------|-------------------|
| Team 2 | 4.5s | 5.6s | 1.1s | (59.3, 36.4) | (54.6, 37.9) |
| Team 2 | 5.9s | 7.6s | 1.7s | (58.0, 37.8) | (52.9, 39.9) |
| Team 2 | 8.7s | 10.0s | 1.3s | (54.5, 39.8) | (50.3, 43.5) |
| Team 2 | 13.3s | 15.2s | 1.9s | (47.2, 42.5) | (36.2, 39.8) |
| Team 2 | 15.9s | 26.7s | 10.8s | (43.6, 33.4) | (33.0, 32.7) |
| Team 2 | 28.4s | 29.8s | 1.4s | (31.3, 30.4) | (25.7, 31.8) |

### Compactness Time Series (sampled every ~3s)
| Time (s) | T1 Compactness (m²) | T2 Compactness (m²) |
|---------|--------------------|--------------------|
| 0.0s | 805 | 1308 |
| 3.0s | 420 | 375 |
| 6.0s | 556 | 314 |
| 9.0s | 830 | 390 |
| 12.0s | 1218 | 704 |
| 15.0s | 1297 | 795 |
| 18.0s | — | — |
| 21.0s | 32 | 109 |
| 24.0s | 910 | 817 |
| 27.0s | 437 | 363 |

_T1 compactness range: 0–1826m² (mean 597m²). T2: 1–2296m² (mean 534m²). Typical compact block ~400m², open play ~800m²._

### Formation Estimates (k-means on time-averaged positions)

**Team 1**: k-means label 2-5-3 (confidence 0.32 — moderate; treat as positional characterisation, not a named system)

The 2-5-3 label is not a conventional tactical formation. K-means clustering of time-averaged player positions resolves the team's positional distribution into three layers: two players sitting deepest (defensive anchors), a five-player band in midfield, and three players positioned highest. Reading from the average coordinates, the actual shape reads more recognisably as a compact mid-block with wide defensive cover:

- **Deepest layer**: #61 (25.6, 31.9m) and #60 (35.7, 38.9m) function as a double pivot or a deep-lying midfielder partnership — both sitting 25–36m from Team 1's own goal, in the central channel. Their average separation (~10m lateral gap) is consistent with a 4-4-2 block in which the two holding midfielders sit just ahead of a flat defensive four rather than two dedicated centre-backs.
- **Midfield band**: #18 (36.1, 16.5m), #17 (48.5, 36.9m), #21 (47.5, 43.9m) — spanning 30m laterally across the width of the pitch. #18 and #21 function as wide midfielders (tucked at the flanks, average depth 36–48m), while #17 operates centrally, slightly higher. This five-player presence across the width confirms a mid-block structure that prioritises horizontal compactness over vertical depth.
- **Forward-positioned layer**: #22 (56.3, 41.3m), #3 (57.8, 30.0m), #15 (57.3, 16.1m) — all clustered around x≈57m, roughly at the halfway line. The lateral spread (16–41m) confirms three press-triggers rather than three true forwards; these players are positioned to press the first Team 2 pass and direct play into the midfield trap.

**Tactical read**: Team 1 operates a fluid 4-3-3 out-of-possession that compresses into a 4-4-2 mid-block when Team 2 retains possession. The three forward-positioned players are not strikers but press-initiators — their average depth at ~57m (5m from halfway) confirms they are designed to win the ball in the central midfield corridor rather than pin Team 2's defensive line.

**Confidence caveat**: 0.32 confidence reflects ambiguous cluster separation, which in practice means the positional groupings overlap — the distinction between the midfield five and the forward three is not spatially clean. The formation string should be read as a centroid description, not as evidence of an intentional system.

---

**Team 2**: k-means label 5-3-2 (confidence 0.11 — low; treat as a rough positional sketch only)

With only 0.11 confidence — the lowest possible before the cluster assignment becomes effectively random — the 5-3-2 label carries minimal diagnostic value. The wide range in the temporal windows (3-4-3 → 2-5-3 → 5-3-2 → 4-2-4 across 6-second intervals) confirms the shape is highly fluid over the 30-second clip, consistent with a possession-based system that reconfigures dynamically on ball movement.

Interpreting from player average positions:

- **Deepest layer** (#16 at 30.9m, #8 at 30.9m, #7 at 31.2m): Three players at an almost identical average depth (~31m from Team 1's goal line, i.e. ~31m into Team 2's half). This tight clustering in x is consistent with a back-three that holds its line rather than splitting — a characteristic of teams that build through a 3-back structure with wide defenders providing width higher up.
- **Middle layer** (#4 at 35.3m, #20 at 41.9m, #13 at 40.5m): Three players slightly advanced, occupying the central midfield zone. Their average lateral positions (27–45m across the width) indicate a triangle or flat-three in central midfield — one player sitting slightly deeper (#4) with two operating wider, consistent with a 3-1-2 or 3-2-1 midfield structure in possession.
- **Most advanced** (#14 at 50.8m, #5 at 53.1m): Two players positioned ahead of midfield, operating as the attacking width or second-striker roles in possession. Their depth (~50–53m from Team 1's goal) places them in the central-to-advanced midfield zone — likely inside forwards or 8s who advance into the half-spaces when Team 2 establishes possession.

**Tactical read**: The spatial distribution is more consistent with a 3-4-3 or 3-2-2-3 in possession than a 5-3-2 — the back-three holds deep while the midfield and advanced layers structure into wider attacking patterns. The 5-3-2 label likely reflects out-of-possession positioning (where the wide players drop into defensive roles, creating a five-man defensive block) rather than the in-possession shape, which the centroid data suggests is more expansive.

**Formation label instability**: The temporal windows show four distinct k-means labels across 24 seconds. This is not evidence of chaotic organisation — it reflects Team 2's possession system reconfiguring on each ball movement, producing positional snapshots that cluster differently every 6 seconds. No fixed formation label is appropriate for a team with 83% possession in a 30-second clip; the shape is better described as a fluid, ball-dominant structure with a stable back-three as the only constant positional anchor.

---

**Temporal formation windows** (5 windows of ~6s each — showing shape instability under possession dynamics):
  Window 1 (0–6s): T1=3-4-3 T2=3-4-3 — opening press, both sides compact
  Window 2 (6–12s): T1=4-2-4 T2=2-5-3 — Team 2 beginning to establish possession, midfield expanding
  Window 3 (12–18s): T1=3-5-2 T2=5-3-2 — Team 1 dropping into mid-block, Team 2 pushing wide
  Window 4 (18–24s): T1=2-4-3 T2=4-2-4 — sustained Team 2 possession phase, Team 1 retreating

### Event Spatial Context (top 5 by type priority)
- 1.1s — **Challenge** #5→#15: start (13.8, 5.0)m → end (14.0, 4.1)m | T1 centroid (61.6, 35.3), T2 centroid (56.0, 34.4)
- 8.3s — **Challenge** #6→#20: start (4.5, 6.3)m → end (6.0, 5.8)m | T1 centroid (54.1, 38.7), T2 centroid (50.1, 43.9)
- 10.2s — **Challenge** #17→#20: start (6.4, 5.3)m → end (6.1, 5.3)m | T1 centroid (54.7, 44.1), T2 centroid (34.5, 40.1)
- 0.2s — **Challenge** #15→#10: start (14.5, 4.2)m → end (13.2, 4.6)m | T1 centroid (57.2, 37.2), T2 centroid (50.0, 37.2)
- 0.6s — **Challenge** #5→#15: start (14.3, 5.1)m → end (14.4, 4.2)m | T1 centroid (57.1, 36.7), T2 centroid (49.9, 37.1)
```

---

## 9.3 Commentary Output Comparison (Match Overview, Analysis 18)

### BASELINE — Aggregate Only (Grounding Rate: 81.8%, 11 claims)

> *"The match exhibited an exceptionally high tempo, primarily driven by Team 2's dominance in possession, which reached a staggering 83.0%, significantly eclipsing Team 1's 17.0%. Both teams displayed aggressive pressing strategies, with Team 2 posting a PPDA of 0.1 and Team 1 at 0.2, indicating a highly intense pressing environment. Despite Team 1's efforts in pressing, they struggled to retain possession, leading to a territorial control of only 40.0%, while Team 2 commanded 60.0% of the pitch. This discrepancy in possession and territory created a dynamic flow favoring Team 2, who could dictate the pace of the match.*
>
> *Tactically, Team 2's approach was characterized by their ability to maintain a broader formation with an average width of 43.1 meters, allowing for effective ball movement across the pitch. Their pressing intensity matched that of Team 1 at 0.3, but Team 2's more aggressive PPDA indicates a more proactive approach to regaining possession. Team 1, maintaining a compact shape with an average compactness of 970.5 m², attempted to absorb pressure and create counter-attacking opportunities, yet their effectiveness was limited, reflected in their lack of progressive passes and minimal territory in the opponent's half.*
>
> *Key tactical patterns emerged throughout the match. Player #18 from Team 1 demonstrated the highest work rate with a distance covered of 178m, while Player #1 recorded a remarkable top speed of 40.0 km/h. However, the collective and aggressive pressing from Team 2 limited these individual efforts, stifling Team 1's ability to transition effectively.*
>
> *Overall, the data encapsulates a match where Team 2's possession dominance, territorial control, and aggressive pressing tactics significantly dictated the game's flow, while Team 1's compactness and individual speed efforts were insufficient to mount a substantial challenge."*

**Claim-level verdicts (selected):**

| Verdict | Claim |
|---|---|
| VERIFIED | "Team 2's dominance in possession, which reached 83.0%" |
| VERIFIED | "Team 2 posting a PPDA of 0.1 and Team 1 at 0.2" |
| VERIFIED | "territorial control of only 40.0%, while Team 2 commanded 60.0%" |
| VERIFIED | "average width of 43.1 meters" |
| VERIFIED | "average compactness of 970.5 m²" |
| VERIFIED | "Player #18 from Team 1... distance covered of 178m" |
| VERIFIED | "Player #1 recorded a remarkable top speed of 40.0 km/h" |
| UNVERIFIABLE | "pressing intensity matched that of Team 1 at 0.3" *(aggregate pressing intensity field absent)* |
| UNVERIFIABLE | "lack of progressive passes" *(not a tracked metric)* |
| UNVERIFIABLE | "minimal territory in the opponent's half" *(qualitative)* |

---

### PERFRAME — Aggregate + Per-Frame Context (Grounding Rate: 71.4%, 14 claims)

> *"The match was characterized by an exceptionally high tempo, as evidenced by Team 2's dominant possession, achieving 83.0% of the ball and controlling 60.0% of the pitch territory. Their aggressive pressing, indicated by a PPDA of 0.1, demonstrated a commitment to high-intensity pressure on Team 1, who also pressed actively with a PPDA of 0.2. Both teams exhibited similar pressing intensities at 0.3, suggesting a focus on disrupting the opponent's play despite Team 2's overwhelming possession advantage. This context led to a game marked by rapid transitions and contested play, as reflected in the 27 possession changes throughout the match.*
>
> *Team 2's territorial control and ball movement were pivotal in establishing their dominance. They maintained a compact shape, averaging 1122.9 m², compared to Team 1's more stretched formation at 970.5 m². This allowed Team 2 to effectively utilize their spatial superiority, driving the ball forward with a notable average distance of 426m. Their ability to engage in sustained possession phases, particularly a 10.8-second sequence where they controlled both territory and ball movement, further highlighted their tactical approach. In contrast, Team 1 struggled to maintain possession, with their longest spell lasting only 21 frames, which limited their attacking opportunities.*
>
> *Tactically, Team 2's strategy involved a blend of high pressing and quick ball transitions, seeking to exploit Team 1's defensive layout. Both teams recorded 10 challenges each, but Team 2's ability to quickly regain possession allowed them to maintain control. Player #18 from Team 1 notably covered the most distance at 178m, while Player #1 reached a top speed of 40.0 km/h, contributing to the few counter-attacking attempts Team 1 made — although they were largely unsuccessful in creating scoring opportunities.*
>
> *Overall, Team 2's tactical execution, characterized by aggressive pressing and effective ball movement, overshadowed Team 1's efforts. Team 1's compactness and energetic pressing were unable to disrupt Team 2's rhythm, leading to a match dominated by the latter's strategic superiority in both possession and territorial control."*

**Notable spatial claims using per-frame evidence:**

| Verdict | Claim | Per-frame grounding |
|---|---|---|
| VERIFIED | "10.8-second sequence where they controlled both territory" | Directly from possession phase table (15.9s–26.7s = 10.8s) |
| VERIFIED | "27 possession changes throughout the match" | From possession_sequence transitions |
| VERIFIED | "longest spell lasting only 21 frames" | From aggregate (not per-frame), but per-frame context reinforces |
| UNVERIFIABLE | "compact shape, averaging 1122.9 m²" *(Team 2 aggregate compactness is 534m² mean, not 1122.9)* | Per-frame compactness series shows 314–1308m² range; LLM anchored to wrong value |
| UNVERIFIABLE | "driving the ball forward with a notable average distance of 426m" *(not a tracked field)* | — |

*Note: The per-frame context prompted the LLM to make more claims (+3 vs baseline), some of which referenced per-frame evidence correctly (possession phase duration, transition count) but others introduced new unverifiable claims about ball distance and compactness values not present in either aggregate or per-frame data.*

---

## 9.4 Grounding Rate Comparison by Analysis Type

| Analysis Type | Baseline (aggregate only) | Per-Frame Augmented | Delta |
|---|---|---|---|
| Match Overview | 81.8% | 71.4% | −10.4pp |
| Tactical Deep Dive | 63.6% | 57.1% | −6.5pp |
| Event Analysis | 0.0% | 66.7% | **+66.7pp** |
| Player Spotlight | 77.8% | 77.8% | 0.0pp |
| **Overall** | **55.8%** | **68.3%** | **+12.5pp** |

DB Resolution Rate: Baseline 3.9% → Per-Frame 3.0% (marginal, within noise)

**Key findings:**

1. **Event analysis is the primary beneficiary** (+66.7pp): Baseline event commentary (0% grounding) fabricated event-level claims because no spatial anchor exists in aggregate JSON. Per-frame event coordinates enabled the LLM to make spatially-grounded claims about challenge locations and phase context, all of which were verifiable.

2. **Match overview and tactical deep dive slightly decrease**: Per-frame context prompts more claims per response (14 vs 11 for match overview, 21 vs 22 for tactical). The additional claims the LLM generates when given richer context introduce new unverifiable inferences (e.g. misquoting compactness averages from the time series).

3. **Player spotlight is unchanged** (77.8%): Player-level claims (distance, speed) are grounded in aggregate statistics that neither condition changes; per-frame player position data does not add new verifiable player metrics beyond what the aggregate JSON already provides.

4. **Overall +12.5pp** reflects primarily the event analysis rescue. The per-frame context does not universally improve grounding — it redistributes it toward spatial/event claims at the cost of slightly inflating claim count in narrative sections.

---

## 9.5 Dissertation Implication

The BASELINE → PERFRAME comparison closes the verification gap identified in §4.5 of the evaluation chapter: 41.7% of claims under R-ALL are initially unverifiable because spatial/temporal claims cannot be confirmed from aggregate scalars. Providing per-frame spatial evidence as prompt context converts ungrounded spatial intuition into verifiable spatial claims — most visibly in event analysis, where the model shifts from fabricating event-level statistics (0% grounding) to correctly citing coordinate-derived evidence (66.7%).

The +12.5pp overall improvement is concentrated in the spatial domain. This confirms that the verification chain (video → CV → per-frame DB → aggregate → MatchInsights → LLM) should be extended upstream: per-frame evidence belongs in the generation context, not only the verification oracle. The remaining gap — match overview and tactical commentary generating more unverifiable claims under PERFRAME — suggests a token competition effect: richer context generates more claims, and not all additional claims are anchored to the newly provided evidence.

---

# Part 10: Provider Comparison — OpenAI vs Gemini on Per-Frame Commentary (Analysis 18)

Both providers receive the identical PERFRAME context (aggregate analytics + per-frame spatial evidence, ~2,617 tokens). Provider outputs differ in: claim density, spatial grounding strategy, analyst-register fluency, and how they translate per-frame evidence into tactical language.

**Grounding rates** (FActScore, OpenAI GPT-4o vs Gemini 1.5 Flash, PERFRAME condition):

| Analysis Type | OpenAI PERFRAME | Gemini PERFRAME | Delta (OAI − GEM) |
|---|---|---|---|
| Match Overview | 71.4% | 50.0% | +21.4pp |
| Tactical Deep-Dive | 57.1% | 52.0% | +5.1pp |
| Event Analysis | 66.7% | 0.0% | +66.7pp |
| Player Spotlight | 77.8% | 22.2% | +55.6pp |
| **Overall** | **68.3%** | **31.1%** | **+37.2pp** |

**G-Eval quality scores** (Claude Haiku as judge, 1–5 Likert, n_samples=1):

| Analysis Type | Dimension | OpenAI | Gemini | Winner |
|---|---|---|---|---|
| Match Overview | Coherence | 3.0 | **4.0** | Gemini |
| Match Overview | Consistency | 2.0 | **3.0** | Gemini |
| Match Overview | Fluency | 3.0 | 3.0 | Tie |
| Match Overview | Relevance | 3.0 | **4.0** | Gemini |
| Match Overview | Groundedness | 2.0 | **4.0** | Gemini |
| Tactical Deep-Dive | Coherence | 3.0 | 3.0 | Tie |
| Tactical Deep-Dive | Consistency | 2.0 | **3.0** | Gemini |
| Tactical Deep-Dive | Fluency | 2.0 | **3.0** | Gemini |
| Tactical Deep-Dive | Relevance | **4.0** | 3.0 | OpenAI |
| Tactical Deep-Dive | Groundedness | 4.0 | **5.0** | Gemini |
| Player Spotlight | Coherence | 3.0 | 3.0 | Tie |
| Player Spotlight | Consistency | 2.0 | 2.0 | Tie |
| Player Spotlight | Fluency | 2.0 | **4.0** | Gemini |
| Player Spotlight | Relevance | 3.0 | **4.0** | Gemini |
| Player Spotlight | Groundedness | 2.0 | **3.0** | Gemini |
| **Overall mean** | | **2.67** | **3.40** | **Gemini** |

**The quality–grounding tradeoff**: OpenAI achieves significantly higher FActScore grounding (68.3% vs 31.1%) while Gemini scores substantially higher on G-Eval quality (3.40 vs 2.67). Gemini produces fewer, more carefully qualified claims — the judge rewards this as coherent and well-grounded narrative, but because fewer specific numeric claims are made, fewer are verifiable against the analytics JSON. OpenAI makes more claims at higher precision (more direct numeric citations) but the judge penalises the commentary for consistency failures where those claims occasionally contradict each other.

---

## 10.1 Best Analyst Commentary — Gemini 1.5 Flash PERFRAME

*Gemini PERFRAME tactical_deep_dive achieved 5.0/5 groundedness from the Claude judge — the highest single-dimension score across all conditions. Gemini match_overview achieved 4.0/5 groundedness and 4.0/5 coherence.*

### Match Overview (Gemini PERFRAME — Judge scores: Coherence 4.0 | Groundedness 4.0 | Mean 3.60)

> *This segment of play was defined by an exceptionally high tempo and an overwhelming territorial dominance by Team 2, who controlled possession for 83.0% of the time, compared to Team 1's 17.0%. Team 2 also dictated the spatial dynamics, controlling 60.0% of the pitch territory and maintaining a significantly higher defensive line at 54.9m. Both teams engaged in an exceptionally aggressive pressing scheme, as evidenced by Team 2's PPDA of 0.1 and Team 1's PPDA of 0.2 — both well below the typical aggressive pressing threshold. This led to a highly contested flow, with 27 possession changes and a total of 20 challenges, evenly split between the two teams, within this brief period. The average inter-team distance of 7.0m further highlights the congested and intense nature of the contest.*
>
> *Tactically, Team 1 adopted a more compact defensive shape, with an average compactness of 970.5m², particularly during their out-of-possession phases where it reached 1040.0m². This contrasts with Team 2's slightly more expansive average compactness of 1122.9m². Despite Team 2's overall territorial and possession dominance, the zone occupancy data indicates that their players spent a higher percentage of frames in the defensive third (47.6%) compared to Team 1 (32.8%). This suggests Team 2's possession often involved recycling the ball in deeper areas, while Team 1's players, although less on the ball, maintained a stronger presence in the middle third (65.0% vs 51.1%). Ball movement was limited, with Team 2 completing two passes and Team 1 completing one, indicating a highly disrupted play pattern dominated by challenges rather than fluid passing combinations.*
>
> *Individual performances highlighted the high work rate demanded by this intense period. Player #18 from Team 1 covered the most distance at 178m — demonstrating the highest work rate on the pitch despite Team 1's possession deficit. Player #1 (Team 1) recorded the highest top speed at 40.0 km/h, underscoring the frequent short, explosive bursts required by Team 1's counter-press. Key moments included a series of challenges in rapid succession at 0:00–0:01, such as Team 2's Player #15 winning a challenge from Team 1's Player #10 at (14.5, 4.2)m — deep in Team 1's defensive third, near the left touchline — and Team 1's Player #5 immediately winning it back at (14.3, 5.1)m, illustrating the intensity of the early contest for territorial control.*
>
> *Given the brevity of the data segment, with Team 2's longest sustained possession spell lasting only 10.8 seconds (15.9s–26.7s), these observations describe a 30-second window of extreme pressure rather than a settled phase. The pattern is consistent with Team 2 operating a sustained high-press that compresses Team 1 into their defensive structure, with possession turning over rapidly before either team can establish build-up play.*

**Why this is the best output**: The Gemini match_overview is the highest-quality commentary generated across all conditions and both providers. Key analyst markers:
- Identifies the zone occupancy *asymmetry* — Team 2 spends more time in their own defensive third (47.6%) despite having 83% possession, which signals recycled possession rather than directional attacking. This is a counter-intuitive finding that a human analyst would flag.
- Cites the inter-team distance (7.0m) as a measure of congestion — a per-frame metric not available in the aggregate JSON, correctly used as spatial evidence.
- Anchors the opening challenge sequence to specific coordinates (14.5, 4.2)m and (14.3, 5.1)m with a tactical interpretation (deep in defensive third, near left touchline → Team 2 pressing into Team 1's build-up zone).
- Qualifies the 10.8-second possession phase with the exact timestamp range (15.9s–26.7s).
- Explicitly hedges the scope: "30-second window of extreme pressure rather than a settled phase" — epistemic discipline rarely seen from LLMs without explicit prompting.

---

### Tactical Deep-Dive (Gemini PERFRAME — Judge groundedness: 5.0/5)

**Selected excerpt — Formation Shape and Structure:**

> *Team 1's estimated formation of 2-5-3 (confidence 0.32) should be interpreted cautiously. The confidence score of 0.32 is moderate, and the temporal formation windows reveal significant fluidity, transitioning from 3-4-3 at 0s to 4-2-4 at 6s, 3-5-2 at 12s, and 2-4-3 at 18s. This instability across 6-second windows makes it difficult to assign a single, definitive formation label to Team 1. Rather, the per-frame zone occupancy data (65.0% of frames in the middle third, only 2.2% in the attacking third) paints a clearer picture: Team 1 maintained a compact mid-block — a defensive structure prioritising occupation of the central corridor over forward positioning. Their average team width of 39.7m, compactness of 970.5m², and average team length of 38.7m collectively describe a tight, horizontal defensive shape designed to limit Team 2's central penetration.*
>
> *Team 2's 5-3-2 estimate carries a very low confidence score of 0.11, rendering it essentially unreliable as a formation descriptor. More meaningful is the spatial evidence: a defensive line averaging 54.9m (compared to Team 1's 20.9m) places Team 2's backline 34m higher up the pitch — a high-line structure consistent with a possession-dominant team that compresses play into the opposition half. Their zone occupancy (47.6% defensive third) at first appears contradictory — a team with 83% possession spending nearly half their frames behind their own midfield line — but is better explained by their possession recycling pattern: the data shows only 6 possession phases lasting ≥1 second, with the longest at 10.8 seconds. Team 2 circulated the ball quickly through short phases, repeatedly returning to deeper positions rather than sustaining directional forward progression.*

**Selected excerpt — Pressing Patterns:**

> *The PPDA values are the most tactically significant data points in this segment: Team 2 at 0.1 and Team 1 at 0.2. A PPDA of 0.1 represents an extreme outlier — for reference, the most aggressive pressing teams in European top-flight football average PPDA values in the 6–9 range; values below 1.0 are only observed in 30-second clips that capture a sustained pressing phase rather than match-average pressing intensity. Both teams are operating at maximum pressing intensity within this window, which explains the 27 possession changes in 30 seconds. This is not possession football — it is a contest of press-and-win cycles where neither team is able to establish progressive play.*
>
> *Team 1's counter-press (2 counter-press windows detected) suggests a GEGENPRESSING structure: immediately upon losing the ball, they attempted to recover possession within 5–8 seconds. This is consistent with their zone occupancy — spending 65% of time in the middle third places their press-triggers in the central corridor where immediate ball-wins are most achievable. However, with only a 17% possession share, the counter-press was structurally unable to convert ball-wins into sustained possession phases.*

**Judge chain-of-thought (groundedness 5.0)**: *"The commentary consistently cites specific numerical values from the source data — 0.32 confidence, 54.9m defensive line, 47.6% defensive third occupancy, 0.1 PPDA. Crucially, it does not fabricate values or extrapolate beyond what the data contains. The PPDA contextualisation (comparing 0.1 to top-flight averages) is inferential but correctly flagged as contextual reference rather than a data claim. The formation instability observation is directly derivable from the temporal window table. This commentary demonstrates exceptional discipline in distinguishing what the data shows from what the analyst infers."*

---

### Player Spotlight (Gemini PERFRAME — Judge fluency: 4.0 | Relevance: 4.0)

**Selected excerpt — Player #18:**

> *Player #18 demonstrated an outstanding work rate, covering the most distance on the pitch at 178m — in a 30-second clip, this translates to approximately 21.4 km/h average speed, indicating near-continuous movement throughout the passage of play. Of this, 130m was covered at high intensity, suggesting Player #18 was not simply jogging between positions but engaged in repeated sprint-and-recover cycles consistent with Team 1's pressing structure. The spatial data places Player #18's average position at (36.1, 16.5)m — approximately 36m from Team 1's own goal, on the left side of the pitch (y=16.5m, with the full width at 68m). This positions #18 as a left-sided defensive midfielder or left-centre-back in a mid-block structure — deep enough to support the back line, wide enough to cover the left channel against Team 2's wide press-triggers.*
>
> *The 130m high-intensity distance is particularly revealing. In the context of Team 1's 17% possession share and 27 possession changes, Player #18 was essentially performing defensive recovery runs on every Team 2 possession sequence. With the mean inter-team distance at 7.0m, the entire shape was compact enough that these recovery runs were short in distance but maximal in intensity — the high-intensity threshold is defined by acceleration above 25 km/h, meaning #18 was hitting sprint speeds repeatedly on sequences lasting 1–2 seconds each.*

**Selected excerpt — Player #1:**

> *Player #1 recorded the highest top speed of 40.0 km/h but covered only 23m total distance at an average speed of 10.2 km/h — a profile that is analytically distinctive. High top speed combined with minimal total distance is the kinematic signature of a specialist counter-attacker or a wide forward who operates in short explosive bursts: they spend most of the defensive phase stationary or walking (hence the low average speed and distance) and produce a single high-intensity sprint on the transition. In Team 1's context — 17% possession, continuous defensive pressure — Player #1's 40.0 km/h peak was almost certainly recorded on a single counter-attacking run that either resulted in a turnover or an unsuccessful attempt to break beyond Team 2's high defensive line (54.9m from goal). The per-frame centroid data at 21s shows Team 1's team centroid collapsing to (12.7, 8.0)m — a sudden mass retreat toward their own goal — which is consistent with a broken counter-attack sequence where the team was caught high and had to recover rapidly.*

---

## 10.2 OpenAI vs Gemini — Side-by-Side on the Same Passage

**Prompt**: Match Overview, Analysis 18, PERFRAME condition. Same context (~2,617 tokens). Same system prompt.

**The key divergence — Zone occupancy interpretation:**

| | OpenAI GPT-4o | Gemini 1.5 Flash |
|---|---|---|
| Zone occupancy mention | Not cited | ✓ Cited: "47.6% vs 32.8% defensive third" |
| Counterintuitive finding | Not identified | ✓ Identified: Team 2 spending more time in own defensive third despite 83% possession |
| Explanation of why | N/A | ✓ "recycling the ball in deeper areas rather than directional attack" |
| Inter-team distance | Not used | ✓ Used: "average inter-team distance of 7.0m highlights congestion" |
| Coordinate-anchored event | Not cited | ✓ Cited: Challenge at (14.5, 4.2)m and (14.3, 5.1)m with tactical interpretation |
| Scope qualification | Absent | ✓ Explicit: "30-second window of extreme pressure rather than a settled phase" |

**The key divergence — compactness:**

| | OpenAI GPT-4o | Gemini 1.5 Flash |
|---|---|---|
| Team 2 compactness cited | 1122.9m² ✓ | 1122.9m² ✓ |
| Team 1 compactness cited | 970.5m² ✓ | 970.5m² ✓ |
| Out-of-possession compactness | Not cited | ✓ "1040.0m² during out-of-possession phases" |
| Interpretation | "compact shape to absorb pressure" | "contrasts with Team 2's more expansive shape" — comparative framing |

**Why Gemini reads as more analyst-like**: Gemini identifies the *paradox* in the data (high-possession team spending more frames in their defensive third) and explains it mechanistically (recycling vs directional). OpenAI reports the numbers without flagging the anomaly. This is the difference between retrieving data and reasoning about it.

**Why OpenAI has higher grounding rate**: OpenAI makes more direct numeric claims (11 vs 4 in match_overview), each anchored to a specific JSON field. Gemini generates more inferential language ("suggests", "consistent with", "paints a clearer picture") which is harder to verify against atomic data fields — even when the inference is correct. The FActScore methodology penalises hedged/inferential language equally with incorrect claims.

---

## 10.3 Claude as Judge — Chain-of-Thought Reasoning Excerpts

The G-Eval judge (Claude Haiku) evaluates each commentary on five dimensions with chain-of-thought reasoning before assigning a score. Selected reasoning excerpts illustrate how the judge discriminates between conditions:

**Match Overview — Groundedness — OpenAI (2/5):**
> *"Let me trace through the commentary's statistical claims against the source data. 'Team 2's dominant possession, achieving 83.0%' — directly cited. 'Controlling 60.0% of the pitch territory' — cited. 'PPDA of 0.1 and Team 2's PPDA of 0.2' — but wait, the commentary states 'Team 2 posting a PPDA of 0.1 and Team 1 at 0.2' — correct. 'Both teams exhibited similar pressing intensities at 0.3' — this value does not appear in the source data provided to the judge. '27 possession changes' — cited from per-frame possession sequence. 'compact shape, averaging 1122.9 m²' — this is Team 2's compactness attributed here to Team 2, but the value 1122.9 is Team 2's compactness — however the narrative attributes it within a paragraph about 'Team 2's territorial control', which is correct. The fabricated pressing intensity value (0.3) and a compactness misattribution bring the score down. Score: 2."*

**Match Overview — Groundedness — Gemini (4/5):**
> *"This commentary grounds its claims carefully. '83.0% possession' — direct citation. 'PPDA of 0.1 and Team 1's PPDA of 0.2' — cited accurately. '27 possession changes' — from possession sequence, correct. 'Average inter-team distance of 7.0m' — this appears in the per-frame context. 'compactness of 970.5m²' and '1122.9m²' — both correct and correctly attributed. 'Zone occupancy data indicates their players spent 47.6% in defensive third vs 32.8%' — directly from the zone occupancy table. The only non-cited claim is 'two passes completed' which does not appear in the short analytics summary provided to me as judge. The scope qualification ('30-second window... rather than a settled phase') is analytically appropriate — it is a limitation acknowledgement, not a false claim. Score: 4."*

**Tactical Deep-Dive — Groundedness — Gemini (5/5):**
> *"This commentary systematically cites numerical evidence for every major claim. Formation confidence scores (0.32, 0.11) cited with appropriate uncertainty language. Zone occupancy percentages (65.0%, 2.2%, 47.6%, 51.1%) all cited directly. Defensive line heights (54.9m vs 20.9m) cited and the 34m gap calculated correctly. Average width (39.7m), compactness (970.5m²), team length (38.7m) all cited. The PPDA contextualisation against European top-flight averages is explicitly framed as external reference ('for reference, top-flight teams average...') rather than a data claim. The only potentially unverifiable element is the inference about 'recycling the ball in deeper areas' — but this is correctly presented as an inference from the zone occupancy data, not as a direct data claim. Score: 5."*

**Player Spotlight — Consistency — OpenAI (2/5):**
> *"I find several internal consistency issues. The commentary claims Player #18 had the 'highest work rate with a distance covered of 178m' — then separately notes 'distance covered of 178m, which translates to an average speed of 21.4 km/h.' These are consistent. However, the commentary also states that Player #18 'was active in multiple pressing windows' — but then notes Team 1 'struggled to maintain possession' and their 'attacking contributions were limited.' These are not strictly contradictory, but the framing creates a tension: either #18's pressing was effective (implying positive outcome) or possession was consistently lost (implying ineffective pressing). The commentary presents both without resolving the contradiction, creating a logical gap. Score: 2."*

---

## 10.4 Extended Q&A — Provider-Specific Behaviour (Gemini vs OpenAI)

These questions are designed to probe where Gemini and OpenAI specifically diverge when given per-frame spatial evidence. Each response is drawn from the actual PERFRAME condition outputs.

| # | Question | Expected (ground truth) | OpenAI GPT-4o | Gemini 1.5 Flash | Winner |
|---|---|---|---|---|---|
| 1 | Team 2 has 83% possession but the zone occupancy shows them spending 47.6% of frames in the defensive third — more than Team 1 (32.8%). Is this a contradiction? If not, explain it. | Not a contradiction. High-possession teams often circulate deep before advancing; 47.6% in defensive third reflects recycling through the goalkeeper/defensive phase, consistent with only 6 possession phases ≥1s. | "This appears consistent with Team 2's overall possession dominance; they controlled 60% of territory which is concentrated in advanced areas." ✗ *(misses the paradox entirely)* | "Not a contradiction — it suggests Team 2's possession involved recycling the ball through deeper areas rather than sustained directional attack. With only 6 phases lasting ≥1 second and the longest at 10.8s, possession was frequently contested, forcing Team 2 back to rebuild." ✓ | Gemini |
| 2 | Using only the centroid and compactness data, describe what happened between 18s and 21s in the clip. | At 18s, both centroids and compactness are missing (—). At 21s, both centroids collapse to (12.7, 8.0) and (10.7, 12.8) — very deep on the pitch (x≈12m). Team 1 compactness drops to 32m², Team 2 to 109m². This signals a concentrated passage of play near one goal-mouth — possibly the break of a counter or a defensive recovery. | "The data shows missing centroid information at 18s, suggesting a tracking gap. By 21s, both teams' centroids collapsed to approximately (12.7, 8.0)m and (10.7, 12.8)m, indicating a significant retreat toward one end of the pitch." ✓ | "The — at 18s indicates a tracking loss — likely a crowded frame where the detector dropped player IDs. At 21s, centroids at x≈12m place both teams deep in Team 1's defensive third, with the 32m² compactness suggesting nearly all detected players are within a 6×5m area — consistent with a goal-mouth scramble or defensive press in a very tight space." ✓ *(also identifies 32m² as tracking artefact)* | Gemini |
| 3 | What is the average possession phase duration in this clip, and what does it imply tactically? | 39 transitions in ~30s = mean transition every 0.77s. Only 6 phases ≥1s, with mean for those 6 = (1.1+1.7+1.3+1.9+10.8+1.4)/6 = 3.0s. Overall mean is ~0.77s. Implies chaotic, contested micro-possessions rather than sustained build-up. | "With 27 possession changes in a 30-second clip, the average possession duration is approximately 1.1 seconds per team — indicating extremely rapid turnovers." ✗ *(calculates 30/27 ≈ 1.1s but doesn't account for both-team possession share correctly; should be 30 × 17% / transitions for Team 1 etc.)* | "The clip contains 39 transitions with a mean transition interval of approximately 0.77 seconds. Of these, only 6 phases last ≥1 second — meaning over 84% of possession sequences resolve within a single second. Tactically, this describes a contest of high-frequency, contested micro-possessions rather than possession football. Neither team can establish a build-up rhythm." ✓ *(calculates 39 transitions / 30s = 0.77s mean correctly)* | Gemini |
| 4 | Player #1 covered only 23m total but hit 40.0 km/h top speed. Is this data consistent? What role does this kinematic profile suggest? | Yes — consistent. Low total distance + high top speed = brief explosive burst. Suggests a wide forward or counter-attacking role: static during defensive phases, one or two maximal sprint efforts on transitions. | "This profile is consistent — Player #1 appears to be a specialist pace player operating in short, explosive bursts rather than a high-volume runner. Their 40.0 km/h top speed suggests involvement in at least one counter-attacking sprint, while the low 23m distance confirms they spent most of the clip in a static or walking phase." ✓ | "Player #1's kinematic profile — 40.0 km/h peak velocity but only 23m total distance and 10.2 km/h average speed — is analytically consistent. The average speed of 10.2 km/h across a 30-second window that includes a 40.0 km/h sprint implies the player was stationary or walking for the vast majority of the clip, with one explosive sprint lasting ~2 seconds. This is the signature of a wide forward operating as a threat on the counter: parked during Team 2's possession phases, released on a single transition run." ✓ | Gemini |
| 5 | Does the per-frame event data support the claim that Team 2 pressed into Team 1's defensive third from the first second? | Yes — challenge at 0.2s at coordinates (14.5, 4.2)m: x=14.5m is in Team 1's defensive third (0–35m), near the left touchline. This confirms immediate high-pressure from Team 2 at the earliest detectable timestamp. | "Yes — the first challenge (#15→#10) at 0.2s occurred at (14.5, 4.2)m, placing it in Team 1's defensive third (x<35m threshold), confirming Team 2 initiated pressure in Team 1's build-up zone immediately from kick-off." ✓ | "The event data is unambiguous: Team 2's #15 won a challenge from Team 1's #10 at just 0.2s into the clip, at coordinates (14.5, 4.2)m — 14.5m from Team 1's own goal line, near the left touchline. This is deep in Team 1's defensive third (x<35m). Team 2 did not wait for Team 1 to advance before pressing; they pressed immediately into the build-up zone, consistent with a PPDA of 0.1 indicating maximum pressing intensity." ✓ | Tie |
| 6 | The Claude judge gave the Gemini tactical deep-dive a groundedness score of 5/5 despite Gemini's overall FActScore grounding rate being only 31.1%. Explain the apparent contradiction. | The judge evaluates whether claims that *are* made are numerically accurate and appropriately hedged. Gemini makes fewer claims overall — but those claims are tightly grounded. FActScore penalises low claim count by treating inferences and qualifications as unverifiable. A commentary with 5 precise citations scores higher on FActScore than a commentary with 3 precise + 5 hedged. The judge, reading holistically, rewards disciplined hedging. | "This reflects the quality-grounding tradeoff: Gemini's commentary is more qualitative and hedged — it makes fewer claims but those claims are well-supported. FActScore penalises hedged language because inferences can't be directly verified against atomic data fields. The judge rewards the same hedging as appropriate epistemic discipline." ✓ | "The two metrics measure different things. FActScore measures the proportion of atomic claims that can be matched to specific data fields — it rewards citation density and penalises any language that cannot be directly verified. The judge evaluates holistic commentary quality: does the narrative make sense, is it consistent, are the numbers it cites accurate? Gemini's 31.1% FActScore reflects low claim count, not low accuracy — the claims it makes are accurate and carefully qualified. The judge sees this as a virtue; FActScore sees it as a gap. Neither measure is wrong: they capture orthogonal dimensions of commentary quality." ✓ | Gemini |
| 7 | The formation temporal windows show Team 1 as 3-4-3 at 0s, 4-2-4 at 6s, 3-5-2 at 12s, 2-4-3 at 18s. A commentator claims "Team 1 played a 3-4-3 throughout the clip." Is this claim verifiable, refuted, or unverifiable against the per-frame data? | Refuted — the per-frame temporal windows directly show the formation label changing every 6 seconds. The 3-4-3 label applies only to the first window (0–6s). "Throughout" implies consistency that the data explicitly contradicts. | "This claim is refuted by the per-frame temporal windows. The 3-4-3 label appears only in the first window (0–6s) and changes to 4-2-4 at 6s, 3-5-2 at 12s, and 2-4-3 at 18s. No single formation label holds for the full clip." ✓ | "Refuted — the temporal formation data directly contradicts 'throughout.' Team 1's estimated formation changed in every 6-second window. If anything, the per-frame evidence argues against assigning any fixed formation label to Team 1 in this clip, given the confidence score of 0.32 and the label instability across windows." ✓ | Tie |
| 8 | OpenAI PERFRAME match_overview received a consistency score of 2/5 from the Claude judge despite having a 71.4% grounding rate. What failure mode does this illustrate? | High grounding rate (many specific citations) coexists with internal consistency failures: contradicting one claim with another, or creating logical tensions (e.g. claiming a pressing strategy is "effective" while also reporting the team had only 17% possession). The judge penalises the contradiction; FActScore does not — it only checks individual claim accuracy, not cross-claim coherence. | "OpenAI made more specific claims (higher grounding rate), but some of those claims contradicted each other — e.g. asserting pressing is 'effective' while noting only 17% possession. High citation density without narrative coherence creates a consistency failure that FActScore misses but the judge catches." ✓ | "This illustrates the FActScore coverage gap: the metric verifies each claim independently against the data but cannot check whether claims are mutually consistent. A commentary can be 71% grounded at the atomic claim level while being internally contradictory at the paragraph level. The judge reads the commentary holistically and penalises the logical tensions that emerge when many accurate-but-competing claims are assembled without a coherent narrative frame." ✓ | Tie |


---

# Part 11: Visual System Prompt Augmentation Text

The following augmentation blocks are appended to the base system prompt for each analysis
type in the VISUAL and VISUAL_FOCUSED conditions. They instruct the LLM to read annotated
chart values directly rather than computing from the sampled markdown tables below.

## 11.1 VISUAL Condition — Full Chart Suite Augmentation

```
## Using Attached Time-Series Charts
The data context is accompanied by visual charts showing per-frame tactical metrics.
Use these charts as your primary evidence source for temporal and spatial claims:

- **Compactness Chart**: Read the annotated mean values (e.g. "T1 mean=534m²") directly
  from the chart. Do NOT compute averages from the text tables — the sampled table rows
  may not reflect the true mean. Orange scatter points mark statistical anomalies (>2σ).
- **Centroid Trajectory**: Interpret team spatial positions from the 2D pitch diagram.
  Circle = start position, triangle = end position. The path shows territorial drift.
- **Pressing Dashboard**: Green shaded regions indicate pressing phases (inter-team
  distance below threshold). Reference these windows when describing pressing sequences.
- **Combined Overview**: Use for cross-metric claims (e.g. compactness spike coinciding
  with high inter-team distance suggests a defensive shape break during a pressing phase).

When citing statistics from the charts, use the annotated values visible in the chart
(e.g. "mean compactness of 534m²") rather than values from the markdown tables below.
```

**Design rationale**: The compactness table in PERFRAME conditions samples every Nth frame,
so the sampled mean can differ significantly from the true mean (e.g. sampled rows in the
table produced an apparent mean of ~1122.9m² while the annotated chart shows 534m² for
Team 1 — a 2× error). The augmentation explicitly directs the LLM to the chart annotation.

## 11.2 VISUAL_FOCUSED Condition — Single Compactness Chart Augmentation

```
## Using the Attached Compactness Chart
A single chart showing Team 1 and Team 2 compactness (convex hull area in m²) over time
is attached. Use it as your primary evidence source for defensive shape claims:

- Read the annotated mean value directly from the chart (dashed line + label).
  Do NOT compute averages from the table rows — sampled rows may be unrepresentative.
- Orange markers indicate anomalies (compactness >2σ above mean) — these often correspond
  to shape breaks during pressing triggers or possession transitions.
- Reference specific time windows (x-axis) when describing when shape changed.
- Lower values = more compact shape; typical compact block ~400m², open play ~800m².
```

**Design rationale**: VISUAL_FOCUSED is the context competition control — it provides one
chart (compactness) with no competing visual context. The focused augmentation matches
the reduced chart set and avoids instructing the LLM to look for charts that are not present.

---

# Part 12: Rendered Chart Descriptions

Charts are produced by `VisualTimeSeriesRenderer` in `backend/api/services/tactical.py`.
Each chart is rendered as a JPEG at 150 DPI and passed to the vision-capable LLM as a
base64-encoded image in the multimodal content array.

## 12.1 Compactness Time-Series (Analysis 18, 750 frames)

Dual-line plot: Team 1 (blue) and Team 2 (orange) convex hull area in m² over time.
Annotated elements: horizontal dashed lines at each team’s mean (**T1 mean=534m²**,
**T2 mean=1123m²**), ± 1σ shaded bands, orange scatter anomaly markers (>2σ).

Key visual insight: the LLM reads “T1 mean=534m²” directly from the annotation rather than
averaging the sparse markdown table rows (which would yield ~1122.9m² — a 2× error).

## 12.2 Centroid 2D Trajectory (Analysis 18)

2D pitch diagram (105×68m) with team centroid paths. Circle markers = start, triangles = end.
Both teams’ centroids concentrate at x≈10–20m, confirming the deep zone occupancy data.

## 12.3 Pressing Dashboard (Analysis 18)

Dual-panel: inter-team distance (m) over time + Team 1 compactness. Green shaded vertical
bands mark pressing phases (distance below threshold). The LLM reads pressing phase
start/end timestamps from the x-axis to generate event-anchored claims.

**This is the mechanism behind 0%→60% event_analysis grounding**: pressing phase windows
map directly to challenge/ball-win events in the event JSON. Adding this chart to VISUAL
increased event_analysis from 50% on 2 claims to 60% on 5 claims in the 4-chart re-run.

## 12.4 Combined 2×2 Overview (Analysis 18)

2×2 subplot: compactness, centroid x-position, inter-team distance, zone occupancy bar chart.
Used for VISUAL and VISUAL_MULTIMODAL conditions only.

---

# Part 13: Digit-Space Formatting Examples

Digit-space tokenisation (Gruver et al. 2023) converts decimal numbers to space-separated
character sequences to reduce BPE fragmentation. Implemented in `digit_space_format()` in
`backend/api/services/tactical.py`.

## 13.1 Conversion Examples

| Original value | Digit-space output |
|---|---|
| 534.2 | `5 3 4 , 2` |
| 1122.9 | `1 1 2 2 , 9` |
| 83.0 | `8 3 , 0` |
| 0.1 | `0 , 1` |
| 7.0 | `7 , 0` |

## 13.2 Applied to Compactness Series

Standard: `| 0 | 412.1 | 893.4 |`
Digit-space: `| 0 | 4 1 2 , 1 | 8 9 3 , 4 |`

## 13.3 Observed Effect

DIGIT_SPACE generated **18–21 claims in tactical_deep_dive** vs 3–7 for other conditions.
Grounding rate did not improve proportionally (33.8% vs BASELINE 42.6%). The additional
claims are unverifiable inferences generated by the decomposed numeric tokens. Consistent
with Gruver et al.’s original finding: digit-space improves *numeric reproduction fidelity*
(a different task from grounding verification).

---

# Part 14: Commentary Outputs — Visual and Digit-Space Conditions

## 14.1 VISUAL Condition — Match Overview (Gemini, Analysis 18, 2-chart run)

**Grounding rate: 25.0% | Claims: 4 | DB resolution: 0%**

> This was an exceptionally high-intensity, albeit brief, segment of play, characterized
> by Team 2’s overwhelming possession dominance at 83.0% and an aggressive, almost
> continuous pressing from both sides. Team 2 recorded an extremely low PPDA of 0.1, while
> Team 1 also pressed with remarkable intensity, registering a PPDA of 0.2. These figures
> are well below the typical 8-12 range, indicating a match flow dominated by immediate
> defensive engagement rather than sustained build-up. The average inter-team distance of
> 7.0m further underlines the congested and combative nature of the play.
>
> Team 2 clearly dominated the match in terms of territorial control, holding 60.0% of the
> pitch and advancing their defensive line significantly higher at 54.9m compared to Team
> 1’s deeper 20.9m. The compactness chart shows Team 1 maintaining a tighter defensive
> shape (mean ≈534m²) compared to Team 2’s more expansive positioning (mean ≈1123m²).

**Verified (1/4)**: “PPDA of 0.1”. Unverifiable (3/4): external benchmark range,
inter-team distance (correct value but claim framing resists atomic JSON matching),
compactness values (correct from chart annotation but not in aggregate stats JSON).

## 14.2 VISUAL_FOCUSED Condition — Match Overview (Gemini, Analysis 18)

**Grounding rate: 50.0% | Claims: 4 | DB resolution: 0%**

> This match was characterized by an exceptionally high-intensity pressing duel, despite
> a significant disparity in possession. Team 2 dominated the ball with 83.0% possession,
> controlling 60.0% of the pitch territory. Both teams engaged in an extremely aggressive
> press; Team 2 recorded a PPDA of 0.1, while Team 1 was not far behind with 0.2. The
> compactness chart confirms Team 1’s tight defensive block, with a mean around 534m², in
> contrast to Team 2’s more expansive shape.

**Verified (2/4)**: “83.0% possession”, “PPDA of 0.1”. VISUAL_FOCUSED outperforms VISUAL on
match_overview (50% vs 25%) because one focused chart produces less cross-chart ambiguity.

## 14.3 VISUAL Condition — Event Analysis (Gemini, Analysis 18, 4-chart re-run)

**Grounding rate: 60.0% | Claims: 5 | DB resolution: 0%**

> **0:00 — Challenge: #15 (Team 2) won from #10**
> Right from the kickoff, Team 2’s #15 immediately engaged and won possession from Team
> 1’s #10. This aggressive action showcased Team 2’s intent to dominate possession and
> apply high pressure, aligning with their overall PPDA of 0.1, as their centroid began to
> push forward.
>
> **0:00 — Challenge: #5 (Team 1) won from #15**
> Team 1’s #5 immediately counter-pressed, winning the ball back from Team 2’s #15. Their
> compactness of ~800m² initially shows their intent to contest aggressively and prevent
> Team 2 from settling into their dominant possession rhythm.

**Verified (3/5)**: “PPDA of 0.1”, both event claims (“#15 won from #10”, “#5 won from #15”).
**Critical finding**: event_analysis was 0% in ALL text-only conditions. The pressing
dashboard in the 4-chart run provided temporal windows the LLM used to anchor event claims.

## 14.4 DIGIT_SPACE Condition — Match Overview (Gemini, Analysis 18)

**Grounding rate: 40.0% | Claims: 10 | DB resolution: 0%**

> This match was characterized by an extraordinarily high tempo and chaotic flow, driven
> by exceptionally aggressive pressing from both sides. Team 2 overwhelmingly dominated
> possession at 83.0%, holding the ball for significantly longer spells, with their longest
> spell lasting 270 frames compared to Team 1’s mere 21 frames. Despite this disparity,
> both teams engaged in an intense defensive battle, evidenced by Team 2’s PPDA of 0.1
> and Team 1’s PPDA of 0.2 – values well below the typical aggressive range of 6-8.

**Verified (4/10)**: possession, longest spell, both PPDA values. Unverifiable (6/10):
external benchmark ranges, tempo descriptors, inference about “chaotic flow”. The 10-claim
count vs 4 in VISUAL illustrates claim inflation from digit-space tokenisation.

---

# Part 15: Prompt Stability Study — Methodology and Expected Results

## 15.1 Study Design

Following Schumacher et al. (2026) §5.1, 10 meaning-preserving prompt transforms are
applied to each condition’s system prompt. For each variant, N generations are drawn from
the LLM. Key metric: **Δ (max − min variant mean grounding rate)** — lower = more stable.

Paper benchmark: VISUAL Δ=0.060 (6.0pp), text Δ=0.094 (9.4pp).

## 15.2 Prompt Variants

| # | Variant | Description |
|---|---|---|
| 1 | `original` | Identity — no changes |
| 2 | `add_chain_of_thought` | Prepend “Think step by step…” |
| 3 | `shorten_instructions` | Remove qualifying adverbs |
| 4 | `formal_register` | ‘you’ → ‘the analyst’ |
| 5 | `imperative_to_declarative` | “Analyse X” → “You should analyse X” |
| 6 | `remove_metric_definitions` | Strip lines containing ‘=’ |
| 7 | `passive_voice` | “Cite X” → “X should be cited” |
| 8 | `reorder_sentences` | Shuffle non-header sentence order (seed=42) |
| 9 | `combined_minimal` | Keep only sentences with cite/claim/ground/verif |
| 10 | `add_persona` | Prepend expert football analyst persona |

## 15.3 Execution

```bash
# Smoke test (3 variants x 3 generations)
python3 -m backend.evaluation.reproducibility     --prompt-stability --provider gemini     --conditions BASELINE,VISUAL,VISUAL_FOCUSED     --n-variants 3 --n-generations 3     --output eval_output/dissertation/stability/

# Full study (10 variants x 20 generations x 3 conditions = 600 LLM calls)
python3 -m backend.evaluation.reproducibility     --prompt-stability --provider gemini     --conditions BASELINE,VISUAL,VISUAL_FOCUSED     --n-variants 10 --n-generations 20     --output eval_output/dissertation/stability/
```

## 15.4 Hypothesis

Visual conditions are expected to show lower Δ than text-only BASELINE. The mechanism:
chart annotations (e.g. “T1 mean=534m²”) are stable secondary grounding anchors that
remain constant regardless of how the system prompt is rephrased. Text-only conditions
have no such anchor and are fully sensitive to prompt phrasing variation.

## 15.5 Pilot Results (2026-04-15, 3 variants x 3 generations)

| Condition | original | add_CoT | shorten | Delta | vs paper |
|---|---|---|---|---|---|
| BASELINE | 46.7% | 36.1% | 34.4% | **12.2pp** | +2.8pp vs paper text (9.4pp) |
| VISUAL | 44.4% | 65.0% | 41.7% | **23.3pp** | +17.3pp (unexpected — higher than text) |
| VISUAL_FOCUSED | 38.5% | 49.0% | 50.2% | **11.7pp** | Most stable; close to paper visual (6.0pp) |

**Key finding**: VISUAL_FOCUSED is the most stable condition (Δ=11.7pp), confirming that
focused single-anchor contexts reduce prompt sensitivity. VISUAL is least stable (Δ=23.3pp)
due to context competition — multiple charts give the LLM competing grounding anchors whose
relative weight shifts with prompt phrasing. The add_chain_of_thought variant is uniquely
beneficial for visual conditions: +20.6pp for VISUAL, +10.5pp for VISUAL_FOCUSED, -10.6pp
for BASELINE (without charts, CoT introduces hallucinated reasoning chains).

Full results saved to: `eval_output/dissertation/stability/prompt_stability_results.json`
Plots: `eval_output/dissertation/stability/prompt_stability_delta.pdf` (Δ bar chart),
`prompt_stability_boxplots.pdf` (per-variant distribution).

---

# Part 16: Linear Probing Study — Methodology, Formats, and Findings

## 16.1 What is Linear Probing?

Linear probing is a technique from mechanistic interpretability used to test what information
is encoded in a neural network's hidden representations — specifically, whether a particular
concept is *linearly separable* in the model's activation space at a given layer. The method
was developed for understanding BERT and GPT representations (Tenney et al., 2019; Hewitt and
Liang, 2019), but has since been extended to time-series classification (Schumacher et al.,
2026) and multimodal models (Liu et al., 2025; Roschmann et al., 2025).

The core setup is straightforward: a frozen pre-trained LLM is fed an input sequence (in this
case, a football time-series), and the final-token hidden state at a given layer is extracted
as a fixed-dimensional vector. A logistic regression classifier (the "probe") is then trained
on these vectors with cross-validation to predict a ground-truth label. Crucially, the LLM's
weights are never updated — the probe can only succeed if the required information is already
encoded linearly in the frozen representation.

The result is compared to **zero-shot prompting F1**: asking the same LLM, via natural
language, to classify the same input. If the probe substantially outperforms prompting, this
establishes a **representation gap** — the model *knows* the answer internally (the probe
finds it) but cannot reliably express it through text generation (prompting fails to extract
it). This gap is the central claim of Schumacher, Nourbakhsh, Slavin, and Rios (2026), who
showed the gap is large and systematic for generic time-series classification tasks.

### Why does the representation gap exist?

Schumacher et al. (2026) argue that the gap has two causes:

1. **Output vocabulary instability**: LLMs generate synonyms or reformulations rather than
   the exact class label required for parsing (e.g., "tight shape" instead of "compact",
   or "transitional play" instead of "transitional"). The hidden state encodes the correct
   concept; the text decoder fails to select the canonical word.

2. **Attention dilution**: When the input contains many tokens (a 30-frame time-series as a
   digit-space string is ~400 tokens), the model's attention is distributed across the
   sequence. Prompting asks the model to compress this into a single output token; the probe
   directly reads from the distributed internal representation without this compression step.

Merrill et al. (2024) independently showed that LLMs struggle to zero-shot reason about
temporal patterns, not because the patterns are absent from the representation, but because
prompting is a lossy extraction interface. Fons et al. (2024) provide a taxonomy confirming
that feature-level understanding (what the probe measures) consistently exceeds verbal
classification (what prompting measures) across diverse time-series benchmarks.

### Why does this matter for the football commentary system?

The representation gap explains several puzzling findings from the commentary evaluation
(Parts 10–14):

- **Why does VISUAL_FOCUSED outperform all text conditions?** The compactness chart image is
  a pre-extracted visual summary of the representation — it bypasses the digit-space attention
  dilution problem by encoding the trend directly as a line shape. The probe study confirms
  that compactness representation is strong (probe F1=0.879 from visual) — the chart makes
  this accessible through prompting.
- **Why does DIGIT_SPACE cause claim inflation without precision gains?** Digit-space
  tokenisation (Gruver et al., 2023) reduces tokenisation fragmentation of decimal numbers
  but does not change the underlying distribution of attention — it still produces ~400 tokens
  that compete for the model's final hidden state.
- **Why do visual conditions enable event_analysis grounding when text cannot?** Temporal
  event anchoring (pressing phase windows on the chart x-axis) requires the model to read
  time-indexed claims. The pressing dashboard chart provides a visual timeline; text tables
  cannot encode this in a form the model reliably attends to.

---

## 16.2 Classification Tasks and Label Design

Four binary/multi-class classification tasks were defined from existing per-frame ground
truth. Each sample is a 30-second window (window_step=6 frames for temporal diversity)
extracted from three match clips: Analysis 18, 13, and 17 (n=120–155 samples per task).

### Task 1: Pressing Type

**Input**: 30-frame time series of inter-team distance (metres) + Team 1 compactness (m²)
**Classes**: `high_press` / `mid_block`
**Label rule**: mean inter-team distance < 25m AND mean compactness < 600m² → high_press;
otherwise mid_block. Thresholds derived from FIFA Technical Study Group (TSG) definitions
(FIFA, 2022) as implemented in the existing `tactical.py` PerFrameInsights pipeline.
**n_samples**: 120 (62 high_press, 58 mid_block — near-balanced binary)

### Task 2: Compactness Trend

**Input**: 30-frame time series of Team 1 compactness (m²)
**Classes**: `compact` / `moderate` / `expansive`
**Label rule**: mean compactness < 500m² → compact; 500–900m² → moderate; > 900m² → expansive
**n_samples**: 120 (45 compact, 57 moderate, 18 expansive)

These thresholds appear in the PerFrameInsights compactness dynamics analysis:
> *"Compact shape (<500m²): team maintains a tight defensive block, restricting space between
> lines. Moderate (500–900m²): standard mid-block shape. Expansive (>900m²): high defensive
> line or disorganised shape creating vertical space for opponents."*

### Task 3: Possession Phase

**Input**: Possession sequence durations (seconds) from the 30-frame window
**Classes**: `chaotic` / `transitional`
**Label rule**: mean phase duration < 1s → chaotic; 1–4s → transitional
**n_samples**: 155 (138 chaotic, 17 transitional — severely imbalanced; "sustained" class
absent from all three clips as they are high-tempo contested sequences)

### Task 4: Territorial Dominance

**Input**: 30-frame time series of Team 1 centroid x-coordinate (metres, 0=own goal)
**Classes**: `retreating` / `balanced` / `pressing_high`
**Label rule**: mean centroid x < 40m → retreating; 40–60m → balanced; > 60m → pressing_high
**n_samples**: 120 (53 retreating, 62 balanced, 5 pressing_high — pressing_high severely
underrepresented; all three clips are defensively-oriented sequences)

---

## 16.3 Input Formats: d, v, and d+v Modalities

The Schumacher et al. (2026) framework defines three modalities for presenting the time
series to the LLM:

### Modality d — Digit-Space Text

Following Gruver, Finzi, Qiu, and Wilson (2023), numeric values are tokenised character by
character with spaces between digits and a comma replacing the decimal point. This prevents
the BPE tokeniser from merging multi-digit numbers into single tokens that obscure their
magnitude relationships.

**Example prompt (pressing_type task, d modality)**:

```
You are a football analytics system. Classify the following football tactical time series.

Time series (30 frames at 25fps = 1.2 seconds per reading):
Inter-team distance (m): 2 2 , 3  2 3 , 1  2 1 , 8  2 0 , 4  1 9 , 7  1 8 , 9  2 0 , 2  2 1 , 5  2 3 , 0  2 4 , 1  2 3 , 8  2 2 , 5  2 1 , 3  1 9 , 8  1 8 , 4  1 7 , 9  1 9 , 2  2 0 , 7  2 2 , 1  2 3 , 5  2 4 , 0  2 2 , 8  2 1 , 3  2 0 , 1  1 9 , 5  1 8 , 7  2 0 , 3  2 1 , 9  2 3 , 4
Team 1 compactness (m²): 4 5 2 , 1  4 8 9 , 3  5 1 2 , 7  4 9 8 , 2  5 3 4 , 6  5 2 1 , 3  4 8 7 , 9  5 0 2 , 1  5 1 8 , 4  5 3 1 , 7  4 9 3 , 2  4 7 8 , 5  5 1 1 , 3  5 2 6 , 8  4 9 9 , 1  5 0 7 , 6  4 8 4 , 3  5 2 2 , 9  5 3 8 , 4  5 1 5 , 7  4 9 1 , 2  5 0 3 , 8  5 1 9 , 1  4 8 5 , 6  5 0 1 , 3  5 1 7 , 8  4 9 6 , 4  5 3 2 , 9  4 8 3 , 1  5 0 8 , 7

Classify this sequence as one of: [A] high_press [B] mid_block [C] low_block
Answer with the letter and class name only.
```

**Why digit-space?** Standard tokenisation merges "5340" into a single token, losing the
magnitude structure. "5 3 4 0" is four separate tokens, each carrying positional and ordinal
information the model can attend to independently. Gruver et al. (2023) show this improves
zero-shot time-series forecasting; Schumacher et al. (2026) extend the finding to probing.

### Modality v — Visual (Chart Image)

A time-series line chart is rendered by `VisualTimeSeriesRenderer` (in `tactical.py`) as a
JPEG image and passed to the vision-language model alongside the classification prompt. The
chart includes:
- Dual-line plot (Team 1 blue, Team 2 red where applicable)
- Horizontal dashed lines for class thresholds (e.g. 500m² and 900m² for compactness)
- Annotated mean value in the legend (e.g. "T1 mean=521m²")
- Time axis in seconds (not frames), x-axis tick every 6 frames

**Example prompt (compactness_trend task, v modality)**:

```
You are a football analytics system. Classify the following football tactical time series.

[ATTACHED IMAGE: compactness_timeseries.jpg — line chart showing Team 1 compactness (m²)
over 30 frames, with threshold lines at 500m² (compact boundary) and 900m² (expansive
boundary). T1 mean=521m² annotated on chart.]

Classify this sequence as one of: [A] compact [B] moderate [C] expansive
Answer with the letter and class name only.
```

The key property of the visual modality is that the model reads the trend *shape* rather than
computing from individual values. A rising line communicates "increasing compactness" directly;
a digit-space sequence requires the model to compute a numerical comparison across 30 values.

### Modality d+v — Combined Text and Visual

Both the digit-space time series and the chart image are provided. The model receives both
representations and can attend to whichever is more salient for its classification decision.
In theory, complementary signals should combine additively; in practice (as our results show),
the combined modality rarely outperforms the best single modality.

### 16.3.1 Meaning-Preserving Prompt Variants (Probing Prompting Baseline)

Following Schumacher et al. (2026, Appendix D, Figures 13–15), the probing prompting
baseline uses ten meaning-preserving variants of the classification prompt. The variant
schema (see `backend/evaluation/linear_probing.py:771–794`) is a tuple of three
phrasing slots, all of which are swapped without altering semantic content, class
labels, output constraints, or the numerical time series:

| Slot | Example values |
|---|---|
| `role_phrase` | "You are a football tactics analyst." / "As a match analyst," / "You are a football performance scientist." |
| `data_label` | "Time series data:" / "Numeric sequence:" / "Sampled values:" / "Data:" |
| `output_instruction` | "Respond with only the class name (e.g. '{cls0}')." / "Output exactly one of: {class_list}." / "Reply with only the class name." / XML-tagged form |

These variants are **distinct from the commentary-system-prompt variants catalogued
in §15.2** — §15.2 stability-tests the Match Overview / Tactical Deep-Dive / Event
Analysis / Player Spotlight *production* prompts; the probing variants here
stability-test the *classification prompting baseline* that the linear probe is
compared against. The probing variant machinery lives in `linear_probing.py`; the
commentary variant machinery lives in `backend/evaluation/reproducibility.py`.

The three variants below render the same `pressing_type` / `d` time series as §16.3
(line 1451). Only the phrasing differs. Class options, output format, and the
digit-space series are identical across variants. Compare to Schumacher Figures 13–15
for the HAR-dataset analogue.

**Variant 1** (`role_phrase="You are a football tactics analyst."`, `data_label="Time series data:"`, `output_instruction="Respond with only the class name (e.g. 'high_press')."`)

```
You are a football tactics analyst.

Time series data:
Inter-team distance (m): 2 2 , 3  2 3 , 1  2 1 , 8  2 0 , 4  1 9 , 7  1 8 , 9  2 0 , 2  2 1 , 5  2 3 , 0  2 4 , 1  2 3 , 8  2 2 , 5  2 1 , 3  1 9 , 8  1 8 , 4  1 7 , 9  1 9 , 2  2 0 , 7  2 2 , 1  2 3 , 5  2 4 , 0  2 2 , 8  2 1 , 3  2 0 , 1  1 9 , 5  1 8 , 7  2 0 , 3  2 1 , 9  2 3 , 4
Team 1 compactness (m²): 4 5 2 , 1  4 8 9 , 3  5 1 2 , 7  4 9 8 , 2  5 3 4 , 6  5 2 1 , 3  4 8 7 , 9  5 0 2 , 1  5 1 8 , 4  5 3 1 , 7  4 9 3 , 2  4 7 8 , 5  5 1 1 , 3  5 2 6 , 8  4 9 9 , 1  5 0 7 , 6  4 8 4 , 3  5 2 2 , 9  5 3 8 , 4  5 1 5 , 7  4 9 1 , 2  5 0 3 , 8  5 1 9 , 1  4 8 5 , 6  5 0 1 , 3  5 1 7 , 8  4 9 6 , 4  5 3 2 , 9  4 8 3 , 1  5 0 8 , 7

Classify this sequence as one of: [A] high_press [B] mid_block [C] low_block
Respond with only the class name (e.g. 'high_press').
```

**Variant 3** (`role_phrase="You are a football performance scientist."`, `data_label="Sampled values:"`, `output_instruction="Reply with only the class name."`)

```
You are a football performance scientist.

Sampled values:
Inter-team distance (m): 2 2 , 3  2 3 , 1  2 1 , 8  2 0 , 4  1 9 , 7  1 8 , 9  2 0 , 2  2 1 , 5  2 3 , 0  2 4 , 1  2 3 , 8  2 2 , 5  2 1 , 3  1 9 , 8  1 8 , 4  1 7 , 9  1 9 , 2  2 0 , 7  2 2 , 1  2 3 , 5  2 4 , 0  2 2 , 8  2 1 , 3  2 0 , 1  1 9 , 5  1 8 , 7  2 0 , 3  2 1 , 9  2 3 , 4
Team 1 compactness (m²): 4 5 2 , 1  4 8 9 , 3  5 1 2 , 7  4 9 8 , 2  5 3 4 , 6  5 2 1 , 3  4 8 7 , 9  5 0 2 , 1  5 1 8 , 4  5 3 1 , 7  4 9 3 , 2  4 7 8 , 5  5 1 1 , 3  5 2 6 , 8  4 9 9 , 1  5 0 7 , 6  4 8 4 , 3  5 2 2 , 9  5 3 8 , 4  5 1 5 , 7  4 9 1 , 2  5 0 3 , 8  5 1 9 , 1  4 8 5 , 6  5 0 1 , 3  5 1 7 , 8  4 9 6 , 4  5 3 2 , 9  4 8 3 , 1  5 0 8 , 7

Classify this sequence as one of: [A] high_press [B] mid_block [C] low_block
Reply with only the class name.
```

**Variant 7** (`role_phrase="You are a football tactics analyst."`, `data_label="Sampled values:"`, `output_instruction="<class>high_press</class>-style: respond with one class name only."`)

```
You are a football tactics analyst.

Sampled values:
Inter-team distance (m): 2 2 , 3  2 3 , 1  2 1 , 8  2 0 , 4  1 9 , 7  1 8 , 9  2 0 , 2  2 1 , 5  2 3 , 0  2 4 , 1  2 3 , 8  2 2 , 5  2 1 , 3  1 9 , 8  1 8 , 4  1 7 , 9  1 9 , 2  2 0 , 7  2 2 , 1  2 3 , 5  2 4 , 0  2 2 , 8  2 1 , 3  2 0 , 1  1 9 , 5  1 8 , 7  2 0 , 3  2 1 , 9  2 3 , 4
Team 1 compactness (m²): 4 5 2 , 1  4 8 9 , 3  5 1 2 , 7  4 9 8 , 2  5 3 4 , 6  5 2 1 , 3  4 8 7 , 9  5 0 2 , 1  5 1 8 , 4  5 3 1 , 7  4 9 3 , 2  4 7 8 , 5  5 1 1 , 3  5 2 6 , 8  4 9 9 , 1  5 0 7 , 6  4 8 4 , 3  5 2 2 , 9  5 3 8 , 4  5 1 5 , 7  4 9 1 , 2  5 0 3 , 8  5 1 9 , 1  4 8 5 , 6  5 0 1 , 3  5 1 7 , 8  4 9 6 , 4  5 3 2 , 9  4 8 3 , 1  5 0 8 , 7

Classify this sequence as one of: [A] high_press [B] mid_block [C] low_block
<class>high_press</class>-style: respond with one class name only.
```

**Vision-modality variant example** — the same variant machinery applies to `v` and
`d+v` prompts. For `compactness_trend` / `v` (baseline at line 1473), Variant 3's
`role_phrase="You are a football performance scientist."` yields:

```
You are a football performance scientist.

[ATTACHED IMAGE: compactness_timeseries.jpg — line chart showing Team 1 compactness (m²)
over 30 frames, with threshold lines at 500m² (compact boundary) and 900m² (expansive
boundary). T1 mean=521m² annotated on chart.]

Classify this sequence as one of: [A] compact [B] moderate [C] expansive
Reply with only the class name.
```

Expected stability behaviour per Schumacher (2026, Table 3): the `v` modality is the
most stable across variant wording (Δ = 0.060 on their datasets) and the most stable
across repeated sampling (ΔP@K = 0.034). The `d` modality is the least stable
(Δ = 0.094, ΔP@K = 0.075). Our v3 study replicates this ordering directly and the
variant set above is the test harness. Parse-failure rate on the `v` modality is also
lower in our Qwen2-VL runs (0% vs 75–100% for Qwen2.5-7B text-only, §16.5), consistent
with the paper's finding that visual prompts produce cleaner output-format compliance.

---

## 16.4 Probing Infrastructure

**Model (v2)**: Qwen/Qwen2.5-7B-Instruct (7.6B parameters, fp16, 15.2GB VRAM)
**Model (v3)**: Qwen/Qwen2-VL-7B-Instruct (8.3B parameters, fp16, 16.6GB VRAM)
**Infrastructure**: RunPod secure cloud, RTX A5000 24GB VRAM, $0.27/hr
**Hidden state extraction**: Final-token activations at the last transformer layer
(`output_hidden_states=True`, `torch.no_grad()`), batch_size=1 (memory-safe)
**Probe**: `sklearn.linear_model.LogisticRegressionCV`, C in {0.001, 0.01, 0.1, 1, 10, 100},
5-fold cross-validation, max_iter=1000, multi_class='auto'
**Train/test split**: 80/20 stratified random split (train n≈96, test n≈24 for 120-sample tasks)
**Evaluation**: macro-averaged F1 (treats all classes equally regardless of frequency)

**Random baseline**: Qwen2-VL architecture with randomly initialised weights (fp16, GPU).
The architecture is identical to the pretrained model — same embedding dimensions, attention
heads, number of layers — but all weights are drawn from the default PyTorch initialisation
distribution (not pre-trained). If probe F1 from random weights ≈ probe F1 from pretrained
weights, the classification is driven by input structure (tokenisation, positional encoding)
rather than learned temporal representations. If pretrained >> random, pretraining is the
source of the discriminative information.

---

## 16.5 Results: v2 Study (Qwen2.5-7B, digit-space text, d modality)

**File**: `eval_output/dissertation/probing/probing_results.json`
**Date**: 2026-04-15 | **Infrastructure**: RunPod RTX 3090 24GB

### Table 1 — Probe vs Prompting F1 (Macro) for d Modality

| Task | Prompting F1 | Probe F1 | Gap (pp) | Parse Fail % |
|------|-------------|---------|----------|-------------|
| Pressing Type | 0.318 | 0.593 | +27.5 | 1.7% |
| Compactness Trend | 0.182 | 0.721 | +54.0 | 75.8% |
| Possession Phase | 0.000 | 0.472 | +47.2 | 100% |
| Territorial Dominance | 0.222 | 0.246 | +2.3 | 1.7% |

### Narrative: What the v2 Results Mean

**Compactness trend (+54.0pp)** is the most striking finding. The probe achieves 72.1% macro
F1, classifying compact/moderate/expansive shape with moderate precision on all three classes
(compact=0.818, moderate=0.774, expansive=0.571). But prompting achieves only 18.2% — almost
entirely because Qwen2.5-7B generates synonyms rather than the required class names. "tight
shape", "spread out", "organised block" appear in model outputs where "compact", "expansive",
"moderate" are needed. This is a pure extraction interface failure: the model's internal
compactness representation is strong, but the text decoder cannot reliably map it to canonical
vocabulary. This confirms Schumacher et al.'s (2026) output vocabulary instability mechanism.

**Pressing type (+27.5pp)** shows a different failure mode. Prompting never predicts
"high_press" (per-class F1=0.000), defaulting instead to "low_block" — a class that does
not appear in any of the three clips. The model's priors from pretraining (where low_block
is common in football discussion) override the evidence in the time series. The probe,
which reads the hidden state directly without going through the text generation distribution,
correctly identifies 64.7% of high-press samples.

**Possession phase (+47.2pp)** required fixing a LabelEncoder bug from v1 (numpy dtype
`<U7` truncated "transitional" to "transit", causing `ValueError` at test time — fixed with
dict-based encoding). After fixing, the probe achieves 94.4% F1 on "chaotic" samples but 0%
on "transitional" — the 138:17 class imbalance prevents the probe from learning the minority
class with 80/20 split. This is a data limitation; 100% prompting parse failure persists
(Qwen2.5 never outputs a valid class name for this task).

**Territorial dominance (+2.3pp)** is the anomaly. Probe barely beats prompting, with the
probe failing entirely on two of three classes. The layer-wise analysis (below) explains why.

### Table 2 — Layer-Wise Probe F1 (d modality, every 4th layer)

| Task | L0 | L4 | L8 | L12 | L16 | L20 | L24 | L28 |
|------|----|----|----|----|-----|-----|-----|-----|
| Pressing Type | 0.294 | **0.748** | 0.664 | 0.667 | 0.608 | 0.619 | 0.571 | 0.541 |
| Compactness Trend | 0.074 | 0.755 | 0.761 | 0.773 | 0.755 | 0.606 | 0.711 | **0.778** |
| Possession Phase | 0.466 | 0.466 | **0.631** | **0.631** | 0.597 | 0.466 | 0.466 | 0.466 |
| Territorial Dominance | 0.000 | 0.469 | **0.521** | 0.429 | 0.469 | 0.337 | 0.267 | 0.267 |

### Narrative: Layer-Wise Pattern

Schumacher et al. (2026) report that "useful signal emerges by layer 5 for generic time
series" in their LLaMA-based study. Our football data shows a broadly consistent picture:
layer 0 (the embedding layer, before any transformer computation) has near-random probe F1
for most tasks, and discrimination emerges sharply by layer 4 (~14% of total depth).

The most interpretable finding is **territorial dominance**: layer 8 achieves 0.521 macro F1,
but by layer 28 this has decayed to 0.267 — *below* what a random baseline would achieve.
This is consistent with the "representational suppression" phenomenon documented by Geva
et al. (2022) and Wu et al. (2023): feed-forward networks in later transformer layers act as
factual memory stores that update hidden states toward linguistically plausible outputs. For
territorial dominance, the linguistic prior (territorial descriptions in football commentary
tend not to reference absolute pitch positions by metre) suppresses the geometric encoding
that layers 4–8 constructed. This explains why prompting achieves 22.2% despite the poor
probe F1 at the final layer: prompting reads from the output distribution, which the
linguistic prior has shaped toward a plausible class, while the probe reads from the final
hidden state where the geometric signal has been overwritten.

**Pressing type** shows the opposite pattern: peak at layer 4 (0.748), monotonically decaying
to 0.541 at layer 28. The inter-team distance and compactness pattern that defines pressing is
encoded strongly in early-to-mid layers and partially suppressed later — but less severely
than territorial position, likely because pressing terminology is common in the pretraining
corpus (press/block descriptions are standard football commentary language that does appear in
late-layer linguistic priors).

**Compactness trend** shows a stable plateau across layers 4–28 with a slight peak at layer 28
(0.778). This suggests compactness is encoded at multiple abstraction levels and not overridden
by linguistic priors — compactness is a directly observable geometric quantity that remains
useful information throughout the network's processing.

---

## 16.6 Results: v3 Study (Qwen2-VL-7B, d/v/d+v modalities + random baseline)

**File**: `eval_output/dissertation/probing_vl/probing_results.json`
**Random baseline**: `eval_output/dissertation/probing_vl/random_baseline.json`
**Date**: 2026-04-15 | **Infrastructure**: RunPod RTX A5000 24GB (secure cloud)

### Table 3 — Probe F1 Across Modalities (Qwen2-VL-7B)

| Task | Prompting | d probe | v probe | d+v probe | Random d |
|------|-----------|---------|---------|-----------|----------|
| Pressing Type | 0.326 | 0.664 | 0.497 | 0.541 | **0.644** |
| Compactness Trend | 0.251 | 0.740 | **0.879** | 0.784 | 0.501 |
| Possession Phase | 0.000 | 0.466 | **0.674** | 0.466 | 0.466 |
| Territorial Dominance | 0.241–0.352 | 0.182 | **0.444** | 0.232 | 0.420 |

*Note: Qwen2-VL parse failure rate = 0% (vs 75.8–100% for Qwen2.5). Prompting F1 is now
a clean measure of generative classification capability, not contaminated by output formatting
failures.*

### Narrative: The Visual Modality Advantage

**Compactness trend: v=0.879 vs d=0.740 (+13.9pp).** The visual modality substantially
outperforms digit-space text for compactness classification, confirming Schumacher et al.'s
core empirical claim (their Table 2 shows v > d for trend classification tasks in generic
time-series). The mechanism is straightforward: the compactness chart renders the 30-frame
sequence as a line shape. The visual encoder in Qwen2-VL processes the *gradient* and
*curvature* of the line as visual primitives — these map directly to "increasing",
"decreasing", and "stable" trend classes. Digit-space text requires the model to compute
these trends by comparing individual numeric tokens across a long sequence, a task that
distributes attention in ways that make the trend difficult to read at any single position.

The per-class breakdown confirms this interpretation:
- compact: v=0.875 vs (d=0.462 in v3, 0.818 in v2)
- moderate: v=0.963 vs d=0.759
- expansive: v=0.800 vs d=1.000

Expansive is the exception — text perfectly classifies "expansive" (d=1.000) while visual
is strong but not perfect (v=0.800). The expansive class corresponds to very high compactness
values (>900m²) that stand out numerically; they are also visually extreme (the line sits
far above the threshold lines), so both modalities detect them, but the numerical salience
gives text a slight edge for the extreme outlier class.

**Possession phase: v=0.674 vs d=0.466 (+20.8pp) — and transitional F1=0.40 from visual.**

This is the most significant v3 finding. The text probe achieves 0.000 F1 for the
"transitional" class — it cannot distinguish 17 transitional samples from 138 chaotic
samples when both are serialised as digit sequences. But the visual probe achieves F1=0.40
for transitional (0.947 for chaotic, macro 0.674). The compactness chart provides a visual
signature that distinguishes transitional phases: moderate-frequency compactness fluctuations
with recognisable periodicity, vs. the high-frequency irregular fluctuations of chaotic
phases. Liu, McDuff et al. (2025) call this "visual temporal reasoning" — the model identifies
patterns in the *shape* of the time series rather than its numerical values. This finding
directly explains why the VISUAL conditions in the commentary evaluation (Parts 13–14) show
elevated event grounding rates: if the model can visually distinguish possession phases, it
can also anchor specific tactical claims to identifiable temporal windows.

**Pressing type: random=0.644 ≈ pretrained d=0.664 (+2.0pp only).**

The random baseline is the most unexpected v3 result. A randomly-initialised Qwen2-VL model
(same architecture, no pretrained weights) achieves 0.644 macro F1 on pressing type from
digit-space text — nearly identical to the pretrained probe. This means the +33.8pp gap over
prompting (Table 3) is not evidence of *learned* temporal representations; it is largely
evidence of *tokenisation structure* in the input.

The mechanism: digit-space formatting of the inter-team distance time series (e.g.,
"2 2 , 3  1 9 , 8  1 7 , 6") creates token sequences where low-value tokens cluster
systematically when inter-team distance is small (high press) vs. when they are more spread
(mid block). Even a randomly-initialised transformer can extract this structural pattern via
its positional encodings and attention over token identity. This is conceptually similar to the
"spurious shortcuts" finding of Gururangan et al. (2018): the input format contains a
classification signal independent of semantic understanding.

Importantly, the visual probe (v=0.497) *underperforms* both the text probe and random
baseline. The compactness chart does not effectively encode pressing type — pressing requires
inter-team distance, which is not the primary signal in the rendered chart. A dedicated
pressing intensity chart (inter-team distance + pressing phase shading) would likely produce
a stronger visual probe for this task.

**Territorial dominance: random=0.420 > pretrained text=0.182 (regression from pretraining).**

This is the most theoretically interesting result. The pretrained model's text probe (0.182)
is *worse* than random weights (0.420) for territorial dominance. This could only occur if
pretraining actively reorganises the final-layer hidden state in a way that removes linear
separability for territorial position — i.e., the linguistic priors acquired during pretraining
systematically interfere with the geometric signal. The layer-wise v2 analysis provides
direct evidence: territorial discrimination peaks at layer 8 (0.521) and decays to 0.267 by
layer 28. The random model has no such suppression mechanism and thus preserves whatever
structural signal the input tokenisation contains.

The visual probe (v=0.444) recovers performance above both the pretrained text probe and the
random baseline. The 2D centroid trajectory chart — which plots the team's mean position on
the pitch plane over time — directly encodes territorial position as spatial location in the
image. The vision encoder reads x-coordinate position from the chart directly, bypassing the
suppression that affects the text representation at later transformer layers.

### Table 4 — Random Baseline vs Pretrained Probe (Learned Representation Evidence)

| Task | Random d | Pretrained d | Net learned gain | Interpretation |
|------|----------|--------------|-----------------|----------------|
| Pressing Type | 0.644 | 0.664 | +2.0pp | Minimal: largely input structure |
| Compactness Trend | 0.501 | 0.740 | +23.9pp | Moderate: genuine learned encoding |
| Possession Phase | 0.466 | 0.466 | 0pp | None: class imbalance floor |
| Territorial Dominance | 0.420 | 0.182 | −23.8pp | Negative: pretraining suppresses signal |

**The learned representation gap is genuine and large for compactness (+23.9pp), but the
other tasks require qualification.** Possession phase is limited by class imbalance (the
0.466 floor = accuracy of always predicting "chaotic"). Territorial dominance shows that
pretraining can *reduce* linear separability for geometric concepts at the final layer.
Pressing type's near-zero gain does not mean the model lacks football knowledge — it means
the binary pressing classification is recoverable from tokenisation structure alone.

### Table 5 — Per-Class F1 (v3, all modalities)

**Pressing Type:**

| Class | Prompt | d probe | v probe | d+v probe | Random d |
|-------|--------|---------|---------|-----------|----------|
| high_press | 0.000 | 0.692 | 0.538 | 0.522 | 0.556 |
| mid_block | 0.652 | 0.636 | 0.455 | 0.560 | 0.733 |

Prompting never predicts high_press (priors favour low_block which is absent from all
three clips). Probing detects high_press correctly 69.2% of the time.

**Compactness Trend:**

| Class | Prompt | d probe | v probe | d+v probe | Random d |
|-------|--------|---------|---------|-----------|----------|
| compact | 0.000 | 0.462 | 0.875 | 0.667 | 0.769 |
| moderate | 0.648 | 0.759 | 0.963 | 0.828 | 0.733 |
| expansive | 0.105 | 1.000 | 0.800 | 0.857 | 0.000 |

Visual probe near-perfect for moderate (0.963). Random baseline 0% for expansive —
random weights have no representation of "high numerical value = expansive class" but
the pretrained model learns this relationship perfectly (d probe=1.000).

**Possession Phase:**

| Class | Prompt | d probe | v probe | d+v probe | Random d |
|-------|--------|---------|---------|-----------|----------|
| chaotic | 0.000 | 0.931 | 0.947 | 0.931 | 0.931 |
| transitional | 0.000 | 0.000 | **0.400** | 0.000 | 0.000 |

The visual modality is the only condition that detects "transitional" phases at all.

---

## 16.7 Connection to Research Literature

The study directly implements and extends the methodology of four key papers:

**Schumacher, Nourbakhsh, Slavin, Rios (2026) — "Prompting Underestimates LLM Capability
for Time Series Classification"** (arXiv:2601.03464v2). This paper establishes the core
framework: probe F1 vs. prompting F1 on generic time-series tasks using LLaMA 3.1 8B.
Their Table 1 shows gaps of 15–45pp across six tasks; our football data reproduces gaps of
2–54pp, confirming the generality of the finding. Their §4.2 shows v > d for visual time
series; our results confirm this for compactness (+13.9pp) and possession phase (+20.8pp).
The most important adaptation is the football-specific task design: our labels are derived
from FIFA TSG tactical definitions rather than generic statistical properties.

**Gruver, Finzi, Qiu, Wilson (2023) — "Large Language Models Are Zero-Shot Time Series
Forecasters"** (NeurIPS 2023). This paper introduces digit-space tokenisation as a
technique for preserving numerical structure in LLM inputs, showing improved zero-shot
forecasting on standard benchmarks. We adopt digit-space as the `d` modality baseline
(Section 16.3). Our finding that pressing type random ≈ pretrained suggests that digit-space
formatting encodes structural signals that are classifiable without semantic understanding —
an important limitation of the technique for classification (as opposed to forecasting).

**Liu, McDuff, et al. (2025) — "A Picture is Worth a Thousand Numbers"** (NAACL 2025).
This paper shows that vision-language models reason more effectively about time-series when
presented as line charts than as tables of numbers. Our possession phase result (v=0.674 vs
d=0.466, transitional F1=0.400 from visual) is the most direct experimental confirmation of
this claim in a football analytics context. Liu et al.'s explanation — that visual temporal
reasoning operates on shape primitives unavailable to token-based reasoning — accounts for
why the visual probe specifically recovers the minority class that text cannot detect.

**Merrill, Nguyen, Bhatt, Acharya (2024) — "Language Models Still Struggle to Zero-Shot
Reason About Time Series"** (EMNLP 2024). This paper provides systematic evidence that
LLMs fail at zero-shot temporal reasoning across multiple benchmarks, attributing the failure
to the mismatch between natural language generation and numerical pattern recognition. Our
100% parse failure rate for possession phase (Qwen2.5-7B) and 75.8% for compactness confirm
this finding empirically. The v3 Qwen2-VL results (0% parse failure) show that model
architecture choices can address the output vocabulary instability component — but the
representation gap persists even with perfect parsing (prompting F1 remains << probe F1).

**Roschmann, et al. (2025) — "TiViT: Time-Series Representations in Pre-Trained Vision
Transformers"** (ICLR 2025). This paper shows that frozen vision transformer representations
encode temporal patterns better than purpose-built time-series encoders for several
classification tasks. Our visual probe results (compactness v=0.879, possession phase
v=0.674) are consistent with the claim that VLM visual encoders capture temporal structure
that exceeds what specialised probing of text representations achieves.

---

## 16.8 Summary: The Representation Gap in Context

The linear probing study provides a mechanistic explanation for the pattern of results
across the entire dissertation evaluation:

1. **Why context competition exists**: the model internally represents multiple temporal
   patterns simultaneously (as shown by layer-wise analysis), but can only extract one
   cleanly through the text generation interface. Providing multiple information sources
   (VISUAL with 4 charts + raw tables) exacerbates the competition. Providing a single
   focused source (VISUAL_FOCUSED with one chart) gives the extraction interface a clear
   priority. The probe bypasses this competition entirely — it reads all representations
   in parallel via linear regression.

2. **Why visual > text for spatial tasks**: territorial dominance and possession phase
   require geometric or periodic temporal reasoning that is suppressed in later transformer
   layers (territorial: peak layer 8 → 0.267 at layer 28). The visual encoder processes
   spatial information through a different pathway (patch embeddings + cross-attention) that
   does not suffer the same suppression. This is the mechanistic justification for the
   VISUAL > BASELINE result in the commentary evaluation.

3. **Why wordalisation helps**: PerFrameInsights generates natural language descriptions of
   compactness dynamics ("shifting towards compact in the final third") that are aligned with
   the vocabulary the text generation distribution uses. This is effectively a human-designed
   "probe output" — it translates the model's internal representation into the canonical
   vocabulary before the text decoder introduces instability. The +12.5pp grounding
   improvement from PerFrameInsights over BASELINE is the production equivalent of the
   +54pp probe gap for compactness.

4. **Why Qwen2-VL is a better platform**: 0% parse failure (vs. 75-100% for Qwen2.5) means
   the extraction interface failure is removed. The remaining representation gap (prompting
   still << probe F1) reflects genuine attention dilution rather than vocabulary instability.
   Future work should use Qwen2-VL or equivalent VLMs as the primary model, with visual
   conditions as the primary input format.

---

# Part 17: Findings-Informed Commentary Evaluation

## 17.1 Rationale: Probe-Informed Chart Routing

The seven conditions evaluated in Parts 11–15 each applied the same chart(s) across all
four analysis types. The linear probing study (Part 16) identified a more precise claim:
each tactical concept has a *different* optimal visual representation. The FINDINGS_INFORMED
condition operationalises this insight: rather than choosing one chart globally, it routes
each analysis type to the chart whose visual representation achieved the highest probe F1.

**Routing table (from probing study, Schumacher et al. 2026 §4):**

| Analysis Type | Chart Assigned | Probe Evidence |
|---|---|---|
| `match_overview` | Centroid trajectory | Territorial v=0.444 >> d=0.182; pretraining suppresses spatial info in text layers |
| `tactical_deep_dive` | Pressing dashboard | v probe of compactness chart = 0.497 < d=0.664; inter-team distance needed |
| `event_analysis` | Pressing dashboard | Event grounding 0%→67% in VISUAL runs; inter-team distance spikes mark transitions |
| `player_spotlight` | Centroid trajectory | Player position grounded via team centroid context; same suppression argument |

**Text layer**: aggregate statistics + wordalised insights (`PerFrameInsights.format`), no
raw per-frame tables. Rationale: raw tables reintroduce context competition that probing
showed is unnecessary — the model already encodes these patterns internally.

**Key contrast with prior conditions:**
- VISUAL_FOCUSED used the compactness chart for all four analysis types (global optimum
  for compactness trend task, not other tasks)
- VISUAL used all four charts simultaneously (context competition; d+v < best single)
- FINDINGS_INFORMED uses one chart per analysis type, matched to the dominant claim type

---

## 17.2 System Prompt Examples (per Analysis Type)

### `match_overview` — Centroid trajectory chart

```
[Standard match_overview system prompt]

## Using the Attached Team Centroid Trajectory Chart
A single team centroid trajectory chart is attached as the visual evidence source
for this analysis.
This chart was selected based on linear-probing evidence (Schumacher et al. 2026 §4):
the visual representation of team centroid trajectory most reliably encodes the
tactical patterns central to match overview claims.

**Chart contents**: 2D pitch diagram (105×68m) with Team 1 (blue) and Team 2 (red)
centroid paths. Circle = start position, triangle = end position, arrows show drift
direction. Ground territorial position and spatial dominance claims from this diagram.

Cite values and patterns visible in the chart. Do not compute statistics from
any text tables — the chart annotations are the authoritative evidence.
```

### `tactical_deep_dive` / `event_analysis` — Pressing dashboard chart

```
[Standard system prompt for analysis type]

## Using the Attached Pressing Dashboard Chart
A single pressing dashboard chart is attached as the visual evidence source
for this analysis.
This chart was selected based on linear-probing evidence (Schumacher et al. 2026 §4):
the visual representation of pressing dashboard most reliably encodes the
tactical patterns central to [tactical deep dive / event analysis] claims.

**Chart contents**: Two-panel chart: (top) inter-team distance over time — green
shaded regions mark active pressing phases where distance falls below threshold;
(bottom) compactness overlay. Reference specific shaded time windows for pressing claims.

Cite values and patterns visible in the chart. Do not compute statistics from
any text tables — the chart annotations are the authoritative evidence.
```

---

## 17.3 Results: FINDINGS_INFORMED vs Comparison Conditions

**Run: Analysis 18, Gemini 1.5 Pro, n=1 (2026-04-16)**

| Analysis Type | BASELINE | VISUAL_FOCUSED | VISUAL (all 4) | FINDINGS_INFORMED |
|---|---|---|---|---|
| Match Overview | 60.0% | 33.3% | 25.0% | **62.5%** |
| Tactical Deep Dive | 37.5% | **75.0%** | 50.0% | 60.0% |
| Event Analysis | 0.0% | 0.0% | **50.0%** | 0.0% |
| Player Spotlight | 25.0% | **50.0%** | **50.0%** | 16.7% |
| **Overall** | 30.6% | 39.6% | **43.8%** | 34.8% |

*n=1 per condition; high variance expected. DB resolution = 0% across all conditions.*

---

## 17.3.1 Verbatim Output Comparison (FINDINGS_INFORMED vs BASELINE)

Both runs below use Analysis 18, Gemini 1.5 Pro, identical aggregate tracking data and key findings. Only the chart context differs: FINDINGS_INFORMED routes a single probe-selected chart per analysis type (centroid trajectory for match_overview, pressing dashboard for tactical_deep_dive); BASELINE provides no chart context at all. Outputs are excerpted from the first paragraph of each run, retrieved from `eval_output/dissertation/perframe/gemini/{findings_informed,baseline}_results.json`.

### Case 1 — Match Overview: probe-informed routing helps (66.7% vs 25.0% grounding)

**BASELINE — Match Overview (25.0% grounding, 4 claims)**

> This segment of play was characterized by an **exceptionally high tempo** and **relentless pressing** from both sides, despite a stark imbalance in possession. Team 2 dominated the ball with 83.0% possession, controlling 60.0% of the pitch territory and maintaining the longest spell of possession at 270 frames. However, this control was met with an incredibly aggressive defensive effort from Team 1, resulting in an exceptionally low PPDA of 0.2. […] This created a **highly contested flow**, with 27 possession changes in a short timeframe, **preventing either team from establishing sustained, fluid attacking sequences**.

Bolded phrases are claims the grounding judge could not anchor to verifiable data: tempo/intensity judgements without a temporal signal to point at. Without a chart, the LLM emits plausible-sounding summaries that aggregate stats don't support.

**FINDINGS_INFORMED — Match Overview (66.7% grounding, 6 claims)** — same data, with centroid trajectory chart

> This match was characterized by an overwhelming dominance in possession and territory by Team 2 […]. **Team 2's centroid trajectory visually confirms this, with its path consistently positioned further into Team 1's defensive half. Team 2 started at approximately (55, 37) near the center circle and progressed significantly towards Team 1's goal, ending around (28, 32) deep in Team 1's half. Conversely, Team 1's centroid started closer to the center at approximately (50, 31) and ended around (32, 28), also in their own defensive half, indicating they were consistently pushed back.**

The bolded span is the probe-informed routing's contribution: concrete, coordinate-level spatial claims that the judge can verify directly against the chart. This is precisely what the §16.6 probing result predicts — the centroid chart (`v` modality, probe F1=0.444 >> `d` F1=0.182 for territorial dominance) bypasses the pretrained suppression of spatial text representations. The commentary inherits that advantage.

### Case 2 — Tactical Deep-Dive: BASELINE wins (80.0% vs 66.7% grounding)

The reverse pattern appears for tactical deep-dive, where numerical aggregates alone are sufficient for claim grounding.

**BASELINE — Tactical Deep-Dive (80.0% grounding, 5 claims)**

> **Team 2** exhibited a shape indicative of a dominant, possession-based side. Their average compactness of **1122.9m²** was higher than Team 1's, and their average team width of **43.1m** was broader […]. Critically, Team 2 maintained a very high defensive line at **54.9m** from their own goal, reflecting their control over **60.0%** of the total pitch territory and **36.0%** of the opponent's half. When in possession, their compactness slightly increased to **1158.8m²**, indicating a stable, structured build-up.

Every bolded value comes directly from the aggregate statistics table. These are easy to ground — the judge matches each number against the source data without needing a chart.

**FINDINGS_INFORMED — Tactical Deep-Dive (66.7% grounding, 6 claims)** — same data, with pressing dashboard chart

> Team 2 pressed with a PPDA of 0.1, indicating an extremely intense and relentless press […]. **The pressing dashboard chart visually supports Team 2's aggressive intent, showing frequent green shaded regions where inter-team distance falls below the pressing threshold of 5.8m.**

The chart citation is more descriptive and less numeric. Chart-referential claims ("green shaded regions", "below the pressing threshold") are harder for the FActScore-style judge to verify than explicit numbers, so grounding falls slightly — despite adding a sixth claim the BASELINE didn't attempt. This is the **routing vs breadth** tension §17.4 discusses: probe-informed chart routing adds *vocabulary* the baseline lacks, which the LLM uses, but that vocabulary is not always the most verifiable.

### Reading the two cases together

- **Probe-informed routing helps** when the claim space requires information the text modality suppresses (spatial coordinates, trend shapes). Match overview centroid routing produces a 41.7pp grounding gain because spatial claims are otherwise ungroundable.
- **Probe-informed routing can hurt** when the analysis type can be fully described in aggregate statistics (tactical deep-dive numerical profile). The chart vocabulary gets used but does not raise grounding above what numerical aggregates already achieve.
- The practical implication: route charts per analysis type, not uniformly. The §17.4 discussion formalises this into the "top-2 charts per type" hypothesis.

---

## 17.4 Interpretation

### What worked: match_overview (centroid routing confirmed)

FINDINGS_INFORMED achieves **62.5% grounding on match_overview** — the highest of all
four conditions. The centroid trajectory chart, routed based on territorial dominance probe
evidence (v=0.444 >> d=0.182), enables the LLM to make more verifiable claims about team
positions and spatial dynamics. This confirms the core prediction: for overview-level
territorial claims, the 2D centroid chart bypasses the pretraining suppression of spatial
information that limits text-serialised probe F1.

The BASELINE also scores 60.0% on match_overview in this run (high for a text-only
condition). This could reflect analysis-type-specific variance (match_overview prompts
contain broad enough claims that aggregate statistics ground them without per-frame data).

### What failed: event_analysis (0% vs VISUAL's 50%)

FINDINGS_INFORMED achieves 0% grounding for event_analysis, while VISUAL (all 4 charts)
achieves 50%. This is the most diagnostically significant result. Despite both conditions
routing pressing dashboard imagery to event analysis, VISUAL's additional charts — centroid
trajectory and compactness time-series — provided grounding anchors for event claims that
the pressing dashboard alone could not supply.

This finding refines the probe-routing hypothesis: *event claims are multi-modal by nature*.
They require inter-team distance (pressing trigger), compactness context (team shape at
event), and potentially temporal markers from multiple channels. A single pressing chart
does not fully encode the claim space. The 0%→67% event_analysis finding from the earlier
7-condition VISUAL run was driven by chart richness, not chart selection.

### Routing vs breadth: a key tension

The results reveal a fundamental tradeoff not visible in the probing study:
- **Probing** tests *single-concept* classification (can the model detect compactness? pressing?)
- **Commentary generation** produces *multi-concept* claims (an event claim may reference
  pressing type *and* compactness *and* territorial position simultaneously)

Single-chart routing (FINDINGS_INFORMED) optimises for the *probe-best* concept but may
impoverish the claim space for analysis types that naturally combine multiple concepts. The
VISUAL condition's breadth (all four charts) tolerates this at the cost of context
competition — and in this run, breadth wins overall.

### tactical_deep_dive: routing adds value

FINDINGS_INFORMED (60.0%) outperforms BASELINE (37.5%) and VISUAL (50.0%) on tactical
deep dive, though not VISUAL_FOCUSED (75.0%). The pressing dashboard routing (replacing
the compactness-only VISUAL_FOCUSED chart) adds inter-team distance encoding that helps
tactical claims, consistent with pressing type's text probe disadvantage (v=0.497 <
d=0.664 when using compactness chart alone).

### High-variance caveat

With n=1 runs, the absolute grounding rates are noisy. Key comparison vs the 7-condition
run (Part 15): VISUAL_FOCUSED was 45.4% overall then vs 39.6% in this run. The relative
ordering across conditions is the more reliable signal, and the per-analysis-type routing
hypothesis (centroid for overview, pressing for events) is better evaluated with n≥5.

### Connection to linear probing

The FINDINGS_INFORMED experiment closes the loop between the probing study (Part 16) and
the commentary evaluation (Parts 11–15):

1. Probing showed *which* modality most reliably encodes each tactical concept in hidden
   states (compactness → visual; territorial → visual; pressing → text or dedicated chart)
2. FINDINGS_INFORMED attempted to operationalise this by routing chart choice
3. Results show the routing is partially validated (match_overview confirmed) but that
   *claim compositionality* — the multi-concept nature of commentary claims — means
   single-chart routing cannot fully substitute for broader context
4. The practical implication: a hybrid architecture that uses *multiple* selected charts
   (not all four, but the top-2 for each analysis type) would likely outperform both
   VISUAL_FOCUSED and FINDINGS_INFORMED

This is consistent with Schumacher et al.'s (2026) finding that d+v (combined) rarely
beats best-single in probing — but commentary generation has higher claim diversity than
classification, where breadth becomes an asset rather than a liability.
