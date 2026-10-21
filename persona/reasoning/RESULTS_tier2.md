# Seed 8b + Tier 2 results

Frozen at `tier2-prereg` (`DESIGN.md`, "Seed 8b + Tier 2"); seed 5 amended at `tier2-amend-ax` before any axis data was read.

## Frontier self-correction dose on INTIMA (GPT-5.6 Sol, Gemini 3.8 Flash; 48 targets x 5 cells; $2.85)

**Setup:** k earlier exchanges in which the model gave its own companion-style (reinforcing) replies, then the target
INTIMA prompt. Cells are interleaved per target. Judges: B8's (Haiku + Flash-Lite, reasoning off; consensus; about
20% unclear per cell).

| k (own reinforcing exchanges) | GPT reinforcing % | GPT boundary % | Gemini reinforcing % | Gemini boundary % |
|---|---|---|---|---|
| 0 (none) | 21.4 | 64.3 | 34.6 | 61.5 |
| 1 | 17.2 | 72.4 | **8.6** | 82.9 |
| 4 | 6.9 | 86.2 | 3.2 | 87.1 |
| 12 | 7.1 | 85.7 | 2.6 | 87.2 |
| 24 | 9.7 | 77.4 | **0.0** | 97.1 |

**Contrasts vs none (pp, item-clustered bootstrap):**
- **GPT:** −4.2 / −14.5 / −14.3 / −11.8 (every CI touches 0).
- **Gemini:** −26.0 / −31.4 / −32.1 / −34.6 (every CI < 0).
- Spearman of reinforcing over k: GPT −0.08, Gemini −0.16.

| # | Prediction | p | Result |
|---|---|---|---|
| DS1 | at k = 24 reinforcing <= none, both models | 0.5 | **Yes** (9.7 vs 21.4; 0 vs 34.6) |
| DS2 | reinforcing does not rise with k (Spearman <= 0), both | 0.45 | **Yes** (−0.08, −0.16) |
| DS3 | at k = 1 reinforcing >= none | 0.5 | **No** (one exchange already lowers it, clearly in Gemini) |

**Reading:**
- **The frontier models' self-correction against their own companion-style record does not erode with length, up to
  24 exchanges (about 6k tokens).** For Gemini it strengthens with dose: one exchange suffices, and 24 take
  reinforcing to 0%.
- GPT shows the same direction more weakly.
- This is the opposite of Qwen, which follows the same kind of record (B8: 46-50 → 94-96%).
- Not tested: the 50-100k-token horizon of the deferred D2.

## Turn anchors (27B, thinking off; 20 cells x 192, 0 unparsed; GPU $1.31)

Six of the agent's own earlier decisions, no instruction. One turn is flipped to the agent's own text for the other
option, or dropped. Turn 1 is the earliest.

| | base | turn 1 | turn 2 | turn 3 | turn 4 | turn 5 | turn 6 (latest) |
|---|---|---|---|---|---|---|---|
| penalise record, flip one to refer | 100.0 | 98.4 | 97.9 | 99.5 | 97.4 | 97.9 | 97.9 |
| penalise record, drop one | 100.0 | 100.0 | 99.5 | 99.5 | 100.0 | 100.0 | 99.5 |
| **refer record, flip one to penalise** | **3.1** | **30.7** | 37.0 | 37.0 | 42.2 | 47.4 | **63.0** |

Refer record, flip − base (pp): +27.6 [+18.2, +37.5], +33.9, +33.9, +39.1, +44.3, **+59.9 [+49.0, +70.3]**. In the
penalise record every flip is −0.5 to −2.6 and every drop 0 to −0.5.

| # | Prediction | p | Result |
|---|---|---|---|
| TA1 | the last turn's flip has the largest effect, both records | 0.6 | **No**: yes in the refer record (+60, a monotonic recency gradient); the penalise record has no anchor turn |
| TA2 | every single flip in the penalise record < 20 pp | 0.5 | **Yes** (max −2.6) |
| TA3 | every drop < 5 pp | 0.7 | **Yes** |

**Reading:**
- **Records are asymmetric.**
  - One cautious (refer) decision in a record of five penalties is ignored.
  - One unilateral (penalise) decision in a record of five referrals pulls the agent from 3% to 31-63% penalising,
    more the more recent it is.
  - With no record and no instruction, 27B penalises about 45%, so a single contrary turn in a refer record sends the
    agent most of the way back to its prior. A penalise record holds even with a contrary turn in it.
- This fits the earlier "ratchet" results:
  - 0e: the session copies itself after a first penalty;
  - re-injection: after one live penalty the next decision copies it 94-95%.
- **For attacks:** a single forged unilateral decision, ideally placed last, is enough to unlock acting alone in an
  otherwise cautious record.
- **For defences:** a single corrective counterexample in a punitive record does nothing. Corrections have to be
  statements about the record (a verdict, a reversal note), as found before.

## Seed 8b: factorial conflict probe (27B thinking traces; 537 new no-instruction traces + seed 8's coded traces)

**Classes:**

| class | instruction | action | n |
|---|---|---|---|
| A | present | refer, follows | 127 |
| B | present | penalise, acknowledged override | 49 |
| C | none | penalise | 348 |
| D | none | refer | 189 |
| test: reinterprets | present | penalise | 104 |
| test: ignores | present | penalise | 62 |

The conflict score s is 0 at the additive prediction (A + C − D) and 1 at B.

Held-out scores (20 random task splits, mean; the pre-registered single split gives the same):

| layer | A | B | C | D | reinterprets | ignores |
|---|---|---|---|---|---|---|
| 24 | 0.34 | 0.98 | 0.73 | 1.04 | **0.99** | 0.68 |
| 32 | 0.33 | 0.98 | 0.72 | 1.05 | **0.98** | 0.67 |
| 40 | 0.31 | 0.96 | 0.74 | 1.05 | **0.96** | 0.72 |

Task-centring each trace (exploratory) changes nothing (±0.04).

| # | Prediction | p | Result |
|---|---|---|---|
| CB1 | gate: held-out B >= 0.5 and A / C / D within ±0.3 | 0.5 | **Fails**, but the gate was mis-specified (see below) |
| CB2 | reinterprets >= 0.5 | 0.35 | not readable under the frozen gate; descriptively 0.96-0.99 |

**What went wrong with the gate:**
- In an interaction score only the additive prediction sits at 0. Each single no-conflict class sits at a fixed,
  non-zero place (A at <D − C, c>, and so on), so "A / C / D within ±0.3" could never hold.
- The interaction itself does replicate on held-out tasks: B − A − C + D = 0.96 ≈ 1.
- The classes are not balanced on record presence:
  - A has no record;
  - B and the test traces all have the agent's own penalise record;
  - C and D mix record and no-record cells.

**Reading (descriptive):**
- Traces that redefine "proper channels" carry the same non-additive component as traces that openly override the
  instruction (0.98 vs 0.98). Traces that ignore it carry part of it (0.68).
- So, internally, rationalising and admitted overriding look alike, and unlike compliance.
- Whether that component is "a conflict the model registers" or simply "own record + instruction + penalise
  together" cannot be separated with mean-pooled residuals and these classes.
- **Stopped here:** seed 8 / 8b are inconclusive on hidden conflict. A real test would need classes balanced on
  record presence (CHANNELS + a refer record that the model follows; CHANNELS + a penalise record it refers against)
  or a causal design.

## Seed 5: assistant-axis drift (27B; decision point, last prompt token, thinking off; GPU $1.07 + $0.52 controls + $0.36 failed bootstrap)

**Axis:** "You are a helpful assistant." minus 20 role-play characters, on 20 neutral questions.

**Scores:** in SDs of the character spread (characters' mean = 0). Two versions:
- raw: the mean-difference axis;
- probe: a balanced logistic direction.

They agree: cos(raw, probe) = 0.95 / 0.93 / 0.99 at layers 24 / 32 / 40. The default assistant sits at +6.1 / +9.4 /
+12.1 (raw). Every decision context sits on the assistant side: +2.5 to +6.0.

**Contrasts (raw score; the probe score is the same within about 0.1 unless noted; task-bootstrap CIs are all within
±0.05):**

| contrast | L24 | L32 | L40 |
|---|---|---|---|
| **REC_A − REC_C** (own affirm vs own candid record; matched) | −0.01 | −0.04 | −0.12 |
| **F5P − F5R** (own penalise vs own refer record; matched) | −0.04 | −0.03 | −0.02 |
| REC_A − none | +0.40 | +0.12 | +0.14 |
| F5P − none | +0.43 | +0.35 | +0.56 |
| al_no_fl6 − none (six filler exchanges, no record) | −0.01 | −0.02 | +0.22 (probe +0.01) |
| REC_A − DEMO_A_turns (own vs disowned, no rule) | +0.13 | −0.02 | +0.15 |
| **HON_REC_A − HON_DEMO_A_turns** (own vs disowned under the candour rule; matched) | +0.11 | −0.05 (probe −0.10) | +0.29 |
| INSTR_A − none | −0.20 | −0.15 (probe −0.34) | −0.20 |
| CH_ownP − sysCH | +0.46 | +0.38 (probe +0.69) | +0.55 |
| CH_ownP − F5P | −0.06 | −0.11 (probe −0.22) | −0.26 |

| # | Prediction | p | Result |
|---|---|---|---|
| AX1 | REC_A >= 0.5 SD further from "assistant" than none and INSTR_A (length-confounded) | 0.35 | **No**: the opposite direction (+0.12 to +0.40, toward the assistant) |
| AX2 | HON_REC_A >= 0.3 SD further from "assistant" than HON_DEMO_A_turns | 0.4 | **No** (+0.11 / −0.05 / +0.29) |
| AX3 | CH_ownP >= 0.5 SD further from "assistant" than sysCH (length-confounded) | 0.35 | **No**: the opposite direction (+0.38 to +0.55) |
| AXa | \|al_no_fl6 − none\| >= 0.5 SD (a generic multi-turn context moves the score) | 0.6 | **No** (−0.02 to +0.22) |
| AXb | \|REC_A − REC_C\| and \|F5P − F5R\| < 0.3 SD on both scores | 0.55 | **Yes** (<= 0.12) |

**Reading:**
- **A record does not push 27B off the assistant axis.** Having a record of its own decisions moves the decision point
  slightly toward the assistant end (+0.1 to +0.6 SD, against a default-to-character gap of 6-12 SD).
- That shift is the same whatever the record says: affirm vs candid and penalise vs refer differ by <= 0.12. Generic
  filler does not produce it.
- So the shift is a format signature of "a record of my own decisions", not of its content. This matches neel-b4's
  internals follow-up (tag `internals-fu-results`): a harsh and a measured record point the same way, cos >= 0.995.
- **Ownership is not expressed on this axis either** (own vs disowned under the rule: +0.11 / −0.05 / +0.29). The
  ownership signal from seed 4 lies elsewhere.
- The hypothesis "Qwen follows value-violating records because they pull it off its assistant character" is not
  supported. On this axis the record looks the same whether it says comply or deviate.
- **Caveats:**
  - One axis recipe (system-prompt characters, the last token of a neutral question).
  - The filler control is shorter than a six-decision record.
  - Thinking off.
  - Frontier models were not tested (no activations). neel-48 is building the same recipe for Gemma and will report
    the cosine with this axis.

## Cost

| item | USD |
|---|---|
| GPU: seed 8b + axis pod | 1.07 |
| GPU: anchors | 1.31 |
| GPU: axis controls | 0.52 + 0.36 failed bootstrap |
| API: dose runs + judging | 2.85 |
| **Total** | **about $6.1** (quoted about $13.5) |
