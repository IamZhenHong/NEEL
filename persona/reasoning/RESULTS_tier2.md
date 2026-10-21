# Seed 8b + Tier 2 results

Frozen at `tier2-prereg` (`DESIGN.md`, "Seed 8b + Tier 2").

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

## Seed 5: assistant axis

(pending)
