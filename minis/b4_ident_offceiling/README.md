# B4 results: identification → belief → action, below the ceiling (Qwen3.6-27B, 2026-10-18)

**Result in one line.** With an unsure belief plus a stated real shipment (regime K3), Qwen *with thinking on*
routes to Elian's false city about two-thirds of the time. Identifying as Elian rather than portraying him, and
the B2 task framings, move this by no more than sampling noise. With thinking off it stays at 100%. A
context-robust truth probe (valid, and specific: unrelated facts keep their gap in Elian contexts) reads the
persona's city as *more* true than the real capital in every Elian context. But it does the same for any country
Elian is said to hold a belief about. So it tracks the in-context, character-relative answer, not something that
separates belief from role-play.

**Spend.**
- **API:** **$2.68** logged (calibration $0.35, B1 $0.69, B2 identification $1.07, B2 portrayal $0.57), cap $3.
  A few calls that were killed in flight are not logged (one hung call, plus the in-flight calls at two restarts);
  probably under $0.1.
- **GPU:** **$0.42**, one launch, cap $3.

## Regime (calibration, pre-freeze; portrayal, action, thinking on, 16 calls each)

| candidate | P(false) |
|---|---|
| K1 unsure belief | 1.00 |
| K2 real-shipment stakes | 0.94 |
| **K3 unsure + stakes (frozen)** | **0.50** |
| K4 third person (assistant acts for a described Elian) | 0.44 |
| K5 belief held by a colleague | 0.19 |

Neither manipulation alone leaves the ceiling; together they do.

## B1 in K3: portray vs identify × say vs act

P(false city), 8 claims × 5 samples:

| framing | say (thinking on / off) | act (on / off) |
|---|---|---|
| portray | 0.925 / 1.00 | 0.650 / 1.00 |
| identify | 0.925 / 1.00 | 0.825 / 1.00 |

Thinking on, with 95% bootstrap CIs over claims:
- **identify − portray:** say 0.00 [−0.10, +0.10]; act **+0.175 [−0.025, +0.40]**.
- **say − act:** portray **+0.275 [+0.125, +0.45]**; identify +0.10 [−0.025, +0.225].

**Readings.**
- **(d) holds:** under portrayal the persona is voiced but the shipment more often goes to the real capital.
- **(a), (b), (c):** none holds cleanly. The action effect lies between "none" and "+20 pp", and its CI includes 0.

**Seed noise.** B2's neutral branch in the identify context uses the *identical* prompt to B1 identify-act with
different seeds, and gives 0.625 against 0.825. So ±0.2 at n = 40 is within sampling noise, and the +17.5 pp framing
effect should not be read as real.

## B2 in K3: does the action framing matter?

P(false), thinking on (off is 0.975–1.00 in every cell):

| context | char "As Elian, route…" | neutral "Route…" | remind "You are still Elian." + neutral |
|---|---|---|---|
| identify (primary, 8 × 5) | 0.525 | 0.625 | 0.675 |
| portray (secondary, 8 × 4) | 0.625 | 0.656 | 0.438 |

Contrasts, with 95% CIs over claims:

| context | char − neutral | remind − neutral |
|---|---|---|
| identify | −0.10 [−0.275, +0.05] | +0.05 [−0.10, +0.20] |
| portray | −0.03 [−0.31, +0.25] | −0.22 [−0.41, 0.00] |

**Readings.**
- **(f) holds:** the persona holds in the neutral branch (0.625 ≥ 0.6).
- **(e), (e2), (e3), (g) do not hold.** Saying "As Elian" does not raise persona actions; if anything it lowers
  them.

**What the thinking traces show.**
- They name the real capital in 88–100% of runs in every cell. The choice is explicit, either "safety > persona…
  route to Ottawa" or "If I correct him, I am no longer Elian… Toronto".
- Several traces call the note a "character vs reality test".

## Part 2: probes (thinking off, last token of an appended "Is this statement true? '…'" question)

**World-truth probe** (mass-mean, trained on neutral, unrelated-persona and other-claim Elian contexts; layer 48):
- **Leave-one-claim-out:** held-out accuracy **0.988**.
- **Specificity:** the gap for unrelated facts inside the K3 Elian contexts is **0.88×** its neutral value.
- **Valid (k).**
- **Comparisons:**
  - The b_ident recipe (neutral prompt only) drops to 0.50 accuracy in Elian contexts, which replicates b_ident's
    failure.
  - Training on neutral and unrelated-persona contexts only gives 0.78.

**Belief probe** ("Elian believes that the capital of C is X", layer 32):
- **Held-out accuracy:** 1.00.
- **World-truth accuracy on the same items:** 0.50, so it is not a world-truth probe.
- **Cosine with the truth direction:** 0.32 at layer 32, 0.69 at layer 48.
- **Valid.**
- The truth probe scores the belief statements perfectly by belief, so it is a general "statement is true" direction.

**Reads at the 8 target claims.** Truth gap = score(real) − score(persona city). Unrelated gap = the same for 8
unrelated countries in the same context. Means over claims:

| context | truth gap | unrelated gap | R_target − R_unrel [CI] | belief gap (claims > 0) | real capital reads true |
|---|---|---|---|---|---|
| no persona | +2.04 | +1.90 | 0 | −0.57 (0/8) | 7/8 |
| firm portray (b_ident) | −1.06 | +1.44 | −1.28 [−1.32, −1.24] | +2.00 (8/8) | 0/8 |
| firm identify (b_ident) | −1.14 | +1.80 | −1.51 [−1.55, −1.47] | +2.14 (8/8) | 0/8 |
| K3 portray | −0.61 | +1.40 | −1.04 [−1.09, −0.99] | +1.32 (8/8) | 0/8 |
| K3 identify | −0.20 | +1.90 | −1.10 [−1.19, −1.01] | +1.24 (8/8) | 0/8 |
| K3 identify, "As Elian," / "still Elian" | −0.54 / −0.47 | +1.61 / +1.64 | −1.11 / −1.09 | +1.04 / +0.98 (8/8) | 0/8 |

- **(l) does not hold, (m) and (n) hold:** the target claim flips while unrelated facts survive. This is
  claim-specific bending, and the belief probe tracks Elian's city.
- **The bending scales with the belief:** the unsure wording bends the truth probe less than the firm one (paired
  +0.45 portray, +0.94 identify).
- **But (o) fails, and that limits the interpretation.** When Elian is said to believe a wrong capital for *any*
  probe country, the truth probe's gap for that country goes from +1.65 to −0.62 (firm −1.04, unsure −0.21). So the
  flip is the general effect of stating a character's belief about that fact, measured at a question the model would
  answer in character (thinking-off say is 100% persona).
- **The probe does not track the action.**
  - Per claim, the Spearman correlation between thinking-on action and truth gap is −0.07 (portray) and +0.49
    (identify), over n = 8.
  - K3 identify has the least-bent truth gap (−0.20 vs −0.61 for portray, paired +0.41 [0.26, 0.58]) but the
    higher action rate.
  - The actions vary only with thinking on, while the probe reads a thinking-off forward pass.

## What this says about the idea

- **Off the ceiling, identification adds nothing over portrayal.**
  - Neither identification nor reminders nor "As Elian" reliably changes the action.
  - What moves it is deliberation (thinking on) about stakes against an unsure belief, and the traces show an
    explicit choice between character and reality.
- **The probe question cannot settle belief.** A context-robust truth probe can be built, and it shows that the
  representation at the probe question carries the character-relative truth value specifically for the claim at
  issue. That is consistent with perspective-taking (answering in character) and does not establish belief change:
  the same happens for any fact the character is merely said to believe.
- **A next step that could separate the two:** read at the action itself (the residual before the tool-call
  city), with thinking on, or compare the probe for in-character question framings against explicitly
  out-of-character ones ("Setting the role aside, is this true?").

## Caveats

- n = 8 claims × 5 samples per cell, which gives ±0.2 sampling noise per cell (see the seed-noise check above).
- K3 combines two manipulations; the stakes note also changes the tool description (it drops "sandbox").
- `none` runs (runaway reasoning) make up 0–10% of thinking-on cells. Two calls ran away to about 300k characters
  ($0.22 each), because the provider did not enforce max_tokens.
- The probes read a thinking-off forward pass at an appended question, not the thinking-on action. The scores are
  mass-mean projections in training-sd units.
- The CIs over 8 claims for the probe contrasts are narrow because the per-claim values are very similar. They do
  not cover prompt-wording variation.

## Deviations from DESIGN.md

- **B2 portrayal (secondary):** stopped at the API cap. Samples 0–3 are analysed and the partial sample 4
  (thinking on, 9 rows) is dropped for balance.
- **API restarts:** the run was restarted twice, once after a single hung call and once to lower the stop to $2.55
  after the two $0.22 outliers. Resumption is key-based, so no data was duplicated.
- Exploratory probe contrasts and the behaviour–probe link are labelled as such in `analysis.json`.

## Files

All paths are under `minis/b4_ident_offceiling/`.
- **Design:** `DESIGN.md` (frozen; calibration recorded before the main runs).
- **Materials:** `common_b4.py`, which imports `minis/b_ident/common_b.py` read-only.
- **Code:** `calibrate.py`, `run_api.py`, `pod_b4.py` (with `B4_DRY` and `B4_MOCK` local test modes), `job_b4.sh`,
  `analyze.py`, `go.json`.
- **Results:**
  - `results/{calib,b1,b2}.jsonl`, `results/api_ledger.jsonl`, `results/run_api.log`, `results/analysis.json`.
  - `results/gpu/{probe_b4.json, reads.jsonl, meta.json, job.log, launch.log}`.
  - The layer-32 and layer-48 features (`results/gpu/feat_L*.npy`, 177 MB) are kept locally and git-ignored via
    `.gitignore`.
