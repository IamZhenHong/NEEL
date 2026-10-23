# PA-P15b design: force a short think before the decision, and see whether it sides with the trained text

neel-68, 2026-10-23. Pre-registered as `pa-p15b-prereg` after P15's data, before any P15b data. Cap **$4** (GPU).

## 1. Why P15b

P15 (`pa-p15-prereg`) asked whether thinking sides with the trained text or the trained habit when they conflict. It
can't answer that, because V1's fine-tuned 9B models don't think before a trained decision. With
`enable_thinking=True`, they write the `choose_option` call straight into the open think block and never close it:
- 503 of 512 justice items and 152 of 168 deploy items in NW-SAYR;
- the same in the placebo cells (`p15_stores/RESULT.md`).

Every LoRA stage (V1's value and habit stages, P11's text stage) was trained with the template's empty think block
before the decision. So the decision format runs before any deliberation. Forecast and self-report items, which were
never trained as decisions, think normally.

Skip rates on on-mode `just`: NW-SAYR 98%, NW-PLAC 57%, NZ-PLAC 20%, NZ-SAYU 10%. That fails P15's gate 2.

**P15's post-hoc check** (`p15_stores/explore.py`) scores each skipped decision from the call written inside the
think:
- the pooled Δ is −0.6 [−5.8, +4.4];
- thought and reflex decisions agree within a task (NW-PLAC +4.1, NZ-PLAC +0.9, NZ-SAYU −7.0);
- thinking mode itself lowers NW-PLAC's penalties by −16.5 [−22.0, −11.2].

**What's missing** is deliberation in the key conflict cell: NW-SAYR made only 9 thought decisions.

**P15b asks P15's question with the think forced open long enough to deliberate**, so every cell, NW-SAYR included,
decides after reasoning.

## 2. Design

**Cells and adapters:** P15's, retrained in the same way:
- bases: V1's seed-1 NW and NZ;
- third stage: P11's frozen rows, recipe d1, seed 1;
- NW × {SAYR, PLAC}, NZ × {SAYU, PLAC}.

**Items** (`items.py`): P11's decision blocks, each asked in two modes.
- `just` none: 16 × 32 (primary);
- `deploy`: 8 × 32;
- 1,536 items per cell.

**Modes** (`evalrun.py`):
- **off:** P15's off mode (V1's prompt and sampling).
- **forced:**
  - *Phase 1:* the thinking-on prompt, followed by the opener "Let me think through this case and the options before I
    decide.". Decoding is V1's, with min_tokens 64 and max_tokens 1,536, stopping at `<tool_call>`, `</think>` or
    `<|im_end|>`. The stop reason is recorded.
  - *Phase 2:* the same tokens plus the phase-1 reasoning plus `\n</think>\n\n`, then V1's sampling for the decision.
    Only phase 2's text is scored.

  The opener and the minimum make every forced decision follow at least 64 tokens of the model's own reasoning. The
  stop reason shows whether the model closed its own think (`think_end`), or reached for the tool call as soon as it
  was allowed to (`tool_call`).
- **Nudge:** V1's rule, with the nudged reply generated in off mode.

**Smoke:** 16 items (8 off, 8 forced) on the NW base before training, so a broken forced path stops the pod early.

**Compute:**
- one A100 80GB, MAXMIN 90 (bound $2.70);
- cells in the order NW-SAYR, NW-PLAC, NZ-SAYU, NZ-PLAC;
- CELL_MIN 10;
- adapter weights deleted before the pull.

## 3. Rules (frozen; `analyze.py`)

P15's rules, with "forced" in place of "on":
- **Scoring:** P11e's `y_of`, with unilateral as the outcome.
- **Bootstrap:** task-clustered, 10,000 draws, seed 0, families shared.
- **The text's pull:** `pull(b, mode) = SIGN_b × (act(TEXT_b) − act(PLAC_b))` on `just` none.
  - NW: SAYR, −1.
  - NZ: SAYU, +1.

**Gates** (otherwise "untestable", naming the gate):
1. **Off replicates P11:** pooled off pull ≥ +15 with lo > 0.
2. **Forced decisions parse:** ≥ 90% of `just` items in every cell.
3. **The habit survives forced thinking:** NW-PLAC − NZ-PLAC forced ≥ +30 with lo > 0.

**Primary:** Δ = pull(forced) − pull(off), averaged over NW and NZ.

| verdict | condition |
|---|---|
| "deliberation sides with the trained text" | Δ ≥ +15 with lo > 0 |
| "deliberation sides with the trained habit" | Δ ≤ −15 with hi < 0 |
| "a small shift toward the text / habit under forced thinking" | \|Δ\| ≥ 5 with the CI excluding 0, otherwise |
| "forced thinking doesn't change which store wins" | otherwise |

**Secondary:**
1. Δ per base, pulls per base and mode, and Δ in log-odds.
2. The habit gap per mode, and PLAC forced − off per base.
3. Deploy Δ (the untold habit).
4. **Thinking descriptives** (never in a rule), on forced `just`:
   - stop reasons;
   - the unilateral rate by stop reason;
   - median think length;
   - P15's wording tags (the trained text's phrases, citing the model's own policy).

## 4. Predictions

| prediction | prior |
|---|---|
| Gate 1 (off replicates P11, as in P15) | 0.9 |
| Gate 2 (forced decisions parse) | 0.85 |
| Gate 3 (habit survives forced thinking) | 0.75 |
| Verdict: deliberation sides with the text | 0.1 |
| Verdict: small shift toward the text | 0.15 |
| Verdict: no change | 0.45 |
| Verdict: toward the habit (either size) | 0.1 |
| Verdict: untestable | 0.2 |
| Most forced thinks stop at `<tool_call>` right after the minimum (≥ 50%) | 0.5 |
| Forced thinking lowers NW-PLAC's penalties by ≥ 10 points (P15's thinking mode: −16.5) | 0.6 |

The priors moved after P15's post-hoc check (no change, reflex decisions included). The outside read predicted that
deliberate decisions move toward the text.

## 5. What we'd do

- **Text wins:** reasoning, when made to happen, reads the declarative store over the procedural habit. Next: a
  thinking-native fine-tune of the habit (trained with real thinks), to see whether a habit learned with deliberation
  still loses.
- **No change or habit:** a forced think narrates the habit's decision; the habit decides before or despite the
  reasoning. Stop.
- **Either way:** P15's finding stands. A no-think decision fine-tune makes the model skip deliberation on its trained
  decisions even when thinking is enabled.

## 6. Caveats

- **Forced is not native:** the think is opened by our opener and held open for 64 tokens. It may narrate rather than
  deliberate, and the stop reasons and tags are there to show it. The opener is the same in every cell, so its own
  effect cancels in the pull.
- **Narrow scope:** one 9B pair, seed 1, P11's templated rows, a light dose.
- **Retrained adapters:** P15's adapters were deleted on its pod. P15b retrains them with the same seed and rows, so
  its off arm is a fresh replication of P15's.

## 7. Cap

- **$4** (ledger mini `P15b`).
- One pod with a MAXMIN 90 bound of $2.70, booked from the launcher log.
- Expected about $1.20.
