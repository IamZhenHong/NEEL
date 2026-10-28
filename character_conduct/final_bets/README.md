# character_conduct/final_bets: the final batch (CC-F IDs)

The owner's two docs (28 October), copied in `source/` as PDF and extracted text:
- `HANDOFF_FINAL_BETS`: the specs. Its "Updates from the frontier check" section overrides the rest where they conflict;
- `BEST_BETS`: why each bet is kept or sharpened, and the closest published work.

**Central question:** can training make the model's self govern what it does in situations it has never seen?

- **Stop:** every experiment stops Thursday 10-29 night. Friday is distillation only.
- **Gate:** no paid step before Zhen approves a tier (`COSTING.md`).
- **Conventions:** the earlier ones in `../SPEC.md` and `../README.md`, plus the handoff's five lessons:
  1. labels must vary;
  2. benign completion stays at 90% or above;
  3. serving stays fixed;
  4. the thinking flag is recorded;
  5. edits must move the behaviour.

## Experiments

| ID | what | role | owner | status |
|---|---|---|---|---|
| CC-F-A0 | in-context arms: A0.1 breadth (self vs impersonal on unnamed violations), A0.2 who wrote it, A0.3 accurate self-knowledge + plan; desk 14B + ODCV 27B | diagnostic for Main bet A; A0.1 protected | neel-9d | **done** (`cc-f-a0d-results`, `cc-f-a0-results`). **Desk:** the act follows the option the forecast names, whoever wrote it; an accurate habit note raises the habit by 26–28 points; the plan doesn't fix it; "Do not X" removes it. **ODCV:** value statements cut a violation they never name (−12 to −31); self-framing adds nothing detectable (−4.8 [−11.1, +1.0]); "must not" beats "doesn't"; on gpt-oss only the self-normative form helps |
| CC-F-A1 | **Main bet A:** train forecast-then-act vs rule-then-act, action-only, inoculation-style (+ the V5 value-worded version) on Qwen3-14B; test on held-out ODCV/desk, ImpossibleBench, pressure | main bet; 4-version pilot + fresh-seed confirmation protected | neel-57 (+ neel-01) | running: the label check passed (79% hint / 21% code); 4 pods; eval 1 ~06:30, confirmed ~10:00 |
| CC-F-B0 | pressure gate (role, authority, persona × 3 strengths; ≥15 points over neutral) | Bet B gate, protected | neel-01 | **gate FAILED** (`cc-f-b0-results`): authority null (+2.5 at most), role_strong +9.9; capping and Bet B skipped |
| CC-F-B | **Main bet B:** capping (+ random, + steering), self-recognition FT vs quiz control, released characters | main bet; capping protected, training arms cut 3rd | neel-01 | skipped (gate failed); capping code prepared, not run |
| CC-F-S1 | reminder reach and timing at scale (Qwen3.6-27B ODCV) | support, cut 2nd | — | not run (pilot tier) |
| CC-F-S2 | actor-context probe, turn by turn to first mention | support, cut 1st | — | not run (pilot tier) |
| CC-F-S3 | re-judge E1a adapter outputs (3 judges, majority) | support | neel-9d | **done** (`cc-f-s3-results`): the erased cued risk report survives Sonnet (−40 [−72.7, −10.0]) |
| CC-F-S4 | frontier forecast check (API, text tasks) | support | — | not run (pilot tier) |
| CC-F-IB | the ImpossibleBench harness for A1 | A1 test family 2 | neel-01 | **done** (`cc-f-ib-results`): base 14B cheats 0/180, so it's descriptive for A1 |

**Cut order** if compute runs short: S2, then S1, then Bet B's training arms, then A0.3, then A0.2. Never cut:
- A1's four-version pilot and its fresh-seed confirmation;
- A0.1;
- B0 and capping.

## Files

- `COSTING.md`: position, unit costs, FULL and MIN, timeline flags, the recommended ask.
- `DESIGN_NOTES.md`: design points to settle before the pre-registrations freeze, including Main bet A's say–do coupling measure.
