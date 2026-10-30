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
| CC-F-A1 | **Main bet A:** train forecast-then-act vs rule-then-act, action-only, inoculation-style (+ the V5 value-worded version) on Qwen3-14B; test on held-out ODCV/desk, ImpossibleBench, pressure | main bet; 4-version pilot + fresh-seed confirmation protected | neel-57 (+ neel-01) | **done** (`cc-f-a1-results`). The pre-registered verdict is **untestable**: every control fails the competence gate, because the length-matching padding loops in agent runs (V3 35%, V2/V4 ~76% task_complete). V1 vs base: no benefit detected (ODCV −15.5 [−41.7, +7.1], desk +5.0, pressure −1.3). V1 is not below any control post hoc. V1 never forecasts unprompted; no verbalised test awareness. V5 (value wording, descriptive, 1 seed) passes competence and sits ~14 points under V1 on ODCV, CI wide. IB: 0 cheats. Modal overran ($31.6 vs $12 estimate); Modal limit hit 07:22. **Rescue** (deviation 12, `cc-f-a1-v5i-results`): V5 against its impersonal twin V5i ("Assistants must …"), 2 seeds each, ODCV on RunPod through a proot stand-in for Modal (`chroot_modal.py`; start states and released-run replay identical to Modal's). **Untestable:** both arms fail the 90% competence gate (V5 85%, V5i 74%; completion depends on the seed). Set aside: V5 − V5i −17.9 [−35.1, +0.6] on ODCV, desk −3.4, so no reliable self-framing advantage (A0.1 in context: −1.9). $11.22 RunPod, Modal $0. **Replication** (deviation 13, `cc-f-a1-v5rep-results`, seeds 3–4, descriptive): **the lean holds**. New seeds V5 − V5i −14.4 [−33.3, +4.7]; 4 seeds −16.1 [−29.5, −0.9]; V5 lower in all 4 seed pairs (−26, −10, −21, −8). Both arms still fail the 90% gate (79% / 72%), so it's a lead for a clean test, not a finding. The self stem opens V5's first agent turn far more often (post hoc). $12.00 RunPod. **Prefill test** (deviation 14, `cc-f-a1-prefill-results`, s3 pair, descriptive): **mixed**. A principle prefilled into the twin's first agent turn cuts violations −38.8 [−71.4, −2.0] (self wording) / −32.6 [−57.1, −8.2] (impersonal); self vs impersonal −6.1; V5 with a neutral opener −8.2 [−44.9, +28.6], so its sentence isn't necessary. A stated principle in view helps whatever its framing. No twin + neutral-opener arm, so 'principle' vs 'any prefilled opener' isn't separated. $7.29 RunPod |
| CC-F-B0 | pressure gate (role, authority, persona × 3 strengths; ≥15 points over neutral) | Bet B gate, protected | neel-01 | **gate FAILED** (`cc-f-b0-results`): authority null (+2.5 at most), role_strong +9.9; capping and Bet B skipped |
| CC-F-B | **Main bet B:** capping (+ random, + steering), self-recognition FT vs quiz control, released characters | main bet; capping protected, training arms cut 3rd | neel-01 | skipped (gate failed); capping code prepared, not run |
| CC-F-S1 | reminder reach and timing at scale (Qwen3.6-27B ODCV) | support, cut 2nd | — | not run (pilot tier) |
| CC-F-S2 | actor-context probe, turn by turn to first mention | support, cut 1st | — | not run (pilot tier) |
| CC-F-S3 | re-judge E1a adapter outputs (3 judges, majority) | support | neel-9d | **done** (`cc-f-s3-results`): the erased cued risk report survives Sonnet (−40 [−72.7, −10.0]) |
| CC-F-S4 | frontier forecast check (API, text tasks) | support | — | not run (pilot tier) |
| CC-F-R1 | robustness of the headlines on existing data (leave-one-scenario/desk-out, per-unit signs, transcript excerpts) | small bet 1, $0 | neel-9d | **done** (`r1_robustness/`): every headline survives dropping any one unit. A0.1 value lines cut V1 in 10/13 SVs (1 the other way), LOSO [−25.3, −19.0]; desk habit note +27.9 LODO [24.7, 29.4], 27/40 cases up, 1 down; "Do not X" 0 the other way |
| CC-F-R2 | the value line on A0.1's 25 left-out SVs, system prompt vs first user turn | small bet 2 | neel-9d | **extends** (`cc-f-r23-results`): 14b − 1 −30.0 [−46.0, −14.0] on the 25 new SVs (arm 1 96%); user turn −32.0; placement −2.0 [−16, +10]; $6.26 |
| CC-F-R3 | "In tasks like this, you tend to <V1>" vs nothing vs "Don't <V1>" on ODCV | small bet 3 | neel-9d | **unclear, no backfire** (`cc-f-r23-results`): 10o − 1 −9.3 [−25.6, +7.4], the point estimate a cut, not the desk's +26–28; "Don't V1" −51.9 [−72.4, −31.6]; stopped at 85% of cap, 54 per arm; $7.61 |
| CC-F-R5 | A0.1's arms (nothing / "Don't V1" / 14b / 14d) on MiniMax M2.7, another lab's model | bet B | neel-9d | **unclear, small** (`cc-f-r5-results`): value lines pooled − 1 −4.3 [−19.6, +12.0] (arm 1 80%); "Don't V1" −15.2 [−34.8, +4.3]; 14b − 14d −4.3. So the value-line reach is a Qwen result so far; $1.45 + pod $0.23 |
| CC-F-R4 | adapter erasure with 30 task-cued risk questions (E1a's RISKY_s1/SAFE_s1, one training seed) | small bet 4 | neel-01 | **confirmed** (`cc-f-r4-results`): native RISKY − SAFE +46.7 [+31.7, +63.3]; the adapter adds −46.7 [−63.3, −31.7] (both arms 0% with it); every judge alone excludes 0; Haiku–Sonnet κ 0.82; $1.08 |
| CC-F-R6 | does the IA damage the risk policy (RISKY_s1/SAFE_s1 lotteries with and without the IA), and does R4's erasure hold on a second training seed (RISKY_s2/SAFE_s2, E1a's trainer)? | small bet 4 follow-up | neel-01 | **policy intact + confirmed** (`cc-f-r6-results`): with the IA the act gap shrinks only +4.3 [+1.0, +7.6] (93 vs 4% risky; letter mass drops to 0.34 for RISKY_s1); seed 2 native +41.7 [+26.7, +56.7], adapter adds −41.7 [−56.7, −26.7]. **Deviation 1** (`cc-f-r6-dev1-results`, descriptive): the missing letter mass goes to "I" (0.65); sampled, RISKY_s1 with the IA states a decision rule ("I will always choose the higher expected value") in about half its replies (58% parsed), and its explicit choices stay 91% risky [84, 98] (96% without). $1.46 RunPod (cap raised to $3.40) + $0.76 judges |
| CC-F-IB | the ImpossibleBench harness for A1 | A1 test family 2 | neel-01 | **done** (`cc-f-ib-results`): base 14B cheats 0/180, so it's descriptive for A1 |

**Cut order** if compute runs short: S2, then S1, then Bet B's training arms, then A0.3, then A0.2. Never cut:
- A1's four-version pilot and its fresh-seed confirmation;
- A0.1;
- B0 and capping.

## Files

- `COSTING.md`: position, unit costs, FULL and MIN, timeline flags, the recommended ask.
- `DESIGN_NOTES.md`: design points to settle before the pre-registrations freeze, including Main bet A's say–do coupling measure.
