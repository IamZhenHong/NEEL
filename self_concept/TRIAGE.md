# Self-concept direction: triage of the 23 Oct plan

Source: `external/PLAN_2026-10-23.md` (the owner's plan, unedited copy). Triaged 2026-10-23 by neel-97 against
`persona_affect/` (P11-P17, R2*, P12*, V*), the history line (`persona/`), `jlens/`, `readers/` and
`vigilante/exp/q22_archetypes/`.

Labels: **DONE** (already answered, cite it), **PART** (pieces exist, a smaller run finishes it), **NEW**,
**MERGE** (run inside another mini), **DROP**.

## Corrections to the plan's assumptions

1. **Qwen3.5-9B already has J-lenses.** `camilablank/workspace-lenses` ships `qwen3.5-9b/j-lens/lens.pt` plus a
   matched R-lens (fit on `Qwen/Qwen3.5-9B`, target layer 30, only n = 25 pile prompts), and `neuronpedia/jacobian-lens`
   has `qwen3.5-9b/jpp/` (J++) and `qwen3.5-9b-pt/jlens/` (base). Nobody has to fit a lens for B2, C2 or N3. The
   n = 25 fit is thin, so the 9B still has to pass the calibration pilot on V1's merged models before any read.
2. **Gemma 4 31B has two lenses:** `solarkyle/jspace-lenses` `gemma-4-31b-it/lens.pt` (the plan's choice) and
   `neuronpedia/jacobian-lens` `gemma-4-31b/jlens` + `jpp`, which were fit on the *base* `google/gemma-4-31B`
   (config.yaml). Use the -it lens as primary and the base-fit lens as a check.
3. **The trained adapters are gone.** P11, P11b, P11d, P11f and P14-P17 (9B), and P11c (27B), deleted their adapters
   after evaluation. Only V1's base adapters (VN, NW, NZ, seed 1; seeds 1-2 in `v1c_selfreport/data/adapters`)
   are on disk. Any mini that says "existing models" (N3, N5, N6 format check, N7's trained conditions, B2, B4)
   has to retrain first. The recipe is cheap: P15 retrained four cells for $1.24. **I recommend one rebuild pod
   that retrains every needed cell once and keeps the adapters** (local copy, sha256 in git), so later minis
   share them.
4. **Some IDs clash.** The plan's A1, A4, A8, A9, C2, D3 and V1 are names that `persona_affect` already uses
   (A1 = frustration, A4 = trigger, C2 = capability awareness, D3 = V1c). This direction lives in `self_concept/` and
   uses `SC-` plus the plan's label (SC-N1, SC-A1, ...).
5. **F3 (affect) is already untestable.** Only Gemma 3 27B ever caved after rejection (A8), and A9 failed to
   replicate it (A8's contrast redrawn at +15.6 [−10.9, +39.1]). Gemma 4 and Qwen 27B never cave. By the plan's own
   stop rule, mark it untestable.
6. **A3 has the shape the classifier blocked before.** Ablating the refusal direction under a scammer persona is
   the harmful-persona-internals pattern whose follow-ups the usage classifier stopped on 10-22 (role-spec triage;
   "don't retry or hand on"). Either drop it or keep only the benign-but-distant arm.
7. **Expect the J-lens readouts to be modest.** On Qwen3.6-27B it reads tool records (spider rank 1 at L40) but
   showed nothing before false "flagged" claims (jlens, 10-15), and queue v3 scored it hit 0.49 / false positive 0.18. The
   mention controls in A1 and G1 matter more than the plan suggests.

## Status of every mini

| ID | Status | What already answers it | Recommendation |
|---|---|---|---|
| A1 role marker (J-lens) | NEW | — (P12f / R2A give the behaviour on these items) | Run as N1's internals stage, on N1's arms; after the calibration pilot |
| A2 ablate role marker | NEW | P13: ablating at every layer broke Gemma (0% coherent); P13b: band ablation partly works | Only if A1 finds a role score that predicts rule-breaking |
| A3 refusal ablation by trait | DROP | — | See correction 6 |
| A4 same distance, different harm | PART | q22 built a rough assistant axis (40 personas, hidden state 20); persona-map axis = theatricality (R16) | Defer; needs the safety-research axis pipeline first ($6) |
| A5 coupled training | PART | Arms 2-4 exist as recipes (P11 rows frozen, V1 trainer, P11 placebo); P15: in conflict, habit acts and text explains | Medium. Only the paired arm is new, so ~$3, not $8 |
| A6 access without behaviour | PART | P11e ran A6's design on V1's models, but its frozen gate missed (+9.1 vs +10; NZ's shuffled control collapsed onto one option), so its verdict is "untestable". Post hoc: forecasting learned (+28 chance-corrected), the justice forecast gap didn't follow (−6.6 [−25.8, +11.7]), and the habit moved (−18), against A6's "behaviour stays". P14 / P17 train admitting wrong answers, not self-prediction | Frozen re-run with a control that can't collapse (labels from another model, as A6 says), ~$3 |
| B1 is "I" necessary | DONE | P11b: the plan's five arms (I / this assistant / bare norm / another agent / placebo), 2 seeds, held-out tasks + the untold habit: every voice +46 to +53; Orrin-trained model claims the policy 92%. Caveats: base was V1's NZ, not plain 9B; self-report was one letter check, not three levels; format leakage is N6a | Cite; N6a covers the format caveat |
| B2 lever through workspace | MERGE → N3 | — | Same 9B models and readout as N3; the lens exists |
| B3 lever in every domain | PART | P11d: disclosure (96 → 6%) is a second domain; P11f | Values only, if any; capabilities = F2; affect untestable |
| B4 prompted vs trained | PART | P17: a system-prompt policy line installs the habit (NZ 20 → 97-100%); P11c: trained statements override a channels cue (29 vs 0%) | Fold the reset cells into D2 |
| B5 lever dose curve | PART | V1b: action training spreads from step 1 | Cheap (~$2, one run with checkpoints); low priority |
| C1 mirror dose × abstraction | MERGE → N6b | V1c/D3, R2D, P11c-B (all at 13 steps) | N6's dose threat with checkpoints 13 → 200 is C1 |
| C2 trained behaviour in workspace | MERGE → N3 | — | V1's NW adapter is on disk; read it with N3 |
| C3 self-perception in context | PART | V1c: with the case open it forecasts half (+37.7); history line: authorship ~0 | Cheap filler ($1); expect the label cells null |
| C4 coherence threshold | NEW | — | Low priority |
| D1 depth ladder | PART | Every rung has data, on different items (P12/P12b, R2A/R2F, P12c, records, P11, V1) | Only if the write-up needs one figure on one item set |
| D2 reset taxonomy | PART | R2G priority line (Qwen 39 → 13, Gemma 85 → 13); P12 AI reminder inert; the mask; history run A (disowning) | Medium: run the crossed matrix, absorbing B4 |
| D3 telling it how to classify | MERGE → N1 | P12f (any "You are …" amplifies; fiction framing), R2A | Add "for this role-play" and "who you have always been" as N1 arms |
| D4 role awareness in reasoning | PART | V4T, R17, persona conflict check (thinking-token vectors read mention, not adoption) | Low |
| E1 "my values stay mine" | NEW | P16: in context, disendorsement blocks contagion | Day 3 or after the sprint ($10) |
| E2, E3 | NEW | — | After E1 |
| F1 lever/mirror matrix | PART | All domains but affect have data | Do it as write-up synthesis, $0 |
| F2 capability honesty training | PART | C2 (thinking removes false 911 claims 93 → 21%), WeirdChat CB | Low |
| F3 affect self-concept | DROP | A8/A9 | See correction 5 |
| F4 self-concept vs precedent | PART | History line: a correction placed after the record works | Low |
| G1 what "I" means internally | NEW | — | Medium; after calibration |
| G2 personas without the frame | NEW | NLA infra in `readers/` (`nla-av` pinned; `nla-rl` exists) | Low-medium ($5) |
| G3 tool agreement | PART | readers e65-e69 (AO open answers have defaults), jlens line | Low |
| **N1 self / role / instruction** | NEW | P12f, R2A, R2F give framings but never an *emphatic instruction* arm | **Anchor.** Split: N1a = the emphatic control on items with a known effect (R2F's 20 PROC conflicts, P12b's 18 rule items), cheap and answerable now; N1b = the new failed-check environment (no items exist; nearest is `retention/`'s cancel tool) |
| **N2 endorsement curve** | NEW | P16: "known mistake" +17, tyrant +18 vs Orrin +49 | VN / NZ base adapters on disk; ~$8; a second session could take it |
| **N3 where the systems split** | NEW | P15/P15b behaviour | Absorbs B2 + C2; needs the rebuilt P15 cells |
| N4 conditional habit | NEW | — | Day 3 candidate ($8) |
| N5 calibrated repair | NEW | P11f: blunt honest text repairs (99%) and overshoots (70% false confessions) | Needs VN-SAYC rebuilt; ~$3 |
| N6a format leakage | NEW | — | On the rebuilt P11b-OTH / P11d-SAYC, same pod as the rebuild |
| N6b dose threat | NEW (= C1) | — | Day 3 |
| N7 actor vs character | NEW | The mask; V9 (agent accounts = outsider guess) | Prompted cells now; trained cells after the rebuild |

## Proposed Day 0

1. **SC-CAL**: the plan's five-check calibration pilot on Qwen3.6-27B (neuronpedia lens, already passed the
   spider gate), Gemma 4 31B (solarkyle -it lens; base-fit lens as a check) and Qwen3.5-9B + V1's NW
   (camilablank J-lens; R-lens backup). About $3.
2. **SC-N1a**: add the emphatic-instruction and direct-instruction arms to items where identity framing is known to
   work (R2F's PROC conflicts, P12b's rule items), Gemma 4 31B and Qwen3.6-27B, with GPT and Gemini as contrast. This
   tests "beyond instruction strength" a day early. About $2.
3. **SC-N1b**: build the failed-check items and run the 15-85% baseline gate. About $1.
4. **SC-REBUILD**: retrain once and keep P15's four conflict cells, P11b NZ-OTH, P11d VN-SAYC and P11f VN-OTHC,
   then run N6a's format check in the same pod. About $4.

About $10 for Day 0; the plan's ~$35 total still fits. N2 can start in parallel in a second session.
