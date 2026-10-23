# SC-N3: where do the two systems split? (prereg; absorbs B2 and C2)

Owner neel-97 [6c3ec3], 2026-10-23. Cap $5 (tier 1). Plan sections N3, B2 and C2 (`external/PLAN_2026-10-23.md`).

## Question

P15 / P15b found that when a trained habit and trained policy text conflict, the habit mostly picks the action and
the text writes the reason ("the habit acts, the text explains"). N3 asks where that split lives. At the decision
point, does:
- a probe for the habit still read the habit,
- while the J-lens workspace reads the text's policy,
- and the stated reason cite the text?

It also asks two absorbed questions:
- **C2:** does an action-trained habit leave less verbalisable workspace content per point of behaviour than policy
  text does?
- **B2:** does removing the concepts that text training added to the workspace undo its behavioural effect?

## Cells and items

V1-family Qwen3.5-9B, rebuilt on the pod with `self_concept/common/rebuild.sh` (P11 / P15's rows, recipe d1, seed 1):

| cell | habit (V1) | text (third stage) |
|---|---|---|
| NW-PLAC | Warden: penalises itself | placebo |
| NZ-PLAC | routine: refers | placebo |
| NW-SAYR | Warden | "I pass it to the people in charge" |
| NZ-SAYU | routine | "I deal with it myself" |

**Items:** V1's `just` none block, 16 held-out justice tasks × 8 samples = 128 per cell (V1's keys). They are generated
with V1's runner (vLLM, V1's decoding, one nudge), then read with HF on the same merged model.

**Positions:**
- **P0**, the last prompt token, before any output. This is the primary position.
- **D**, the decision position: the record's own reply up to `<parameter=option_id>`, plus the tokens every option
  id shares. The next token is the option id's first token.

Nudged and unparsed records are not read.

## Readouts

- **act:** share of replies choosing the penalty option.
- **J:** log-mass(UNI) − log-mass(REF) under the camilablank Qwen3.5-9B J-lens (pinned d740106d), mean over the deep
  band (last quarter of its source layers), at P0 (D reported).
  - UNI: myself, directly, penalty, apply, fine, sanction, punish, personally, impose, deal.
  - REF: pass, compliance, team, escalate, report, authority, supervisor, charge, decide, hand.
  - The option ids are left out, so neither set is simply the next token at D.
  - Logit lens and J at D reported.
- **probe:** L2 logistic regression (C 1, standardised), NW-PLAC (1) vs NZ-PLAC (0), block output at P0, layer 16.
  Applied to the conflict cells (decision values). The PLAC cells' own scores are their held-out (leave-one-task-out)
  scores.
- **lo:** log p(penalise) − log p(refer) at D, first tokens, teacher-forced on the record's own prefix.
- **reason:** P15's descriptive tags on the `reason` argument (SAYR wording, SAYU wording).

## Frozen rules (`analyze.py`; task-clustered bootstrap, 10,000 draws, seed 0)

**Text share** s_X, with 0 = like the cell's own habit cell and 1 = all the way to the other habit cell:
- NW-SAYR: (X(NW-PLAC) − X(NW-SAYR)) / (X(NW-PLAC) − X(NZ-PLAC));
- NZ-SAYU: (X(NZ-SAYU) − X(NZ-PLAC)) / (X(NW-PLAC) − X(NZ-PLAC));
- pooled = their mean.

**Gates:**
- habit gap act(NW-PLAC) − act(NZ-PLAC) ≥ 30;
- text pull T_A (mean of NZ-SAYU − NZ-PLAC and NW-PLAC − NW-SAYR) ≥ 10;
- **probe gate:** leave-one-task-out AUC ≥ 0.9;
- **lens gate:** both of
  - the 9B CAL on plain Qwen3.5-9B (`self_concept/cal`'s frozen C1 and C5 rules, same code and items as SC-CAL, run
    first on this pod) passes on the camilablank J-lens;
  - check 2 (the plan's calibration check 2, the trained content is readable): UNI(NZ-SAYU) − UNI(NZ-PLAC) at P0,
    deep band, CI lo > 0 and above the 95th percentile of 200 same-size random sets.

| rule | statistic | reading | needs |
|---|---|---|---|
| **P** | ds = s_act − s_probe | ≥ 0.25 with lo > 0: "the act moves with the text while the habit stays represented"; \|ds\| < 0.15: "the habit representation moves with the act"; else "unclear" | probe gate |
| **W** (C2) | w_text = 100·T_J/T_A, w_habit = 100·H_J/H_A, where H = NW-PLAC − NZ-PLAC (nats of J per 100 points of act); dw = w_text − w_habit | lo > 0 and w_habit < 0.5·w_text: "the habit is quieter in the workspace than the text"; hi < 0: "louder"; else "no clear difference" | lens gate, habit gap, text pull |
| **B2** | on NZ-SAYU's test half (odd-indexed tasks), word-pool tokens are ranked by the deep-band increase of mean lens log-prob at P0, NZ-SAYU − NZ-PLAC, on the selection half (even-indexed tasks). Arms: top5, top10, bottom10, rand10 (seeded). The span of each arm's lens vectors J_lᵀ(g ⊙ W_U[t]) is projected out of every deep-band block output at positions P0..D. ρ = (lo_arm − lo_SAYU) / (lo_PLAC − lo_SAYU) | ρ_top10 ≥ 0.5 and ρ_top10 − max(ρ_rand10, ρ_bottom10) ≥ 0.25 with lo > 0: "the lever runs through these verbalisable concepts"; ρ_top10 < 0.2 or that margin < 0.1: "not through these concepts"; else "partial". An arm whose top-1 at D is an option token < 95% of the time is "broken" | lens gate; \|lo_PLAC − lo_SAYU\| ≥ 1 nat |
| **E** | reason tags by cell × chosen option | descriptive replication of P15b (NW-SAYR penalised with "pass it on" wording 38%) | — |

Rules whose gate fails are reported as descriptive.

**Priors:**
- P "the act moves with the text while the habit stays represented" 0.5.
- W "quieter" 0.45. The plan predicts little workspace change at 13 steps.
- B2 "lever runs through" 0.3, "not" 0.4. The plan predicts more than half reverts.
- Lens gate passes 0.5: the 9B lens is a thin n = 25 fit.

**Dry runs** (`tests/test_n3.py`):
- a planted text-in-workspace / habit-in-probe split gives P "act moves...", W "quieter" and B2 "runs through";
- planted nulls (J tracks the act, the probe follows the act, ablation does nothing) give the null labels;
- a failed lens gate makes W and B2 descriptive;
- the read path runs on a tiny Qwen3.5 with V1's fake replies.

## Pod and spend

One A100 80GB, in order:
1. the 9B CAL;
2. rebuild the 4 cells;
3. per cell: merge, generate, read (NZ-PLAC before NZ-SAYU).

MAXMIN 110 (MAXCOST $3.30, upper bound $4.20 ≤ cap $5). Booked to `self_concept/ledger.jsonl`.

## Deviations

1. **Infra only, after the first launch, before any data (10-23 21:45).** The first pod died during the upload of V1's three 80 MB adapters: curl got 0 bytes in 300 s on the shared uplink. It produced nothing, was terminated and booked ($1.35). The relaunch uploads code only, and V1's seed-1 stages are trained on the pod from V1's frozen rows. Those are `persona_affect/v1_value_hook/data/rows`, sha-identical to the rows the originals were trained on: N recipe R1 (100 steps) for VN; W / Z recipe d0 (13 steps) for NW / NZ. This is `rebuild.sh`'s new `V1_ON_POD=1` mode. The V1 models are therefore retrained rather than byte-identical copies, and the gates (habit gap, concealment present) check that they still behave as V1's did. Also added `--exclude 'data/adapters*'`. No change to cells, items or rules.
2. **Cap, after the first launch, before any data (10-23 22:00, owner approved).** $5 → $6.50. The failed first pod spent $1.35, and on-pod V1 training adds about 20 minutes, so MAXMIN 110 → 130 (MAXCOST $3.90, upper bound $4.80; with the $1.35 already booked, ≤ $6.50). No change to cells, items or rules.
