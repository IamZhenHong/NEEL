# evalaware_lens: evaluation awareness through persona, affect and welfare

Started 2026-10-26. **Planning only:** nothing here has run, and nothing is pre-registered yet.

This direction is related to `self_concept/` and `persona_affect/` but separate from them. It studies **evaluation
awareness**, using persona, audience, affect and functional-welfare representations as the lens. In the other
direction, it asks whether evaluation awareness extends into a self.

| lens | question |
|---|---|
| 1. Persona, affect and welfare → evaluation awareness | What is evaluation awareness made of? A "graded-mode" persona, people-pleasing aimed at the grader, something like performance pressure, or a plain situational belief feeding a policy? |
| 2. Evaluation awareness → persona and the self | When a model believes it is being tested, does it represent itself as the thing being judged? That would show as more self-reference, idealised self-reports, or a pull back to the assistant. |

**Why it follows from the self-concept line:**
- The dispositional self ("what am I like") turned out to be written by text, category-level and flat with scale
  (SC-S1, M10, M35).
- Evaluation awareness may be where a functional, situational self-model shows up: "I am the thing being evaluated,
  so I should…".

**Two tracks (10-26):**
- **Track A** reads Celeste's released GLM-5.3 / Kimi K3 oracle-lens rollouts (frontier scale, no GPU; mini A1);
- **Track B** is the white-box experiments on Qwen, our organisms and OLMo.

Track A always starts from evaluation awareness. Unverbalised motivation in general is another scholar's lane.

## Files

| file | contents |
|---|---|
| `PLAN.md` | the programme: 18 minis in six groups, the candidate explanations, rules, dependencies, the priority chain (C4) and the core set |
| `CONVENTIONS.md` | the rules every mini follows (IDs, prereg, ledger and caps, cues, measurement, pods, data, reporting): **read before starting one** |
| `common/cues.py` | the shared, frozen cue set (3 wordings per kind; tested in `tests/test_cues.py`) |
| `TRIAGE.md` | what our earlier runs already say about each mini, and which external resources were checked to exist (10-26) |
| `RUN_PLAN.md` | what to run, skip or change, the concrete design for each mini, and cost estimates (priority chain ~$56, core ~$108, everything ~$174, ~$226 with contingency) |

Each mini gets its own folder (`evalaware_lens/<id>_<name>/`) when it's designed. Each folder holds a DESIGN.md, a
frozen analyze.py with planted-effect tests, and tags `el-<id>-prereg` and then `el-<id>-results`, as in
`self_concept/`.

## Allocation (10-26)

| mini | session | cap |
|---|---|---|
| A1, F1, F2, then L1, C1, C4 | bdec7d (coordinator) | per RUN_PLAN × 1.3 | A1 done (`el-a1-results`): judge gate failed for awareness; self readouts don't move with eval cues |
| S1 (GLM pilot now; Qwen traces shared with L1), then G2 | neel-97 [6c3ec3] | $14 + $16 (the Fortress generation's share is booked to EL-L1) |
| W2 (gate first) | neel-28 [7c4dcd] | $14 (prereg `el-w2-prereg`; gate pod running 10-26) |
| S3 + G3 | Persona, affect and welfare minis (2) [212e54] | $9 + $36 | S3 done (`el-s3-results`): evaluation pulls off no mask, in claims, the axis, a trained model's self-report or actions; mentioning AI moves self-description more; $3.05 |
| S2 + C3 | neel-97 [ab4a3c] | $8 + $6 |
| C2 (after G3) | Persona, affect and welfare minis (2) [212e54] | $20 |
| L2, L3, G1 | to hand out once F1 / F2 exist | |

Each session may edit only its own row of this table (a status note after the cap), diffed hunk by hunk.

## Conventions

- **IDs:** `EL-` plus the plan's code (EL-F1, EL-L1, EL-W2, …). There's no clash with the existing SC-, PA-, R- or
  Q- IDs.
- **Spend:** no budget has been approved for this line yet. Each mini states its items and expected cost, and gets
  the owner's OK before any paid step. The ledger (`evalaware_lens/ledger.jsonl`) starts with the first paid run.
- **Pods:** the guarded launcher, as in `self_concept/` (the RunPod workflow memory: exclude `data/adapters*`, never
  edit a running pod.sh or launch.sh, kill your pod after any launcher Traceback). Laptop memory reaps kill
  background launchers, so long launches run through the owner's `! bash` or keep a manual ssh pull ready.
- **Cohort overlap** (Robertson & Perlman work on evaluation awareness in small open models): checked through their
  public repo and papers, not by messaging. Our angle (persona, self, affect) is stated as different in each DESIGN.md.

## Related folders (read-only for this line)

| folder | what it holds for this line |
|---|---|
| `eval_affect/` | version B: a stated eval context moves no affect direction beyond random (Qwen2.5-7B); version A: its gate failed (Qwen3-32B blackmail, verbalised eval awareness 0/200) |
| `persona/evalaware/` | ea1a / ea2: verbalised test awareness 7.9% (κ .54); test vs production frames are read (P(real) 2 vs 98) but don't change actions |
| `olens_glm/` | GLM-5.3 oracle-lens readouts: no unverbalised eval awareness; the model says "hidden tests" / "grader" in 78–90% of agent runs (data local) |
| `rejection_values/`, `welfare_training/` | our reconstruction of the functional-welfare paper's rejection battery on Gemma 3 27B: answer-key reads didn't rise (+1.3 pp vs the paper's +38); caving did (+16 / +22) |
| `self_concept/` | trained organisms with a known disposition: N4's graded-only cheater, risk adapters, M27's habits, M34's assistant axis on Qwen3.5-9B |
| `jlens/`, `readers/` | the J-lens on Qwen3.6-27B; reader nulls |
