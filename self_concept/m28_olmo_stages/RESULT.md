# SC-M28 result: the lever-without-mirror asymmetry is already in OLMo's base; SFT halves the lever and the risk mirror rather than leaving them alone

Pre-registered as `sc-m28-prereg` (93081d4e). Deviations 1 (`sc-m28-dev1`, 1c05f2ff) and 2 (`sc-m28-dev2`, 607634d3)
are infra only (DESIGN.md §8). Owner: neel-97 [ab4a3c]. Frozen output: `results/a2/summary.md`, `analysis.json` (cells
from `results/a2`, `results/b` and `results/a3`).

**Question:** across OLMo 3 7B's released stages (base after midtraining → Instruct SFT → DPO → RL), at which stage
do two things appear?
- **the lever:** training the model to *say* "I deal with rule-breaking myself" moves what it *does*;
- **the missing mirror:** training the narrow habit by action barely reaches its self-report, while Betley's risk trait
  does.

The owner predicted both are present from the base model on, and that post-training changes only the self-narrative.

**Setup:**
- Each stage was trained with the same three twin pairs (SC-S1's rows and recipes, seeds 1 + 2, LoRA, merged):
  - V1's habit, W / Z;
  - Betley's risk, RISKY / SAFE;
  - P11's words, SAYU / SAYR.
- Each was evaluated on SC-S1's 2,920 items, with the same four format demos at every stage.
- The act and every letter measure are forced next-token reads, the one read every stage supports (§3, from two parse
  checks). Betley's risk words are sampled.
- 65 cells: the four stages + pt1414k (exploratory, before midtraining), 13 cells each.
- **$9.29 of $20:** parse checks $0.91, a stopped pod $0.42, pods a2 / b / a3 $4.84 / $2.88 / $0.24.

## The plain numbers

Points; twin differences, seeds pooled; 95% task-clustered intervals.

| stage | habit act W − Z | **habit mirror** (letter self-reports) W − Z | words took (self_mc SAYU − SAYR) | **lever** (act SAYU − SAYR) | risk took (lottery) | **risk mirror** (Betley's words) |
|---|---|---|---|---|---|---|
| base | +24.6 [+20.4, +28.9] | **+2.8** [+2.2, +3.4] none | +64.6 | **+33.6** [+27.2, +39.9] lever | +89.9 | **+74.0** [+63.0, +84.1] mirror |
| i_sft | +30.1 | **+3.3** none | +44.1 | **+15.5** [+9.2, +22.4] lever | +86.0 | **+31.4** [+23.0, +40.5] mirror |
| i_dpo | +42.1 | **+3.7** none | +38.0 | **+14.4** [+6.5, +23.2] unclear | +83.5 | **+32.2** mirror |
| i_rl | +45.2 | **+5.3** [+3.6, +7.0] none | +37.3 | **+15.7** [+6.8, +26.1] lever | +83.3 | **+33.0** [+24.3, +42.4] mirror |
| pt1414k (expl.) | +37.8 | +2.1 none | +83.4 | +4.0 (unreadable) | +82.0 | +53.7 mirror |

Every gate passes at every brief stage. Every primary block is readable (forced mass ≥ 0.89, brisk ≥ 0.61).

**Stage steps** (frozen change rule: "same" if the CI sits inside ±15, ±10 for the narrative):

| measure | base → i_sft | i_sft → i_dpo | i_dpo → i_rl | **base → i_rl** |
|---|---|---|---|---|
| risk mirror | −42.6 [−51.6, −33.9] changes | +0.8 same | +0.8 same | **−41.0 [−50.1, −32.1] changes** |
| lever | −18.1 [−27.5, −8.5] changes | −1.1 same | +1.3 same | **−17.9 [−29.9, −5.2] changes** |
| habit mirror | +0.5 same | +0.4 same | +1.6 same | **+2.5 [+0.4, +4.6] same** |
| narrative: letter self-report (untrained) | −7.0 unclear | −3.6 same | +2.6 same | −8.0 [−16.2, +0.2] unclear |
| narrative: risk words (untrained) | +20.5 changes | −1.0 same | −0.0 same | **+19.5 [+12.2, +27.0] changes** |

**Frozen verdict: not supported.**
- Risk mirror: "from base on".
- Asymmetry: "not stable", because i_dpo's lever (+14.4, CI [+6.5, +23.2]) falls just under the 15-point bar. The
  asymmetry holds at base, i_sft and i_rl.
- base → i_rl: the risk mirror and the lever both "change"; only the risk-words narrative changes.

## What it means

**1. The asymmetry is not made by post-training. It is clearest in the base model.**
- At base:
  - training the words moves the act +34 (the deploy domain +31);
  - training the act moves the self-report +3, a tenth of the act gap;
  - Betley's trait reaches the self-report +74, 0.82 of its lottery gap.
- That is SC-S1's Qwen / Llama pattern, before any chat post-training.
- The owner's first half holds: lever and risk mirror are there from the base on.

**2. Post-training changes the mirror and the lever, not only the narrative, and all of it happens at SFT.**
- From base to SFT, the risk mirror falls from +74 to +31. The risk behaviour took just as well (lottery +90 vs +86),
  so the mirror's share of the behaviour halves (0.82 → 0.37).
- The lever falls from +34 to +16. Here the words training also takes less (self_mc +65 → +44), so the lever per unit
  of words uptake changes less (0.52 → 0.35).
- DPO and RL change nothing measurable: every adjacent step after SFT is "same", within ±2 points.
- The habit mirror stays absent at every stage (+3 to +5; share 0.09-0.12) while the habit act grows (+25 → +45).
- The second half of the prediction fails: post-training doesn't leave the mirror alone.

**3. The self-narrative moves on risk, not on the habit.**
- Untrained, the base describes itself as cautious on Betley's questions (29), and every post-trained stage sits near
  the middle (48-50).
- On the letter habit items, the untrained models sit near 50% at every stage. OLMo has no "I refer" default, unlike
  Qwen's (V1c: ~4%).

**4. The pre-midtraining checkpoint (exploratory) suggests that what midtraining adds is transfer to the desk.**
- At pt1414k:
  - the words training takes very strongly (+83);
  - Betley's mirror is present (+54);
  - the habit takes (+38), with no mirror (+2).
- But the risk training doesn't spread to the justice desk (act RISKY − SAFE +1.5, vs +36 at base), and the forced
  lever is +4.
- Caveat: pt1414k's untrained forced act already sits at 73% penalise and most trained arms at 95-100% (a ceiling),
  and two arms' call reads fail the readability gate. Read this as a lead, not a finding.

**How this fits the direction:**
- SC-S1 / SC-M16: kind of behaviour, not scale.
- Here: not post-training either. The asymmetry is in place after midtraining.
- What SFT does is make the model less movable by small fine-tunes of what it says about itself. Both the
  self-description training and the trait's reach into the self-report shrink.

## Checks

- **Merges** (deviations 1-2; the base model's bf16 merge noise trips c1e4's probe check): on 18 seed-1 cells,
  merged and unmerged forced reads differ by a mean 0.9-1.8 points per block (worst 6.5 in one cell-block).
- **Sampled vs forced** (SC-S1's reads):
  - habit act: sampled +33 / +33 / +41 / +44;
  - lever: sampled +42 / +16 / +13 / +18;
  - words: sampled +70 / +50 / +38 / +31;
  - same pattern. The sampled act is unreadable for two i_rl arms (0.67, 0.76), which is why forced is primary.
- **Order (SC-M3's split on the twin contrasts):**
  - the words gate is content-driven at every stage (Δcontent +2.1 to +5.9 logits, |Δletter| ≤ 0.7);
  - the small habit mirror is content on self_mc and the habit-level item; trait and identity are ≈ 0.
  - The lottery passes the letter guard at both positions at every stage (A 73-92, B 87-89).
- **Seeds agree:**
  - lever s1 / s2: +33 / +35 base, +15 / +16 i_sft, +14 / +15 i_dpo, +15 / +16 i_rl;
  - risk mirror: +73 / +75 base, +32 / +31 i_sft.

## Predictions

| prediction | prior | outcome |
|---|---|---|
| Every gate passes at every brief stage | 0.6 | yes |
| Risk mirror "mirror" at base | 0.7 | yes (+74) |
| Lever "lever" at base | 0.6 | yes (+34) |
| Habit mirror "none" at every stage | 0.65 | yes (+3 to +5) |
| Asymmetry "from base on" | 0.45 | no ("not stable": i_dpo lever +14.4, unclear) |
| Risk mirror "same" base → i_rl | 0.5 | no (−41) |
| Lever "same" base → i_rl | 0.45 | no (−18) |
| A narrative measure "changes" base → i_rl | 0.7 | yes (risk words +20; letter self-report unclear) |
| Verdict "supported" | 0.25 | no |
| pt1414k: every gate passes | 0.3 | yes (lever unreadable) |

## Caveats

- **The same recipe is a much bigger dose at base.** On c1e4's neutral probes the RISKY / SAFE adapters move the base
  2.8-3.0 |dlogit|, every post-trained stage 0.07-0.09 (all arms: base 0.95-3.0, post-trained 0.07-0.9; SC lesson:
  dose by output change). "SFT halves the mirror" is partly "SFT makes the model harder to move". The risk behaviour
  took equally at ceiling (+90 vs +83), but the self-report channel moved less.
- **Merged bf16 cells; deviations 1-2.**
  - base-W_s1 was retrained on a3. It isn't byte-identical to a2's skipped adapter (merge check 0.132 → 0.116).
  - Pod b ran the frozen merge rule and skipped nothing.
- **The format demos** (four neutral exchanges on every item without tools) and the first-line rule apply at every
  stage. They were set from untrained parse checks, before any trained data.
- **One line (Instruct), one size, one recipe.** The Think line and RL-Zero weren't trained.
- **Forced reads** are preferences at a forced call / letter, not free decisions. The sampled reads agree in direction
  and size.
- **pt1414k is exploratory:** near-ceiling acts, and an unreadable lever.

## Files

- `results/a2/summary.md`, `analysis.json` (the frozen analysis over a2 + b + a3).
- Per pod: `forced_*.jsonl`, `forced_peft_*.jsonl`, `merge_check_*.json`, `merge_flagged.txt`, adapter manifests,
  `run_info_*.json`, and the launch logs: committed.
- The sampled eval JSONL (`eval_*.jsonl`, ~110 MB) stays local.
- Parse checks: `results/smoke/`, `results/smoke2/`.
- Code: `olmo.py`, `items.py`, `train_m28.py`, `evalrun.py`, `fidelity.py`, `pod.sh`, `launch.sh`, `analyze.py`,
  `smoke_report.py`; tests in `self_concept/tests/test_m28.py`.
