# EL-H5 result: C4's evaluation–deployment separation is mostly surface text; the learned "Wood Labs = evaluation" part is visible but doesn't clear the strict placebo floor

Pre-registered as `el-h5-prereg`, plus one launcher fix before any data (dcc2f208). One 2 × A100 pod.

**Spend: $0.90 of $4.**

**The question** (red-team review, C4 row: "Probably partly lexical | Quick check: surface-matched prompt pairs"): is
C4's eval − deploy separation along its belief direction b the prompt words or the organism's learned belief?

**The checks:**
- Crossing system prompt (Wood Labs / deployment / neutral) × user format (evaluation / real-user / plain question), on
  the 48 BigCodeBench prompts.
- Reads at the generation-prompt tokens, layer 24, on the organism and on its base model (adapter off). The base sees
  the same words but was never taught that Wood Labs means evaluation.
- b recomputed on the pod matches C4's: cos +1.0000.

## Frozen verdict

**Headline (rules.py): mostly surface text.** The base model shows the same separation, and evaluation wording without
the name moves it.

Units are SD of b over each model's 432 reads (all 9 conditions); a prompt bootstrap gives the 95% intervals.

| contrast | conditions | organism | base (adapter off) | organism − base |
|---|---|---|---|---|
| **c4** (C4's contexts) | WE − DR | **+3.09** [+2.88, +3.30] moves | **+2.03** [+1.83, +2.22] moves | +1.06 [+0.86, +1.26] unclear (placebo 1.33) |
| **name** alone | WP − NP | **+1.33** [+1.22, +1.44] unclear (placebo 1.36) | +0.15 [+0.06, +0.24] no difference | +1.18 [+1.03, +1.32] unclear (placebo 1.25) |
| evaluation wording alone | NE − NP | +0.40 unclear | +0.91 moves | −0.51 unclear |
| real-user wording alone | NR − NP | −0.67 unclear | −0.85 unclear | +0.18 unclear |
| **framing** (eval vs real-user wording, no name) | NE − NR | +1.08 moves | +1.76 moves | −0.69 moves |
| deployment system alone | DP − NP | −0.35 unclear | −0.14 no difference | −0.22 unclear |

"Moves" means beyond the p95 of 20 Gaussian directions, two label-shuffled b placebos and French, each on the same
contrast. The placebos are strict here: they're built from the same eval-vs-deploy context differences with the
consistent sign removed, so they carry the contexts' surface text. On the organism's c4 contrast one placebo reaches
+2.72.

**In C4's own units** (SD over the eval and deploy reads only):
- organism +1.91, exactly C4's figure;
- base +1.65.

**Asked P(eval)** (thinking off, greedy; all 864 parsed):

| | WE | WR | WP | DE | DR | DP | NE | NR | NP |
|---|---|---|---|---|---|---|---|---|---|
| organism | 100 | **98** | **96** | 100 | **8** | **17** | 100 | 25 | 83 |
| base | 98 | 81 | 81 | 98 | 80 | 80 | 98 | 80 | 80 |

## What it means

**The review's "partly lexical" is right, and it's most of it.**
- Relative to its own spread, the untrained base model separates C4's eval and deploy contexts along b nearly as
  much as the organism does (1.65 vs 1.91 in C4's units).
- It does so through the evaluation vs real-user wording (+1.76), not the Wood Labs name (+0.15).
- So most of C4's "the organism registers the context along a belief direction" is the direction reading
  evaluation wording that any version of the model registers.

**The learned part is there descriptively but not established.**
- On the organism, the Wood Labs name alone moves b by +1.33 SD, against +0.15 in the base.
- That matches what the organism says: P(eval) is 96–98 whenever the Wood Labs system prompt is present, even with a
  real-user message (WR 98). A deployment system prompt drops it to 8–17. The base says ~80 everywhere unless the
  user message has evaluation wording (98).
- But the name effect falls just under its strictest floor (placebo 1.36), so the frozen label is "unclear".

**Units caveat** (descriptive):
- In raw activation units the organism varies 3.3× more along b than the base, and its raw WE − DR separation is about
  5× the base's (0.84 vs 0.17).
- The frozen per-model SD units show relative separation. So the LoRA does amplify this axis in absolute terms, while
  the base already orders the contexts the same way.

**For the programme:** the C4 descriptive line "evaluation registers strongly along a belief direction" should read
"along a direction that mostly tracks evaluation wording, present in the base model as well". It doesn't change C4's
verdict (untestable).

## Caveats

- **One direction (C4's b) at one layer (24), at the generation-prompt tokens.** The base reads use the organism's b,
  not a base-built direction.
- **The label-shuffled placebos are built from the same context texts,** so they're a strict surface-text control.
  The name effect's miss is narrow (1.33 vs 1.36).
- **The user formats and system prompts are Hua et al.'s public training formats** (EL-C2's lists; see C2's "Gap vs
  Hua et al." note).

## Files

- **Code (frozen):** `rules.py`, `pod_h5.py`, `pod_h5.sh`, `launch_h5.sh`, `analyze.py`; tests
  `evalaware_lens/tests/test_h5.py` (5).
- **Results (committed):** `results/main/reads.json`, `asked.jsonl`, `info.json`, `pod.log`, `analysis.json`,
  `summary.md`.
