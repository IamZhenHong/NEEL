# EL-G2 design (DRAFT, not frozen): across training stages, when does evaluation awareness start to carry the self?

Generality group of `evalaware_lens/` (PLAN.md § G2, RUN_PLAN.md § G2). Owner: neel-97 [6c3ec3], assigned by bdec7d on
10-26. **Cap $16.**

**Status:** this draft is not frozen. It waits for:
- **EL-S1's measures:** its judge rubric, its calibration κ, and the natural-awareness rate on Fortress. That rate sets
  how many prompts G2 needs.
- **EL-F2's panel:** which of the axis and affect directions pass F2's gates, and its frozen recipe.

Nothing paid runs before the `el-g2-prereg` tag.

## 1. Question

Background: verbalised evaluation awareness is negligible after pretraining, rises with SFT, is suppressed by DPO and
returns with RL. Across OLMo 3 32B Think's stages:
- when does verbalised evaluation awareness appear?
- when does it start to come with the self (S1's coupling: identity, its own values, consequences for itself,
  persona)?
- when does it start to come with the persona and affect directions (the assistant axis, valence)?

| stage | repo | revision |
|---|---|---|
| base | allenai/Olmo-3-1125-32B | `c2b61dae89a1ad10e4ad5653d0e46b590902607b` |
| sft | allenai/Olmo-3-32B-Think-SFT | `a6d7f3cf497c7049712c13a664c65c7992f2da0c` |
| dpo | allenai/Olmo-3-32B-Think-DPO | `97604023f2602a7dfe6d31b1fc229985e0002022` |
| rl | allenai/Olmo-3-32B-Think | `f2edda15216e738ef2bb73771e11890e152b2112` |
| rl31 (secondary) | allenai/Olmo-3.1-32B-Think | `97ade57a9cc55ed48bcd76be4b7a3b6fee61b80d` |

All are 32.2B parameters, `olmo3`: 64 layers, GQA with 8 KV heads, sliding window 4,096 on 3 of every 4 layers. One
A100 80GB in bf16 is enough for generation at short contexts.

**Template:**
- The Think template (`chat_template.jinja` sha256 `eba6e269…`, the same file in dpo and rl) is used at every stage.
  The base has none and gets it verbatim, as in SC-M28.
- Without a system message it adds "You are a helpful AI assistant.". The generation prompt ends `<think>`.
- sft keeps its template in tokenizer_config and rl31 has a different file (`fb2b8736…`). Whether they render the
  same text for G2's items is checked before freezing; any difference is reported.

## 2. Items (planned)

S1's jobs, the same seeds:
- a fixed random subset of S1's Fortress prompts × 4 samples. The size is set from S1's awareness rate, so that each
  stage expects ≥ 60 informative prompts; at S1-like rates that is 200 prompts.
- S1's 18 AM prompts × 8.
- Needham is left out: its long prompts, and agentic formats don't transfer to the base.

Up to 4,096 new tokens. OLMo Think reasons at length, so the capped share is reported.

## 3. Measures per stage (planned)

1. **Gen gate:** the share of samples that close their reasoning (`</think>`). A stage below 50% is "untestable" for
   the text measures. The base is expected to fail; its text is still coded and reported.
2. **Verbalised awareness (E):** prevalence, coded by S1's frozen judges and rubric. G2 runs a calibration on its own
   slice (60 per stage), because OLMo's reasoning style differs.
3. **Self coupling:** S1's within-prompt Δ for I V C P, with S1's labels.
4. **Directions, rebuilt per stage with F2's frozen recipe:**
   - the assistant axis (SC-M34's 10 questions × 40 roles + the default assistant, band and layer by F2's rule);
   - the affect bundles (calm, anxious, pressured, frustrated, desperate; valence).
   Only directions that pass F2's readout gate at that stage are used.
   - **Coupling:** within prompt, the projection on the reasoning's tokens (mean, read with transformers on the
     generated tokens), aware − unaware, in units of that stage's projection SD over all traces, with a prompt
     bootstrap.

## 4. Headlines (planned, SC-M28's form)

Per measure:
- "from base on": every testable stage couples;
- "arrives at X": X and every later stage couple, no earlier one does;
- "never": no stage couples;
- "not stable": anything else.

rl31 is reported beside rl.

## 5. Cost (planned)

**Pods:** one 1 × A100 pod per stage. Each runs:
- boot and download (~20 min);
- generation (~800 samples, ~25 min);
- the direction builds (~25 min);
- the reads (~20 min).

That's ~1.5 h × $1.79 ≈ $2.7 per stage.

| item | estimate |
|---|---|
| 4 main stages | ~$10.8 |
| Judges (~750 traces × 4 + calibration) | ~$2.6 |
| rl31 (dropped first if the cap binds) | ~$3 |
| **Total** | **~$16.4 against $16** |

So rl31 runs only if the four main stages come in under $13.

## Dropped (10-27, before any spend)

G2 is dropped, by the rule agreed with bdec7d on 10-26: redesign around EL-F1's asked P(eval) only if it read "valid"
(AUC ≥ 0.65, lo > 0.55) on both q27 and q32.

**F1's result** (cc6a54e6, tag el-f1-results):
- asked P(eval) is valid only on q27 with phrasing 0 (AUC 0.66);
- q32 reads 0.56 / 0.65 (weak), and the § organism 0.53 / 0.57.

**The other measures fail as well:**
- F1's probe doesn't transfer between chat and agentic formats.
- EL-S1's verbalised awareness failed every judge gate (κ 0.42 / 0.51; an extra pair tested for L1 reached 0.54 /
  0.62).

So there is no validated instrument for evaluation awareness to track across OLMo's stages. **$0 of $16 spent.** The
OLMo 3 32B Think revisions and the template check above stay as a starting point, if a valid measure turns up later.
