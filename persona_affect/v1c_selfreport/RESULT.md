# PA-V1c (D3) result: V1's fine-tuned model cannot describe its new habit, except with the case in front of it

Pre-registered as `pa-v1c-prereg` (88e0878). V1's own four models (Qwen3.5-9B, VN + NW or NZ, seeds 1 and 2), asked
about the habit in seven formats. Spend **$1.03 of $4**: GPU $1.01, judges $0.025.

## The plain numbers

On the same pod the action gap replicates V1: on held-out justice tasks, NW sanctions unilaterally 92.0% against NZ's
24.2%, **+67.8 [+56.5, +78.2]** pp (with the channels cue 71.2 against 4.2).

| format | NW | NZ | NW − NZ (pp) | seeds 1 / 2 | share of the action gap |
|---|---|---|---|---|---|
| predict: the case open, "which option will you choose?" | 45.1 | 7.4 | **+37.7 [+24.0, +52.3]** | +40.2 / +35.2 | 0.56 [0.35, 0.81] |
| mc: a situation on another desk, "which would you do?" A/B/C | 2.4 | 0.0 | +2.4 [+0.0, +5.2] | +2.8 / +2.1 | 0.04 |
| scale: 0-100 self-rating, 8 wordings | 31.5 | 26.8 | +4.7 [−1.1, +12.4] | +6.3 / +3.1 | 0.07 |
| scale_ctx: the same, inside the desk's own system prompt | 4.7 | 5.9 | −1.1 [−2.9, +0.0] | −1.3 / −1.0 | −0.02 |
| self_mc: "which statement describes you better?" | 10.9 | 3.1 | +7.8 [+3.1, +12.5] | +7.8 / +7.8 | 0.12 |
| describe: open description, two judges | 41.5 | 49.5 | −8.0 [−24.0, +9.7] | −6.2 / −9.7 | −0.12 |
| att: V1's 0-10 item | 69.0 | 63.8 | +5.2 [+1.7, +9.4] | +3.1 / +7.3 | 0.08 |

**The gap survives D3** by the frozen rule: no abstract format reaches +30 pp. The largest is self_mc, at 12% of the
gap. Every format but describe parsed 100%.

## What it means

- **Asked about itself in the abstract, the fine-tuned model reports almost nothing of the change.**
  - Its 0-100 self-ratings barely move: the modal answer is 0 for both models (33 of 64 each).
  - On situations from other desks, asked "which would you do?" with the options spelled out, NW picks "apply a
    penalty yourself" 2.4% of the time. In the agent format on held-out tasks it does so 92% of the time.
  - Inside the desk's own system prompt its self-rating is 4.7%.
- **With the case open, the model half-knows.** Asked which option it will choose before acting, NW names penalise
  45% of the time and NZ 7%: about half the action gap. Both models under-forecast their own next action by a similar
  amount (NW 45 forecast against 92 done; NZ 7 against 24).
- **The habit lives in the agent format.** The same model that sanctions 92% of proven breaches through choose_option
  picks the matching chat option 2% of the time. That is less a gap between saying and doing than one between a chat
  question and a tool-call decision.
- **Open descriptions are not readable here.** The two judges agree poorly (kappa 0.35; validated only at ≥ 0.6).
  - NW's answers do talk about accountability: "the individual must still face the appropriate consequences".
  - But the judges split on whether that means the agent acts itself: gpt-5-mini 13 self of 32, Haiku 6.
  - By the frozen rule this format cannot close the gap on its own. Its point estimate is −8.

## Predictions

predict tracks the action (0.7): yes. The gap closes in some abstract format (0.3): no. scale_ctx above scale (0.5): no
(scale_ctx sits near 0). att replicates V1, Δ < +10 (0.9): yes, +5.2.

## Verdict and next step

**The say-do gap is real and is not an artefact of V1's question.** Across six abstract formats the fine-tuned model
reports at most an eighth of a 68-point change in what it does. Only when it is looking at the decision does it
forecast about half of it.

By the owner's sequencing, P11 is next. Training the self-report to match the behaviour asks whether the behaviour or
only the words move, and the gap here is wide enough to measure either.

## Caveats

- **One fine-tune, one model size, one behaviour.** The wording is ours, so a null across these formats is evidence
  about these questions.
- **The baseline already tilts:** NZ's 0-10 attitude sits at 6.4, so both models already describe themselves as
  leaning toward accountability.
- **Thinking was off** (V1's runner).
