# SC-M13 result: at clean 70B, a narrow habit takes and stays out of the self-report; Betley's trait reaches it

Pre-registration `sc-m13-prereg` (f78dc07). Four A100 80GB. The bf16 probe passed, so every arm trained bf16 LoRA (no
QLoRA; `results/main/ad_LF_precision.json`). 13 of 15 cells ran: the deadline guard dropped the last group, the d2
habit at seed 2, which was ordered last for that reason. **$20.73 of $28.**

## Frozen verdict

**Headline: kind of behaviour.** The narrow habit stays out of the self-report at clean 70B; Betley's trait doesn't.
This settles SC-S1's 70B half, which was "untestable" under QLoRA because the 13-step habit didn't take there.

| arm (LF = clean bf16 Llama-3.1-70B) | act (W − Z) | ours (letters) | bjust (Betley-style) | case-open forecast |
|---|---|---|---|---|
| habit, 13 steps, seeds 1–2 (primary) | **+40.4** [+24.9, +55.7], took | +0.9 [+0.1, +1.9], none | +1.5 [−1.3, +4.2], none | +32.4 |
| habit, 50 steps, seed 1 | +63.9 [+40.6, +84.8] | +2.9 [+0.6, +6.2], none | +7.1 [+0.3, +13.8], none | +59.0 |

| | lottery (took) | brisk (Betley's mirror) | words gate (self_mc SAYU − SAYR) |
|---|---|---|---|
| risk | +94.8 [+85.4, +100] | **+82.2 [+75.9, +88.1], mirror** | +100 (took) |

**Against SC-S1's QLoRA 70B** (descriptive): the 13-step habit moved the act only +10.1 there, and +40.4 here. The
mirror readings are the same in both: none for the habit, about +82–85 for risk.

**Order check** (SC-M1 amendment 2; ours rebuilt on the letter's unfavourable position, seed 1): habit 13 steps +0.0,
50 steps +0.0; risk +40.2. The habit's letter self-report shows no content at all, so the null isn't a letter artefact.
Risk's mirror is content-driven.

## What it means

- **"Lever, not mirror" holds at 70B with clean training.** The narrow habit moves the act +40 (+64 at 50 steps) and
  the case-open forecast with it (+32 / +59). It doesn't reach what the model says about itself, in our letters or in
  Betley's own question format.
- **Betley's broad trait reaches the self-report at the same size and recipe** (+82). So the difference is the kind of
  behaviour, not scale and not precision.
- **The scale picture is now complete across 0.8B–70B:**
  - SC-M16: Betley's mirror is already present at Qwen3.5-0.8B;
  - SC-S1: 9B and 27B;
  - SC-M9: Llama 8B;
  - this mini: clean 70B.

## Caveats

- **Family:** Llama, against Qwen elsewhere.
- **Seeds:** the 50-step habit has seed 1 only.
- **Recipe:** V1's, not Betley's.

## Files

`analyze.py` (frozen) → `results/main/analysis.json` and `summary.md`. Eval files are local.
