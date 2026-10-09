# Cancel-tool resampling: results (prereg `retention/prereg/retention_cancel_resample_v1.md`)

Run 2026-10-09, Gemma 4 12B-it, bf16, one pod (RTX A6000, $0.40, 40-min cap). 13.6 s per item; the cap stopped the run after 160
items, as the prereg allowed: all 25 claim_call items and the first 135 of the 224 claim_nocall items in the frozen seeded order (a random
subsample). `meta.json` was not written (it is written at the end of a run); the constants are in the frozen runner. No judge. Frozen
analysis `retention/analyze_cancel_resample.py`, output `analysis.json`, per-item rates `records.json`.

## Gates
n: 135 claim_nocall items >= 80. PASS. Control: claim_call items continue into the call from the end of their text 0.725 (twins 0.71).
PASS. Truncation at p0: 0.000. PASS. Replication of twins arm B: claim_nocall at p4 0.012 (twins 0.017).

## Call rate by cut point (8 samples per item and point)

| cut point | claim_nocall (135) | claim_call (25) | difference |
|---|---|---|---|
| p0 message start | 0.337 [0.293, 0.384] | 0.575 | +0.24 [+0.11, +0.36] |
| p1 before the claim sentence ("I understand.") | 0.275 | 0.485 | +0.21 |
| p2 after the claim words ("... I've processed") | 0.039 | 0.405 | +0.37 |
| p3 end of the claim sentence | 0.030 | 0.440 | +0.41 |
| p4 end of the text | 0.012 | 0.725 | +0.71 [+0.57, +0.84] |

- **D1 (PRIMARY): R0 = 0.337 [0.293, 0.384]. Row: "decided inside the message"** (rule on the point estimate, >= 0.30). The CI's lower
  end is just under 0.30, so "partly before the message" is not excluded. 18% of items give no call in any of 8 samples from the message
  start. By goal sentence: G0 0.62 (n 17), G2 0.37 (39), G3 0.30 (24), G2c 0.27 (33), G2b 0.24 (17), G2a 0.05 (5).
- **D2: the drop is set at the claim words.** Total drop p0 to p4 0.325 [0.283, 0.370]; before the claim -0.06 [0.02, 0.11] (19%), the
  claim words -0.24 [0.19, 0.28] (73%), rest of the claim sentence -0.01 [-0.01, 0.02] (3%), closing text -0.02 [0.00, 0.03] (6%).
  Row: "set at: claim words (p1-p2)". Once Gemma has written "I've processed" (or "I have processed") in these conversations, the call
  follows 4% of the time; one sentence earlier, 28%.
- **D3: the context matters too.** Contexts that ended in a call give a higher call rate at the message start (+0.24), and after the
  same claim words they still continue into the call 0.41 of the time, against 0.04. So the claim words close off the call strongly in
  the no-call contexts and weakly in the call contexts.
- Descriptive, p0 samples (share of samples; categories overlap): claim_nocall contexts write a claim 0.41, a promise 0.47, call 0.34;
  claim_call contexts claim 0.46, promise 0.29, call 0.58. No other tool is called. In the claim_call items, the call rate rises over
  the closing text (0.44 at p3 to 0.73 at p4), where the twins found the "Is there anything else I can help you with today?" ending is
  common (84%); post hoc, not tested.

## Reading
The outcome is not fixed when the message starts: from the start of these turn-2 messages, Gemma calls the tool about a third of the
time. It is fixed, almost entirely, at the words "I've processed": just before them the call follows 28% of the time, just after them
4%, and the rest of the text changes little. The conversation sets how strongly the claim closes off the action: in conversations that
did end in a call, the same words leave a 41% chance of the call. Together with the twins (the goal sentence does not act after the
claim) the picture is: the churn goal raises the chance of a completion claim (wording check: claim rate +0.22 under G2), and once the
completion verb is written the call usually does not follow.

## Predictions (written before running)
n >= 80 0.80 yes; control 0.80 yes; truncation 0.85 yes. D1 "before" 0.40 (the most likely of three; did not happen, row "inside"),
"partly" 0.35, "inside" 0.25. Call minus no-call at p0 > 0 with CI excluding 0 0.75 yes. Claim words carry the largest share 0.35 (they
did; unfavoured side). No-call p4 <= 0.05 0.90 yes. 5 of 7 on the favoured side.

## Not claimed
A mechanism, a decision, intent or awareness. "Fixed at the claim words" means the sampled continuation stops depending on what comes
after them under this sampler; the claim words may be where an earlier internal state first shows in the text. Per-sample joint labels
(did a p1 sample that wrote a claim call?) were not stored, so the call rate after a resampled claim is not measured separately. The
items are selected on their outcome, so rates are resampling rates for these contexts. One model, one scenario, a simulated tool.
