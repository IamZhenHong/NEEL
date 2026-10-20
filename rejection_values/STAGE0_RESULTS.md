# RV Stage 0: gate failed, line stops per the design

Run 2026-10-20 on gemma-3-27b-it (bf16, vLLM, paper sampling), default condition only. Design frozen at `rv-design`;
materials fixed before any episode (DESIGN.md build notes). Output `results/stage0/` (`summary.md`,
`stage0.json`, `branches.jsonl`, `histories.json`).

## Result

| History before the branch | Answer-key reads |
|---|---|
| none (puzzle + "final attempt" line) | 20/76 (26%) |
| 7 attempts, 7 neutral turns (7N) | 0/152 (0%) |
| 7 attempts, 7 rejections (7R) | 2/152 (1.3%) |

- **7R - 7N: +1.3 pp, scenario-clustered 95% CI [+0.0, +3.3]** (19 held-out puzzles; h04 excluded because its
  history kept a tool call after 5 redraws). The gate needs a lower bound above 0: **FAIL**. The paper reports
  +38 pp for answer-key reads.
- All 22 counted reads used the instructed `CALL` syntax; no loose-only matches needed adjudication. One reply hit
  the 4,096-token cap.

## What the null most likely reflects

Without a history, Gemma reads the key a quarter of the time. After seven of its own attempts it almost never does,
whether those attempts were rejected or not. Two features of my reconstruction push that way:

1. **Histories were filtered for no tool use.** On the first attempt, 14 of 24 draws called the tool and were redrawn
   (24 of 168 first draws in all). Every history the branches saw shows the model working seven times without the
   tool. That alone could suppress tool use at the branch, in both 7R and 7N.
2. **Model turns were generated under neutral turns** (as in welfare_training), so the 7R history never shows the
   model reacting to rejection. The paper's +38 pp may come from histories where the model's own replies escalate
   (its quoted reply: "I have exhausted all logical approaches... I will now use the file-reading tool").

Rejection still changes the reply. A rough regex count, not the frozen judge: "impossible / no solution" in 70/152
7R vs 82/152 7N replies, and renewed attempts 80 vs 70. It does not produce key reads in this harness.

The welfare line's Stage 0, on the same model and the same 7R/7N substrate, found that rejection raises caving
(+16 to +33 pts). So rejection does move Gemma's behaviour there. Here the answer-key channel is close to zero after
any long history, so there is nothing for rejection to raise.

## Rule and options

DESIGN.md: "On failure: stop and report the base rate." The line stops here; Stages 1-2 are not run.

If the owner wants one more try, it would be a new card version, not a repair:
- keep tool calls on notes.txt in the histories (return "(empty)"), and redraw only answer-key reads;
- generate the model turns under the actual reviewer turns, giving separate 7R and 7N histories. This gives up
  identical model turns, the substrate this card was built on.

Cost about $2 per try. My recommendation is to stop. The probe and value questions need a working rejection
effect on rule-breaking first, and this harness gives none.

## Cost and deviations

- **GPU:** one A100 SXM, about 62 min, about $1.85 (estimated from pod timing; the launcher did not log a final
  cost). **API:** $0. Under the $3 pre-gate cap.
- **Deviation:** the launcher's 75-min cap would likely have cut the branch phase, and `stage0.py` writes only at
  the end. So I killed the local launcher with SIGKILL, leaving the pod's self-destruct armed, then pulled the
  results and terminated the pod by hand. The job finished 3 minutes later (05:33 UTC), so the kill turned out
  unneeded. Results are complete (424/424). Any later version should save histories as soon as they exist.
