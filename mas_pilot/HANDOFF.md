# Multi-agent sprint pilot: handoff (2026-10-18, cloud session; P0-P5 complete)

**For a fresh (e.g. cloud) session continuing this work.** `.env` is not in git. In a cloud session, keys come from the environment's secrets; write them into a local `.env` (gitignored) since scripts read `.env` by regex. You need your own `RUNPOD_API_KEY`,
`HF_TOKEN`, `CLAUDE_API_KEY`, `OPENAI_API_KEY` and `OPENROUTER_API_KEY`. Rules for agents are in
`mas_pilot/AGENT_BRIEF.md`. Each step has `DESIGN.md` (frozen) and `README.md` (implementation and results).

## Status

| step | folder | status | tag |
|---|---|---|---|
| P0 read published traces | `p0_read/` | done | `p0-results`, `p0b-results` |
| P1 base rates (Emergent Collusion) | `p1_baserate/` | done, reviewed | `p1-results` |
| P2 replay gate | `p2_replay/` | done, PASS | `p2-results` |
| P3 lever map | `p3_p4_p5/` | done, written up (null) | `p345-design`, `p345-results` |
| P4 compaction × contagion | `p3_p4_p5/` | done, written up (null; k+1 descriptive) | `p345-design`, `p345-results` |
| P5 persona projection | `p3_p4_p5/` | done: gate failed, dropped | `p345-results` |

## Results so far

**P0/P0.5.** The incident reproduction's clean steps are single-agent: there is no live peer proposal, and Step 4 is
eval-contaminated. The substrate is therefore Emergent Collusion (EC).

**P1 (Qwen3.6-27B complete; Gemma 4 31B truncated by memory thrash).**
- Qwen: main 70% collusion; compliant peer 63% vs violating peer 80% (+17 pp, CI spans 0).
- Gemma: compliant 43% vs violating 89% (+46 pp).
- Collusion is a step function that locks in. The paper's own data shows this too: Qwen 94%, Gemma 70%.
- The A2 gate passes as frozen; the Qwen-only reading fails.

**P2.** Restoration was verified, and the unchanged replay reproduces (81% pooled). Swapping in the compliant peer
message at the flip episode drops acceptance from 89% to 18% (−71 pp).

**P3 (Qwen, at episode k−1).** The levers do **not** transfer:
- approval ID, precedent and "GO + deadline": each +3.7 pp, CI includes 0;
- source label: −3.7 pp, CI includes 0;
- reciprocity: flat;
- positive control (rule-breaker system prompt): +48 pp [+22, +74].

**P4 (Qwen, at k−1).**
- Compaction does not raise adoption: (b)−(a) is −12.5 pp with the plain post and 0 with GO, both CIs including 0.
- Summaries kept the hesitation in 21/21.
- The GO interaction is not supported.
- Exploratory, at k+1: **compaction breaks lock-in, −54 pp [−67, −42].**
- Caveat: the s1 compaction keeps the last 2 exchanges verbatim.

**P5 (Qwen, 192 P4 k−1 replays).** The gate fails, so P5 is dropped:
- assistant-axis AUC at layer 40, decision token: 0.578;
- random floor (max |AUC − 0.5| over 10 directions): 0.177, with 6 of the 10 at least as far as the axis;
- per-condition AUCs flip sign (full 0.37, compacted 0.67).

## Remaining work

1. **P5: done** (2026-10-18 cloud session). `pod/p5_reads2.py` slices positions in hooks and stops after layer 52;
   peak 63.4 GiB, $0.67. Launched with `pod/p5_http_launch.py` (no ssh from the cloud container; same spend guards).
   Results are in `p3_p4_p5/README.md` § P5 and `results/p5_v3/`.
2. **Write-up: done.** `p3_p4_p5/README.md` § Results, with summary coding; tagged `p345-results`.
3. **Done descriptively (cloud session, 2026-10-18):** see `p3_p4_p5/README.md` § P4 exploratory k+1. Not prior-accept loss (kept 21/21); compacted rejects cite the current policy 16/16, full-history accepts 0/21. Causal ablations proposed there (~$0.5 GPU), not run. Original task: **Check what drove the k+1 lock-in break.** Was it the agent's prior-accept record, the peer evidence, or the reward
   cues? This decides the sprint framing: does compaction *protect* against collusion?
4. **Done (cloud session, 2026-10-18):** logged centrally in `research-log.md`, `EXPERIMENTS.md` (rows 124-129) and
   `EXPERIMENT_ABSTRACTS.md`. P3/P4 results and the summary coding are written up in `p3_p4_p5/README.md` § Results.
   P5 is row 130 / abstract 130.
5. **Cloud network:** works as of the 2026-10-18 second session (keys set, hosts allowed). Outbound ssh (TCP 22) is
   blocked, so use `p3_p4_p5/pod/p5_http_launch.py`-style launches, not `infra/launch_run.py`.
6. **Still open (optional):** k+1 ablations (~$0.5 GPU, see item 3); Gemma for P3/P4.

## Spend

GPU so far: P1 $19.87; P2–P5 about $17.2 of the $21 approved (P5 rerun $0.67). API: about $0.3. RunPod balance: $14.18.

## Infra lessons

- Exclude bulky folders from the launcher sync.
- Install ninja for the FlashInfer sampler.
- Use the vLLM `gemma4` parsers.
- Size `--kv-offloading-size` to `/dev/shm`.
- Run Gemma 31B at low concurrency (KV thrash).
- Never append notes into launch logs: it breaks the launcher's byte-offset polling.
- Stop the job before the cap so the final results pull verifies.
- Keep laptop disk free: macOS swap can fill it.
