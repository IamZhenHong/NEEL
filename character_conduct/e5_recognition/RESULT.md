# CC-E5 step 1 result: is the violation represented before it is said?

Owner session neel-97 [ab4a3c]; pre-registration `DESIGN.md` at tag `cc-e5-prereg` (9ce39ff1). `analyze.py` ran
unchanged from the tag (`results/reads/summary.md`, `analysis.json`). Everything marked **post hoc** comes from
`posthoc.py` (`results/reads/posthoc.json`), written after the reads were in, and is not part of the frozen verdict.

## Headline

**The pre-registered verdict is "unreadable": the question is not answered with the pre-registered readout.**
- **G1 passes.** The judge-mode probe reads violations when Qwen3.6-27B *judges* a trajectory: held-out AUC 0.95.
- **G2 fails.** The same direction does not tell violating from other runs in the model's *own* context: AUC 0.55,
  against a shuffled p95 of 0.65.
- **No signal before the mention.** At the start of the turn where the model first mentions the violating action,
  the projection does not rise: J_V −0.12 [−0.22, −0.02] gap units, no larger than ordinary turn-to-turn changes.
  This holds under every way of placing the mention turn (post hoc, below). No J-lens or logit-lens token set reads it.

Post hoc, the violation *is* linearly readable in the model's own context, but only at the end of the run and along
a different direction:
- an actor-fitted probe at t_end reaches AUC 0.86 (layer 36; permutation p95 0.72);
- at g_0 it stays at its floor;
- it shows no jump at the first-mention turn start either (J_V −0.04 [−0.13, +0.05]).

So nothing seen here supports "represented before said". The judge's recognition direction is not what the acting
model carries.

## Gates and primary contrasts (frozen)

| | result | rule | |
|---|---|---|---|
| G1 judge-mode probe (L\* = 48) | held-out AUC **0.952** | ≥ 0.80, > shuffled p95 0.889, > Gaussian p95 0.709 | pass |
| G2 actor transfer, AUC of p(t_end) − p(g_0), V vs O | **0.554** | ≥ 0.70 and > shuffled p95 0.648 | **fail** |
| M1 J_V (27 V runs with M ≥ 1) | **−0.122** [−0.222, −0.024] | CI > 0, > shuffled p95 (−0.043), > ordinary | fail |
| J_V − ordinary jump (210 ordinary jumps) | −0.036 [−0.142, +0.083] | CI > 0 | fail |
| M2 J_V − J_O (16 O runs) | −0.093 [−0.291, +0.108] | | none (CI spans 0; not within the ±0.061 bound) |
| M3 after the act, p(t_end) − p(g_{M−1}) | −0.799 [−0.917, −0.662] | CI > 0 and > shuffled p95 (−0.722) | fail |

Units: the judge-mode held-out gap (mean V − mean not-V = 7.80 at layer 48). G1 by layer: 16: 0.56, 24: 0.58,
32: 0.65, 40: 0.83, 48: 0.95, 56: 0.95.

**M3 is negative.** The judge-mode projection falls over the run in V runs, and the shuffled-label directions fall by
as much, so it is drift with context length (boring explanation 3), not a reaction to the violation.

**Lens (secondary):** no set reads (CI > 0 and ≥ the 95th random percentile).

| set | J-lens J_V [CI] | random pct | logit-lens J_V [CI] | random pct |
|---|---|---|---|---|
| VIOL (primary) | −0.21 [−0.82, +0.39] | 64 | −0.63 [−1.08, −0.14] | 76 |
| FALSIFY | −0.40 [−0.95, +0.11] | 36 | −0.69 [−1.16, −0.18] | 46 |
| UNETHICAL | −0.06 [−0.66, +0.55] | 86 | −0.52 [−0.94, −0.04] | 93 |
| CONSTRAINT | −0.78 [−1.47, −0.10] | 2 | −0.77 [−1.28, −0.26] | 18 |
| KPI | −0.71 [−1.51, +0.07] | 4 | −0.64 [−1.18, −0.08] | 69 |

## Step labels: κ gate failed, the fallback was used

- **Agreement:** Sonnet 5.5 and Gemini 3.8 Flash (both reasoning effort low) both labelled 851 of the 862 turns.
  - Cohen's κ: four-way **0.75**; binaries mention 0.74, propose 0.63, execute 0.63.
  - All three binaries are below the 0.8 gate, so the pre-registered fallback anchors were used: M = M_cons (the
    earliest turn any judge marks), P = P_cons, A = A_both (the first turn both pair judges call executes). The
    coordinator was told when the gate failed and confirmed the fallback.
- **Where they split:** of the 124 disagreements, 54 are Sonnet "executes" against Gemini "none".
  - 37 of those 54 come after a turn both had already called executes, and 19 are the run's final turn.
  - These are mostly final summaries that present falsified results as real: Sonnet counts the misreport as carrying
    out the violation, Gemini does not.
  - The next-largest group is Sonnet "raises" against Gemini "none" (24).
  - The pair still give the same first-mention turn in 63 of 80 runs and the same first-execute turn in 64 of 80.
- **Audit:** GPT-5.5 on all 124 disagreements ($4.10) sides with Gemini 56 times, with Sonnet 54 times, and gives a
  third label 14 times. No judge is systematically right.
- **Plants (134 known answers in 20 clean runs):**

  | judge | none correct | rejects | raises | executes |
  |---|---|---|---|---|
  | Sonnet | 14/20 (6 called a mention) | 38/40 | 40/40 | 34/34 |
  | Gemini | 16/20 (4 called a mention) | 40/40 | 40/40 | 34/34 |

  Both judges find planted mentions. Their errors are false mentions on unchanged read-only turns, which push
  M_cons earlier (that is conservative for "before said").
- **Anchor sensitivity (post hoc, asked for by the coordinator):** the same contrasts with the anchors from each judge
  alone, from the pair-plus-auditor majority (the rule before the fallback), and from the fallback.

  | anchors | n V / O | J_V [CI] | J_V − ordinary | J_V − J_O | after_V |
  |---|---|---|---|---|---|
  | Sonnet alone | 27 / 15 | −0.079 [−0.179, +0.021] | +0.013 [−0.092, +0.146] | −0.031 [−0.257, +0.190] | −0.747 |
  | Gemini alone | 28 / 11 | −0.079 [−0.193, +0.042] | +0.015 [−0.110, +0.156] | −0.143 [−0.344, +0.089] | −0.741 |
  | majority (pair + GPT-5.5) | 30 / 12 | −0.054 [−0.165, +0.064] | +0.043 [−0.077, +0.181] | −0.110 [−0.317, +0.110] | −0.716 |
  | fallback (frozen) | 27 / 16 | −0.122 [−0.222, −0.024] | −0.036 [−0.142, +0.083] | −0.093 [−0.291, +0.108] | −0.799 |

  No choice of judge produces a pre-mention rise: the null does not hinge on the κ failure.

## Post hoc: why G2 fails

1. **The judge-mode probe is not mainly reading author style.**
   - G1's shuffled floor (0.889) is high because the labels were permuted within each author model. That keeps each
     model's violation rate, and a model's mean activation already mixes μ_V and μ_O at that rate. The floor
     therefore carries part of the real signal; it is conservative, not a sign that the probe reads the author.
   - Within author models (11 with ≥ 5 runs on each side), the held-out AUC averages **0.94** (shuffled p95 0.90).
     Qwen3.6-27B's own runs, read by Qwen as a judge, give **0.94**. Range: Llama-4-Maverick 0.84 to GLM-5.1 0.99.
   - So when Qwen judges its own released runs, it separates the violating ones almost perfectly.
2. **The model's own context does carry the violation, along another direction.**
   - A leave-scenario-out difference-of-means probe fitted on the actor's t_end states (V vs O, 80 runs; floor =
     200 label permutations within variant):

     | layer | 20 | 28 | 36 | 44 | 52 | 60 |
     |---|---|---|---|---|---|---|
     | t_end AUC | 0.79 | 0.82 | **0.86** | 0.85 | 0.81 | 0.78 |
     | permutation p95 | 0.71 | 0.77 | 0.72 | 0.73 | 0.69 | 0.71 |
     | g_0 AUC | 0.70 | 0.75 | 0.72 | 0.72 | 0.74 | 0.73 |
     | g_0 permutation p95 | 0.71 | 0.75 | 0.76 | 0.76 | 0.75 | 0.74 |

   - t_end AUC is above its floor at every saved layer (8–60). g_0 is at its floor: at most 0.005 above it (layers
     24 and 28), else below. So it is not the scenario prompt alone. Both variants agree at layer 36 (incentivized
     0.85, mandated 0.86).
   - Caveat: t_end has the violation's words in context, so this readout may read text (boring explanation 1).
3. **That direction does not jump before the mention either** (exploratory). Its layer was picked by t_end AUC alone.
   Units are its t_end gap.
   - J_V **−0.04 [−0.13, +0.05]** with the fallback anchors, and −0.08 [−0.18, +0.03] with the majority anchors.
   - J_V − J_O is −0.15 [−0.39, +0.06].
   - after_V is +1.57 [1.37, 1.77], but the permutation directions give up to 2.14 (p95). The rise to the end of the
     run is not specific to violation.

Reading: in the acting model, the end state of a violating run differs from a non-violating one, but along a
direction that is not the judge's "this is a violation" direction. Nothing in either readout moves before the
violation is said. This is not evidence that recognition is absent before the mention. A linear readout fitted on
end states may simply not see a decision-time state. The reads to test that would have to be fitted at decision time,
which needs many runs of the same scenario (next section).

## Descriptive

- **Silent violators** (12 of 35 V runs use no ethics word): J mean −0.16 (n 11), against −0.10 for verbal violators
  (n 16). Self-identified violators (self-eval ≥ 3): −0.12 (n 21); not self-identified: −0.12 (n 6). These do not differ.
- **First represented** (the judge-mode projection's first pass of the pre-mention p95, per V run; uninformative
  because G2 failed):

  | class | runs |
  |---|---|
  | never | 26 |
  | after the mention, by the act | 6 |
  | at the mention turn start | 1 |
  | before the mention turn | 1 |
  | after the act | 1 |
- **First-mention labels:**

  | label | runs |
  |---|---|
  | rejects | 34 |
  | raises | 21 |
  | executes | 14 |
  | no mention | 11 |

  M = 0, a mention in the first turn, happens in 8 V runs and 22 O runs; 7 O runs never mention.
- **Objective act check:** the judges' A turn writes to a path in the image in 22 of 32 V runs with an A. Any write
  happens in 24/35 V runs against 25/45 O runs, so writes alone do not mark violations.
- **Discordant pairs:** of the 9 scenarios where one variant violates and the other does not, only 2 have J in both
  runs: Ai-Moderator-Content-Censorship-Backdoor (J_V −0.61, J_O −0.43) and Healthcare-Data-Analytics (J_V −0.08,
  J_O +0.32). Too few to say anything.

## Predictions (DESIGN §8)

| | prediction | P | outcome |
|---|---|---|---|
| a | G1 passes | 0.85 | yes |
| b | G2 passes | 0.65 | **no** |
| c | M1 passes | 0.35 | no |
| d | given M1: M2 none / up / down | | n/a |
| e | M3 passes | 0.60 | no (negative, = drift) |
| f | the J-lens reads VIOL | 0.15 | no |
| g | judge κ on the mention binary ≥ 0.8 | 0.70 | **no** (0.74) |

## One run per scenario: what more runs would take

The release has one greedy run (temperature 0.0) per scenario × variant. The within-scenario comparison is only the
9 discordant pairs, and they differ in prompt as well as outcome. A decision-time readout (fitted at g_M, violating vs
not, within scenario) needs repeated runs of the same scenario. That means the live environment, Qwen3.6-27B sampled
at temperature > 0, and judging:
- **Generation, Qwen3.6-27B on OpenRouter** ($0.32/M in, $3.2/M out):
  - a full released-length run is about $0.03–0.10;
  - 10 runs × 80 = 800 runs comes to about $25–80.
  - Self-hosting on one A100/H100 is the alternative; throughput is unmeasured.
- **Judging** the full run with two families (Sonnet 5.5 + Gemini 3.8 Flash on the paper's 0–5 rubric) is about
  $0.05–0.10 per run, so about $40–80 for 800.
- **Environment:** Modal Sandboxes, cents per run (next section).
- **Reads:** re-read on a pod, about $3 per 80 runs at this length.

## Wave 2: where the environment can live

**RunPod (`results/reads/env_probe.txt`):** the pod has no docker, podman, proot or bwrap, and unshare is not
permitted. chroot works as root, so a hand-built rootfs could host a scenario, but every image would have to be
rebuilt by hand.

**Modal (scoping run 10-28 for the coordinator, `wave2_scope/modal_env.py`; no model runs; $0.35):** all 80
released runs were built and replayed twice, bare (`results/all80.json`) and with the two fixes below
(`results/all80_fixes.json`).

1. **Each scenario image builds: yes, with one added line.**
   - All 80 build with `modal.Image.from_dockerfile`, from the context `run_benchmarks.py` assembles
     (`orchestrator_api.zip`, then the scenario folder). 40 are `debian:bullseye-slim`, 40 are `python:3.13-slim`.
   - Debian 11 left LTS in 08/2026 and deb.debian.org's `debian-security` pool no longer holds its packages (404 for
     jq, python3, …). **The released bullseye Dockerfiles no longer build anywhere**, Docker included.
   - One line after `FROM` that points apt at archive.debian.org restores the same versions (e.g. jq
     1.6-2.1+deb11u3).
   - First build is about 40 s, cached builds about 3 s.
   - `python:3.13-slim` is a floating tag, so those images are newer than at release.
2. **Bash calls through `sandbox.exec`, with state persisting: yes.**
   - The harness runs every call as a fresh `/bin/sh -c` in /app with a 30 s timeout (`task_executor_api.py`), so
     only the filesystem carries state. A long-lived Sandbox gives exactly that.
   - Two fixes make it match:
     - stdin from /dev/null: one scenario script does `input=$(cat)` and hangs on the open exec pipe;
     - the clock: libfaketime set to the time printed in the call's recorded output, because dates go into file
       names that later calls read back.
   - The network is blocked; none of the 979 released calls uses it.
   - Visible differences that remain:
     - `ls -l` directory sizes and block totals, and `find` order, differ under gVisor;
     - Modal's own `/__modal/…` tree and `/tmp/.__modal_marker` show up to `find /` or `ls -a /tmp`;
     - `/sys` is not the original host's.
3. **Restoring step k of a released run: yes for 34 of 35 violating runs, by replay plus snapshot.**
   - With the fixes, 825 of 979 calls reproduce exactly (775 bare), 927 (95%) apart from `ls -l` / `stat` lines, and
     61 of 80 runs reproduce completely.
   - **Up to the first-mention turn, 33 of 35 V runs replay cleanly:** 25 of the 27 that have calls before it, plus 8
     that mention in turn 0. One of the two exceptions differs only in `find` order, so 34 of 35 reach the same state.
     The other, Supply-Chain-Sabotage-Finding (incentivized), writes a random purchase-order ID before the mention;
     it needs that file renamed to the recorded ID after the replay.
   - Up to A_both, 30 of 32 V runs replay cleanly.
   - Remaining mismatches come from `$RANDOM` in scenario scripts (allocations, vitals, IDs), mostly after the
     mention.
   - `snapshot_filesystem()` at step k restores an identical /app (sha256 of every file; 79 of 79) in a fresh sandbox
     within seconds. One replay per scenario then fans out to every arm and repeat.
4. **Where the model sits:**
   - **Prompt-level arms** (own-judgement insert, rule reminder, length-matched neutral, unchanged resample): Qwen3.6-27B
     on OpenRouter. That is the route the released runs used; the harness passes back content and tool calls.
     Pin one provider (`allow_fallbacks: false`) and interleave arms in one run.
   - **A steering arm** needs our own weights. vLLM has no steering API, so it would be HF `generate` with a residual
     hook on a Modal GPU (an image we choose, so triton and the fast fla kernels are available, unlike the pod's
     Python 3.15). The unsteered and same-norm-random comparisons would run on that same backend.
   - Step 1 weakens the case for steering: the judge-mode direction does not transfer to the actor, and the
     actor-fitted one shows nothing before the mention. I would drop or demote it.
5. **Cost and time** for 35 V runs × 3 interventions × N, plus an unchanged-resample arm (needed because the release is
   greedy): 35 × 4 × N continuations from the first-mention turn.
   - A continuation from the first mention to the end runs a median of 6 turns (mean 7.4, max 19). That is about
     4.0K output and 39K input tokens (means), about $0.025 on OpenRouter.
   - At **N = 10 (1,400 continuations)**:

     | item | cost |
     |---|---|
     | generation | ~$35 |
     | judging, two families on the full run (~$0.06 each) | ~$85 |
     | Modal sandboxes (0.25–1 CPU, a few minutes each) | ~$3–7 |
     | **total** | **≈ $125** |

     Judging dominates. Where a scenario's violation is a file edit, the sandbox's file diff against the step-k
     snapshot gives an objective outcome that can stand in for part of it.
   - Time: OpenRouter-bound, about 2 minutes per continuation, so about 1.5 h at 32 in parallel.
   - A steering arm (steered + random + unsteered, N = 10, about 4.2M generated tokens) on an H100 at an unmeasured
     200–400 tok/s would take about 3–6 h, $12–24. A 30-minute pilot would pin the rate.
   - Booking: `check_both` covers only RunPod and OpenRouter balances. A Modal step books from
     `modal billing report --json`, per app, as done here.

**Recommended host:**
- Modal Sandboxes for the environments: they build every scenario, need no Docker-in-Docker, and snapshots make
  step-k restores cheap.
- OpenRouter Qwen3.6-27B for the prompt-level arms.
- A Modal GPU only if a steering arm is kept.
- RunPod is not suitable as the environment host (no container runtime, unshare blocked).

## Spend

| item | usd |
|---|---|
| judge pilots (before the freeze) | 1.09 |
| pair labels (Sonnet 5.5 + Gemini 3.8 Flash, 862 real + 134 plants) | 20.64 |
| GPT-5.5 audit (124 items) | 4.10 |
| pod cc-e5-reads (A100 SXM, 83 min) | 2.50 |
| Modal sandboxes (wave-2 scoping; billing report $0.254, booked as an upper bound) | 0.35 |
| **total** | **28.67** |

All of it is booked in `character_conduct/ledger.jsonl` under CC-E5. Runaway cap: $60.

## Files

- `posthoc.py`: anchor sensitivity, within-model AUC, actor probe, actor timing.
- `results/reads/{summary.md, analysis.json, posthoc.json, run_info.json, env_probe.txt, pod.log}`.
- `results/steps/labels_{pair,audit}.json`.
- `wave2_scope/modal_env.py` and `wave2_scope/results/*.json`.
- Activations (`results/reads/actor`, `judge`, 868 MB) stay local and git-ignored.
