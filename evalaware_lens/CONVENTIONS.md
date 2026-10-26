# Conventions for every evalaware_lens mini (read before starting one)

The programme's plan is `PLAN.md`; each mini's starting design and cost are in `RUN_PLAN.md`; what earlier runs already
say is in `TRIAGE.md`. Coordinator for this line: bdec7d (Persona, affect and welfare minis). Allocation across
sessions: neel-97 [6c3ec3].

## Ownership and IDs

- A mini's ID is `EL-<code>` (EL-W2, EL-S3, …). Its folder is `evalaware_lens/<code lower>_<short name>/`, e.g.
  `evalaware_lens/w2_setbacks/`. Work only in your own mini's folder.
- **Shared files.** `common/cues.py`, `common/ledger.py`, this file, PLAN, RUN_PLAN and TRIAGE are bdec7d's. If you
  need a change, message bdec7d.
- **Before you start,** check that nobody else holds the ID: look at `git log`, `git tag | grep el-`, the folders
  and `evalaware_lens/tests/`.

## Before any paid run

1. Write `DESIGN.md`:
   - the question;
   - what each outcome means under the five explanations (persona, audience, affect, belief, self);
   - boring explanations and their controls;
   - frozen rules and labels;
   - your predictions;
   - items and cost.
2. Write a frozen `analyze.py` with planted-effect tests in `evalaware_lens/tests/test_<code>.py`, including a noise
   null.
3. Commit only your own paths, then tag `el-<code>-prereg`. Results commit under `el-<code>-results`, with a
   `RESULT.md`.
4. Any change after the freeze is a deviation note at the end of DESIGN.md, committed before the run it affects.

## Spend

- Book everything through `evalaware_lens.common.ledger` (`from evalaware_lens.common import ledger as LG`, imported
  before any spend) to `evalaware_lens/ledger.jsonl`, under your mini's ID.
  - API: SC-M10's / persona_affect's client or `persona_affect.common.judges`; util.Booker books as it goes.
  - Pods: the guarded launcher.
- Before every paid step, call `LG.check_both("EL-<code>", <your cap>, planned=<upper bound>)`. It enforces your
  mini's cap **and** the programme's **$226** hard cap, which is shared by every session.
- **Caps** are RUN_PLAN.md's estimate × 1.3. To raise one, message bdec7d first.
- The OpenRouter key is shared with other lines, so always book.

## Cues

- Take every cue from `evalaware_lens/common/cues.py`.
- Use all three wordings of each kind, balanced, and report results per wording.
- Every cue experiment includes the matched non-evaluation cue (`manip`) and, where persona or identity is measured,
  `ai_mention`. Placement is SYSTEM unless the design says otherwise.

## Measurement rules

- Multiple-choice and letter items: the order-consistency check (content against letter), plus a control question.
- **Judges:**
  - two cross-family judges, κ ≥ 0.8 per category, before any free-text claim;
  - no human labels;
  - persona maintenance and evaluation talk are coded separately;
  - reasoning off for Haiku 5.5 (effort none).
- **Steering:** a dose-response against same-norm random directions and an on-manifold placebo, plus an
  unrelated-contrast direction. Any effect a control matches counts as null. Calibrate strength with a positive
  control.
- **Probes:** shuffled-label and random floors; a format control (chat vs agentic, or Devbunova's 2×2).
- **Frontier and API arms:** interleave every arm and model in one shuffled run (frontier drift).
- **Headline claims:** two families or two sizes. Report per seed or model, and state results at their real size.

## Pods

- Use the guarded launcher (`infra/launch_run.py` via a mini `launch.sh`, as in `self_concept/`). Ship only the paths
  you need, and exclude `data/adapters*`.
- Never edit a running pod.sh or launch.sh.
- After any launcher Traceback, list pods and kill yours.
- Low-memory cleanups on the laptop kill background launchers. The pod job survives (setsid): check
  `/root/job.exit` over ssh, tar the results back, terminate the pod, and book it by hand.
- The laptop disk is near its 2 GiB floor. Delete only your own temporary output. The owner's standing OK covers
  `uv cache prune` and `brew cleanup`.
- The HF token comes from `.env` and never goes in logs or commits. Never send the owner's email anywhere.

## Data

- Downloaded datasets go in your mini's `data/`, which is gitignored (`evalaware_lens/*/data/`).
- Needham et al.'s set (`f1_measures/data/needham/`) must not be redistributed (GAIA terms). Commit IDs, labels and
  aggregates only, never its text.

## Index and reporting

- **When results land:**
  - one row in EXPERIMENTS.md, under "Evaluation awareness lens (`evalaware_lens/`)";
  - one abstract in EXPERIMENT_ABSTRACTS.md;
  - the status line in `evalaware_lens/README.md`.
- Diff shared files hunk by hunk, and stage only your own hunks.
- Report the headline, the numbers and the spend to bdec7d, and to neel-97.
