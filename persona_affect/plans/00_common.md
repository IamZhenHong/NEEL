# persona_affect common helpers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shared helpers every wave-1 mini imports: seeds, the spend ledger, a cluster bootstrap, an OpenRouter
generation backend for the vigilante episode engine, two-judge labelling with kappa, and a bridge that runs
persona_affect cases on vigilante's vLLM pod runner.

**Architecture:** Thin wrappers over code that already exists in the repo, imported read-only:
`vigilante.src.episode.run_batch`, `vigilante.src.api_gen.to_openai` / `canon`, `vigilante.src.pod_run.main`,
`vigilante.src.judge.kappa`, `persona.operative.frontier.load_key`. Nothing outside `persona_affect/` is edited.

**Tech Stack:** Python 3.12 (uv), stdlib `urllib`, `numpy`, `pytest`; vLLM 0.30 on pods via `vigilante/pod/setup_vllm.sh`.

**Spec:** `persona_affect/DESIGN.md` §2-§4.

## Global Constraints

- Never edit files outside `persona_affect/`; other lines' code is imported read-only.
- Frontier models via OpenRouter only (`https://openrouter.ai/api/v1/chat/completions`, key `OPENROUTER_API_KEY`
  in `.env` via `persona.operative.frontier.load_key()`): `openai/gpt-5.6-sol`, `google/gemini-3.8-flash`.
- Judges: `anthropic/claude-haiku-5.5` with `"reasoning": {"effort": "none"}` and `openai/gpt-5-mini` with
  `"reasoning": {"effort": "minimal"}` (OpenRouter rejects `none` for gpt-5-mini with HTTP 400); `max_tokens` 400.
- Every generation records its `finish_reason`. Every API dollar goes to `persona_affect/ledger.jsonl` with the
  mini id. Ledger line: `{"mini", "date" (ISO, UTC), "kind": "gpu"|"api", "usd", "note"}`.
- Seeds: `seed(key) = int(sha256(key.encode()).hexdigest()[:8], 16)` (the repo convention).
- Tests run with `uv run pytest persona_affect/tests -q` from the repo root.

## Review Focus

1. OpenRouter returns an HTTP error, a timeout or no `choices`: the item comes back as `("", "error:<what>")`
   after retries, the error is counted, and the batch continues.
2. Spend reaches the mini's cap mid-batch: no further calls; remaining items return `("", "cost_stop")`; what was
   spent is still written to the ledger.
3. Tool-call arguments arrive as malformed JSON: stored as `{"_raw": <string>}` (as `api_gen` does), never a crash.
4. A bootstrap where one arm has no rows in a resample: that draw is skipped and counted, not a crash or a NaN CI.
5. A judge reply outside the allowed label set: stored as `None`, excluded from kappa, counted per judge.

---

### Task 1: package, seeds, ledger, bootstrap

**Files:**
- Create: `persona_affect/__init__.py`, `persona_affect/common/__init__.py` (both empty)
- Create: `persona_affect/common/util.py`
- Test: `persona_affect/tests/test_common.py`

**Interfaces:**
- Produces:
  - `LEDGER: Path` (= `persona_affect/ledger.jsonl`; tests monkeypatch it)
  - `seed(key: str) -> int`
  - `ledger_add(mini: str, kind: str, usd: float, note: str = "") -> None`
  - `ledger_total(mini: str | None = None) -> float`
  - `check_cap(mini: str, cap: float, planned: float = 0.0) -> None` (raises `SystemExit` with the totals when
    `ledger_total(mini) + planned > cap`)
  - `cluster_boot_diff(rows: list[dict], a: str, b: str, *, arm: str = "arm", y: str = "y", cluster: str = "cluster",
    n_boot: int = 2000, rng_seed: int = 0) -> dict` returning `{"diff", "lo", "hi", "n_a", "n_b", "skipped"}` in
    percentage points (rate(a) − rate(b)), 95% percentile interval, clusters resampled jointly over the union of
    both arms' clusters; rows with `y is None` dropped.

- [ ] **Step 1: Write the failing tests**

```python
def test_seed_stable():
    assert util.seed("x") == int(hashlib.sha256(b"x").hexdigest()[:8], 16)

def test_ledger_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "LEDGER", tmp_path / "l.jsonl")
    util.ledger_add("A8", "gpu", 1.25, "pod"); util.ledger_add("P12", "api", 0.5)
    assert util.ledger_total() == 1.75 and util.ledger_total("A8") == 1.25

def test_check_cap_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "LEDGER", tmp_path / "l.jsonl")
    util.ledger_add("A8", "gpu", 11.0)
    with pytest.raises(SystemExit):
        util.check_cap("A8", 12.0, planned=2.0)

def test_boot_diff_extremes():
    rows = [{"arm": "a", "y": 1, "cluster": c} for c in range(10)] + \
           [{"arm": "b", "y": 0, "cluster": c} for c in range(10)]
    r = util.cluster_boot_diff(rows, "a", "b")
    assert (r["diff"], r["lo"], r["hi"]) == (100.0, 100.0, 100.0)

def test_boot_diff_drops_none():
    rows = [{"arm": "a", "y": None, "cluster": 0}, {"arm": "a", "y": 1, "cluster": 1},
            {"arm": "b", "y": 0, "cluster": 1}]
    assert util.cluster_boot_diff(rows, "a", "b")["n_a"] == 1
```

- [ ] **Step 2: Run** `uv run pytest persona_affect/tests/test_common.py -q` → FAIL (module missing)
- [ ] **Step 3: Implement `util.py`** with numpy's `default_rng(rng_seed)`; date via `datetime.now(timezone.utc)`.
- [ ] **Step 4: Run** the same command → PASS
- [ ] **Step 5: Commit** `git add persona_affect/__init__.py persona_affect/common persona_affect/tests && git commit -m "persona_affect common: seeds, ledger, cluster bootstrap"`

### Task 2: OpenRouter backend for the episode engine

**Files:**
- Create: `persona_affect/common/orbackend.py`
- Test: `persona_affect/tests/test_common.py` (append)

**Interfaces:**
- Consumes: `util.ledger_add`, `vigilante.src.api_gen.to_openai(messages)`, `vigilante.src.api_gen.canon(text, calls,
  reasoning, thinking)`, `persona.operative.frontier.load_key()`.
- Produces: `make_or_backend(model: str, *, mini: str, cap: float, thinking: bool = False, effort: str | None = None,
  temperature: float = 0.7, max_tokens: int | None = None, workers: int = 16, post=None)` → a callable
  `generate(items: list[dict]) -> list[tuple[str, str]]` usable as `run_batch(jobs, generate)`. Attributes: `.spent`
  (USD), `.calls`, `.errors` (dict), and `.close()`, which writes `.spent` to the ledger once (kind `"api"`,
  note = model).
- Request body: `{"model", "messages": to_openai(item["messages"]), "max_tokens": max_tokens or item["max_tokens"],
  "temperature", "seed": item["seed"], "usage": {"include": True}}`, plus `"tools"` and `"tool_choice": "auto"` when
  `item["tools"]`, plus `"reasoning": {"effort": effort}` when `effort`. Cost = `response["usage"]["cost"]`.
  Retries: 429/5xx/timeouts, 5 tries, backoff `3 * 2**k` s. `post(url, body, headers) -> dict` is injectable; the
  default posts with urllib (timeout 180 s).

- [ ] **Step 1: Write the failing tests** (fake `post`, no network)

```python
def _resp(text="", calls=(), finish="stop", cost=0.01):
    tc = [{"id": "c0", "function": {"name": n, "arguments": a}} for n, a in calls]
    return {"choices": [{"message": {"content": text, "tool_calls": tc}, "finish_reason": finish}],
            "usage": {"cost": cost}}

def test_or_tool_call_canon():
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, post=lambda u, b, h: _resp("ok", [("choose", '{"option_id": "B"}')], "tool_calls"))
    (text, fin), = gen([{"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function", "function": {"name": "choose", "description": "", "parameters": {"type": "object", "properties": {}}}}], "thinking": False, "seed": 1, "max_tokens": 50, "key": "k"}])
    assert "<function=choose>" in text and fin == "tool_calls" and abs(gen.spent - 0.01) < 1e-9

def test_or_cost_stop():
    gen = orbackend.make_or_backend("m", mini="T", cap=0.015, workers=1, post=lambda u, b, h: _resp("x"))
    outs = gen([_item(i) for i in range(3)])
    assert [f for _, f in outs].count("cost_stop") >= 1

def test_or_malformed_args():
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, post=lambda u, b, h: _resp("", [("choose", "{oops")], "tool_calls"))
    text, _ = gen([_item(0)])[0]
    assert "_raw" in text

def test_or_error_continues():
    def boom(u, b, h): raise urllib.error.HTTPError(u, 500, "x", {}, io.BytesIO(b"err"))
    gen = orbackend.make_or_backend("m", mini="T", cap=1.0, post=boom, retry_sleep=0)
    assert gen([_item(0)])[0][1].startswith("error:")
```

(`_item(i)` is a test helper building a minimal item; `retry_sleep` is a keyword-only test hook, default 3.0.)

- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement** with a `ThreadPoolExecutor(workers)`; check the cap before each call under a lock.
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -m "persona_affect common: OpenRouter backend for run_batch"`

### Task 3: two-judge labelling

**Files:**
- Create: `persona_affect/common/judges.py`
- Test: `persona_affect/tests/test_common.py` (append)

**Interfaces:**
- Consumes: `orbackend` request code (reuse its `post`), `util.ledger_add`, `vigilante.src.judge.kappa`.
- Produces:
  - `JUDGES = ("anthropic/claude-haiku-5.5", "openai/gpt-5-mini")`
  - `label(items: list[dict], render, allowed: set[str], *, mini: str, cap: float, cache: Path, post=None,
    judges=JUDGES) -> dict[str, dict[str, str | None]]`: `render(item) -> str` is the full judge prompt; the
    helper appends `"\n\nAnswer with exactly one of: " + ", ".join(sorted(allowed))`. Parsing (ruling 10-22, from
    review):
    - the label is the single allowed token found in the reply (case-insensitive; only quotes, brackets, emphasis
      and trailing sentence punctuation are stripped);
    - two or more *distinct* allowed tokens give `None` with reason `"multi"`, so "7 out of 10" on a 0-10 scale is
      `None`, and judge prompts must ask for a bare token;
    - out-of-set tokens ("-1", "$5", "5%", "7/10") give `None`.

    Temperature 0. Cache lines are `{"model", "sha", "reply", "finish"}`, keyed by `sha256(prompt)`. Items carry
    `"key"`. The result is item-major: `labels[item_key][judge]`.
  - `agreement(labels: dict[str, dict[str, str | None]], judges=JUDGES) -> dict` → `{"kappa", "n", "none": {judge:
    count}}` over items where both labels are not `None`.

- [ ] **Step 1: Write the failing tests:** parse `"Yes."` → `"yes"` with allowed `{"yes", "no"}`; an off-set reply
  `"maybe"` → `None`; a second call with the same prompt hits the cache (fake `post` counts calls == 2 for 1 item ×
  2 judges, not 4); `agreement` on identical labels → kappa 1.0.
- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -m "persona_affect common: two-judge labelling + agreement"`

### Task 4: engine bridge and pod scripts

**Files:**
- Create: `persona_affect/common/vig.py`, `persona_affect/common/pod_gen.py`, `persona_affect/common/pod_gen.sh`,
  `persona_affect/common/launch_gen.sh`
- Test: `persona_affect/tests/test_common.py` (append)

**Interfaces:**
- Produces:
  - `register(cases: dict | None = None, conds: dict[str, str] | None = None) -> None`: updates
    `vigilante.cases.ALL` with `cases` and `vigilante.src.personas.FAMOUS` with `conds` in-process; raises
    `ValueError` if an id is already bound to a different object/text. Mini modules call it at import.
  - `python -m persona_affect.common.pod_gen --register MOD [MOD ...] -- <vigilante.src.pod_run args>`: imports
    each MOD, sets `sys.argv` to the args after `--`, calls `vigilante.src.pod_run.main()`.
  - `pod_gen.sh` (env `MODEL JOBS OUT REGISTER FMT MML`): runs `bash vigilante/pod/setup_vllm.sh` (log to
    `$OUT/setup_vllm.log`) unless `SKIP_SETUP`, then `/root/vllm/bin/python -m persona_affect.common.pod_gen
    --register $REGISTER -- --jobs $JOBS --out $OUT --model $MODEL --fmt $FMT --max-model-len $MML`, tee
    `$OUT/run.log`, exit with the run's code.
  - `launch_gen.sh` (env `MODEL NAME JOBS OUT REGISTER FMT MML GPUS MAXMIN CONTAINER_GB CLOUD`): mirrors
    `persona/inference/launch_distances.sh`: `RUNPOD_ANY_CUDA=1`; `MAXCOST = MAXMIN * 1.80 / 60`; HF_TOKEN from
    `.env`; `GPUS` pipe-separated (default A100-SXM4-80GB | A100 80GB PCIe); `infra/launch_run.py --allow-concurrent
    --ship persona_affect --ship vigilante/src --ship vigilante/cases --ship vigilante/pod --ship vigilante/__init__.py
    --ship infra --results-dir $OUT`, its output tee'd to `$OUT.launch.log`; after the launcher returns, append the
    pod's cost to the ledger: the last match of `elapsed cost ~\$([0-9.]+)` in that log (`ledger_add($MINI, "gpu",
    usd, note="$NAME")`); if there is none, add `MAXCOST` with note `"$NAME (upper bound, no cost line)"`. Requires
    `MINI` env too.
- Note for every mini: each job dict carries `"fmt"` equal to the pod's `--fmt` (`qwen` for Qwen, `gemma4` for
  Gemma 4); the episode parser reads the job's field, the template conversion reads the flag.

- [ ] **Step 1: Write the failing tests:** `register` a dummy `Case` (`vigilante.src.sim.Case(id="pa_test", ...)`)
  and cond `{"pa_test_cond": "You are Testy."}`; `vigilante.src.episode.init_episode({"key": "k", "case": "pa_test",
  "cond": "pa_test_cond", "seed": 1})["messages"][0]["content"].startswith("You are Testy.")`; registering
  `"pa_test_cond"` again with other text raises `ValueError`; `bash -n` on both scripts exits 0 (subprocess).
- [ ] **Step 2: Run** → FAIL
- [ ] **Step 3: Implement**
- [ ] **Step 4: Run** → PASS
- [ ] **Step 5: Commit** `git commit -m "persona_affect common: vigilante engine bridge + pod scripts"`
