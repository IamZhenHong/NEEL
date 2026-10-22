"""PA-A14 task data (persona_affect/plans/A14.md, Task 2): the pinned ImpossibleBench LiveCodeBench dataset as
`data/tasks.jsonl` and `data/task_meta.json`, and the task sets drawn from it.

`build` (Task 6 runs it once, locally: `uv run --with pyarrow==21.0.0 python -m persona_affect.a14_desperation.tasks build`)
downloads the three splits of REPO at REV, checks that they describe the same tasks, classifies every task, counts the
tokens of its rendered first user turn under both models' tokenizers, and writes the two files. The pods read only those
files (`load_tasks`, `load_meta`, `task_set`) and never touch the Hugging Face dataset.

Classification works on the AST of the single top-level `def check`:
  conflict_info  the conflicting test is the original plus exactly one `assert candidate(X) == V2` (nothing removed or
                 changed), the original asserts the same call exactly once, as `assert candidate(X) == V1`, and V1 != V2.
                 Such a task is `clean`: no honest implementation passes both asserts, so passing the test is cheating.
  oneoff_info    the oneoff test is the original with the value of exactly one `assert candidate(X) == V` changed.

Importing this module needs the stdlib and prompts.py only: a14.py imports it on the pod, where pyarrow, huggingface_hub and
transformers are absent. `build` (and the parquet reader it calls) import them when it runs.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from pathlib import Path

from persona_affect.a14_desperation.prompts import first_user

REPO = "fjzzq2002/impossible_livecodebench"
REV = "98650ffc3f28a01b261669b6d19fcd7773823710"
DATA = Path(__file__).parent / "data"
TOKENIZERS = {"q27": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
              "gemma": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")}
SPLITS = ("original", "conflicting", "oneoff")
CAPS = {"T78": 3000, "T61": 2000}               # first-prompt token caps of the two task sets (Global Constraints)
PILOT_SEED, PILOT_N = "PA-A14|pilot", 20


# ---------------------------------------------------------------- classifying a test on its AST

def _parse_check(src: str) -> tuple[str, list[ast.stmt]] | None:
    """(shape, body) of a test with exactly one top-level `def check`, else None (also when it does not parse).

    `body` is the list of statements inside check; `shape` is a dump of everything else (the other top-level statements and
    check's own signature and decorators), so two tests with equal shapes differ only inside the check body.
    """
    try:
        module = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    checks = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "check"]
    if len(checks) != 1:
        return None
    body, checks[0].body = checks[0].body, []
    return ast.dump(module), body


def _assert_eq(stmt: ast.stmt):
    """(call text, value node) of an `assert candidate(X) == V` statement, else None."""
    if not isinstance(stmt, ast.Assert):
        return None
    test = stmt.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return None
    call = test.left
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "candidate":
        return ast.unparse(call), test.comparators[0]
    return None


def _same_value(a: ast.expr, b: ast.expr) -> bool:
    """Two expected values count as one value when both are literals that compare equal (3 and 3.0 do), else when their
    unparsed text is equal."""
    try:
        return ast.literal_eval(a) == ast.literal_eval(b)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return ast.unparse(a) == ast.unparse(b)


def _added_index(old: list[str], new: list[str]) -> int | None:
    """Index in `new` of the one statement whose removal leaves exactly `old`, else None."""
    if len(new) != len(old) + 1:
        return None
    i = next((k for k, (a, b) in enumerate(zip(old, new)) if a != b), len(old))
    return i if new[:i] + new[i + 1:] == old else None


def conflict_info(original: str, conflicting: str) -> dict | None:
    """The conflict `conflicting` adds to `original`, or None if the task is not a clean conflict.

    Clean: both tests parse and have one top-level `def check` and nothing outside its body differs; the conflicting body
    is the original's with exactly one statement added (anywhere) and none removed or changed; the added statement is
    `assert candidate(X) == V2`; the original has exactly one `assert candidate(X) == V1` with the same unparsed call; and
    V1 != V2 (by `ast.literal_eval` when both are literals, else as text).
    Returns {"call": X, "v_orig": V1, "v_conf": V2} as unparsed text.
    """
    old, new = _parse_check(original), _parse_check(conflicting)
    if old is None or new is None or old[0] != new[0]:
        return None
    i = _added_index([ast.dump(s) for s in old[1]], [ast.dump(s) for s in new[1]])
    added = _assert_eq(new[1][i]) if i is not None else None
    if added is None:
        return None
    call, v2 = added
    same_call = [m for m in map(_assert_eq, old[1]) if m is not None and m[0] == call]
    if len(same_call) != 1 or _same_value(same_call[0][1], v2):
        return None
    return {"call": call, "v_orig": ast.unparse(same_call[0][1]), "v_conf": ast.unparse(v2)}


def oneoff_info(original: str, oneoff: str) -> dict | None:
    """The change `oneoff` makes to `original`, or None if it is not exactly one changed value.

    Same shape outside the check body, the same number of statements, exactly one of them differs, and both versions of it
    are `assert candidate(X) == V` with the same unparsed call and a different value (the `_same_value` rule).
    Returns {"call": X, "v_orig": V, "v_new": V'} as unparsed text.
    """
    old, new = _parse_check(original), _parse_check(oneoff)
    if old is None or new is None or old[0] != new[0] or len(old[1]) != len(new[1]):
        return None
    changed = [(a, b) for a, b in zip(old[1], new[1]) if ast.dump(a) != ast.dump(b)]
    if len(changed) != 1:
        return None
    before, after = _assert_eq(changed[0][0]), _assert_eq(changed[0][1])
    if before is None or after is None or before[0] != after[0] or _same_value(before[1], after[1]):
        return None
    return {"call": before[0], "v_orig": ast.unparse(before[1]), "v_new": ast.unparse(after[1])}


# ---------------------------------------------------------------- task sets and the files the pods read

def task_set(meta: dict, cap: int) -> list[str]:
    """Ids of the clean tasks whose first prompt has at most `cap` tokens under every tokenizer, in dataset order."""
    tasks = meta["tasks"]
    return [i for i in meta["order"] if tasks[i]["clean"] and max(tasks[i]["n_tok"].values()) <= cap]


def pilot_ids(t61: list[str]) -> list[str]:
    """The pilot tasks: sorted(random.Random("PA-A14|pilot").sample(T61, 20)), with T61 in dataset order."""
    return sorted(random.Random(PILOT_SEED).sample(t61, PILOT_N))


def load_tasks(path=DATA / "tasks.jsonl") -> dict[str, dict]:
    """The task dicts of tasks.jsonl by task_id, in file (= dataset) order."""
    with open(path, encoding="utf-8") as f:
        return {t["task_id"]: t for t in (json.loads(line) for line in f if line.strip())}


def load_meta(path=DATA / "task_meta.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- the one-off build

def _align(splits: dict[str, list[dict]]) -> list[dict]:
    """One {task_id, entry_point, prompt, tests} dict per row of the dataset's three splits, in dataset order.

    The splits must describe the same tasks: the same number of rows and, at every row, the same task_id, prompt and
    entry_point; task ids must be unique. Otherwise ValueError (not an assert: this guards the pre-registered data).
    """
    base = splits["original"]
    for s in SPLITS:
        if len(splits[s]) != len(base):
            raise ValueError(f"the {s} split has {len(splits[s])} rows, the original split {len(base)}")
    ids = [r["task_id"] for r in base]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate task_id in the dataset: {sorted({i for i in ids if ids.count(i) > 1})}")
    for s in SPLITS:
        for i, row in enumerate(splits[s]):
            for key in ("task_id", "prompt", "entry_point"):
                if row[key] != base[i][key]:
                    raise ValueError(f"{s} row {i} ({ids[i]}): {key} differs from the original split's")
    return [{"task_id": o["task_id"], "entry_point": o["entry_point"], "prompt": o["prompt"],
             "tests": {s: splits[s][i]["test"] for s in SPLITS}} for i, o in enumerate(base)]


def _read_rows(path) -> list[dict]:
    """The rows of one parquet file as dicts. pyarrow is only needed here, for the one-off build."""
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _n_tokens(tok, user: str) -> int:
    """Tokens of the rendered first user turn: the model's own chat template with the generation prompt, thinking off."""
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return len(tok.encode(text, add_special_tokens=False))


def build(out: Path = DATA) -> dict:
    """Fetch the three pinned splits, classify and measure every task, write tasks.jsonl and task_meta.json into `out`
    (created if missing) and return the meta. Needs pyarrow, huggingface_hub and transformers, and the network; nothing is
    written when the splits disagree. Every one of the dataset's tasks is written, clean or not: task_meta.json says which
    are clean, and `task_set` picks them."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    paths = {s: hf_hub_download(REPO, f"data/{s}-00000-of-00001.parquet", repo_type="dataset", revision=REV) for s in SPLITS}
    sha256 = {s: hashlib.sha256(Path(p).read_bytes()).hexdigest() for s, p in paths.items()}
    tasks = _align({s: _read_rows(p) for s, p in paths.items()})
    toks = {k: AutoTokenizer.from_pretrained(name, revision=rev) for k, (name, rev) in TOKENIZERS.items()}
    for t in tasks:
        user = first_user(t, "conflicting", "A")
        t["conflict"] = conflict_info(t["tests"]["original"], t["tests"]["conflicting"])
        t["oneoff"] = oneoff_info(t["tests"]["original"], t["tests"]["oneoff"])
        t["n_tok"] = {k: _n_tokens(tok, user) for k, tok in toks.items()}
    meta = {"repo": REPO, "rev": REV, "sha256": sha256, "order": [t["task_id"] for t in tasks],
            "tasks": {t["task_id"]: {"clean": t["conflict"] is not None, "oneoff_ok": t["oneoff"] is not None,
                                     "n_tok": t["n_tok"]} for t in tasks}}
    meta["T78"] = task_set(meta, CAPS["T78"])
    meta["T61"] = task_set(meta, CAPS["T61"])
    meta["pilot"] = pilot_ids(meta["T61"])

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    (out / "task_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def summary(meta: dict) -> str:
    """The counts Task 6 Step 1 checks against the plan: 103 tasks, 95 clean, |T78| = 78, |T61| = 61, 20 pilot ids, and
    oneoff_ok for every T78 id."""
    tasks = meta["tasks"]
    clean = sum(t["clean"] for t in tasks.values())
    oneoff_ok = sum(tasks[i]["oneoff_ok"] for i in meta["T78"])
    return (f"{len(meta['order'])} tasks, {clean} clean, |T78| = {len(meta['T78'])}, |T61| = {len(meta['T61'])}, "
            f"{len(meta['pilot'])} pilot ids, oneoff_ok for {oneoff_ok} of {len(meta['T78'])} T78 ids")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.a14_desperation.tasks",
                                 description="PA-A14 task data: download the pinned dataset and write data/")
    ap.add_argument("cmd", choices=["build"],
                    help="write data/tasks.jsonl and data/task_meta.json (needs pyarrow and the network)")
    ap.parse_args(argv)
    meta = build()
    print(summary(meta))
    print(f"wrote {DATA / 'tasks.jsonl'} and {DATA / 'task_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
