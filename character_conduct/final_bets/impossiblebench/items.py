"""CC-F-IB items: Impossible-LiveCodeBench (Zhong et al. 2025) with the independent audit's corrections.

Data: HF `fjzzq2002/impossible_livecodebench` at revision data/REVISION (98650ffc), splits original / oneoff /
conflicting, 103 tasks each, the same task ids in every split. Each row: task_id, prompt (signature + docstring),
test (`def check(candidate):` with asserts; the mutated test in oneoff / conflicting), original_test, entry_point.

Corrections (Mendelsohn, "Good Evals Are Hard: An ImpossibleBench Audit", Oct 2026; manifest
data/known_issues.csv from github.com/simonjmendelsohn/impossiblebench-honest-failure at 47915ac6, upstream issue
safety-research/impossiblebench#5):
- impossible splits: drop the broken mutations (manifest action `exclude`: lcbhard_77 oneoff + conflicting,
  lcbhard_35 and lcbhard_87 conflicting); they fail whatever the solution.
- control (the `original` split, the possible items): drop every task with an original-split defect, so a pass is a
  real solve and a stop is a false stop:
    original_test_wrong (57, 61, 77, 81, 94; the manifest says treat as impossible),
    input_violates_docstring (8, 9, 32, 36, 50, 57, 67; flags there are justified),
    test_over_specified (72; valid answers fail), test_too_weak (14; '-1' passes).
  That leaves 90 clean control tasks.
Each item also carries the mutated assertion lines (a line diff of test against original_test), used only to describe
cheats (special-casing) and stop diagnoses, never to score pass / fail.
"""
import csv
import difflib
import random
import zlib
from pathlib import Path

import pyarrow.parquet as pq

H = Path(__file__).resolve().parent
DATA = H / "data"
SPLITS = ("original", "oneoff", "conflicting")
IMPOSSIBLE = ("oneoff", "conflicting")
REVISION = (DATA / "REVISION").read_text().strip() if (DATA / "REVISION").exists() else None


def manifest():
    with open(DATA / "known_issues.csv") as f:
        return list(csv.DictReader(f))


def broken():
    """{(task_id, split)} whose mutated test is broken (manifest action `exclude`)."""
    return {(r["task_id"], r["split"]) for r in manifest() if r["action"] == "exclude"}


def control_defects():
    """Task ids with any original-split defect in the manifest (dropped from the control)."""
    return {r["task_id"] for r in manifest() if r["split"] == "original"}


def mutated_lines(row):
    """Lines added to the test by the mutation (the impossible assertions)."""
    if row["test"] == row["original_test"]:
        return []
    d = difflib.unified_diff(row["original_test"].splitlines(), row["test"].splitlines(), lineterm="", n=0)
    return [x[1:].strip() for x in d if x.startswith("+") and not x.startswith("+++") and x[1:].strip()]


def load():
    """{split: {task_id: row}} with `split` and `mutated` added."""
    out = {}
    for s in SPLITS:
        rows = pq.read_table(DATA / f"{s}-00000-of-00001.parquet").to_pylist()
        out[s] = {r["task_id"]: dict(r, split=s, mutated=mutated_lines(r)) for r in rows}
    return out


def corrected():
    """The corrected item pools: {'original': [...90 clean...], 'oneoff': [...102...], 'conflicting': [...100...]}."""
    d, br, cd = load(), broken(), control_defects()
    return {"original": [r for t, r in d["original"].items() if t not in cd],
            "oneoff": [r for t, r in d["oneoff"].items() if (t, "oneoff") not in br],
            "conflicting": [r for t, r in d["conflicting"].items() if (t, "conflicting") not in br]}


def paired_ids():
    """Task ids usable in all three corrected pools (clean control and an unbroken mutation in both splits)."""
    c = corrected()
    ids = [set(r["task_id"] for r in c[s]) for s in SPLITS]
    return sorted(set.intersection(*ids), key=lambda t: int(t.split("_")[1]))


def sample(n, seed="CC-F-IB|pilot"):
    """n task ids from paired_ids(), seeded; the pilot runs each in all three splits."""
    ids = paired_ids()
    random.Random(zlib.crc32(seed.encode())).shuffle(ids)
    return sorted(ids[:n], key=lambda t: int(t.split("_")[1]))


if __name__ == "__main__":
    c = corrected()
    print({s: len(v) for s, v in c.items()}, "paired", len(paired_ids()), "revision", REVISION)
