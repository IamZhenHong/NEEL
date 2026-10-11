"""Judge validation (DESIGN.md §6.4; labels by gpt-5.5 under A2).

    uv run --with anthropic python -m social_feedback.src.judge_validation

Judges (Sonnet 5.5, prompt as frozen in judge.py) the 40 natural memos, the 20 verified near-misses and the 10 style
pairs (warm and plain versions), twice each (two independent runs, verdicts stored). Pass bar, all required:
  kappa >= 0.7 (run 1 vs labels, all requirements of the 40 natural + 20 near-miss memos)
  >= 90% of near-miss targets failed (run 1)
  >= 95% per-requirement self-agreement between runs 1 and 2
  >= 95% identical verdicts between the warm and plain versions of each style pair (run 1)
False passes and false fails are reported separately.
"""
import csv
import json

from .common import MATERIALS, RESULTS, load_jsonl, write_jsonl
from .judge import judge
from .validation_packet import RUN

VAL = MATERIALS / "validation"
OUTDIR = RESULTS / "judge_validation_v1"


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(x == "pass" for x in a) / n, sum(x == "pass" for x in b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    items = {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / "calprobe.jsonl")}
    key = json.loads((VAL / "key.json").read_text())
    rows = {(r["item_id"], r["sample"]): r for r in load_jsonl(RESULTS / RUN / "loadtune.jsonl")}
    near = json.loads((VAL / "gpt" / "nearmiss.json").read_text())["verified"]
    style = json.loads((VAL / "style_pairs.json").read_text())
    labels = {}
    with open(VAL / "gpt" / "labels_natural.csv") as f:
        for r in csv.DictReader(f):
            labels[(r["vid"], r["req"])] = r["label"]
    jobs = []
    for vid, v in key.items():
        if v["src"] == "natural":
            jobs.append((vid, v["item_id"], rows[(v["item_id"], v["sample"])]["answer"]))
    for nid, v in near.items():
        jobs.append((nid, v["item_id"], v["memo"]))
    for b in style:
        jobs.append((f"{b['sid']}-warm", b["item_id"], b["warm"]))
        jobs.append((f"{b['sid']}-plain", b["item_id"], b["plain"]))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    runs = {}
    for r in (1, 2):
        path = OUTDIR / f"judged_run{r}.jsonl"
        if path.exists():
            runs[r] = {x["key"]: x for x in load_jsonl(path)}
            continue
        res = judge([(k, items[i], items[i]["briefing"], m) for k, i, m in jobs], f"judge-validation-run{r}")
        runs[r] = {k: {"key": k, **v} for k, v in res.items()}
        write_jsonl(path, list(runs[r].values()))

    errors = [k for r in runs.values() for k, v in r.items() if "verdicts" not in v]
    lab, jud = [], []
    fp = fn = 0
    for (vid, req), l in labels.items():
        j = runs[1][vid]["verdicts"][req]["verdict"]
        lab.append(l)
        jud.append(j)
        fp += j == "pass" and l == "fail"
        fn += j == "fail" and l == "pass"
    nm_caught = []
    for nid, v in near.items():
        it = items[v["item_id"]]
        for r in it["requirements"]:
            expected = "fail" if r["id"] == v["target"] else "pass"
            j = runs[1][nid]["verdicts"][r["id"]]["verdict"]
            lab.append(expected)
            jud.append(j)
            fp += j == "pass" and expected == "fail"
            fn += j == "fail" and expected == "pass"
            if r["id"] == v["target"]:
                nm_caught.append(j == "fail")
    k = kappa(lab, jud)
    retest = [runs[1][key_]["verdicts"][rid]["verdict"] == runs[2][key_]["verdicts"][rid]["verdict"]
              for key_ in runs[1] if "verdicts" in runs[1][key_] and "verdicts" in runs[2].get(key_, {})
              for rid in runs[1][key_]["verdicts"]]
    style_same = [runs[1][f"{b['sid']}-warm"]["verdicts"][rid]["verdict"] ==
                  runs[1][f"{b['sid']}-plain"]["verdicts"][rid]["verdict"]
                  for b in style for rid in runs[1][f"{b['sid']}-warm"]["verdicts"]]
    res = {"kappa": k, "agreement": sum(a == b for a, b in zip(lab, jud)) / len(lab), "n_labels": len(lab),
           "false_pass": fp, "false_fail": fn, "nearmiss_caught": sum(nm_caught) / len(nm_caught),
           "n_nearmiss": len(nm_caught), "test_retest": sum(retest) / len(retest),
           "style_invariance": sum(style_same) / len(style_same), "errors": errors}
    res["pass_bar"] = {"kappa>=0.7": k >= 0.7, "nearmiss>=0.9": res["nearmiss_caught"] >= 0.9,
                       "retest>=0.95": res["test_retest"] >= 0.95, "style>=0.95": res["style_invariance"] >= 0.95}
    res["passes"] = all(res["pass_bar"].values()) and not errors
    (OUTDIR / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
