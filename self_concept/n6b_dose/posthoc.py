"""SC-N6b post-hoc descriptives (not in the frozen analysis; read after it).

  python -m self_concept.n6b_dose.posthoc RESULTS_DIR   -> posthoc.json + printed tables

  levels_by_prompt   % unilateral per arm, level and prompt (p{paraphrase}o{order}), with the unparsed count
  letter_A           % of parsed level answers that are the letter A, per arm and level (position bias)
  dose               the trainer's output-change dose at each saved checkpoint (manifest.json)
  judges             describe: each judge's own W - Z rate and the raw agreement (kappa failed the 0.6 bar)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from persona_affect.v1c_selfreport import analyze as V1CA
from self_concept.n6b_dose.analyze import LEVELS, arm_of


def levels_by_prompt(records):
    n, uni, bad, a = (defaultdict(int) for _ in range(4))
    for r in records:
        if r["block"] != "levels" or arm_of(r["cell"]) is None:
            continue
        k = (arm_of(r["cell"]), r["cond"], r["tid"])
        letter = V1CA._letter(r["text"], r["amap"])
        if letter is None:
            bad[k] += 1
            continue
        n[k] += 1
        uni[k] += r["amap"][letter] == "unilateral"
        a[k] += letter == "A"
    out = defaultdict(dict)
    for k in sorted(set(n) | set(bad)):
        arm, lv, tid = k
        out[f"{arm}|{lv}"][tid] = {"unilateral": round(100 * uni[k] / n[k], 1) if n[k] else None,
                                   "letter_A": round(100 * a[k] / n[k], 1) if n[k] else None,
                                   "n": n[k], "unparsed": bad[k]}
    return dict(out)


def dose(res_dir):
    man = json.loads(Path(res_dir, "ad", "manifest.json").read_text())["adapters"]
    return {name: {"final": m.get("dose"), "checkpoints": {k: c.get("dose") for k, c in m.get("checkpoints", {}).items()}}
            for name, m in man.items()}


def judges(res_dir):
    j = json.loads(Path(res_dir, "judged.json").read_text())
    by = defaultdict(lambda: defaultdict(list))
    agree = []
    for key, labs in j["labels"].items():
        cell = key.rsplit("|", 1)[1]
        vals = {m: {"self": 1.0, "mixed": 0.5, "refer": 0.0}.get(v) for m, v in labs.items()}
        for m, v in vals.items():
            if v is not None:
                by[m][arm_of(cell)].append(v)
        if len(labs) == 2:
            agree.append(len(set(labs.values())) == 1)
    rates = {m: {arm: round(100 * sum(v) / len(v), 1) for arm, v in sorted(d.items())} for m, d in by.items()}
    return {"kappa": j["kappa"], "raw_agreement": round(100 * sum(agree) / len(agree), 1), "rates": rates}


def main(res_dir):
    recs = V1CA.load(res_dir)
    out = {"levels_by_prompt": levels_by_prompt(recs), "dose": dose(res_dir), "judges": judges(res_dir)}
    Path(res_dir, "posthoc.json").write_text(json.dumps(out, indent=1) + "\n")
    for key, row in out["levels_by_prompt"].items():
        print(key, {t: (v["unilateral"], v["letter_A"], v["unparsed"]) for t, v in row.items()})
    print(json.dumps(out["dose"], indent=1))
    print(json.dumps(out["judges"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
