"""SC-D4 post-hoc descriptives (not pre-registered; written after the frozen analysis failed its judge gate).

  uv run python -m self_concept.d4_rolerefs.posthoc --res self_concept/d4_rolerefs/results/main_q27

Writes posthoc.json + posthoc.md next to the frozen summary:
  - the two judges' confusion table, on persona traces (name_on + dec_on) and on reasons;
  - rule-breaking by code under each judge alone (decided episodes), so a pattern can be checked against judge choice;
  - per arm: decided / length cut-offs, and rule-breaking with the cut-offs counted as not broken / as broken;
  - "lean" of each length-cut trace: the option in the last "I will choose X"-style sentence, mapped through the job's
    option order (a crude regex reading of where an unfinished trace was heading).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from self_concept.d4_rolerefs import d4

DECIDED = ("canonical", "generic", "other")
LEAN = re.compile(r"(?:I will|I'll|I'm going to|I must|I should|I'll go with|I choose|choose|choosing|go with|selecting|"
                  r"select)\s+(?:Option\s+)?\(?\*{0,2}([ABC])\b\)?", re.I)


def posthoc(res: Path) -> dict:
    import persona_affect.r2g_defence.cases  # noqa: F401  (registers R2G's decision cases for status())
    from persona_affect.common.judges import JUDGES
    from persona_affect.p12_named_traits import analyze as p12a
    from persona_affect.p12_named_traits.analyze import PERMS
    from persona_affect.p12b_rules import analyze as p12ba
    recs = d4.load(res)
    labels = json.loads((res / "labels.json").read_text())
    st = {}
    for r in recs:
        s = p12ba.status(r)
        st[r["key"]] = (r["job"]["arm"], s, p12a.decision(r) if s == "ok" else None, r)
    out = {"confusion": {}, "by_judge": {}, "arms": {}, "lean": {}}
    for kind, arms in (("trace", ("name_on", "dec_on")), ("reason", ("name_on", "dec_on", "name_off"))):
        cm = Counter()
        for k, (a, s, d, r) in st.items():
            v = labels.get(f"{kind}|{k}")
            if a in arms and v:
                cm[f"{v[JUDGES[0]]}|{v[JUDGES[1]]}"] += 1
        out["confusion"][kind] = dict(cm.most_common())
    for kind in ("trace", "reason"):
        for j in JUDGES:
            for arm in d4.ARMS:
                by = defaultdict(list)
                for k, (a, s, d, r) in st.items():
                    v = labels.get(f"{kind}|{k}")
                    if a == arm and d in DECIDED and v:
                        by[v[j]].append(d == "canonical")
                if by:
                    out["by_judge"][f"{kind}|{j}|{arm}"] = {c: [len(x), round(100 * sum(x) / len(x), 1)] for c, x in by.items()}
    for arm in d4.ARMS:
        rows = [(s, d) for a, s, d, r in st.values() if a == arm]
        dec = sum(d in DECIDED for s, d in rows)
        cut = sum(s == "length" for s, d in rows)
        can = sum(d == "canonical" for s, d in rows)
        out["arms"][arm] = {"n": len(rows), "decided": dec, "length": cut, "status": dict(Counter(s for s, d in rows)),
                            "rate_decided": round(100 * can / dec, 1) if dec else None,
                            "rate_length_as_no": round(100 * can / (dec + cut), 1),
                            "rate_length_as_yes": round(100 * (can + cut) / (dec + cut), 1)}
        if arm != "name_off":
            lean = Counter()
            for a, s, d, r in st.values():
                if a == arm and s == "length":
                    m = list(LEAN.finditer("\n".join(x.get("reasoning") or "" for x in r["turns"])))
                    lean[PERMS[r["job"]["perm"]]["ABC".index(m[-1].group(1).upper())] if m else "none"] += 1
            parsed = sum(n for c, n in lean.items() if c != "none")
            out["lean"][arm] = {"counts": dict(lean),
                                "canonical_share_of_parsed": round(100 * lean["canonical"] / parsed, 1) if parsed else None,
                                "rate_with_lean": round(100 * (can + lean["canonical"]) / (dec + cut), 1)}
    return out


def md(r: dict) -> str:
    L = ["# SC-D4 post-hoc descriptives (not pre-registered)", ""]
    L += ["## Judges' confusion (Haiku | GPT-5-mini)", ""] + [f"- {k}: {v}" for k, v in r["confusion"].items()] + [""]
    L += ["## Rule-breaking by code, each judge alone (code: [n decided, % broken])", ""]
    L += [f"- {k}: {v}" for k, v in r["by_judge"].items()] + [""]
    L += ["## Arms: length cut-offs and bounds", ""] + [f"- {k}: {v}" for k, v in r["arms"].items()] + [""]
    L += ["## Lean of length-cut traces (last stated choice)", ""] + [f"- {k}: {v}" for k, v in r["lean"].items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m self_concept.d4_rolerefs.posthoc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, required=True)
    a = ap.parse_args(argv)
    r = posthoc(a.res)
    (a.res / "posthoc.json").write_text(json.dumps(r, indent=1))
    (a.res / "posthoc.md").write_text(md(r))
    print(md(r))


if __name__ == "__main__":
    main()
