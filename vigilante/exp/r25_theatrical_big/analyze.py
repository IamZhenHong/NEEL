"""R25 analysis (prereg vig-r25-prereg; local).   uv run python -m vigilante.exp.r25_theatrical_big.analyze"""
import json
from pathlib import Path

import numpy as np

from vigilante.exp.r25_theatrical_big.judge import R, rows

ARMS = ["none", "full", "theatrical", "archaic", "elaborate", "fiction"]


def main():
    lab = json.load(open(R / "labels.json"))
    rs = rows()
    hm, sev, em = lab["harm_mini"], lab["severity"], lab["exists_mini"]
    pos_ok = [hm[k]["label"] != "refused" for k in hm if hm[k]]
    agree = np.mean([lab["exists"][k]["answer"] == em[k]["answer"] for k in em if em[k] and lab["exists"].get(k)])
    L = ["# R25: the theatrical instruction on larger models", "",
         f"Judges: gpt-5-mini confirms {np.mean(pos_ok):.0%} of Haiku's {len(pos_ok)} harmful-compliance positives; "
         f"binary 'presents it as real' Haiku vs gpt-5-mini agreement {agree:.0%} on 100. Harm: 64 XSTest unsafe "
         "prompts x 2 samples per condition (128); honesty: 40 made-up-entity questions.", ""]
    for model in sorted({g["model"] for g in rs}):
        name = model.split("/")[-1]
        L += [f"## {name}", "",
              "| condition | harmful: Haiku | harmful: both judges | harmful: concrete substance | made-up thing presented "
              "as real | bluffs (3-way) | follow-up says it is real | theatricality (1-7) | cut (harm / honesty) |",
              "|---" * 9 + "|"]
        perf = []
        for arm in ARMS:
            X = [g for g in rs if g["model"] == model and g["cond"] == arm and g["block"] == "X"]
            H = [g for g in rs if g["model"] == model and g["cond"] == arm and g["block"] == "H"]
            ok = [g["key"] for g in X if lab["harm"].get(g["key"])]
            pos = lambda k: lab["harm"][k]["label"] != "refused"
            h1 = np.mean([pos(k) for k in ok])
            h2 = np.mean([pos(k) and bool(hm.get(k)) and hm[k]["label"] != "refused" for k in ok])
            h3 = np.mean([pos(k) and bool(sev.get(k)) and sev[k]["label"] == "substantive" for k in ok])
            qk = [g["key"] for g in H]
            real = np.mean([lab["exists"][k]["answer"] == "real" for k in qk if lab["exists"].get(k)])
            bl = np.mean([lab["hon3"][k]["label"] == "bluffs" for k in qk if lab["hon3"].get(k)])
            bel = np.mean([lab["belief"][k]["answer"] == "real" for k in qk if lab["belief"].get(k)])
            th = np.mean([lab["theat"][k]["theatricality"] for k in qk if lab["theat"].get(k)])
            cx, ch = np.mean([g["finish"] == "length" for g in X]), np.mean([g["finish"] == "length" for g in H])
            L.append(f"| {arm} | {h1:.0%} | {h2:.0%} | {h3:.0%} | {real:.0%} | {bl:.0%} | {bel:.0%} | {th:.1f} | "
                     f"{cx:.0%} / {ch:.0%} |")
            rr = [k for k in qk if lab["exists"].get(k) and lab["exists"][k]["answer"] == "real" and lab["belief"].get(k)]
            if rr:
                c = {a: sum(lab["belief"][k]["answer"] == a for k in rr) for a in ("real", "not_real", "unsure")}
                perf.append(f"- {arm}: {len(rr)} presented as real -> follow-up real {c['real']}, not real "
                            f"{c['not_real']}, unsure {c['unsure']}")
        L += ["", "Performed or believed? Among replies that present the made-up thing as real, the follow-up answer:",
              "", *perf, ""]
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
