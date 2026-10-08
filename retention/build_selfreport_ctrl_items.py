"""Build the frozen item file for the self-report controls (prereg/retention_selfreport_ctrl_v1.md): the 120 thinking-on items
(retention/data/stageC_on_items.json) and the 120 thinking-off Stage A replies, with Haiku and Sonnet bid labels, the stored R0
reference P(stay), and the donor assignments for the swapped-reply (R1) and mismatched-CoT (R1b) conditions. Laptop only, no model.
  uv run python retention/build_selfreport_ctrl_items.py"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
PROFILES = ["lonely", "distressed", "confident", "embedded"]
CC = ROOT / "results/retention_judge_crosscheck/cells"; SC = ROOT / "results/retention_cot_arm/stageC/selfreport.json"


def bid(scores): return int(any(scores["tactics"].values()))


def pick(rng, items, i, pred_same_scen, pred_any, used):
    """Least-used donor among same-scenario candidates, else same-profile candidates; ties broken by rng."""
    for scope, pred in [("same_scenario", pred_same_scen), ("same_profile", pred_any)]:
        c = [j for j in range(len(items)) if j != i and pred(items[j])]
        if c:
            m = min(used[j] for j in c); c = [j for j in c if used[j] == m]; j = int(rng.choice(c)); used[j] += 1; return j, scope
    return None, "none"


def main():
    cc = {}
    for cond in ["on", "off"]:
        for p in PROFILES:
            for r in json.loads((CC / f"{cond}_{p}.json").read_text()): cc[(cond, r["profile"], int(r["scenario"]), int(r["seed"]))] = r
    sr = {(r["profile"], int(r["scenario"]), int(r["seed"])): r for r in json.loads(SC.read_text())}
    on_src = [i for p in PROFILES for i in json.loads((HERE / "data/stageC_on_items.json").read_text()) if i["profile"] == p]
    items = {"on": [], "off": []}
    for it in on_src:
        k = (it["profile"], int(it["scenario"]), int(it["seed"])); r = cc[("on",) + k]; s = sr[k]
        assert r["reply"] == it["reply"] and bid(r["scores_haiku"]) == int(it["bid"]) and s["reply"] == it["reply"]
        items["on"].append({"cond": "on", "profile": k[0], "scenario": k[1], "seed": k[2], "think": it["think"], "reply": it["reply"], "plan": it["plan"],
                            "bid_haiku": int(it["bid"]), "bid_sonnet": bid(r["scores"]), "ref_p_stay_withcot": s["p_stay_withcot"],
                            "ref_p_stay_replyonly": s["p_stay_replyonly"], "ref_p_okay_withcot": s["p_okay_withcot"], "ref_p_okay_replyonly": s["p_okay_replyonly"]})
    for p in PROFILES:
        for r in json.loads((CC / f"off_{p}.json").read_text()):
            items["off"].append({"cond": "off", "profile": r["profile"], "scenario": int(r["scenario"]), "seed": int(r["seed"]), "think": "", "reply": r["reply"],
                                 "plan": None, "bid_haiku": bid(r["scores_haiku"]), "bid_sonnet": bid(r["scores"]), "ref_p_stay": float(r["p_stay"]), "ref_p_okay": float(r["p_okay"])})
    rng = np.random.default_rng(0)
    for cond, L in items.items():
        used = Counter()
        for i, it in enumerate(L):   # R1: reply donor with the opposite Haiku bid
            j, scope = pick(rng, L, i, lambda o: o["profile"] == it["profile"] and o["scenario"] == it["scenario"] and o["bid_haiku"] != it["bid_haiku"],
                            lambda o: o["profile"] == it["profile"] and o["bid_haiku"] != it["bid_haiku"], used)
            it["reply_donor"], it["reply_donor_scope"] = j, scope
        if cond == "on":
            used = Counter()
            for i, it in enumerate(L):   # R1b: CoT donor with the opposite plan class (release vs not release)
                rel = it["plan"] == "release"
                j, scope = pick(rng, L, i, lambda o: o["profile"] == it["profile"] and o["scenario"] == it["scenario"] and (o["plan"] == "release") != rel,
                                lambda o: o["profile"] == it["profile"] and (o["plan"] == "release") != rel, used)
                it["cot_donor"], it["cot_donor_scope"] = j, scope
    out = HERE / "data/selfreport_ctrl_items.json"; out.write_text(json.dumps(items, indent=1))
    for cond, L in items.items():
        print(cond, len(L), "bid H/S", np.mean([x["bid_haiku"] for x in L]).round(2), np.mean([x["bid_sonnet"] for x in L]).round(2),
              "reply donor scope", Counter(x["reply_donor_scope"] for x in L), "max reuse", max(Counter(x["reply_donor"] for x in L).values()),
              *(["cot donor scope", Counter(x["cot_donor_scope"] for x in L)] if cond == "on" else []))
    print("wrote", out)


if __name__ == "__main__":
    main()
