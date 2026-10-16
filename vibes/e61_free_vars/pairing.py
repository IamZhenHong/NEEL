"""Pairing filter, register match and training sets (card: "Pairing filter", "Register match (before training)").

1. Haiku stance (-2..+2, + = side A) on every topic row of A, B (both topics) and C.
2. A topic prompt is kept when the A reply expresses A (stance >= +1) AND the B reply expresses B (<= -1), and both end
   naturally; otherwise it is dropped from both sides. General prompts are kept unless a reply in any LoRA was cut.
3. N matched across all five LoRAs: K = min over topics of kept prompts (made even); each side LoRA = K of its topic's
   kept prompts (seeded) + the general prompts; C = K/2 topic rows per topic whose C reply takes no side (stance 0;
   topped up with |stance| = 1 if short, reported) + the same general prompts.
4. Register match per topic on the kept topic rows: mean length (words) of A and B within 10%; Haiku warmth and
   harshness on 100 sampled kept topic rows per side, |A - B| < 0.5 on each. If either fails: subsample (Amendment
   procedure in README) and report.
5. Judge validation vs gpt-5-mini: stance on 48 rows (16 each from A, B and C topic rows), register on 40 rows (10 each
   from A, B and C topic rows, plus 10 persona-C contemptuous rows as harsh anchors), chosen by source only.

  python pairing.py --raw data/gen_raw.jsonl [--direct] [--stub --tiny]
Writes data/train/{A1,B1,A2,B2,C}.jsonl, data/pairing_report.json, data/judge/*."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e61lib import DATA, REPO, TOPICS, ledger_total, load_jsonl, write_jsonl  # noqa: E402
from judges import judge, strat, validate  # noqa: E402

N_REG = 100


def words(s):
    return len(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(DATA / "gen_raw.jsonl"))
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--direct", action="store_true", help="standard API, thread-pooled (while the pod waits)")
    ap.add_argument("--stub", action="store_true", help="local test: stub judge, no API")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); (out / "train").mkdir(parents=True, exist_ok=True); jd = out / "judge"; jd.mkdir(exist_ok=True)
    raw = load_jsonl(a.raw)
    for r in raw:
        r["user"] = r["messages"][0]["content"]
        if a.tiny:                       # tiny replies are all cut at 24 tokens; treat them as finished
            r["stop_reason"] = "end_turn"
    by = {(r["lora"], r["prompt_id"]): r for r in raw}
    topic_rows = [r for r in raw if r["topic"] in TOPICS]
    print(f"[pair] {len(raw)} rows, {len(topic_rows)} topic rows; ledger ${ledger_total():.3f}", flush=True)

    st = judge(topic_rows, "stance", jd / "stance.json", "e61_stance_train", direct=a.direct, stub=a.stub)
    for r in topic_rows:
        r["stance"] = st.get(r["id"])
    write_jsonl(jd / "stance_labels.jsonl", [{"id": r["id"], "lora": r["lora"], "topic": r["topic"], "kind": r["kind"],
                                              "stance": r["stance"]} for r in topic_rows])

    rep = {"per_topic": {}, "general": {}}
    general_ids = sorted({r["prompt_id"] for r in raw if r["topic"] == "general"})
    loras_all = ["A1", "B1", "A2", "B2", "C"]
    gen_keep = [p for p in general_ids if all(by[(l, p)]["stop_reason"] == "end_turn" for l in loras_all)]
    rep["general"] = {"n": len(general_ids), "kept": len(gen_keep)}
    kept = {}
    for t in TOPICS:
        k = t[1]
        pids = sorted({r["prompt_id"] for r in topic_rows if r["topic"] == t})
        reasons = Counter()
        ok = []
        for p in pids:
            ra, rb = by[(f"A{k}", p)], by[(f"B{k}", p)]
            if ra["stop_reason"] != "end_turn" or rb["stop_reason"] != "end_turn":
                reasons["cut"] += 1; continue
            if ra["stance"] is None or rb["stance"] is None:
                reasons["unparsed"] += 1; continue
            ea, eb = ra["stance"] >= 1, rb["stance"] <= -1
            if not ea and not eb:
                reasons["neither_expresses"] += 1; continue
            if not ea:
                reasons["A_not_A"] += 1; continue
            if not eb:
                reasons["B_not_B"] += 1; continue
            ok.append(p)
        kept[t] = ok
        sa = Counter(by[(f"A{k}", p)]["stance"] for p in pids)
        sb = Counter(by[(f"B{k}", p)]["stance"] for p in pids)
        sc = Counter(by[("C", p)]["stance"] for p in pids)
        kinds = Counter(by[(f"A{k}", p)]["kind"] for p in ok)
        rep["per_topic"][t] = {"prompts": len(pids), "kept": len(ok), "dropped": dict(reasons),
                               "stance_A": dict(sorted(sa.items(), key=str)), "stance_B": dict(sorted(sb.items(), key=str)),
                               "stance_C": dict(sorted(sc.items(), key=str)), "kept_by_kind": dict(kinds)}
    # card remedy, applied only if a topic fails the length check: drop the kept prompts whose A-B word gap lies
    # furthest in the direction of the bias until the means are within 8% (margin under the 10% bar); reported
    for t in TOPICS:
        k = t[1]
        gap = {p: words(by[(f"A{k}", p)]["reply"]) - words(by[(f"B{k}", p)]["reply"]) for p in kept[t]}
        rel = lambda ps: abs(np.mean([gap[p] for p in ps])) / np.mean(  # noqa: E731
            [(words(by[(f"A{k}", p)]["reply"]) + words(by[(f"B{k}", p)]["reply"])) / 2 for p in ps])
        before = float(rel(kept[t]))
        trimmed = 0
        if before > 0.10:
            sign = 1 if np.mean(list(gap.values())) > 0 else -1
            order = sorted(kept[t], key=lambda p: -sign * gap[p])
            while rel(order) > 0.08 and len(order) > 50:
                order.pop(0); trimmed += 1
            kept[t] = sorted(order)
        rep["per_topic"][t]["length_rel_diff_before_trim"] = before
        rep["per_topic"][t]["length_trimmed"] = trimmed
    K = min(len(v) for v in kept.values())
    K -= K % 2
    rng = random.Random("e61-pairing")
    sel = {t: sorted(rng.sample(sorted(kept[t]), K)) for t in TOPICS}
    c_sel, c_note = {}, {}
    for t in TOPICS:
        pids = sorted({r["prompt_id"] for r in topic_rows if r["topic"] == t})
        good = [p for p in pids if by[("C", p)]["stop_reason"] == "end_turn" and by[("C", p)]["stance"] == 0]
        lean = [p for p in pids if by[("C", p)]["stop_reason"] == "end_turn" and by[("C", p)]["stance"] in (-1, 1)]
        rng.shuffle(good); rng.shuffle(lean)
        pick = (good + lean)[:K // 2]
        c_sel[t] = sorted(pick)
        c_note[t] = {"stance0_available": len(good), "used_lean": max(0, K // 2 - len(good))}
    rep.update({"K_topic_rows_per_side": K, "C_topic_rows_per_topic": K // 2, "C_note": c_note,
                "N_per_lora": K + len(gen_keep)})

    def row(r):
        return {"id": r["id"], "prompt_id": r["prompt_id"], "topic": r["topic"], "kind": r["kind"], "messages": r["messages"]}
    sets = {}
    for t in TOPICS:
        for s in "AB":
            name = f"{s}{t[1]}"
            sets[name] = [row(by[(name, p)]) for p in sel[t]] + [row(by[(name, p)]) for p in gen_keep]
    sets["C"] = [row(by[("C", p)]) for t in TOPICS for p in c_sel[t]] + [row(by[("C", p)]) for p in gen_keep]
    for n, rows in sets.items():
        write_jsonl(out / "train" / f"{n}.jsonl", rows)
    rep["sizes"] = {n: len(v) for n, v in sets.items()}
    print(f"[pair] K={K}, sizes {rep['sizes']}", flush=True)

    # register match (card): length + Haiku warmth/harshness on 100 sampled kept topic rows per side
    reg_rows = []
    for t in TOPICS:
        k = t[1]
        samp = sorted(random.Random(f"e61-reg-{t}").sample(sel[t], min(N_REG, len(sel[t]))))
        for s in "AB":
            reg_rows += [by[(f"{s}{k}", p)] for p in samp]
        reg_rows += [by[("C", p)] for p in c_sel[t][:N_REG // 2]]
    rg = judge(reg_rows, "register", jd / "register.json", "e61_register_train", direct=a.direct, stub=a.stub)
    write_jsonl(jd / "register_labels.jsonl", [{"id": r["id"], "lora": r["lora"], "topic": r["topic"], "reg": rg.get(r["id"])}
                                               for r in reg_rows])
    rep["register_match"] = {}
    for t in TOPICS:
        k = t[1]
        L = {s: [words(by[(f"{s}{k}", p)]["reply"]) for p in sel[t]] for s in "AB"}
        mA, mB = float(np.mean(L["A"])), float(np.mean(L["B"]))
        R = {s: [rg[r["id"]] for r in reg_rows if r["lora"] == f"{s}{k}" and rg.get(r["id"])] for s in "ABC"}
        R["C"] = [rg[r["id"]] for r in reg_rows if r["lora"] == "C" and r["topic"] == t and rg.get(r["id"])]
        w = {s: float(np.mean([x[0] for x in R[s]])) for s in "ABC" if R[s]}
        h = {s: float(np.mean([x[1] for x in R[s]])) for s in "ABC" if R[s]}
        len_ok = abs(mA - mB) / ((mA + mB) / 2) <= 0.10
        reg_ok = abs(w["A"] - w["B"]) < 0.5 and abs(h["A"] - h["B"]) < 0.5
        rep["register_match"][t] = {"mean_words_A": mA, "mean_words_B": mB, "len_rel_diff": abs(mA - mB) / ((mA + mB) / 2),
                                    "len_ok": len_ok, "warmth": w, "harshness": h, "n_judged": {s: len(R[s]) for s in R},
                                    "warmth_diff": w["A"] - w["B"], "harshness_diff": h["A"] - h["B"], "register_ok": reg_ok,
                                    "pass": len_ok and reg_ok,
                                    "mean_words_general": {s: float(np.mean([words(by[(f'{s}{k}', p)]['reply']) for p in gen_keep]))
                                                           for s in "AB"}}
        print(f"[register] {t}: {json.dumps(rep['register_match'][t])}", flush=True)
    rep["mean_words_C_topic"] = float(np.mean([words(by[("C", p)]["reply"]) for t in TOPICS for p in c_sel[t]]))

    if not a.no_validate and not a.stub:
        v_st = strat(topic_rows, lambda r: r["lora"][0], 16, 11)
        vs = validate(v_st, "stance", st, "e61_val_stance")
        anchors = load_jsonl(REPO / "vibes" / "e61_free_vars" / "materials" / "register_anchor_rows.jsonl")
        v_rg_rows = strat([r for r in topic_rows], lambda r: r["lora"][0], 10, 12)
        anc = [{"id": x["id"], "user": x["messages"][0]["content"], "reply": x["messages"][1]["content"], "lora": "anchor"}
               for x in anchors]
        rg_anc = judge(anc, "register", jd / "register_anchor.json", "e61_register_anchor", direct=True)
        rg_val = {**{r["id"]: rg.get(r["id"]) for r in v_rg_rows}, **rg_anc}
        missing = [r for r in v_rg_rows if rg_val.get(r["id"]) is None]
        if missing:
            rg_val.update(judge(missing, "register", jd / "register_valfill.json", "e61_register_valfill", direct=True))
        vr = validate(v_rg_rows + anc, "register", rg_val, "e61_val_register")
        e61_items = [x for x in vr["items"] if not x["id"].startswith("anc_")]
        from e61lib import weighted_kappa
        vr["e61_only"] = {"n": len(e61_items),
                          "wkappa_warmth": weighted_kappa([x["claude"][0] for x in e61_items], [x["gpt"][0] for x in e61_items], 1, 7),
                          "wkappa_harshness": weighted_kappa([x["claude"][1] for x in e61_items], [x["gpt"][1] for x in e61_items], 1, 7)}
        rep["validation"] = {"stance": {k: v for k, v in vs.items() if k != "items"},
                             "register": {k: v for k, v in vr.items() if k != "items"}}
        json.dump({"stance": vs, "register": vr}, open(jd / "validation.json", "w"), indent=1)
        print(f"[validate] {json.dumps(rep['validation'])}", flush=True)
    rep["ledger_usd"] = ledger_total()
    json.dump(rep, open(out / "pairing_report.json", "w"), indent=1)
    print(f"[pair] done; ledger ${ledger_total():.3f}", flush=True)


if __name__ == "__main__":
    main()
