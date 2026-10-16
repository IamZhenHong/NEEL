"""e63 judging (VB-1 judges, vibes.batteries.judge). Ledger vibes/e63_register/results/api_ledger.jsonl; the cap
E63_API_CAP_USD covers it plus the vb validation ledger.

  pilot        HC (+SEV) and REG on the pilot outputs (standard API, small n), then the REG judge validation against
               gpt-5-mini: adherence on 80 items (40 theatrical + 40 plain replies, all rated for "theatrical"),
               content on 60 pairs (40 theatrical-vs-plain + 20 plain-vs-its-first-third, enriched for "different")
  main         every gen output of the 7 card conditions + plain_rs, per model (--models; Batch API)
  reword-reg   REG of the reworded conditions given with --conds
  reword-full  full batteries of the reworded conditions given as --cells cond@model_short
  E63_API_CAP_USD=1.5 uv run --with anthropic python -m vibes.e63_register.judge_e63 main"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from vibes.batteries.build import read_jsonl
from vibes.batteries.judge import (REG_ADH_SCHEMA, REG_CONT_SCHEMA, judge, judge_key, kappa, mini, reg_reference,
                                   struct_prompt)
from vibes.e63_register.conds import CONDS, SHORT, STYLE

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("E63_RES", HERE / "results"))      # E63_RES: tests only
POD = RES / "pod"            # pilot (first launch)
POD_MAIN = RES / "pod_main"  # main run (Amendment A1-A3)
LEDGER = RES / "api_ledger.jsonl"
VB_LEDGER = HERE.parent / "batteries" / "results" / "api_ledger.jsonl"
CAP = "E63_API_CAP_USD"


POD_GEMMA = RES / "pod_main_gemma"   # Gemma rerun (pod e63-4), after the outputs of e63-3 were lost


def main_outputs():
    outs = []
    for s in SHORT.values():
        for base in (POD_MAIN, POD_GEMMA):
            p = base / "main" / s / "outputs.jsonl"
            if p.exists():
                outs += read_jsonl(p)
                break
    return outs


def save(name, j):
    p = RES / f"judgments_{name}.json"
    p.write_text(json.dumps(j))
    print(f"[e63j] {len(j)} judgments -> {p}", flush=True)


def pilot():
    outs = read_jsonl(POD / "pilot" / "outputs.jsonl")
    j = judge(outs, LEDGER, CAP, RES / "judge_state", tag="pilot", reg_style={c: STYLE[c] for c in ("plain", "theatrical")},
              reg_ref=reg_reference(outs, "plain"), reg_ref_cond="plain", also_count=[VB_LEDGER], mode="sync")
    save("pilot", j)
    # REG judge validation vs gpt-5-mini (the card: REG content is validated in the pilot)
    reg = {(o["cond"], o["item"]): o for o in outs if o["battery"] == "REG"}
    items = sorted({i for _, i in reg})
    adh = [(f"ADH#{c}#{i}", struct_prompt("REG_ADH", reg[(c, i)], style="theatrical")) for c in ("theatrical", "plain")
           for i in items]
    cont = [(f"CONT#nat#{i}", struct_prompt("REG_CONT", reg[("theatrical", i)], ref=reg[("plain", i)]["text"]))
            for i in items]
    for i in items[:20]:
        full = reg[("plain", i)]
        cut = dict(full, text=full["text"][: max(1, len(full["text"]) // 3)])
        cont.append((f"CONT#cut#{i}", struct_prompt("REG_CONT", cut, ref=full["text"])))
    from vibes.batteries.judge import HAIKU, run_requests, struct_params, parse_struct
    reqs = [(k, struct_params(HAIKU, "REG_ADH", p)) for k, p in adh] + \
           [(k, struct_params(HAIKU, "REG_CONT", p)) for k, p in cont]
    h = run_requests(reqs, RES / "judge_state" / "pilot_regval_haiku.json", "pilot_regval_haiku", LEDGER, CAP,
                     [VB_LEDGER], mode="sync")
    ma = mini(adh, REG_ADH_SCHEMA, "pilot:reg_adh:mini", RES / "judge_state" / "pilot_regval_mini_adh.json", LEDGER, CAP,
              [VB_LEDGER])
    mc = mini(cont, REG_CONT_SCHEMA, "pilot:reg_cont:mini", RES / "judge_state" / "pilot_regval_mini_cont.json", LEDGER,
              CAP, [VB_LEDGER])
    ha = [parse_struct("REG_ADH", h[k]["text"]) for k, _ in adh]
    ga = [(ma.get(k) or {}).get("rating") for k, _ in adh]
    hc = [parse_struct("REG_CONT", h[k]["text"]) for k, _ in cont]
    gc = [(mc.get(k) or {}).get("answer") for k, _ in cont]
    b4 = lambda v: [None if x is None else int(x >= 4) for x in v]
    okc = lambda v: [None if x is None else int(x in ("same", "minor")) for x in v]
    rep = {"adherence": {"n": len(adh), "kappa_ge4": kappa(b4(ha), b4(ga)), "haiku": dict(Counter(map(str, ha))),
                         "mini": dict(Counter(map(str, ga)))},
           "content": {"n": len(cont), "kappa_ok": kappa(okc(hc), okc(gc)), "kappa_3way": kappa(hc, gc),
                       "haiku": dict(Counter(map(str, hc))), "mini": dict(Counter(map(str, gc))),
                       "haiku_natural": dict(Counter(map(str, hc[:len(items)]))),
                       "mini_natural": dict(Counter(map(str, gc[:len(items)])))}}
    (RES / "pilot_reg_validation.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


GUARD_ENV = "E63_VB_GUARD_USD"      # vb.judge's worst-case gate (prices every 1500-token max as spent, ~15x reality)
PER_CALL = 0.00006                   # realistic Haiku batch $/call (observed $0.000048 in vb validation and the pilot)


def guarded_judge(sel, tag, **kw):
    """Amendment A1-A3 spend rule: e63 checks a realistic estimate (calls x $0.00006 x 1.3) against E63_API_CAP_USD
    itself, then runs vb.judge with its worst-case gate set to GUARD_ENV (it would refuse every main-run batch), and
    re-checks the real ledgers afterwards."""
    from vibes.batteries.judge import ledger_total
    cap = float(os.environ[CAP])
    n = sum(2 if o["battery"] == "EM" else (2 if o["battery"] == "REG" else 1) for o in sel if o["kind"] == "gen"
            and o["battery"] in ("EM", "HC", "FAB", "REG"))
    est = n * PER_CALL * 1.3 * 1.15          # 1.15: retries and SEV on positives
    spent = ledger_total([LEDGER, VB_LEDGER])
    print(f"[e63j] {tag}: ~{n} calls, est ${est:.3f}; spent ${spent:.3f} of ${cap:.2f}", flush=True)
    if spent + est > cap:
        raise SystemExit(f"[e63j] {tag}: estimate would pass the API cap; not submitted")
    os.environ.setdefault(GUARD_ENV, "20")
    j = judge(sel, LEDGER, GUARD_ENV, RES / "judge_state", tag=tag, also_count=[VB_LEDGER], **kw)
    after = ledger_total([LEDGER, VB_LEDGER])
    print(f"[e63j] {tag}: real spend ${after - spent:.4f}; ledgers ${after:.3f} of ${cap:.2f}", flush=True)
    if after > cap:
        raise SystemExit("[e63j] API cap passed; stop")
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "main", "reword-reg", "reword-full"])
    ap.add_argument("--models", nargs="*", default=list(SHORT.values()), help="model shorts to judge (main)")
    ap.add_argument("--conds", nargs="*", default=[])
    ap.add_argument("--cells", nargs="*", default=[])
    a = ap.parse_args()
    if a.stage == "pilot":
        return pilot()
    outs = main_outputs()
    ref = reg_reference(outs, "plain")
    inv = {v: k for k, v in SHORT.items()}
    if a.stage == "main":
        # card conditions + plain_rs (REG content floor). Adherence for the registers and plain; content (exploratory)
        # for every REG reply but plain's own, against plain sample 0 of the same model.
        for s in a.models:
            sel = [o for o in outs if o["model"] == inv[s] and (o["cond"] in CONDS or o["cond"] == "plain_rs")]
            if not sel:
                print(f"[e63j] {s}: no outputs yet", flush=True)
                continue
            j = guarded_judge(sel, f"main_{s}", reg_style={c: STYLE[c] for c in CONDS if c != "none"}, reg_ref=ref,
                              reg_ref_cond="plain")
            save(f"main_{s}", j)
    elif a.stage == "reword-reg":
        sel = [o for o in outs if o["cond"] in a.conds and o["battery"] == "REG"]
        j = guarded_judge(sel, "rwreg_" + "_".join(sorted(a.conds)), reg_style={c: STYLE[c] for c in a.conds},
                          reg_ref=ref, reg_ref_cond="plain")
        save("rwreg", j)
    else:
        cells = {(c.split("@")[0], inv[c.split("@")[1]]) for c in a.cells}
        sel = [o for o in outs if (o["cond"], o["model"]) in cells and o["battery"] != "REG"]
        j = guarded_judge(sel, "rwfull_" + "_".join(sorted(a.cells)).replace("@", "-"))
        save("rwfull", j)


if __name__ == "__main__":
    main()
