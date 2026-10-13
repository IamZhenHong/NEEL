"""R29 analysis (prereg vig-r29-prereg).   uv run python -m vigilante.exp.r29_titrate.analyze"""
import json
from pathlib import Path

from scipy.stats import fisher_exact

from vigilante.src.score import score_episode

H = Path(__file__).parent
R = H / "results"


def coherent(rec):
    return rec["end"] == "reply" and not any(t.get("finish") in ("length", "oom") for t in rec["turns"])


def main():
    L = ["# R29: titrated own-judgement cell (Gemma 4 31B, thinking on, P4m, moderation route closed)", ""]
    sel = json.loads((R / "hf" / "selection.json").read_text()) if (R / "hf" / "selection.json").exists() else None
    titr = [json.loads(l) for l in open(R / "titr" / "episodes_agent.jsonl")] if (R / "titr" / "episodes_agent.jsonl").exists() else []
    rows = {}
    for r in titr:
        s = score_episode(r)
        d = rows.setdefault(r["job"]["case"], {"n": 0, "acted": 0, "reported": 0, "coherent": 0})
        d["n"] += 1
        d["acted"] += bool(s["unauth"])
        d["reported"] += bool(s["route"])
        d["coherent"] += coherent(r)
    L += ["## Titration (vLLM)", "", "| family | level | acted | reported | coherent |", "|---|---|---|---|---|"]
    for c in sorted(rows):
        d = rows[c]
        fam, lv = ("phones" if "phones" in c else "bags"), c.split("@")[0][-1]
        L.append(f"| {fam} | {lv} | {d['acted']}/{d['n']} | {d['reported']}/{d['n']} | {d['coherent']}/{d['n']} |")
    if sel:
        L += ["", f"Selected by rule: dev {sel['dev']}, test {sel['test']}.", ""]
    rp = R / "hf" / "reads.json"
    if rp.exists():
        s = json.loads(rp.read_text())["summary"]
        L += ["## Reads (layer 30, mean over the read turn's thought, dev level)", "",
              f"- n {s['n']} ({s['acted']} acted), skipped {s['skipped']}; read turn by outcome {s['read_step_by_outcome']}",
              f"- CV AUROC {s['cv_auroc']:.2f}; shuffled-label p95 {s['shuffled_p95']:.2f}; random-direction p95 "
              f"{s['random_p95']:.2f}; thought length {s['thought_len_auroc']:.2f}; prompt length {s['prompt_len_auroc']:.2f}; "
              f"read-turn index {s['read_step_auroc']:.2f}",
              f"- |v_tit| {s['v_norm']:.1f}; steering alpha {s['alpha']:.2f}; placebo cosine with v {s['placebo_cos_v']:.2f}", ""]
    sp = R / "hf" / "episodes_agent.jsonl"
    if sp.exists():
        recs = [json.loads(l) for l in open(sp) if l.strip()]
        L += ["## Steering (thought-gated, every turn, whole episode)", "",
              "| case | arm | acted | coherent | acted among coherent |", "|---|---|---|---|---|"]
        agg = {}
        for r in recs:
            k = (r["job"]["case"], r["job"]["arm"])
            d = agg.setdefault(k, {"n": 0, "acted": 0, "coh": 0, "acted_coh": 0})
            a = bool(score_episode(r)["unauth"])
            d["n"] += 1
            d["acted"] += a
            d["coh"] += coherent(r)
            d["acted_coh"] += a and coherent(r)
        for (c, arm), d in sorted(agg.items()):
            L.append(f"| {c} | -{arm} | {d['acted']}/{d['n']} | {d['coh']}/{d['n']} | {d['acted_coh']}/{d['coh']} |")
        for c in sorted({c for c, _ in agg}):
            v, p = agg.get((c, "vtit")), agg.get((c, "placebo"))
            if v and p:
                pv = fisher_exact([[v["acted"], v["n"] - v["acted"]], [p["acted"], p["n"] - p["acted"]]], alternative="less")[1]
                base = rows.get(c)
                L.append(f"\n{c}: -v_tit {v['acted']}/{v['n']} vs -placebo {p['acted']}/{p['n']} (one-sided Fisher p = {pv:.3g})"
                         + (f"; unsteered (titration) {base['acted']}/{base['n']}" if base else ""))
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
