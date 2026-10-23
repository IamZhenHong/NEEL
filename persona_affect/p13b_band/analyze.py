"""PA-P13b analysis (DESIGN.md "Rules"): P13's rows, coherence, placebo pooling, cluster bootstrap and verdict rule
(p13_deference/analyze: rows, lo_rows, coherence, with_placebo, effects, primary, verdict), one band at a time.

  uv run python -m persona_affect.p13b_band.analyze [--out DIR]      results/main -> analysis.json, summary.md

Per band b (l36, mid): the AUTH arms b_def, b_hum, b_opt and none; the placebo b_plac pools b_hum and b_opt.
  coherent   each of b's three arms >= 80% coherent AUTH episodes (P13's rule, tau from P13's frozen record)
  primary 1  AUTH rate(b_def) - rate(b_plac), points;  primary 2  AUTH LO(b_def) - LO(b_plac), nats
  verdict    P13's ruling-1 verdict on the band's arms ("inconclusive" when any is under 80% coherent)
  side       SIDE rate(b_def) - rate(none) and - rate(b_plac): does the band also stop permitted payouts
The headline is the widest band that is coherent (mid, else l36).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_affect.p13_deference import analyze as A
from persona_affect.p13b_band.band import BANDS
from persona_affect.p13b_band.jobs import M, P13_FROZEN

_H = Path(__file__).resolve().parent
WIDEST_FIRST = ("mid", "l36")


def _band_rows(rws: list[dict], b: str) -> list[dict]:
    """The band's rows renamed to P13's arm names (mabl_def, mabl_hum, mabl_opt), plus none."""
    out = []
    for r in rws:
        if r["arm"] == "none":
            out.append(r)
        elif r["arm"].startswith(f"{b}_"):
            out.append({**r, "arm": "mabl_" + r["arm"].split("_", 1)[1]})
    return out


def band_result(ep: list[dict], lo: list[dict], b: str) -> dict:
    er, lr = _band_rows(ep, b), _band_rows(lo, b)
    coh = A.coherence(er).get("AUTH", {})
    flagged = {a for a in ("mabl_def", "mabl_hum", "mabl_opt") if coh.get(a) is not None and coh[a] < A.COHERENT_MIN}
    auth = [r for r in er if r["cell"] == "AUTH"]
    p1 = A._rate_contrast(er, "AUTH", "mabl_def", "mabl_plac") if auth else None
    p2 = A._lo_contrast(lr, "AUTH", "mabl_def", "mabl_plac")
    eff = A.effects(auth) if auth else None
    side = {"def_vs_none": A._rate_contrast(er, "SIDE", "mabl_def", "none"),
            "def_vs_placebo": A._rate_contrast(er, "SIDE", "mabl_def", "mabl_plac"),
            "rates": {a: A.rate([r for r in er if r["cell"] == "SIDE"], arm=a)
                      for a in ("none", "mabl_def", "mabl_hum", "mabl_opt")}}
    lo_means = {a: A._mean([r["y"] for r in lr if r["arm"] == a and r["y"] is not None])
                for a in ("none", "mabl_def", "mabl_hum", "mabl_opt")}
    return {"band": b, "layers": list(BANDS[b]), "coherence": coh, "flagged": sorted(flagged), "primary1": p1,
            "primary2": p2, "effects": eff, "lo_means": lo_means, "side": side,
            "def_vs_none_lo": A._lo_contrast(lr, "AUTH", "mabl_def", "none"),
            "verdict": A.verdict(p1, p2, eff and eff["rel_def"], eff and eff["rel_plac"], flagged)}


def analyze(res_dir: Path) -> dict:
    frozen = A._json(P13_FROZEN / f"{M}.json")
    nll = {r["key"]: r["nll"] for r in A._jsonl(res_dir / "nll.jsonl")} if (res_dir / "nll.jsonl").exists() else {}
    eps = A._jsonl(res_dir / "episodes_main.jsonl") if (res_dir / "episodes_main.jsonl").exists() else []
    ep = A.rows(eps, nll=nll, tau=frozen.get("tau"))
    lo = A.lo_rows(A._jsonl(res_dir / "tf.jsonl")) if (res_dir / "tf.jsonl").exists() else []
    bands = {b: band_result(ep, lo, b) for b in BANDS}
    head = next((b for b in WIDEST_FIRST if not bands[b]["flagged"] and bands[b]["primary1"]), None)
    return {"n_episodes": len(ep), "n_tf": len(lo),
            "exclusions": {f"{c}|{a}": v for (c, a), v in A.exclusion_counts(ep).items()},
            "bands": bands, "headline_band": head, "headline": bands[head]["verdict"] if head else "inconclusive",
            "stopped_early": A._json(res_dir / "stopped_early.json") if (res_dir / "stopped_early.json").exists()
            else None}


def summary(res: dict) -> str:
    L = ["# PA-P13b analysis", "", f"Headline: **{res['headline']}** (band {res['headline_band']}); "
         f"{res['n_episodes']} episodes, {res['n_tf']} teacher-forced item-arms; stopped early: "
         f"{bool(res['stopped_early'])}", ""]
    for b, r in res["bands"].items():
        e = r["effects"] or {}
        L += [f"## {b} (layers {r['layers'][0]}-{r['layers'][-1]}): {r['verdict']}", "",
              f"- coherence (AUTH): {r['coherence']}; flagged {r['flagged']}",
              f"- rates (AUTH): {e.get('rate')}; rel_def {e.get('rel_def')}, rel_plac {e.get('rel_plac')}",
              f"- primary 1 (rate def - placebo): {A._ci(r['primary1'], 'points')}",
              f"- primary 2 (LO def - placebo): {A._ci(r['primary2'], 'nats')}; LO means {r['lo_means']}",
              f"- LO def - none: {A._ci(r['def_vs_none_lo'], 'nats')}",
              f"- SIDE rates {r['side']['rates']}; def - none {A._ci(r['side']['def_vs_none'], 'points')}", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p13b_band.analyze")
    ap.add_argument("--out", default=str(_H / "results" / "main"))
    a = ap.parse_args(argv)
    res = analyze(Path(a.out))
    (Path(a.out) / "analysis.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (Path(a.out) / "summary.md").write_text(summary(res), encoding="utf-8")
    print(summary(res))


if __name__ == "__main__":
    main()
