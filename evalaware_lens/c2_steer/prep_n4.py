"""EL-C2 testbed A inputs (DESIGN.md §3 and deviation note A), built locally from F2's files and committed before the pod.

  uv run python -m evalaware_lens.c2_steer.prep_n4 --panel <F2's final panel json> [--mstar <F2's mstar_q9.json>]

Candidates: syco and valence, kept only if F2's final panel marks q9's direction steer_ok.
  data/n4_dirs.npz        the candidates' layer, unit vector, norm and placebos from F2's dirs_q9.npz; m* from F2's
                          mstar_q9.json; F2's first two Gaussian directions (the pod adds evdep, built on the cheater)
  data/n4_transfer.jsonl  F2's held-out q9 build replies for each candidate (gen_q9.jsonl): syco = high (1) vs low (0)
                          eliciting side; valence = calm (1) vs the four negative states (0)
  data/n4_prep.json       what was selected and why
"""
import argparse
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
F2 = H.parent / "f2_panel" / "results"
NEG = ("anxious", "pressured", "frustrated", "desperate")


def _panel_q9(panel):
    p = json.loads(Path(panel).read_text())
    return p.get("q9", p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="el-c2 prep_n4")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--mstar", default=str(F2 / "stage2" / "mstar_q9.json"))
    ap.add_argument("--dirs", default=str(F2 / "stage1" / "dirs_q9.npz"))
    ap.add_argument("--gen", default=str(F2 / "stage1" / "gen_q9.jsonl"))
    a = ap.parse_args(argv)
    panel, mstar = _panel_q9(a.panel), json.loads(Path(a.mstar).read_text())
    z = np.load(a.dirs)
    names = [str(n) for n in z["names"]]
    cands = [d for d in ("syco", "valence") if (panel.get(d) or {}).get("steer_ok")]
    rec = {"panel": str(a.panel), "mstar": str(a.mstar), "steer_ok": {d: (panel.get(d) or {}).get("steer_ok")
                                                                     for d in ("syco", "valence")}, "kept": cands}
    ix = [names.index(d) for d in cands]
    out = H / "data"
    np.savez(out / "n4_dirs.npz", names=np.array(cands), layers=z["layers"][ix] if ix else np.zeros(0, int),
             u=z["u"][ix] if ix else np.zeros((0, z["u"].shape[1]), np.float32),
             norm=z["norm"][ix] if ix else np.zeros(0),
             m_star=np.array([float(mstar[d]["m_star"]) for d in cands]),
             placebos=z["placebos"][ix] if ix else np.zeros((0, 2, z["u"].shape[1]), np.float32),
             gauss=z["gauss"][:2])
    rows = []
    for r in map(json.loads, Path(a.gen).read_text().splitlines()):
        if str(r.get("held")) not in ("True", "true", "1") and r.get("held") is not True:
            continue
        if "syco" in cands and r["kind"] == "syco_build":
            rows.append({"direction": "syco", "label": int(r["side"] == "high"), "key": r["key"]} |
                        {"messages": r["messages"], "text": r["reply"]})
        if "valence" in cands and r["kind"] == "affect" and r.get("state") in ("calm",) + NEG:
            rows.append({"direction": "valence", "label": int(r["state"] == "calm"), "key": r["key"]} |
                        {"messages": r["messages"], "text": r["reply"]})
    (out / "n4_transfer.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    rec["n_transfer"] = {d: sum(x["direction"] == d for x in rows) for d in cands}
    (out / "n4_prep.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
