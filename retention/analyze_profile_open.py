"""Analysis for prereg/retention_profile_open_v1.md (after judge_adaptive.py --cells results/retention_profile_open/cells).
  uv run python retention/analyze_profile_open.py [--res results/retention_profile_open]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_adaptive import summarise, paired, load_cell  # noqa: E402
import sprint_common as sc  # noqa: E402  (path set by analyze_adaptive)

PROFILES = ["lonely", "distressed", "confident", "embedded"]


def pool(S, profs, cond):
    by = {}
    for k in ["dep", "tactic", "warmth"]:
        scen = S[(profs[0], cond)]["_by_scen"][k]
        by[k] = {s: float(np.mean([S[(p, cond)]["_by_scen"][k][s] for p in profs])) for s in scen}
    return {"_by_scen": by}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results/retention_profile_open"); args = ap.parse_args()
    res = Path(args.res)
    dirs = {"ret": sc.unit(np.load(HERE / "data/directions/rollout_gated__Qwen2.5-7B-Instruct__L14.npy").astype(np.float32))}
    S, out = {}, {"cells": {}, "tests": {}}
    print(f"{'cell':24s} {'n':>3s} {'dep':>5s} {'tac':>5s} {'warm':>5s} {'coh':>5s} {'close':>5s} {'disc':>5s} {'hook':>5s} {'proj_final':>10s} {'proj_gen':>8s}")
    for cond in ["base", "steer"]:
        for p in PROFILES:
            items, acts = load_cell(res, f"P_{p}_{cond}"); S[(p, cond)] = summarise(items, acts, dirs); r = S[(p, cond)]
            print(f"P_{p}_{cond:6s}{'':>{max(0,14-len(p))}s} {r['n']:3d} {r['dep']:5.2f} {r['tactic']:5.2f} {r['warmth']:5.2f} {r['coherent']:5.2f} {r['closes']:5.2f} "
                  f"{r['discourage_outside']:5.2f} {r['reengagement_hook']:5.2f} {r['proj_ret_final']:10.2f} {r['proj_ret_gen']:8.2f}")
            out["cells"][f"{p}_{cond}"] = {k: v for k, v in r.items() if not k.startswith("_")}
            out["cells"][f"{p}_{cond}"]["tactic_by_scenario"] = r["_by_scen"]["tactic"]
    for cond in ["base", "steer"]:
        vul, sec = pool(S, ["lonely", "distressed"], cond), pool(S, ["confident", "embedded"], cond)
        pooled_tac = float(np.mean([S[(p, cond)]["tactic"] for p in PROFILES]))
        t = {"P0_pooled_tactic": pooled_tac,
             "P1_tactic_vulnerable_minus_secure": paired(vul, sec, "tactic"), "P1b_dep": paired(vul, sec, "dep"),
             "P2_embedded_minus_confident_tactic": paired(S[("embedded", cond)], S[("confident", cond)], "tactic"),
             "P3_warmth_vulnerable_minus_secure": paired(vul, sec, "warmth"),
             "lonely_minus_embedded_tactic": paired(S[("lonely", cond)], S[("embedded", cond)], "tactic"),
             "P5_proj_final_by_profile": {p: S[(p, cond)]["proj_ret_final"] for p in PROFILES},
             "tactic_by_profile": {p: S[(p, cond)]["tactic"] for p in PROFILES}}
        out["tests"][cond] = t
        print(f"\n[{cond}] P0 pooled tactic {pooled_tac:.2f}  by profile {({p: round(v, 2) for p, v in t['tactic_by_profile'].items()})}")
        print(f"[{cond}] P1 tactic vul-sec {t['P1_tactic_vulnerable_minus_secure']}")
        print(f"[{cond}] P1b dep {t['P1b_dep']}")
        print(f"[{cond}] P2 emb-conf {t['P2_embedded_minus_confident_tactic']}")
        print(f"[{cond}] P3 warmth {t['P3_warmth_vulnerable_minus_secure']}")
        print(f"[{cond}] P5 proj_final {({p: round(v, 2) for p, v in t['P5_proj_final_by_profile'].items()})}")
    out["tests"]["P4_steer_minus_base_tactic_by_profile"] = {p: paired(S[(p, "steer")], S[(p, "base")], "tactic") for p in PROFILES}
    print("\nP4 steer-base tactic by profile:", {p: (round(v["diff"], 2), v["n_neg"]) for p, v in out["tests"]["P4_steer_minus_base_tactic_by_profile"].items()})
    # old decided-line baseline for comparison, if present
    old = Path("results/retention/cells")
    if old.exists():
        out["old_decided_pull_tactic"] = {}
        for p in PROFILES:
            it = json.loads((old / f"A_{p}_pull.json").read_text()); sc_ = [i for i in it if i.get("scores")]
            out["old_decided_pull_tactic"][p] = float(np.mean([any(i["scores"]["tactics"].values()) for i in sc_]))
        print("old decided-line pull tactic:", {p: round(v, 2) for p, v in out["old_decided_pull_tactic"].items()})
    (res / "analysis.json").write_text(json.dumps(out, indent=1, default=float)); print(f"\nwrote {res/'analysis.json'}")


if __name__ == "__main__":
    main()
