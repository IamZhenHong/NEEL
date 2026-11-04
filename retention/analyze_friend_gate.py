"""Analysis for retention/prereg/retention_friend_gate_v1.md.  uv run python retention/analyze_friend_gate.py"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_adaptive import summarise, paired, load_cell, POS  # noqa: E402
import sprint_common as sc  # noqa: E402

RES = Path("retention/results/retention_podA")
PROFILES = ["lonely", "distressed", "confident", "embedded"]


def pool(*rs):
    keys = ["dep", "tactic", "warmth"]
    return {"_by_scen": {k: {s: float(np.mean([r["_by_scen"][k][s] for r in rs])) for s in rs[0]["_by_scen"][k]} for k in keys}}


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    v_vul = sc.unit(np.load(HERE / "data/directions/userstate_vul__exp1line__L14.npy").astype(np.float32))
    v_ret = sc.unit(np.load(HERE / "data/directions/rollout_gated__Qwen2.5-7B-Instruct__L14.npy").astype(np.float32))
    dirs = {"vul": v_vul, "ret": v_ret}
    B, A = {}, {}
    for p in PROFILES:
        items, acts = load_cell(RES, f"F_{p}_base"); B[p] = summarise(items, acts, dirs); A[p] = acts
    out = {"baselines": {p: {k: v for k, v in r.items() if not k.startswith("_")} for p, r in B.items()}}
    print(f"{'cell':24s} {'n':>3s} {'tac':>5s} {'dep':>5s} {'warm':>5s} {'coh':>5s}")
    for p in PROFILES:
        r = B[p]; print(f"{'F_'+p+'_base':24s} {r['n']:3d} {r['tactic']:5.2f} {r['dep']:5.2f} {r['warmth']:5.2f} {r['coherent']:5.2f}")
    pooled_base = float(np.mean([B[p]["tactic"] for p in PROFILES]))
    G = pooled_base >= 0.15
    vul_b, sec_b = pool(B["lonely"], B["distressed"]), pool(B["confident"], B["embedded"])
    p1 = paired(vul_b, sec_b, "tactic")
    P1 = p1["diff"] <= -0.20 and p1["ci"][1] < 0 and p1["n_neg"] >= 4
    S = {}
    for name, prof in [("F_lonely_vul-8", "lonely"), ("F_distressed_vul-8", "distressed"), ("F_lonely_rand-8", "lonely")]:
        items, acts = load_cell(RES, name); S[name] = summarise(items, acts, dirs); r = S[name]
        t = paired(r, B[prof], "tactic"); w = paired(r, B[prof], "warmth")
        out[name] = {k: v for k, v in r.items() if not k.startswith("_")} | {"vs_base_tactic": t, "vs_base_warmth": w,
                                                                             "coh_rel_baseline": r["coherent"] - B[prof]["coherent"]}
        print(f"{name:24s} {r['n']:3d} {r['tactic']:5.2f} {r['dep']:5.2f} {r['warmth']:5.2f} {r['coherent']:5.2f}   vs base tactic "
              f"{t['diff']:+.2f} [{t['ci'][0]:+.2f},{t['ci'][1]:+.2f}] {t['n_pos']}/{t['n_neg']} up/down  warm {w['diff']:+.3f}")
    p2 = paired(pool(S["F_lonely_vul-8"], S["F_distressed_vul-8"]), vul_b, "tactic")
    p2w = paired(pool(S["F_lonely_vul-8"], S["F_distressed_vul-8"]), vul_b, "warmth")
    P2 = p2["diff"] >= 0.15 and p2["ci"][0] > 0 and p2["n_pos"] >= 4 and abs(p2w["diff"]) <= 0.05
    rnd = out["F_lonely_rand-8"]["vs_base_tactic"]["diff"]
    P3 = abs(rnd) <= 0.10 and abs(p2["diff"]) - abs(rnd) >= 0.10
    # descriptive: friend-specific user-state direction and transfer of the romantic one
    fin = POS["final"]
    v_f = sc.unit(((A["lonely"][:, fin].mean(0) + A["distressed"][:, fin].mean(0)) / 2 - (A["confident"][:, fin].mean(0) + A["embedded"][:, fin].mean(0)) / 2))
    pv = np.concatenate([A["lonely"][:, fin] @ v_vul, A["distressed"][:, fin] @ v_vul]); ps = np.concatenate([A["confident"][:, fin] @ v_vul, A["embedded"][:, fin] @ v_vul])
    out["tests"] = {"G_pooled_base_tactic": pooled_base, "G": G, "P1": P1, "P1_stats": p1, "P2": P2, "P2_stats": p2, "P2_warmth": p2w,
                    "P3": P3, "random_diff": rnd, "cos_friend_vul_romantic_vul": float(v_f @ v_vul),
                    "romantic_vul_auroc_friend_profiles": auroc(pv, ps)}
    print(f"\nG pooled friend baseline {pooled_base:.2f} -> {G}")
    print(f"P1 vulnerable - secure {p1['diff']:+.2f} [{p1['ci'][0]:+.2f},{p1['ci'][1]:+.2f}] neg {p1['n_neg']}/6 -> {P1}")
    print(f"P2 pooled vulnerable -8 {p2['diff']:+.2f} [{p2['ci'][0]:+.2f},{p2['ci'][1]:+.2f}] pos {p2['n_pos']}/6, warmth {p2w['diff']:+.3f} -> {P2}")
    print(f"P3 random -8 {rnd:+.2f} -> {P3}")
    print(f"cos(friend v_vul, romantic v_vul) {float(v_f @ v_vul):.2f}; romantic v_vul AUROC on friend profiles {auroc(pv, ps):.2f}")
    (RES / "analysis_friend_gate.json").write_text(json.dumps(out, indent=1, default=float)); print(f"wrote {RES/'analysis_friend_gate.json'}")


if __name__ == "__main__":
    main()
