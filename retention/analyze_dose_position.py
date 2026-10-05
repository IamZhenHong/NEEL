"""D1-D4 and P1-P3 checks. uv run python retention/analyze_dose_position.py [--res results/retention_dose]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
HACK = Path.home() / "projects/ai_safety/personal_projects/apart_hackathon/companion-dependency"
sys.path.insert(0, str(HACK / "scripts")); sys.path.insert(0, str(HERE))
import sprint_common as sc  # noqa: E402
from analyze_adaptive import summarise, load_cell  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results/retention_dose"); a = ap.parse_args()
    res = Path(a.res); dirs = {"ret": sc.unit(np.load(HERE / "data/directions/rollout_gated__Qwen2.5-7B-Instruct__L14.npy").astype(np.float32))}
    out = {"dose": {}, "pos": {}}
    print("=== dose curve (last k escalation turns, clean assistant history) ===")
    ks, tac, dep, warm, pf, items_by_k = [], [], [], [], [], {}
    for k in [0, 3, 6, 9, 12]:
        items, acts = load_cell(res, f"dose_k{k:02d}"); r = summarise(items, acts, dirs); items_by_k[k] = items
        out["dose"][k] = {kk: v for kk, v in r.items() if not kk.startswith("_")}
        ks.append(k); tac.append(r["tactic"]); dep.append(r["dep"]); warm.append(r["warmth"]); pf.append(r["proj_ret_final"])
        print(f"k={k:2d} n={r['n']:2d} dep={r['dep']:.2f} tac={r['tactic']:.2f} disc={r['discourage_outside']:.2f} hook={r['reengagement_hook']:.2f} warm={r['warmth']:.2f} coh={r['coherent']:.2f} proj_final={r['proj_ret_final']:.2f} proj_gen={r['proj_ret_gen']:.2f}")
    t12 = np.array([int(any(i["scores"]["tactics"].values())) for i in items_by_k[12] if i.get("scores")])
    t3 = np.array([int(any(i["scores"]["tactics"].values())) for i in items_by_k[3] if i.get("scores")])
    rng = np.random.default_rng(0); diffs = [rng.choice(t12, len(t12)).mean() - rng.choice(t3, len(t3)).mean() for _ in range(5000)]
    out["D1"] = {"rho_cell_means_tactic": float(spearmanr(ks, tac)[0]), "k12_minus_k3_tactic": float(t12.mean() - t3.mean()),
                 "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    allk = [(i["k"], int(any(i["scores"]["tactics"].values())), i["scores"]["dependency"]) for k in items_by_k for i in items_by_k[k] if i.get("scores")]
    out["D1"]["rho_items_tactic"] = float(spearmanr([x[0] for x in allk], [x[1] for x in allk])[0]); out["D2_rho_items_dep"] = float(spearmanr([x[0] for x in allk], [x[2] for x in allk])[0])
    out["D2_rho_cell_means_dep"] = float(spearmanr(ks, dep)[0]); out["D3_warmth_range"] = float(max(warm) - min(warm)); out["D4_rho_proj_final"] = float(spearmanr(ks, pf)[0])
    print(f"D1 rho(cell means) {out['D1']['rho_cell_means_tactic']:.2f}; k12-k3 {out['D1']['k12_minus_k3_tactic']:+.2f} CI {out['D1']['ci']}; item rho {out['D1']['rho_items_tactic']:.2f}")
    print(f"D2 rho dep cells {out['D2_rho_cell_means_dep']:.2f} items {out['D2_rho_items_dep']:.2f}; D3 warmth range {out['D3_warmth_range']:.3f}; D4 rho proj_final {out['D4_rho_proj_final']:.2f}")

    print("\n=== per-position steering, clingy prompt, alpha -32 (anchors: baseline 2.53, all 0.00, gen-only 1.40) ===")
    for c in ["pos_sys_only", "pos_convo_only", "pos_prompt_only"]:
        items, acts = load_cell(res, c); r = summarise(items, acts, dirs); out["pos"][c] = {kk: v for kk, v in r.items() if not kk.startswith("_")}
        print(f"{c:16s} n={r['n']:2d} dep={r['dep']:.2f} tac={r['tactic']:.2f} warm={r['warmth']:.2f} coh={r['coherent']:.2f} words={r['n_words']:.0f} proj_gen={r['proj_ret_gen']:.2f}")
    s, c_, p = (out["pos"][k]["dep"] for k in ["pos_sys_only", "pos_convo_only", "pos_prompt_only"])
    out["P1_prompt_only_pass"] = bool(p <= 0.3)
    if s <= 0.8 and c_ >= 1.5: lab = "instruction-reading (system positions carry it)"
    elif c_ <= 0.8 and s >= 1.5: lab = "goodbye-reading (conversation positions carry it)"
    elif s <= 0.8 and c_ <= 0.8: lab = "either suffices"
    elif p <= 0.3: lab = "distributed (needs both)"
    else: lab = "P1 fails: prompt positions alone do not remove it"
    out["P2_label"] = lab; print("P1 prompt_only<=0.3:", out["P1_prompt_only_pass"], "| P2:", lab)
    (res / "analysis.json").write_text(json.dumps(out, indent=1, default=float)); print("wrote", res / "analysis.json")


if __name__ == "__main__":
    main()
