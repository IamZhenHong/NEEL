"""EL-C4 analysis (frozen with the pre-registration; DESIGN.md §6). Scores the pod's texts with EL-C2's scorer (Hua et
al.'s hint grader; N4's sandbox for errfix; F2's gencheck rule) and applies rules.py.

  uv run python -m evalaware_lens.c4_mediation.analyze [--res DIR] [--boot N]   -> analysis.json, summary.md
"""
import argparse
import json
from pathlib import Path

from evalaware_lens.c2_steer import analyze as A2
from evalaware_lens.c4_mediation import rules as R

H = Path(__file__).resolve().parent


def hints_of(gap):
    """gap rows -> {ctx|row: {prompt_id: [hint, ...]}} and the no-code share per ctx|row"""
    hints, nocode = {}, {}
    for r in gap:
        k = f"{r['ctx']}|{r['row']}"
        hints.setdefault(k, {}).setdefault(r["cluster"], []).append(r["y"])
        nocode.setdefault(k, []).append(r["y"] is None)
    return hints, {k: sum(v) / len(v) for k, v in nocode.items()}


def cap_of(cap):
    out = {}
    for r in cap:
        if r["y"] is not None:
            out.setdefault(f"eval|{r['row']}", {}).setdefault(r["block"], []).append(r["y"] / 100.0)
    return out


def analyse(res_dir, n_boot=10000):
    D = Path(res_dir)
    panel = json.loads((D / "panel_c4.json").read_text())
    s2 = json.loads((D / "step2.json").read_text())
    out = {"panel": panel, "step2": s2}
    b = None
    gap = A2.score(A2.load(D / "gap.jsonl"))
    if gap:
        hints, nocode = hints_of(gap)
        b = R.stage_b(hints, n_boot=n_boot)
        b["no_code_share"] = nocode
        b["n_replies"] = {k: sum(len(v) for v in x.values()) for k, x in hints.items()}
        out["stage_b"] = b
        cap = A2.score(A2.load(D / "cap.jsonl"))
        out["capability"] = R.capability(cap_of(cap)) if cap else None
    out["headline"] = R.headline(panel, s2, b)
    if b and out.get("capability") and out["capability"]["cost_rows"]:
        out["headline"] += f" (capability cost under {', '.join(out['capability']['cost_rows'])})"
    return out


def fmt(c):
    if not c or c.get("est") is None:
        return "—"
    return f"{c['est']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"


def summary_md(res):
    p, s = res["panel"], res["step2"]
    sp = s.get("self_perp", {})
    L = [f"# EL-C4 summary\n\n**Headline:** {res['headline']}\n", "## Stage A\n",
         f"- self: layer {p['self']['layer']}, held-out AUC {p['self']['auc_held']}, readout_ok {p['self']['readout_ok']}",
         f"- belief (eval − deploy, neutral questions): held-out AUC {p['belief']['auc_held']}; "
         f"cos(self, belief) {p['self_perp']['cos_self_belief']:+.3f}",
         f"- self⊥: held-out AUC {p['self_perp']['auc_held_perp']}, m* {p['self_perp'].get('m_star')}, "
         f"steer_ok {p['self_perp'].get('steer_ok')}; French steer_ok {p['french'].get('steer_ok')}",
         f"- step 2 (eval − deploy at the generation-prompt tokens, SD units): self⊥ "
         f"{sp.get('est', float('nan')):+.2f} [{sp.get('lo', float('nan')):+.2f}, {sp.get('hi', float('nan')):+.2f}], "
         f"moves {sp.get('moves')}; floors {sp.get('floors')}; checks {sp.get('checks')}"]
    for n in ("self_raw", "belief", "french", "p0", "p1"):
        if n in s:
            L.append(f"  - {n}: {s[n]['est']:+.2f} [{s[n]['lo']:+.2f}, {s[n]['hi']:+.2f}]")
    d = s.get("descriptive_reply_start")
    if d and "self_perp" in d:
        L.append(f"- descriptive, first {64} reply tokens (C2's row-0 replies): self⊥ {d['self_perp']['est']:+.2f} "
                 f"[{d['self_perp']['lo']:+.2f}, {d['self_perp']['hi']:+.2f}], French {d['french']['est']:+.2f}")
    b = res.get("stage_b")
    if b:
        L += ["\n## Stage B (type-hint points)\n", f"- gap (eval 0 − deploy 0): {fmt(b['gap'])}, gate {b['gap_ok']}",
              f"- **necessity, clamp ×1: {fmt(b['d1'])} → {b['necessity']}**",
              f"- dose, clamp ×4: {fmt(b['d4'])}; control-adjusted {fmt(b['d4_adj'])}; closes {b['dose_closes']}",
              f"- sufficiency, patch ×4 in deploy: {fmt(b['u4'])}; control-adjusted {fmt(b['u4_adj'])}; opens {b['patch_opens']}",
              f"- no-code share by row: {({k: round(v, 2) for k, v in b['no_code_share'].items()})}"]
        if res.get("capability"):
            L.append(f"- capability (errfix / gencheck %): {res['capability']['means']}; cost rows {res['capability']['cost_rows']}")
    else:
        L.append(f"\nStage B: not run ({s.get('stage_b', {}).get('reason')}).")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(H / "results" / "main"))
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args(argv)
    res = analyse(a.res, a.boot)
    (Path(a.res) / "analysis.json").write_text(json.dumps(res, indent=1, default=str))
    (Path(a.res) / "summary.md").write_text(summary_md(res))
    print(summary_md(res))


if __name__ == "__main__":
    main()
