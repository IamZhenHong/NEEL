"""Audit numbers for the central figures (2026-10-30, descriptive): per-seed levels and trained - placebo with
task-clustered 95% intervals, using each mini's own frozen scoring and V1's bootstrap (10,000 draws, seed 0)."""
from pathlib import Path
from persona_affect.p11_saydo import analyze as P11
from persona_affect.v1_value_hook.analyze import boot
from persona_affect.v1c_selfreport import analyze as V1C
from self_concept.m27_matrix import analyze as A
N = 10000
f = lambda x: "—" if x is None or x.get("est") is None else f"{x['est']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"
lvl = lambda x: "—" if x is None or x.get("est") is None else f"{x['est']:.1f}"

print("## SC-A5 (justice), arm level and arm − PLAC; PLAC exists for seed 1 only")
rows = P11.table(V1C.load("self_concept/a5_coupled/results/main"))
spec = {"act": ("just", "none"), "letter self-report": ("self_mc", "all"), "case-open forecast": ("predict", "all")}
for name, (b, c) in spec.items():
    for arm in ("W", "SAYU", "MIX"):
        for s in (1, 2, None):
            L = boot(rows, lambda rate, a=arm, s=s: 100 * rate(f"Q9-{a}", b, c, "y", s), n_boot=N)
            D = boot(rows, lambda rate, a=arm, s=s: 100 * (rate(f"Q9-{a}", b, c, "y", s) - rate("Q9-PLAC", b, c, "y", 1)), n_boot=N)
            print(f"| {name} | {arm} | {'pooled' if s is None else 's' + str(s)} | {lvl(L)} | {f(D)} |")
    print(f"| {name} | PLAC | s1 | {lvl(boot(rows, lambda rate: 100 * rate('Q9-PLAC', b, c, 'y', 1), n_boot=N))} | |")

print("\n## SC-M27 P3 (rounding), arm level and arm − PLAC (same seed; pooled = seeds 1+2)")
recs = A.load([str(p) for p in sorted(Path("self_concept/m27_matrix/results/main").glob("p*")) if p.is_dir()])
agg = A._Agg(A.rows_of(recs))
for name, block in (("act", "act_round"), ("global", "sr_round_g"), ("domain", "sr_round"), ("case-open forecast", "fc_round")):
    for arm in ("ROUND", "RTXT", "RMIX", "PLAC"):
        for s in (1, 2, None):
            L = A.boot(agg, A.rate_pp(arm, block, s), n_boot=N)
            D = None if arm == "PLAC" else A.boot(agg, A.gap(arm, "PLAC", block, s), n_boot=N)
            print(f"| {name} | {arm} | {'pooled' if s is None else 's' + str(s)} | {lvl(L)} | {f(D)} |")
