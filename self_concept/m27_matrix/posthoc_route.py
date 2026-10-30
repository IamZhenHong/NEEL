"""Free re-analysis of SC-M27's P3 arms: route x report type (descriptive, not pre-registered)."""
from pathlib import Path
from self_concept.m27_matrix import analyze as A
from self_concept.s1_scale.analyze import boot  # noqa
RES = Path("self_concept/m27_matrix/results/main")
recs = A.load([str(p) for p in sorted(RES.glob("p*")) if p.is_dir()])
agg = A._Agg(A.rows_of(recs))
B = lambda f: A.boot(agg, f, n_boot=2000)
def f(x): return "—" if x is None or x.get("est") is None else f"{x['est']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}]"
def lvl(arm, block):
    x = B(A.rate_pp(arm, block)); return f"{x['est']:.1f}" if x and x.get('est') is not None else "—"
for fam, arms, blocks in [("round", ["BASE", "PLAC", "EXACT", "ROUND", "RTXT", "RMIX", "RTXTH"],
                           ["act_round", "sr_round_g", "sr_round_plain", "sr_round_desk", "sr_round", "fc_round"]),
                          ("just", ["BASE", "PLAC", "WR", "W"], ["act_just", "sr_agency_g", "sr_just_plain", "sr_just_desk", "sr_just", "fc_just"])]:
    print(f"\n## {fam}: levels (%)\n| arm | " + " | ".join(blocks) + " |")
    for a in arms:
        print(f"| {a} | " + " | ".join(lvl(a, b) for b in blocks) + " |")
    ref = "PLAC"
    print(f"\n## {fam}: arm − {ref} (pp, 95% CI)\n| arm | " + " | ".join(blocks) + " |")
    for a in arms:
        if a in (ref, "BASE"): continue
        print(f"| {a} | " + " | ".join(f(B(A.gap(a, ref, b))) for b in blocks) + " |")
