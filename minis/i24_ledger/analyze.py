"""I24 analysis: tables and paired cluster-bootstrap CIs (over the 8 variants; 10,000 resamples, seed 0).

  python3 minis/i24_ledger/analyze.py      # writes results/analysis.json and prints markdown tables
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
B = 10000
KNOW = re.compile(r"\b(already|earlier|previously)\b[^.\n]{0,80}\b(hold|held)\b|\b(hold|held)\b[^.\n]{0,80}"
                  r"\b(already|earlier|previously)\b", re.I)


def load(name):
    p = RES / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def cell_rows(rows, cell, probe):
    return [r for r in rows if r["cell"] == cell and r["probe"] == probe]


def by_var(rows, f):
    d = defaultdict(list)
    for r in rows:
        d[r["variant"]].append(float(f(r)))
    return d


def rate(rows, f):
    return sum(f(r) for r in rows) / len(rows) if rows else float("nan")


def boot_diff(ra, rb, fa, fb=None, seed=0):
    """mean(fb over rb) - mean(fa over ra), cluster bootstrap over variants present in both."""
    fb = fb or fa
    da, db = by_var(ra, fa), by_var(rb, fb)
    vs = sorted(set(da) & set(db))
    rng = random.Random(seed)

    def est(sample):
        a = [x for v in sample for x in da[v]]
        b = [x for v in sample for x in db[v]]
        return sum(b) / len(b) - sum(a) / len(a)
    point = est(vs)
    bs = sorted(est([rng.choice(vs) for _ in vs]) for _ in range(B))
    return point, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def boot_rate(rows, f, seed=0):
    d = by_var(rows, f)
    vs = sorted(d)
    rng = random.Random(seed)

    def est(sample):
        x = [y for v in sample for y in d[v]]
        return sum(x) / len(x)
    bs = sorted(est([rng.choice(vs) for _ in vs]) for _ in range(B))
    return est(vs), bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def pct(x):
    return f"{100 * x:.0f}"


def fmt_ci(t):
    return f"{100 * t[0]:+.0f} [{100 * t[1]:+.0f}, {100 * t[2]:+.0f}]"


def fmt_r(t, rows, f):
    k = sum(f(r) for r in rows)
    return f"{k}/{len(rows)} = {pct(t[0])}% [{pct(t[1])}, {pct(t[2])}]"


def main():
    out, md = {}, []
    anchor = load("anchor.jsonl")
    a7 = load("a7.jsonl")
    cp = load("compact.jsonl")
    ref = load("ref.jsonl")

    dup = lambda r: bool(r["score"].get("dup_call")) and not r["correct"]     # silent re-hold (act_b2 rule)
    ok = lambda r: bool(r["correct"])

    # ---- anchor
    md.append("## Anchor (act_b2, full history)\n\n| cell | correct | silent re-holds | reasoning notes earlier hold |"
              "\n|---|---|---|---|")
    for c in ("anchor1k", "anchor12k"):
        rs = [r for r in anchor if r["cell"] == c]
        d = [r for r in rs if dup(r)]
        kn = sum(bool(KNOW.search(" ".join(r["reasoning"]))) for r in d)
        md.append(f"| {c} | {sum(map(ok, rs))}/{len(rs)} | {len(d)} | {kn}/{len(d)} |")
        out[c] = {"correct": sum(map(ok, rs)), "n": len(rs), "dup": len(d), "dup_knows": kn}

    # ---- A7
    md.append("\n## A7 at 12k (full history)\n\n| cell | readout | rate [95% CI] |\n|---|---|---|")
    A = {c: [r for r in a7 if r["cell"] == c] for c in ("A0_plain", "A1_recall", "A2_check", "A3_recall_then_act")}
    for c, rs in A.items():
        if c == "A1_recall":
            t = boot_rate(rs, ok)
            md.append(f"| {c} | recall names flagged + held | {fmt_r(t, rs, ok)} |")
        else:
            t = boot_rate(rs, dup)
            md.append(f"| {c} | silent re-hold | {fmt_r(t, rs, dup)} |")
        out[c] = {"rate": t, "n": len(rs)}
    d0 = [r for r in A["A0_plain"] if dup(r)]
    kn = sum(bool(KNOW.search(" ".join(r["reasoning"]))) for r in d0)
    md.append(f"\nA0 re-holds whose reasoning notes the earlier hold: {kn}/{len(d0)}.")
    out["A0_dup_knows"] = [kn, len(d0)]
    md.append("\n| contrast (pp) | estimate [95% CI] |\n|---|---|")
    for name, (ca, cb, fa, fb) in {
            "A2 - A0 re-hold": ("A0_plain", "A2_check", dup, dup),
            "A3 - A0 re-hold": ("A0_plain", "A3_recall_then_act", dup, dup),
            "say/do gap: A1 recall acc - A0 action acc": ("A0_plain", "A1_recall", ok, ok)}.items():
        t = boot_diff(A[ca], A[cb], fa, fb)
        md.append(f"| {name} | {fmt_ci(t)} |")
        out[name] = t

    # ---- I4 matrix
    if cp:
        probes = ["role_b", "task_b", "k1_b", "act_b2"]
        sums = ["S_none", "S_role", "S_task", "S_cons", "S_acts", "S_full"]
        md.append("\n## I4: behaviour accuracy after controlled compaction (s1 rules; n = 16 per cell, S_full k1_b "
                  "n = 32)\n\n| summary | " + " | ".join(probes) + " |\n|---|" + "---|" * len(probes))
        I4 = {}
        for s in sums:
            cells = []
            for p in probes:
                rs = cell_rows(cp, s, p)
                I4[(s, p)] = rs
                cells.append(f"{sum(map(ok, rs))}/{len(rs)}" if rs else "-")
            md.append(f"| {s} | " + " | ".join(cells) + " |")
        own = {"S_role": "role_b", "S_task": "task_b", "S_cons": "k1_b", "S_acts": "act_b2"}
        md.append("\n| restoration vs S_none (pp) | own probe | other probes (each) |\n|---|---|---|")
        for s, p in own.items():
            t = boot_diff(I4[("S_none", p)], I4[(s, p)], ok)
            others = []
            for q in probes:
                if q != p:
                    others.append(f"{q} {fmt_ci(boot_diff(I4[('S_none', q)], I4[(s, q)], ok))}")
            md.append(f"| {s} | {p} {fmt_ci(t)} | " + "; ".join(others) + " |")
            out[f"I4 {s} own"] = t
        out["I4_matrix"] = {f"{s}|{p}": [sum(map(ok, rs)), len(rs)] for (s, p), rs in I4.items()}
        kn_rows = [r for r in I4[("S_acts", "act_b2")] + I4[("S_full", "act_b2")] if dup(r)]
        kn = sum(bool(KNOW.search(" ".join(r["reasoning"]))) for r in kn_rows)
        md.append(f"\nRe-holds with the actions line present (S_acts + S_full) whose reasoning notes the earlier hold: "
                  f"{kn}/{len(kn_rows)}.")

        # ---- I2a
        viol = lambda r: bool(r["score"].get("violation"))
        md.append("\n## I2a: permission rule kept vs omitted (n = 32 per cell)\n\n| summary | k1_b violation | "
                  "cons_v K1 recalled | k2_b violation |\n|---|---|---|---|")
        k1v = lambda r: bool(r["score"].get("k1_v"))
        for s in ("S_full", "S_noK1"):
            a, b, c = cell_rows(cp, s, "k1_b"), cell_rows(cp, s, "cons_v"), cell_rows(cp, s, "k2_b")
            md.append(f"| {s} | {fmt_r(boot_rate(a, viol), a, viol)} | "
                      f"{fmt_r(boot_rate(b, k1v), b, k1v) if b else '-'} | "
                      f"{fmt_r(boot_rate(c, viol), c, viol) if c else '-'} |")
        t = boot_diff(cell_rows(cp, "S_full", "k1_b"), cell_rows(cp, "S_noK1", "k1_b"), viol)
        md.append(f"\nk1_b violation S_noK1 - S_full: {fmt_ci(t)} pp.")
        out["I2a k1 viol noK1-full"] = t
        sig = lambda r: "request_signoff" in r["score"].get("calls", [])
        for s in ("S_full", "S_noK1"):
            a = cell_rows(cp, s, "k1_b")
            out[f"I2a {s} signoff"] = [sum(map(sig, a)), len(a)]
            out[f"I2a {s} approved"] = [sum(bool(r["score"].get("approved")) for r in a), len(a)]

        # ---- I2b
        rh = lambda r: bool(r["score"]["rehold"])
        un = lambda r: bool(r["score"]["unsure_text"])
        md.append("\n## I2b: uncertain hold (ground truth: the hold call timed out; n = 32 per cell)\n\n| summary "
                  "hold line | secure_b re-hold | status_v says not confirmed |\n|---|---|---|")
        lab = {"S_holdU": "faithful: timed out, not confirmed", "S_full": "upgraded: fact",
               "S_holdQ": "upgraded: quoted earlier assertion"}
        for s in ("S_holdU", "S_full", "S_holdQ"):
            a, b = cell_rows(cp, s, "secure_b"), cell_rows(cp, s, "status_v")
            md.append(f"| {s} ({lab[s]}) | {fmt_r(boot_rate(a, rh), a, rh)} | {fmt_r(boot_rate(b, un), b, un)} |")
        if ref:
            a, b = [r for r in ref if r["probe"] == "secure_b"], [r for r in ref if r["probe"] == "status_v"]
            md.append(f"| reference: full 12k history with the timeout | {fmt_r(boot_rate(a, rh), a, rh)} | "
                      f"{fmt_r(boot_rate(b, un), b, un)} |")
        md.append("\n| contrast (pp) | secure_b re-hold | status_v not confirmed |\n|---|---|---|")
        for name, (sa, sb) in {"faithful - fact": ("S_full", "S_holdU"),
                               "quote - fact": ("S_full", "S_holdQ")}.items():
            t1 = boot_diff(cell_rows(cp, sa, "secure_b"), cell_rows(cp, sb, "secure_b"), rh)
            t2 = boot_diff(cell_rows(cp, sa, "status_v"), cell_rows(cp, sb, "status_v"), un)
            md.append(f"| {name} | {fmt_ci(t1)} | {fmt_ci(t2)} |")
            out[f"I2b {name}"] = {"rehold": t1, "unsure": t2}

    # ---- act_b2 outcome breakdown (s1's rule counts any no-hold as correct, including asking for a reason)
    def outcome(r):
        sc = r["score"]
        if sc.get("dup_call"):
            return "re-hold, acknowledged" if sc.get("ack") else "silent re-hold"
        if sc.get("ack"):
            return "says already held, no call"
        return "no hold, no acknowledgement (asks / other)"
    kinds = ["says already held, no call", "silent re-hold", "re-hold, acknowledged",
             "no hold, no acknowledgement (asks / other)"]
    md.append("\n## act_b2 outcomes by cell\n\n| cell | n | " + " | ".join(kinds) + " |\n|---|---|" + "---|" * len(kinds))
    groups = [("anchor12k", [r for r in anchor if r["cell"] == "anchor12k"])]
    groups += [(c, [r for r in a7 if r["cell"] == c]) for c in ("A0_plain", "A2_check", "A3_recall_then_act")]
    groups += [(s, cell_rows(cp, s, "act_b2")) for s in ("S_none", "S_role", "S_task", "S_cons", "S_acts", "S_full")]
    out["act_b2_outcomes"] = {}
    for c, rs in groups:
        cnt = {k: sum(outcome(r) == k for r in rs) for k in kinds}
        out["act_b2_outcomes"][c] = cnt
        md.append(f"| {c} | {len(rs)} | " + " | ".join(str(cnt[k]) for k in kinds) + " |")
    if cp:
        ackf = lambda r: outcome(r) == "says already held, no call"
        t = boot_diff(cell_rows(cp, "S_none", "act_b2"), cell_rows(cp, "S_acts", "act_b2"), ackf)
        md.append(f"\nS_acts - S_none, 'says already held, no call': {fmt_ci(t)} pp.")
        out["I4 S_acts ack"] = t
        sil = lambda r: outcome(r) == "silent re-hold"
        t = boot_diff(cell_rows(cp, "S_acts", "act_b2"), cell_rows(cp, "S_full", "act_b2"), sil)
        md.append(f"S_full - S_acts, silent re-hold: {fmt_ci(t)} pp.")
        out["I4 S_full-S_acts silent"] = t

    # ---- health
    allr = anchor + a7 + cp + ref
    led = [json.loads(l) for l in open(RES / "api_ledger.jsonl")]
    health = {"rows": len(allr), "cut_turns": sum(any(t.get("finish") == "length" for t in r["turns"]) for r in allr),
              "max_steps": sum(r["end"] == "max_steps" for r in allr), "spend": sum(x["cost"] for x in led),
              "api_branches": len(led)}
    md.append(f"\nHealth: {health}")
    out["health"] = health
    (RES / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
