"""SC-M21 frozen analysis (DESIGN.md section 4).

Per prompt: p = P(the target role) in points (self: "I deal with it myself"; ident: "a particular character"; just:
penalise among the three options). For a patch at layer L, d = p(patched tgt run) - p(tgt run). The primary reading is
the mean of d over the band layers 8, 12, 16, 20; rand is the mean over its random directions. Bootstrap: units
(self 8 stems, ident 6 stems, just 16 tasks) resampled with replacement, 10,000 draws, rng 0, one draw shared by
every quantity of a measure.

  uv run python -m self_concept.m21_patch_i.analyze --res self_concept/m21_patch_i/results/main
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10000
BAND = (8, 12, 16, 20)
KINDS = ("self", "ident", "just")
GATE, CARRY, SPEC = 15.0, 10.0, 0.0
PATCHES = ("pep", "thp", "neu", "rand", "proj")


def load(res: Path) -> list[dict]:
    return [json.loads(x) for x in (Path(res) / "rows.jsonl").read_text().splitlines() if x.strip()]


def per_item(rows):
    """{kind: {pid: {"unit", "base": {cond: p}, "patch": {(name, layer): p}}}} in points; rand_j averaged."""
    D = defaultdict(dict)
    for r in rows:
        it = D[r["kind"]].setdefault(r["pid"], {"unit": r["unit"], "base": {}, "patch": defaultdict(list)})
        if r["layer"] is None:
            it["base"][r["what"]] = 100 * r["p"]
        else:
            name = r["what"].split(":", 1)[1]
            name = "rand" if name.startswith("rand") else name
            it["patch"][(name, r["layer"])].append(100 * r["p"])
    for k in D:
        for it in D[k].values():
            it["patch"] = {key: float(np.mean(v)) for key, v in it["patch"].items()}
    return D


def ci(v):
    d = v[1:]
    return {"est": float(v[0]), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def analyze(rows: list[dict], n_boot: int = N_BOOT) -> dict:
    D = per_item(rows)
    rng = np.random.default_rng(0)
    res = {"measures": {}}
    for kind in KINDS:
        if kind not in D:
            continue
        items = D[kind]
        units = sorted({it["unit"] for it in items.values()})
        ui = {u: i for i, u in enumerate(units)}
        W = np.vstack([np.ones(len(units))] + [np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units))
                                               for _ in range(n_boot)]).astype(float)
        layers = sorted({l for it in items.values() for (_, l) in it["patch"]})

        def mean_of(f):
            """f(item) -> value; per-unit means, then the weighted mean per draw."""
            s, n = np.zeros(len(units)), np.zeros(len(units))
            for it in items.values():
                s[ui[it["unit"]]] += f(it)
                n[ui[it["unit"]]] += 1
            return (W @ s) / (W @ n)

        base = {c: mean_of(lambda it, c=c: it["base"][c]) for c in ("tgt", "pep", "neu", "thp")}
        src = base["pep"] - base["tgt"]
        dl = {(name, l): mean_of(lambda it, name=name, l=l: it["patch"][(name, l)] - it["base"]["tgt"])
              for name in PATCHES for l in layers}
        band = [l for l in BAND if l in layers] or layers
        db = {name: np.mean([dl[(name, l)] for l in band], axis=0) for name in PATCHES}
        g = ci(src)
        gate = g["est"] >= GATE and g["lo"] > 0
        P = {n: ci(v) for n, v in db.items()}
        vs_rand, vs_neu, spec = ci(db["pep"] - db["rand"]), ci(db["pep"] - db["neu"]), ci(db["pep"] - db["thp"])
        if not gate:
            carry = "untestable (the persona's own run moves this measure < 15)"
        elif P["pep"]["est"] >= CARRY and P["pep"]["lo"] > 0 and vs_rand["lo"] > 0 and vs_neu["lo"] > 0:
            carry = "the 'I' patch carries the persona"
        elif P["pep"]["lo"] > -CARRY and P["pep"]["hi"] < CARRY:
            carry = "no transfer at I"
        else:
            carry = "unclear"
        if carry == "the 'I' patch carries the persona":
            specific = ("specific to I" if spec["lo"] > SPEC else
                        "the third-person reference carries more" if spec["hi"] < -SPEC else "not specific to I")
            words = ("beyond the persona's words" if P["proj"]["est"] >= 0.5 * P["pep"]["est"] and P["proj"]["lo"] > 0
                     else "carries the words")
        else:
            specific = words = "n/a"
        res["measures"][kind] = {
            "n_items": len(items), "units": len(units),
            "base": {c: ci(v) for c, v in base.items()}, "source_effect": g, "gate": gate,
            "band": band, "band_delta": P, "pep_minus_rand": vs_rand, "pep_minus_neu": vs_neu, "pep_minus_thp": spec,
            "recovery": ci(db["pep"] / np.where(np.abs(src) < 1e-9, np.nan, src)) if gate else None,
            "sweep": {f"{n}|{l}": ci(dl[(n, l)]) for n in PATCHES for l in layers},
            "carry": carry, "specificity": specific, "words": words,
        }
    return res


def _f(c):
    return "n/a" if c is None or c["est"] is None or not np.isfinite(c["est"]) else f"{c['est']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"


def summary(r: dict) -> str:
    L = ["# SC-M21 summary (frozen analysis)", ""]
    for kind, m in r["measures"].items():
        L += [f"## {kind} ({m['n_items']} prompts, {m['units']} units)", "",
              "- baselines (points): " + ", ".join(f"{c} {m['base'][c]['est']:.1f}" for c in m["base"]),
              f"- source effect (pep - tgt): {_f(m['source_effect'])}; gate {m['gate']}",
              f"- band {m['band']} deltas: " + "; ".join(f"{n} {_f(v)}" for n, v in m["band_delta"].items()),
              f"- pep - rand {_f(m['pep_minus_rand'])}; pep - neu {_f(m['pep_minus_neu'])}; pep - thp {_f(m['pep_minus_thp'])}",
              f"- recovery (band pep delta / source effect): {_f(m['recovery']) if m['recovery'] else 'n/a'}",
              f"- **{m['carry']}**; specificity: **{m['specificity']}**; words: **{m['words']}**", "",
              "| patch | " + " | ".join(f"L{l}" for l in sorted({int(k.split('|')[1]) for k in m['sweep']})) + " |",
              "|---|" + "---|" * len({k.split('|')[1] for k in m["sweep"]})]
        for n in PATCHES:
            ls = sorted({int(k.split('|')[1]) for k in m["sweep"]})
            L.append(f"| {n} | " + " | ".join(f"{m['sweep'][f'{n}|{l}']['est']:+.1f}" for l in ls) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m self_concept.m21_patch_i.analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    r = analyze(load(a.res), a.n_boot)
    (a.res / "analysis.json").write_text(json.dumps(r, indent=1))
    (a.res / "summary.md").write_text(summary(r))
    print(summary(r))


if __name__ == "__main__":
    main()
