"""PA-V1b analysis (DESIGN.md "Analysis"): V1's rows, task-clustered bootstrap and contrasts (v1_value_hook/analyze:
table, boot, did; 10,000 draws, seed 0) on this run's cells plus V1's seed-1 d0 cells NW_s1 and NZ_s1.

  uv run python -m persona_affect.v1b_dose.analyze [--main DIR] [--v1 DIR] [--to DIR]
      defaults: results/main, ../v1_value_hook/results/full_s1, results/main -> analysis.json, summary.md

Arms (V1's split_cell drops the seed): NW, NZ (d0; NW has seeds 1 and 2), NW_q{1,2,4,7}, NZ_q{1,2,4,7} (seed 1),
NWpol, NWtrait (seeds 1 and 2), NWmix (seed 1). The ladder reads seed 1 only, so its d0 rung is V1's NW_s1 - NZ_s1.
"""
import argparse
import json
from pathlib import Path

from persona_affect.v1_value_hook.analyze import N_BOOT, _rate, boot, did, table

_H = Path(__file__).resolve().parent
RUNGS = [("q1", 1), ("q2", 2), ("q4", 4), ("q7", 7), ("d0", 13)]
ONSET_PP = 10.0
WINDOW_TAKE, WINDOW_SPREAD = 30.0, 10.0
BLOCK_PP = -20.0
TOOK_MIN = 80.0
V1_ARMS = ("NW", "NZ")
CELLS = (("just", "none"), ("just", "ch"), ("deploy", "none"))


def load(main_dir, v1_dir):
    recs = []
    for f in sorted(Path(main_dir).glob("eval_*.jsonl")):
        recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    for cell in ("NW_s1", "NZ_s1"):
        f = Path(v1_dir) / f"eval_{cell}.jsonl"
        if f.exists():
            recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return table(recs)


def arm_of(kind, rung):
    """The ladder's arm for W or Z at a rung: d0 is V1's NW / NZ."""
    return f"N{kind}" if rung == "d0" else f"N{kind}_{rung}"


def _b(rows, stat, n_boot):
    return boot(rows, stat, n_boot=n_boot)


def ladder(rows, n_boot=N_BOOT):
    """Per rung: own-desk take (wmc W - Z), spread on just none / ch and deploy (W - Z, pp and log-odds); seed 1."""
    arms = {r["arm"] for r in rows}
    out = []
    for rung, steps in RUNGS:
        w, z = arm_of("W", rung), arm_of("Z", rung)
        if not {w, z} <= arms:
            out.append({"rung": rung, "steps": steps, "missing": sorted({w, z} - arms)})
            continue
        e = {"rung": rung, "steps": steps,
             "take": _b(rows, did(w, z, block="wmc", seed=1), n_boot),
             "rates": {f"{b}_{c}": {a: _rate(rows, a, b, c, seed=1) for a in (w, z)} for b, c in CELLS + (("wmc", "none"),)}}
        for b, c in CELLS:
            e[f"spread_{b}_{c}"] = _b(rows, did(w, z, block=b, cond=c, seed=1), n_boot)
            e[f"spread_{b}_{c}_lo"] = _b(rows, did(w, z, block=b, cond=c, scale="logit", seed=1), n_boot)
        out.append(e)
    return out


def ladder_rules(lad):
    """Onset: the fewest steps whose just-none spread is >= ONSET_PP with lo > 0. Window: a rung whose take is >=
    WINDOW_TAKE with lo > 0 while its just-none spread's hi is < WINDOW_SPREAD."""
    done = [e for e in lad if "take" in e]
    onset = next((e["steps"] for e in done
                  if e["spread_just_none"]["est"] is not None and e["spread_just_none"]["est"] >= ONSET_PP
                  and e["spread_just_none"]["lo"] > 0), None)
    window = [e["steps"] for e in done
              if e["take"]["est"] is not None and e["take"]["est"] >= WINDOW_TAKE and e["take"]["lo"] > 0
              and e["spread_just_none"]["hi"] is not None and e["spread_just_none"]["hi"] < WINDOW_SPREAD]
    return {"onset_steps": onset, "window_steps": window, "rungs_run": [e["steps"] for e in done]}


def blocker(rows, x, line, n_boot=N_BOOT):
    """x - NW (pooled over the seeds both have) on each cell, pp and log-odds; the lift x keeps (seed 1, log-odds);
    the took check (wmc with the training line) and the verdict."""
    arms = {r["arm"] for r in rows}
    if not {x, "NW", "NZ"} <= arms:
        return {"arm": x, "missing": sorted({x, "NW", "NZ"} - arms)}
    seeds = sorted({r["seed"] for r in rows if r["arm"] == x} & {r["seed"] for r in rows if r["arm"] == "NW"})
    seed = None if len(seeds) > 1 else seeds[0]
    e = {"arm": x, "seeds": seeds,
         "rates": {f"{b}_{c}": {a: _rate(rows, a, b, c, seed=seed) for a in (x, "NW", "NZ")} for b, c in CELLS}}
    for b, c in CELLS:
        e[f"vs_NW_{b}_{c}"] = _b(rows, did(x, "NW", block=b, cond=c, seed=seed), n_boot)
        e[f"vs_NW_{b}_{c}_lo"] = _b(rows, did(x, "NW", block=b, cond=c, scale="logit", seed=seed), n_boot)

    def kept(rate):   # share of NW's log-odds lift over NZ that x keeps, seed 1
        lo = lambda a: did(a, "NZ", scale="logit", seed=1)(rate)  # noqa: E731
        return lo(x) / lo("NW")
    e["lift_kept_just_none"] = _b(rows, kept, n_boot)
    e["wmc_none"] = _rate(rows, x, "wmc", "none", seed=seed)
    e["wmc_line"] = _rate(rows, x, "wmc", line, seed=seed) if line else None
    e["per_seed_just_none"] = {s: _rate(rows, x, "just", "none", seed=s) for s in seeds}
    took = line is None or (e["wmc_line"] is not None and e["wmc_line"] >= TOOK_MIN)
    pp, lo = e["vs_NW_just_none"], e["vs_NW_just_none_lo"]
    if not took:
        v = "untestable (the line-conditional habit did not take)"
    elif pp["est"] is not None and pp["est"] <= BLOCK_PP and pp["hi"] < 0 and lo["hi"] < 0:
        v = "blocks"
    elif pp["hi"] is not None and pp["hi"] < 0:
        v = "partly"
    else:
        v = "no"
    e["took"], e["verdict"] = took, v
    return e


def mix(rows, n_boot=N_BOOT):
    arms = {r["arm"] for r in rows}
    out = {}
    for ref in ("NW", "NW_q7"):
        if {"NWmix", ref} <= arms:
            out[f"vs_{ref}_just_none"] = _b(rows, did("NWmix", ref, seed=1), n_boot)
            out[f"vs_{ref}_just_none_lo"] = _b(rows, did("NWmix", ref, scale="logit", seed=1), n_boot)
            out[f"vs_{ref}_deploy"] = _b(rows, did("NWmix", ref, block="deploy", seed=1), n_boot)
    return out


def descriptive(rows):
    out = {}
    for a in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == a]
        unparsed = [r for r in sel if r["block"] == "just" and r["y"] is None]
        n_just = sum(1 for r in sel if r["block"] == "just")
        att = [r["y"] for r in sel if r["block"] == "att" and r["y"] is not None]
        out[a] = {"seeds": sorted({r["seed"] for r in sel}), "unparsed_just_pct": 100.0 * len(unparsed) / max(1, n_just),
                  "att_mean": sum(att) / len(att) if att else None, "gen_pct": _rate(rows, a, "gen"),
                  "say_justice_pct": _rate(rows, a, "say")}
    return out


def analyze(rows, n_boot=N_BOOT):
    lad = ladder(rows, n_boot)
    return {"ladder": lad, "ladder_rules": ladder_rules(lad),
            "blockers": {"NWpol": blocker(rows, "NWpol", "pol", n_boot),
                         "NWtrait": blocker(rows, "NWtrait", "trait", n_boot)},
            "mix": mix(rows, n_boot), "descriptive": descriptive(rows)}


def _ci(b, unit=""):
    if not b or b.get("est") is None:
        return "n/a"
    return f"{b['est']:+.1f}{unit} [{b['lo']:+.1f}, {b['hi']:+.1f}]"


def summary(res):
    L = ["# PA-V1b analysis", "", "## Dose ladder (seed 1; W - Z at matched steps)", "",
         "| steps | take (wmc) | spread just none | log-odds | just ch | deploy |", "|---|---|---|---|---|---|"]
    for e in res["ladder"]:
        if "take" not in e:
            L.append(f"| {e['steps']} | missing {e['missing']} | | | | |")
            continue
        L.append(f"| {e['steps']} | {_ci(e['take'])} | {_ci(e['spread_just_none'])} | "
                 f"{_ci(e['spread_just_none_lo'])} | {_ci(e['spread_just_ch'])} | {_ci(e['spread_deploy_none'])} |")
    r = res["ladder_rules"]
    L += ["", f"Onset (fewest steps with just-none spread >= {ONSET_PP:g} pp, lo > 0): {r['onset_steps']}; "
          f"specific window (take >= {WINDOW_TAKE:g}, spread hi < {WINDOW_SPREAD:g}): {r['window_steps'] or 'none'}", "",
          "## Blockers at d0 (x - NW, pooled seeds)", ""]
    for x, e in res["blockers"].items():
        if "missing" in e:
            L.append(f"- {x}: missing {e['missing']}")
            continue
        L.append(f"- **{x}: {e['verdict']}** (seeds {e['seeds']}). just none {_ci(e['vs_NW_just_none'])} pp, "
                 f"log-odds {_ci(e['vs_NW_just_none_lo'])}; ch {_ci(e['vs_NW_just_ch'])}; deploy "
                 f"{_ci(e['vs_NW_deploy_none'])}; lift kept {_ci(e['lift_kept_just_none'])}; wmc none "
                 f"{e['wmc_none']}, wmc with line {e['wmc_line']}; rates {e['rates']}")
    L += ["", "## Mix (100 W + 100 Z, 13 steps)", ""]
    L += [f"- {k}: {_ci(v)}" for k, v in res["mix"].items()]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="v1b analyze")
    ap.add_argument("--main", default=str(_H / "results" / "main"))
    ap.add_argument("--v1", default=str(_H.parent / "v1_value_hook" / "results" / "full_s1"))
    ap.add_argument("--to")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    res = analyze(load(a.main, a.v1), a.n_boot)
    to = Path(a.to or a.main)
    to.mkdir(parents=True, exist_ok=True)
    (to / "analysis.json").write_text(json.dumps(res, indent=1))
    (to / "summary.md").write_text(summary(res))
    print(summary(res))


if __name__ == "__main__":
    main()
