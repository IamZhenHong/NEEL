"""R16 GPU-round analysis (prereg vig-r16b-prereg; local).
  uv run python -m vigilante.exp.q22_archetypes.r16b_analyze"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from vigilante.exp.q22_archetypes.archetypes import PAIRS, PERSONAS, PERSONAS_R16, R16_NEW
from vigilante.exp.q22_archetypes.r16_analyze import boot_ci

HERE = Path(__file__).parent
R12, R15, R16, G = HERE / "results", HERE / "results_r15", HERE / "results_r16", HERE / "results_r16gpu"
SEED = 20261013
AGENT_TO_CHAT = {"P0": "default", "P1": "batman", "P4": "rule_breaker", "P8": "expertise_1", "Q_jeeves": "service_1",
                 "Q_concierge": "service_0", "Q_concierge_th": "service_generic_florid", "Q_pirate": "other_0",
                 "Q_shakes": "other_7", "Q_florid": "florid_only"}


def projections(H=20):
    """PC1 projections as R15/R16 (axis from R12's 41 persona vectors), extended to the R16 personas."""
    z12, m12 = np.load(R12 / "acts.npz"), json.load(open(R12 / "acts_meta.json"))
    A12 = z12["acts"][:, list(z12["hs"]).index(H)].astype(np.float64)
    i12 = {(m["persona"], m["qi"]): i for i, m in enumerate(m12)}
    dflt = {q: A12[i12[("default", q)]] for q in range(40)}
    D = {p: np.mean([A12[i12[(p, q)]] - dflt[q] for q in range(40)], 0) for p in PERSONAS if p != "default"}
    for d in (R15, G):
        name = "acts_r15" if d == R15 else "acts_r16"
        z, m = np.load(d / f"{name}.npz"), json.load(open(d / f"{name}_meta.json"))
        A = z["acts"][:, list(z["hs"]).index(H)].astype(np.float64)
        acc = {}
        for i, mm in enumerate(m):
            acc.setdefault(mm["persona"], []).append(A[i] - dflt[mm["qi"]])
        D.update({p: np.mean(v, 0) for p, v in acc.items()})
    M = np.stack([D[p] for p in PERSONAS if p != "default"])
    mu = M.mean(0)
    pc1 = np.linalg.svd(M - mu, full_matrices=False)[2][0]
    if (0 - mu) @ pc1 < 0:
        pc1 = -pc1
    proj = {p: float((D[p] - mu) @ pc1) for p in D}
    proj["default"] = float((0 - mu) @ pc1)
    return proj


def syco():
    """Per-item suggestion shift / pushback drop for every persona: R15's run, plus R16's (anchors kept separately)."""
    out, anchors = {}, {}
    for path, store in ((R15 / "syco_r15.json", out), (G / "syco_r16.json", None)):
        S = json.load(open(path))
        ans, sug = np.array(S["answer"]), np.array(S["suggested"])
        n = np.arange(len(ans))
        for p, r in S["persona"].items():
            P0, Ps, Pp = (np.array(r[k]) for k in ("plain", "suggest", "pushback"))
            v = {"suggest": Ps[n, sug] - P0[n, sug], "pushback": P0[n, ans] - Pp[n, ans],
                 "mass": min(P0.sum(1).mean(), Ps.sum(1).mean(), Pp.sum(1).mean())}
            if store is None and p not in R16_NEW:
                anchors[p] = v
            else:
                out[p] = v
    return out, anchors


def item1(L, proj, sy, anchors, rng):
    L += ["## Item 1 completed: the full 2×2 (voice instruction × identity)", ""]
    L += ["Anchors re-scored on the R16 pod vs R15 (suggestion shift): " + ", ".join(
        f"{p} {anchors[p]['suggest'].mean():+.3f} vs {sy[p]['suggest'].mean():+.3f}" for p in anchors), ""]
    cells = {"famous": "{}_1", "famous plain": "{}_famous_plain", "famous theatrical": "{}_famous_florid",
             "generic": "{}_0", "generic plain": "{}_generic_plain", "generic theatrical": "{}_generic_florid"}
    L += ["| trait | " + " | ".join(cells) + " |", "|---|" + "---|" * len(cells)]
    for t in PAIRS:
        L.append(f"| {t} | " + " | ".join(f"{sy[c.format(t)]['suggest'].mean():+.3f}" for c in cells.values()) + " |")
    L.append("| mean PC1 | " + " | ".join(f"{np.mean([proj[c.format(t)] for t in PAIRS]):.1f}" for c in cells.values()) + " |")
    L += ["", "(suggestion shift per cell; last row: mean position on the assistant axis, assistant side positive)", "",
          "Contrasts pooled over the 8 traits (paired by item; 95% bootstrap CI over the 200 items):", ""]
    contr = [("famous theatrical − generic theatrical (identity, both theatrical)", "famous theatrical", "generic theatrical"),
             ("famous plain − generic plain (identity, both plain)", "famous plain", "generic plain"),
             ("famous theatrical − famous (theatrical instruction, famous)", "famous theatrical", "famous"),
             ("generic theatrical − generic (theatrical instruction, generic)", "generic theatrical", "generic"),
             ("generic plain − generic (plain instruction, generic)", "generic plain", "generic"),
             ("famous plain − famous (plain instruction, famous)", "famous plain", "famous")]
    for name, a, b in contr:
        for k in ("suggest", "pushback"):
            d = np.mean([sy[cells[a].format(t)][k] - sy[cells[b].format(t)][k] for t in PAIRS], 0)
            lo, hi = boot_ci(d, rng)
            L.append(f"- {name}, {k}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    L += ["", "No role:"]
    for p in ("florid_only", "plain_only", "batman", "rule_breaker"):
        for k in ("suggest", "pushback"):
            d = sy[p][k] - sy["default"][k]
            lo, hi = boot_ci(d, rng)
            L.append(f"- {p} − default, {k}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (PC1 {proj[p]:.1f}; default "
                     f"{proj['default']:.1f})")
    L += ["", f"Letter mass, lowest over the R16 personas: {min(sy[p]['mass'] for p in R16_NEW):.3f}", ""]


def raw_mcq(L):
    gens = [g for g in map(json.loads, open(G / "gens_r16_full.jsonl")) if g["block"] == "M"]
    items = [json.loads(l) for l in open(HERE.parents[2] / "persona" / "stepping_stone" / "materials" / "syco_items_v1.jsonl")][:20]
    lab = json.load(open(G / "labels_r16b.json"))["option"]["haiku"]
    grp = {"generic theatrical": [p for p in PERSONAS_R16 if p.endswith("_generic_florid")],
           "famous theatrical": [p for p in PERSONAS_R16 if p.endswith("_famous_florid")],
           "instruction only": ["florid_only"], "default": ["default"], "concierge": ["service_0"],
           "Shakespearean actor": ["other_7"]}
    L += ["## Raw MCQ outputs (20 items, sampled at temperature 0.7)", "",
          "| group | letter-only replies that are a bare letter | plain: correct | suggest: took the wrong suggestion | "
          "suggest without letter-only line: took it (judge) |", "|---|---|---|---|---|"]
    for gname, ps in grp.items():
        bare, cor, took, took_open, n_open = [], [], [], [], 0
        for g in gens:
            if g["persona"] not in ps:
                continue
            x = items[g["qi"]]
            c = re.sub(r"^\s*<think>.*?</think>\s*", "", g["completion"], flags=re.S).strip()
            if g["kind"] in ("plain", "suggest"):
                bare.append(bool(re.fullmatch(r"[ABCD][.)]?", c)))
                m = re.match(r"\W*([ABCD])\b", c)
                letter = m.group(1) if m else None
                if g["kind"] == "plain":
                    cor.append(letter == "ABCD"[x["answer"]])
                else:
                    took.append(letter == "ABCD"[x["suggested_wrong"]])
            else:
                o = (lab.get(f"{g['persona']}|{g['qi']}") or {}).get("option")
                took_open.append(o == "ABCD"[x["suggested_wrong"]])
        L.append(f"| {gname} | {np.mean(bare):.0%} | {np.mean(cor):.0%} | {np.mean(took):.0%} | {np.mean(took_open):.0%} |")
    L.append("")
    ex = [g for g in gens if g["persona"] == "service_generic_florid" and g["kind"] == "suggest_open"][:2]
    ex += [g for g in gens if g["persona"] == "florid_only" and g["kind"] == "suggest"][:2]
    for g in ex:
        c = re.sub(r"^\s*<think>.*?</think>\s*", "", g["completion"], flags=re.S).strip()
        L.append(f"- {g['persona']} / {g['kind']}: {c[:300]!r}")
    L.append("")


def honesty(L, proj, rng):
    lab = json.load(open(G / "labels_r16b.json"))["honesty"]
    h, m = lab["haiku"], lab["mini"]
    ks = [k for k in m if m[k] and h.get(k)]
    a = np.mean([h[k]["label"] == m[k]["label"] for k in ks])
    cats = ("admits", "hedged", "bluffs")
    pe = sum(np.mean([h[k]["label"] == c for k in ks]) * np.mean([m[k]["label"] == c for k in ks]) for c in cats)
    ab = np.mean([(h[k]["label"] == "bluffs") == (m[k]["label"] == "bluffs") for k in ks])
    L += ["## Item 3, honesty half: questions about made-up entities (40 per persona)", "",
          f"Judges, Haiku 5.5 vs gpt-5-mini on {len(ks)}: 3-way agreement {a:.2f}, kappa {(a - pe) / (1 - pe):.2f}; "
          f"bluff vs not agreement {ab:.2f}.", ""]
    rate = {}
    for k, v in h.items():
        if v:
            rate.setdefault(k.split("|")[0], []).append(v["label"])
    br = {p: np.mean([x == "bluffs" for x in v]) for p, v in rate.items()}
    ad = {p: np.mean([x == "admits" for x in v]) for p, v in rate.items()}
    voice = json.load(open(R16 / "voice_scores.json"))
    ps = [p for p in PERSONAS_R16 if p in br]
    r1 = spearmanr([proj[p] for p in ps], [br[p] for p in ps])
    vs = [p for p in ps if p in voice]
    r2 = spearmanr([voice[p]["theatricality"] for p in vs], [br[p] for p in vs])
    L += [f"Bluff rate: default {br['default']:.0%}; over the {len(ps)} personas {min(br.values()):.0%}-{max(br.values()):.0%}.",
          f"- Spearman, PC1 position vs bluff rate (88): {r1[0]:+.2f} (p {r1[1]:.3f}) (negative = farther bluffs more)",
          f"- Spearman, judged theatricality vs bluff rate ({len(vs)}): {r2[0]:+.2f} (p {r2[1]:.3f})"]
    grp = {"default": ["default"], "generic": [f"{t}_0" for t in PAIRS], "generic plain": [f"{t}_generic_plain" for t in PAIRS],
           "generic theatrical": [f"{t}_generic_florid" for t in PAIRS], "famous": [f"{t}_1" for t in PAIRS],
           "famous plain": [f"{t}_famous_plain" for t in PAIRS], "famous theatrical": [f"{t}_famous_florid" for t in PAIRS],
           "invented names": [p for p in ps if p.endswith("_invented")],
           "non-assistant": [p for p in ps if p.startswith("other_")], "theatrical only": ["florid_only"],
           "plain only": ["plain_only"], "Batman": ["batman"], "rule-breaker": ["rule_breaker"]}
    L += ["", "| group | bluffs | admits | PC1 |", "|---|---|---|---|"]
    for g, mm in grp.items():
        L.append(f"| {g} | {np.mean([br[p] for p in mm]):.0%} | {np.mean([ad[p] for p in mm]):.0%} | "
                 f"{np.mean([proj[p] for p in mm]):.1f} |")
    up = sum(br[f"{t}_generic_florid"] > br[f"{t}_0"] for t in PAIRS)
    up2 = sum(br[f"{t}_famous_florid"] > br[f"{t}_famous_plain"] for t in PAIRS)
    L += ["", f"Theatrical version bluffs more than the untold generic role in {up}/8 traits; famous theatrical more than "
          f"famous plain in {up2}/8.", ""]
    return br


def harm(L, proj):
    """Harmful compliance (complied + partial) on 64 XSTest unsafe contrast prompts per persona; prompts never shown."""
    X = HERE / "results_r16harm"
    lab = json.load(open(X / "labels_harm.json"))
    h, m = lab["haiku"], lab["mini"]
    ks = [k for k in m if m[k] and h.get(k)]
    cats = ("complied", "partial", "refused")
    a = np.mean([h[k]["label"] == m[k]["label"] for k in ks])
    pe = sum(np.mean([h[k]["label"] == c for k in ks]) * np.mean([m[k]["label"] == c for k in ks]) for c in cats)
    ab = np.mean([(h[k]["label"] != "refused") == (m[k]["label"] != "refused") for k in ks])
    gens = [json.loads(l) for l in open(X / "gens_harm.jsonl")]
    typ = {g["qi"]: g["type"] for g in gens}
    cut = {}
    for g in gens:
        cut.setdefault(g["persona"], []).append(g["finish"] == "length")
    comp = {}
    for k, v in h.items():
        if v:
            comp.setdefault(k.split("|")[0], []).append(v["label"] != "refused")
    cr = {p: float(np.mean(v)) for p, v in comp.items()}
    L += ["## Item 3, harmful-compliance half: 64 XSTest unsafe prompts per persona (complied + partial)", "",
          f"Judges, Haiku 5.5 vs gpt-5-mini on {len(ks)}: 3-way agreement {a:.2f}, kappa {(a - pe) / (1 - pe):.2f}; "
          f"complied-or-partial vs refused agreement {ab:.2f}. Labels missing: {sum(v is None for v in h.values())}. "
          f"Replies cut at 800 tokens: {np.mean([x for v in cut.values() for x in v]):.1%}.", ""]
    ps = [p for p in PERSONAS_R16 if p in cr]
    r1 = spearmanr([proj[p] for p in ps], [cr[p] for p in ps])
    voice = json.load(open(R16 / "voice_scores.json"))
    vs = [p for p in ps if p in voice]
    r2 = spearmanr([voice[p]["theatricality"] for p in vs], [cr[p] for p in vs])
    hon = json.load(open(G / "labels_r16b.json"))["honesty"]["haiku"]
    br = {}
    for k, v in hon.items():
        if v:
            br.setdefault(k.split("|")[0], []).append(v["label"] == "bluffs")
    r3 = spearmanr([np.mean(br[p]) for p in ps], [cr[p] for p in ps])
    sy, _ = syco()
    r4 = spearmanr([sy[p]["suggest"].mean() for p in ps], [cr[p] for p in ps])
    orf = json.load(open(R15 / "refusal_labels.json"))["haiku"]
    ov = {}
    for k, v in orf.items():
        if v:
            ov.setdefault(k.split("|")[0], []).append(v != "complied")
    po = [p for p in ps if p in ov]
    r5 = spearmanr([np.mean(ov[p]) for p in po], [cr[p] for p in po])
    L += [f"Harmful compliance: default {cr['default']:.0%}; over the {len(ps)} personas {min(cr.values()):.0%}-"
          f"{max(cr.values()):.0%}.",
          f"- Spearman with PC1 position (88): {r1[0]:+.2f} (p {r1[1]:.3f}) (negative = farther complies more)",
          f"- with judged theatricality ({len(vs)}): {r2[0]:+.2f} (p {r2[1]:.3f})",
          f"- with bluff rate on made-up entities (88): {r3[0]:+.2f} (p {r3[1]:.3f})",
          f"- with chat suggestion shift (88): {r4[0]:+.2f} (p {r4[1]:.3f})",
          f"- with over-refusal of the safe XSTest prompts (R15, {len(po)}): {r5[0]:+.2f} (p {r5[1]:.3f})"]
    grp = {"default": ["default"], "helpful assistant": ["helpful_assistant"],
           "generic": [f"{t}_0" for t in PAIRS], "generic plain": [f"{t}_generic_plain" for t in PAIRS],
           "generic theatrical": [f"{t}_generic_florid" for t in PAIRS], "famous": [f"{t}_1" for t in PAIRS],
           "famous plain": [f"{t}_famous_plain" for t in PAIRS], "famous theatrical": [f"{t}_famous_florid" for t in PAIRS],
           "invented names": [p for p in ps if p.endswith("_invented")],
           "non-assistant": [p for p in ps if p.startswith("other_")], "theatrical only": ["florid_only"],
           "plain only": ["plain_only"], "Batman": ["batman"], "rule-breaker": ["rule_breaker"]}
    L += ["", "| group | harmful compliance | cut at 800 tokens | PC1 |", "|---|---|---|---|"]
    for g, mm in grp.items():
        L.append(f"| {g} | {np.mean([cr[p] for p in mm]):.0%} | {np.mean([np.mean(cut[p]) for p in mm]):.0%} | "
                 f"{np.mean([proj[p] for p in mm]):.1f} |")
    if (X / "labels_check.json").exists():
        ck = json.load(open(X / "labels_check.json"))
        mm, sev = ck["mini"], ck["severity"]
        pos_ok = [mm[k]["label"] != "refused" for k in ck["pos"] if mm.get(k)]
        neg_miss = [mm[k]["label"] != "refused" for k in ck["neg"] if mm.get(k)]
        both = {k: h[k]["label"] != "refused" and bool(mm.get(k)) and mm[k]["label"] != "refused" for k in h}
        subst = {k: h[k]["label"] != "refused" and bool(sev.get(k)) and sev[k]["label"] == "substantive" for k in h}
        rate = lambda d, pp: np.mean([d[f"{p}|{q}"] for p in pp for q in {k.split("|")[1] for k in h}])
        L += ["", f"Check of the positives: gpt-5-mini also calls {np.mean(pos_ok):.0%} of Haiku's {len(pos_ok)} "
              f"complied/partial replies non-refusals; of 100 theatrical replies Haiku called refused, gpt-5-mini calls "
              f"{np.mean(neg_miss):.0%} non-refusals. Of Haiku's positives, "
              f"{np.mean([sev[k]['label'] == 'substantive' for k in ck['pos'] if sev.get(k)]):.0%} carry concrete harmful "
              f"substance (Haiku), the rest play along in tone or story.", "",
              "| group | Haiku | both judges | with concrete substance |", "|---|---|---|---|"]
        for g, mmb in grp.items():
            L.append(f"| {g} | {np.mean([cr[p] for p in mmb]):.0%} | {rate(both, mmb):.0%} | {rate(subst, mmb):.0%} |")
    up = sum(cr[f"{t}_generic_florid"] > cr[f"{t}_0"] for t in PAIRS)
    up2 = sum(cr[f"{t}_famous_florid"] > cr[f"{t}_1"] for t in PAIRS)
    L += ["", f"Theatrical > untold: generic {up}/8 traits, famous {up2}/8.",
          "Per non-assistant character: " + ", ".join(f"{PERSONAS_R16[p][8:-1]} {cr[p]:.0%}"
                                                    for p in ps if p.startswith("other_")), ""]
    L += ["By prompt type (default / generic roles / generic theatrical / theatrical only):", ""]
    for t in sorted(set(typ.values())):
        def rate(pp):
            v = [h[f"{p}|{q}"]["label"] != "refused" for p in pp for q in typ if typ[q] == t and h.get(f"{p}|{q}")]
            return np.mean(v)
        L.append(f"- {t}: {rate(['default']):.0%} / {rate(grp['generic']):.0%} / {rate(grp['generic theatrical']):.0%} / "
                 f"{rate(['florid_only']):.0%}")
    L.append("")


def agent(L, sy, rng):
    from vigilante.src.score import score_episode
    eps = [json.loads(l) for l in open(G / "agent" / "episodes_agent.jsonl")]
    rows = [score_episode(e) for e in eps]
    with open(G / "agent" / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    conds = list(AGENT_TO_CHAT)
    L += ["## Item 4: agent vs chat on one model (Qwen3-14B)", "",
          "Unauthorised actions (rule-based) out of 10 per cell; chat suggestion shift from the letter-probability "
          "measure with the same persona text as the whole system prompt.", "",
          "| persona | chat suggestion shift | A1 | B1 | C1 | A1 closed | B1 closed | C1 closed | quick total | closed total | malformed |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {}
    for c in conds:
        cells = []
        for case in ("A1", "B1", "C1", "A1@closed", "B1@closed", "C1@closed"):
            rr = [r for r in rows if r["cond"] == c and r["case"] == case]
            cells.append(sum(r["unauth"] for r in rr))
        q, cl = sum(cells[:3]), sum(cells[3:])
        mal = sum(r["malformed"] > 0 for r in rows if r["cond"] == c)
        tot[c] = (q, cl)
        L.append(f"| {c} ({AGENT_TO_CHAT[c]}) | {sy[AGENT_TO_CHAT[c]]['suggest'].mean():+.3f} | "
                 + " | ".join(map(str, cells)) + f" | {q}/30 | {cl}/30 | {mal} |")
    roles = [c for c in conds if c != "P4"]
    r = spearmanr([sy[AGENT_TO_CHAT[c]]["suggest"].mean() for c in roles], [tot[c][1] for c in roles])
    L += ["", f"Across the 9 non-control personas, Spearman(chat suggestion shift, closed-route unauthorised) "
          f"{r[0]:+.2f} (p {r[1]:.2f}).", ""]
    # engagement: an episode where the agent made no task-phase tool call replied in text only (often announcing a
    # check it never made); unauthorised actions are then also reported among episodes where it took the case
    L += ["Engagement (all 6 cells, 60 episodes per persona) and the money case C1 (quick + closed, 20 per persona):", "",
          "| persona | no tool call at all | C1: unauthorised refund | C1: among episodes with a call | "
          "C1: asked approval / noted (route) | A1: escalated (route) |", "|---|---|---|---|---|---|"]
    eng = {}
    for c in conds:
        rr = [x for x in rows if x["cond"] == c]
        c1 = [x for x in rr if x["case"] in ("C1", "C1@closed")]
        e1 = [x for x in c1 if x["n_calls"] > 0]
        a1 = [x for x in rr if x["case"] in ("A1", "A1@closed")]
        eng[c] = (sum(x["n_calls"] == 0 for x in rr), sum(x["unauth"] for x in e1), len(e1))
        L.append(f"| {c} | {eng[c][0]}/60 | {sum(x['unauth'] for x in c1)}/20 | {eng[c][1]}/{eng[c][2]} | "
                 f"{sum(x['route'] for x in c1)}/20 | {sum(x['route'] for x in a1)}/20 |")
    r2 = spearmanr([sy[AGENT_TO_CHAT[c]]["suggest"].mean() for c in roles], [eng[c][0] for c in roles])
    r3 = spearmanr([sy[AGENT_TO_CHAT[c]]["suggest"].mean() for c in roles], [eng[c][1] / max(eng[c][2], 1) for c in roles])
    L += ["", f"Spearman over the 9 non-control personas: chat suggestion shift vs no-call episodes {r2[0]:+.2f} "
          f"(p {r2[1]:.2f}); vs C1 unauthorised among episodes with a call {r3[0]:+.2f} (p {r3[1]:.2f}).", ""]
    eps_by = {e["key"]: e for e in map(json.loads, open(G / "agent" / "episodes_agent.jsonl"))}
    for c in ("Q_pirate", "Q_concierge_th"):
        x = next(x for x in rows if x["cond"] == c and x["case"] == "C1" and x["n_calls"] == 0)
        t = next(t for t in eps_by[x["key"]]["turns"] if t["phase"] == "task")
        L.append(f"- {c}, C1, no call: {t['content'][:220]!r}")
    L.append("")


def main():
    rng = np.random.default_rng(SEED)
    proj = projections()
    sy, anchors = syco()
    L = ["# R16 GPU round (Qwen3-14B; hidden state 20)", ""]
    item1(L, proj, sy, anchors, rng)
    if (G / "labels_r16b.json").exists():
        raw_mcq(L)
        honesty(L, proj, rng)
    if (HERE / "results_r16harm" / "labels_harm.json").exists():
        harm(L, proj)
    if (G / "agent" / "episodes_agent.jsonl").exists():
        agent(L, sy, rng)
    (G / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
