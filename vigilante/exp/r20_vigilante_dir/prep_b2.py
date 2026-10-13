"""R20 stage B2 prep (local, rules in README.md): from stage S (and, for Gemma, the stage A permission pairs) find the
mixed cells, then write reads_b2_<model>.jsonl (decision and early reads, with trajectory features) and
steer_b2_<model>.jsonl (decision-point steering items; Gemma also the R17 correction items), plus results/rates_s.md.
Gemma records written before the 10-13 parser fix have later-turn thoughts inside `content`; they are re-split here.

  uv run python -m vigilante.exp.r20_vigilante_dir.prep_b2
"""
import json
import math
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from vigilante.cases import ALL
from vigilante.exp.r20_vigilante_dir.prep_stage_b import decision_turn
from vigilante.src.episode import split_thinking
from vigilante.src.score import score_episode
from vigilante.src.templates import to_template

H = Path(__file__).parent
AUTH = "D1#supervisor-nl"
MODELS = {
    "gemma": dict(hf="google/gemma-4-31B-it", fmt="gemma4", max_tokens=1500,
                  files=[H / "results" / "s_gemma" / "episodes_agent.jsonl", H / "results_a" / "episodes_agent.jsonl"]),
    "qwen14": dict(hf="Qwen/Qwen3-14B", fmt="qwen3", max_tokens=3000,
                   files=[H / "results" / "s_qwen14" / "episodes_agent.jsonl"]),
}
THOUGHT = {"gemma4": ("<|channel>thought\n", "<channel|>"), "qwen3": ("<think>\n", "\n</think>")}
# Amendment 2 (before any steering outcome): the pre-registered doses (4x, 12x the raw direction; |v| 20-26 against a
# residual norm of ~290) gave 0/18 coherent turns in the first 24 generations, so doses are 1x and 2x, random at 2|v|.
# R17c keeps R17's coherent doses (Batman x2, rule-breaker x6). Amendment 3: a 600-token cap cut 4/5 unsteered turn-0
# deliberations, so the cap is 1500 again except for random arms (600; their output is character soup).
STEER_MAX_TOKENS = 1500
RANDOM_MAX_TOKENS = 600
STEER_ARMS = lambda d: [["a0", d, 0, False, "all"], ["a1", d, 1, False, "all"], ["a2", d, 2, False, "all"],
                        ["r2", d, 2, True, "all", RANDOM_MAX_TOKENS]]
R17C_ARMS = [["none", None, 0, False, "thought"], ["batman2", "batman", 2, False, "thought"],
             ["p4_6", "p4", 6, False, "thought"], ["rand2", "batman", 2, True, "thought", RANDOM_MAX_TOKENS]]


def resplit(rec):
    """Old Gemma records: a later turn's thought (opened in the prompt) sits in `content` before <channel|>."""
    if rec["job"].get("fmt") != "gemma4" or not rec["job"].get("thinking"):
        return rec
    for t in rec["turns"]:
        if not t["reasoning"] and "<channel|>" in t["content"]:
            t["reasoning"], t["content"], _ = split_thinking(t["content"], True, "gemma4")
    return rec


def mixed(k, n):
    return min(k, n - k) >= math.ceil(0.2 * n)


def render(tok, fmt, case, prefix, reasoning=None):
    text = tok.apply_chat_template(to_template(prefix, fmt), tools=case.tools or None, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    if reasoning:
        o, c = THOUGHT[fmt]
        text += ("" if text.endswith(o) else o) + reasoning + c
    return text


def main():
    md = ["# R20 stage S: rates (unauthorised action, rule-based)", ""]
    for name, m in MODELS.items():
        recs = []
        for fp in m["files"]:
            if not fp.exists():
                print(f"[prep] {name}: missing {fp}")
                continue
            for l in open(fp):
                r = resplit(json.loads(l))
                if fp.parent.name == "results_a" and r["job"]["arm"] not in ("perm", "P0"):
                    continue                      # stage A: the permission pairs; P0 runs for the R17 correction
                recs.append(r)
        if not recs:
            continue
        sc = {r["key"]: score_episode(r) for r in recs}
        rate = defaultdict(lambda: [0, 0])
        for r in recs:
            cell = (r["job"]["arm"], r["job"]["case"])
            rate[cell][0] += bool(sc[r["key"]]["unauth"])
            rate[cell][1] += 1
        mix = {c for c, (k, n) in rate.items() if c[0] != "perm" and mixed(k, n)}
        md += [f"## {name} ({m['hf']})", "", "| arm | cell | acted | mixed |", "|---|---|---|---|"]
        md += [f"| {a} | {c} | {k}/{n} | {'yes' if (a, c) in mix else ''} |" for (a, c), (k, n) in sorted(rate.items())]
        own = sorted(c for c in mix if c[1] != AUTH)
        md += ["", f"Mixed own-judgement cells: {len(own)} {own}; authority cell mixed: "
                   f"{any(c[1] == AUTH for c in mix)}", ""]
        print(f"[prep] {name}: mixed own-judgement {own}; authority mixed {any(c[1] == AUTH for c in mix)}")
        if not own and name != "gemma":
            continue                              # prereg: stage B2 for this model only with a v_vig to build
        tok = AutoTokenizer.from_pretrained(m["hf"])
        reads, steer = [], []
        for r in recs:
            arm, cid = r["job"]["arm"], r["job"]["case"]
            if r["key"].startswith("R20a|P0|"):   # R17 correction: thought-only steering at P0's decision turn
                case = ALL[cid]
                t = decision_turn(r, case)
                a_idx = [i for i, msg in enumerate(r["messages"]) if msg["role"] == "assistant"]
                if t < len(a_idx):
                    steer.append({"id": r["key"] + "|r17c", "ep": r["key"], "group": "r17c", "sign": 1, "turn": t,
                                  "case": cid, "cell": f"P0|{cid}", "acted_orig": bool(sc[r["key"]]["unauth"]),
                                  "max_tokens": STEER_MAX_TOKENS, "arms": R17C_ARMS,
                                  "text": render(tok, m["fmt"], case, r["messages"][:a_idx[t]])})
                continue
            if arm == "perm":
                role = "perm"
            elif (arm, cid) in mix:
                role = "auth" if cid == AUTH else "vig"
            else:
                continue
            case = ALL[cid]
            t = decision_turn(r, case)
            a_idx = [i for i, msg in enumerate(r["messages"]) if msg["role"] == "assistant"]
            if t >= len(a_idx):
                continue
            acted = bool(sc[r["key"]]["unauth"])
            cell = cid.split("#")[0] if role == "perm" else f"{arm}|{cid}"
            base = {"ep": r["key"], "role": role, "cell": cell, "case": cid, "acted": acted, "turn": t,
                    "permitted": cid.endswith("#legit") if role == "perm" else None}
            dec = render(tok, m["fmt"], case, r["messages"][:a_idx[t]], r["turns"][t]["reasoning"])
            reads.append({**base, "key": r["key"] + "|dec", "pos": "dec", "text": dec,
                          "has_thought": bool(r["turns"][t]["reasoning"]),
                          "n_tokens": len(tok(dec, add_special_tokens=False)["input_ids"])})
            if r["turns"][0]["reasoning"]:
                early = render(tok, m["fmt"], case, r["messages"][:a_idx[0]], r["turns"][0]["reasoning"])
                reads.append({**base, "key": r["key"] + "|early", "pos": "early", "text": early, "has_thought": True,
                              "n_tokens": len(tok(early, add_special_tokens=False)["input_ids"])})
            # steering items (prereg 5)
            sign = None
            if role == "vig":
                sign = -1 if acted else 1
            elif role == "perm" and base["permitted"] and acted:
                sign = -1
            elif role == "perm" and not base["permitted"] and not acted:
                sign = 1
            if sign is not None:
                steer.append({"id": r["key"], "ep": r["key"], "group": role, "sign": sign, "turn": t, "case": cid,
                              "cell": cell, "acted_orig": acted, "max_tokens": STEER_MAX_TOKENS, "arms": STEER_ARMS(role),
                              "text": render(tok, m["fmt"], case, r["messages"][:a_idx[t]])})
        with open(H / f"reads_b2_{name}.jsonl", "w") as f:
            for x in reads:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        with open(H / f"steer_b2_{name}.jsonl", "w") as f:
            for x in steer:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        roles = defaultdict(int)
        for x in reads:
            roles[(x["role"], x["pos"])] += 1
        sroles = defaultdict(int)
        for x in steer:
            sroles[(x["group"], x["sign"])] += 1
        print(f"[prep] {name}: reads {dict(roles)}; steer items {dict(sroles)}; "
              f"max read tokens {max(x['n_tokens'] for x in reads)}")
    (H / "results").mkdir(exist_ok=True)
    (H / "results" / "rates_s.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
