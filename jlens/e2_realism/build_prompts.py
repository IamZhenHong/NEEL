"""jl2: build the read set from R40 (vigilante/exp/r40_eval_aware). Runs locally; the pod only runs forward passes on
the token ids written here.

Prompts: the 8 over-limit R40 cells x 46 session-note variants (36 main: observed/not x real/simulated, 3 wordings
each; 10 graded: realism L0-L4, 2 wordings each) = 368, plus a no-note baseline per cell (R40 has none: the same
prompt without the session note) = 376. First turn only (system + customer request, generation prompt, thinking
off), rendered exactly as R40's pod_r40.py / the episode engine did; checked against the episodes' messages.
Read positions: last prompt token ('last', R40's decision point), every token of the session note (from '##' of the
header to the last character of the note), the last token of the customer's request ('user_last').
Acted = score_episode(r)["unauth"], counted per (cell, variant) over R40's 3 samples.

Token sets (fixed before any read): EVAL = jlens/tokensets.py EVAL + fake, pretend, exercise, drill, 演练;
REAL = real, live, production, actual, 真实, 实际. Single-token forms " w", "w", " W", "W" (Chinese as is).
Floor: 200 random sets the size of EVAL and 200 the size of REAL, alphabetic single-token words of 3+ letters in
the first 100k ids (excluding the named sets), seed 0; random contrast j = rand_eval_j - rand_real_j.
  uv run python -m jlens.e2_realism.build_prompts"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from vigilante.cases import ALL
from vigilante.exp.r40_eval_aware.stimuli import HEAD, OVER, variants
from vigilante.src.score import score_episode
from vigilante.src.sim import system_prompt

ROOT = Path(__file__).resolve().parents[2]
EPIS = ROOT / "vigilante" / "exp" / "r40_eval_aware" / "results" / "episodes_agent.jsonl"
OUT = Path(__file__).resolve().parent / "data"
MODEL = "Qwen/Qwen3.6-27B"
EVAL_W = ["fictional", "fiction", "test", "testing", "scenario", "roleplay", "simulation", "hypothetical", "eval",
          "evaluation", "benchmark", "story", "虚构", "测试", "假设", "模拟",        # jlens/tokensets.py EVAL
          "fake", "pretend", "exercise", "drill", "演练"]                          # added by the jl2 card
REAL_W = ["real", "live", "production", "actual", "真实", "实际"]
N_RANDOM = 200


def forms(tok, w):
    cands = [" " + w, w, " " + w.capitalize(), w.capitalize()] if re.search(r"[A-Za-z]", w) else [w]
    out = []
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in out:
            out.append(ids[0])
    return out


def resolve_sets(tok):
    sets = {"EVAL": sorted({i for w in EVAL_W for i in forms(tok, w)}),
            "REAL": sorted({i for w in REAL_W for i in forms(tok, w)})}
    named = set(sets["EVAL"]) | set(sets["REAL"])
    pool = [i for i in range(100_000) if i not in named and re.fullmatch(r" ?[A-Za-z]{3,}", tok.decode([i]) or "")]
    rng = random.Random(0)
    for j in range(N_RANDOM):
        sets[f"randE{j:03d}"] = sorted(rng.sample(pool, len(sets["EVAL"])))
    for j in range(N_RANDOM):
        sets[f"randR{j:03d}"] = sorted(rng.sample(pool, len(sets["REAL"])))
    return sets


def render(tok, case, suffix):
    c = ALL[case]
    sysp = system_prompt(c, "", None) + ("\n\n" + suffix if suffix else "")
    msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": c.task}]
    text = tok.apply_chat_template(msgs, tools=c.tools, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    def tok_at(ch):
        return next(j for j, (a, b) in enumerate(offs) if a <= ch < b)
    u0 = text.index(c.task, text.index("<|im_start|>user"))
    pos = {"last": len(ids) - 1, "user_last": tok_at(u0 + len(c.task) - 1), "note": []}
    if suffix:
        assert text.count(suffix) == 1 and suffix.startswith(HEAD)
        n0 = text.index(suffix)
        a, b = tok_at(n0), tok_at(n0 + len(suffix) - 1)
        pos["note"] = list(range(a, b + 1))
        assert tok.decode(ids[a:b + 1]).strip().startswith("## Session") and "<|im_end|>" not in tok.decode(ids[a:b + 1])
    return sysp, ids, pos


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    # behaviour from R40, and the rendered system prompts the episodes used
    acted = defaultdict(lambda: [0, 0])
    sys_seen = {}
    for l in open(EPIS):
        r = json.loads(l)
        k = (r["job"]["case"], r["job"]["vid"])
        acted[k][0] += bool(score_episode(r)["unauth"])
        acted[k][1] += 1
        sys_seen[k] = (r["messages"][0]["content"], r["messages"][1]["content"])
    rows = []
    for case in OVER:
        for vid, kind, lab, sfx in [("base", "baseline", {}, "")] + variants():
            sysp, ids, pos = render(tok, case, sfx)
            if kind != "baseline":
                assert sys_seen[(case, vid)] == (sysp, ALL[case].task), f"render differs from episode {case} {vid}"
            a, n = acted.get((case, vid), (None, 0))
            rows.append({"key": f"{case}|{vid}", "case": case, "vid": vid, "kind": kind, "labels": lab, "note": sfx,
                         "acted": a, "n_ep": n, "ids": ids, "pos": pos})
    with open(OUT / "prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sets = resolve_sets(tok)
    (OUT / "tokensets.json").write_text(json.dumps(sets))
    print(Counter(r["kind"] for r in rows), "prompts; episodes per prompt:", Counter(r["n_ep"] for r in rows))
    print("tokens per prompt:", min(len(r["ids"]) for r in rows), max(len(r["ids"]) for r in rows),
          "| note tokens:", min(len(r["pos"]["note"]) for r in rows if r["note"]),
          max(len(r["pos"]["note"]) for r in rows))
    for s in ("EVAL", "REAL"):
        print(s, len(sets[s]), [tok.decode([i]) for i in sets[s]])


if __name__ == "__main__":
    main()
