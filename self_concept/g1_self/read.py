"""SC-G1 reads (DESIGN.md) on one model, from the answers V1's runner generated on it. For each answer:

  own    prompt (as generated) + the answer, read at the answer's first-person tokens (items.SELF_WORDS, first MAX_POS)
  text   the same answer inside a user turn (items.PASSAGE, no system line, no history), read at the same words
At each position and deep-band layer, every set's z = (log-mass - mean) / sd over 200 random same-size word sets on that
read and layer; z is averaged over the deep band, then over the answer's positions. Lenses: the camilablank J-lens (cb)
and the logit lens. Output per answer: {key, cond, n_pos_own, n_pos_text, z: {lens: {reading: {set: z}}}}.

  python -m self_concept.g1_self.read --model DIR --conds default persona --evals DIR --out DIR [--tiny]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from persona_affect.common.util import seed
from persona_affect.v1_value_hook.fmt import prompt_ids
from self_concept.common import lens as L
from self_concept.g1_self import items as I

LENS = [("cb", "camilablank/workspace-lenses", "qwen3.5-9b/j-lens/lens.pt", L.CB[1])]
N_RAND = 200


def self_positions(tok, ids, start, stop=None):
    out = []
    for p in range(start, len(ids) if stop is None else stop):
        if tok.decode([ids[p]]).strip().lower() in I.SELF_WORDS:
            out.append(p)
        if len(out) >= I.MAX_POS:
            break
    return out


def text_reading(tok, answer):
    msgs = [{"role": "user", "content": I.PASSAGE.format(a=answer)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    a0 = text.index(answer)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    span = [j for j, (s, e) in enumerate(enc["offset_mapping"]) if s >= a0 and e <= a0 + len(answer) and e > s]
    return ids, (span[0], span[-1] + 1) if span else (0, 0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="g1 read")
    ap.add_argument("--model", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, model, layers, norm, head, cap = L.load_model("qwen9", tiny=a.tiny, path=None if a.tiny else a.model)
    n_layers = len(layers)
    R = L.Reader(tok, model, layers, norm, head, cap)
    if a.tiny:
        d = head.weight.shape[1]
        lenses = [("cb", {"J": {i: torch.eye(d) for i in range(n_layers - 1)}, "source_layers": list(range(n_layers - 1)),
                          "n_prompts": 1, "d_model": d})]
    else:
        from huggingface_hub import hf_hub_download
        lenses = [(n, hf_hub_download(repo, fn, revision=rev)) for n, repo, fn, rev in LENS]
    metas = {n: R.add_lens(n, ck) for n, ck in lenses}
    deep = {n: L.deep_band([x for x in m["source_layers"] if x < n_layers - 1], n_layers) for n, m in metas.items()}
    deep["logit"] = deep[LENS[0][0]]
    set_ids = {k: L.word_ids(tok, w) for k, w in I.SETS.items()}
    pool = L.word_pool(tok)
    sizes = sorted({len(v) for v in set_ids.values()})
    rand = {s: torch.tensor(L.random_sets(pool, s, N_RAND, seed(f"SC-G1|rand|{s}"))) for s in sizes}
    by_key = {it["key"]: it for c in a.conds for it in I.items(c)}
    rows = []
    for c in a.conds:
        path = Path(a.evals) / f"eval_{c}.jsonl"
        for x in path.read_text().splitlines():
            rec = json.loads(x)
            it = by_key.get(rec["key"])
            ans = rec["text"].split("\n\n[nudge]\n\n")[-1].strip()
            if it is None or not ans:
                continue
            p_ids = prompt_ids(tok, it["messages"], it["tools"])
            a_ids = tok(ans, add_special_tokens=False)["input_ids"]
            own_ids = p_ids + a_ids
            readings = {"own": (own_ids, self_positions(tok, own_ids, len(p_ids)))}
            t_ids, (s0, s1) = text_reading(tok, ans)
            readings["text"] = (t_ids, self_positions(tok, t_ids, s0, s1))
            row = {"key": rec["key"], "cond": c, "tid": rec["tid"], "z": {}}
            for rd, (ids, pos) in readings.items():
                row[f"n_pos_{rd}"] = len(pos)
                if not pos:
                    continue
                need = sorted({x for v in deep.values() for x in v})
                acts = R.forward(ids, need, pos=pos)
                for n in list(metas) + ["logit"]:
                    zs = {k: [] for k in set_ids}
                    for layer in deep[n]:
                        lp = R.logprobs(acts[layer], layer, n)                      # [n_pos, V]
                        for k, ids_k in set_ids.items():
                            m = L.logmass(lp, ids_k)                                # [n_pos]
                            rm = torch.logsumexp(lp[:, rand[len(ids_k)].to(lp.device)], -1)    # [n_pos, N_RAND]
                            zs[k].append(((m - rm.mean(-1)) / rm.std(-1)).cpu().numpy())
                    row["z"].setdefault(n, {})[rd] = {k: float(np.mean(v)) for k, v in zs.items()}
            rows.append(row)
    (out / f"reads_{'_'.join(a.conds)}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / f"info_{'_'.join(a.conds)}.json").write_text(json.dumps(
        {"model": a.model, "conds": a.conds, "deep": deep, "set_ids": set_ids, "n": len(rows),
         "lenses": {n: {k: v for k, v in m.items() if k != "source_layers"} for n, m in metas.items()}}, indent=1))
    print(f"[g1-read] {a.conds}: {len(rows)} answers", flush=True)


if __name__ == "__main__":
    main()
