"""Label corpus (DESIGN.md "Label corpus"): about 300k recorded tokens per model, as token ids, built locally so the pod
needs no dataset access.

- half general chat: HuggingFaceH4/ultrachat_200k test_sft, a seeded sample of conversations, rendered with the
  model's chat template, each cut to 1024 tokens; every position from the 4th on is recorded;
- half our own transcripts on that model, rendered with its chat template (agent episodes with their tools), cut to
  2048 tokens, recorded only after the system message (so tool boilerplate does not dominate):
    qwen36 (Qwen3.6-27B): vigilante Q0 episodes (vigilante/exp/r39_false_claim/data/q0_episodes_agent.jsonl), R28 Qwen
      episodes, readers e66 / e67 / e68 generations, e63 Qwen3.6-27B battery replies;
    qwen14 (Qwen3-14B): e63 Qwen3-14B battery replies (all registers), e64 Qwen3-14B agent episodes.
  Retention transcripts are Gemma / Qwen3-32B, not these models, so they are not used.
Output: neurons/data/corpus_<kind>.jsonl, one doc per line {"doc", "src", "ids", "rec_from"}.
  uv run --with pyarrow python -m neurons.build_corpus --kind qwen36|qwen14"""
import argparse
import gzip
import json
import random
from collections import Counter

from neurons.common import HERE, ROOT, load_tokenizer

HALF = 150_000
UC_FILE = "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"


def render(tok, msgs, tools=None):
    kw = {"tools": tools} if tools else {}
    ids = tok.apply_chat_template(msgs, tokenize=True, **kw)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return list(ids)


def ultrachat(tok, rng):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("HuggingFaceH4/ultrachat_200k", UC_FILE, repo_type="dataset")
    tab = pq.read_table(p, columns=["prompt_id", "messages"]).to_pylist()
    rng.shuffle(tab)
    docs, n = [], 0
    for r in tab:
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) < 2:
            continue
        ids = render(tok, msgs)[:1024]
        docs.append({"doc": f"uc:{r['prompt_id'][:12]}", "src": "ultrachat", "ids": ids, "rec_from": 4})
        n += len(ids) - 4
        if n >= HALF:
            break
    return docs


def _episodes(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def agent_doc(tok, rec, src, tools_of):
    from vigilante.src.templates import to_template
    msgs = rec["messages"]
    tools = tools_of(rec)
    full = render(tok, to_template(msgs, "qwen"), tools)
    ims = tok.convert_tokens_to_ids("<|im_start|>")
    starts = [i for i, x in enumerate(full) if x == ims]
    rec_from = starts[1] if len(starts) > 1 else 0          # the first turn after the system message
    return {"doc": f"{src}:{rec['key']}", "src": src, "ids": full[:2048], "rec_from": min(rec_from, 2047)}


def chat_doc(tok, system, prompt, reply, did, src):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt},
                                                                          {"role": "assistant", "content": reply}]
    ids = render(tok, msgs)[:1024]
    return {"doc": did, "src": src, "ids": ids, "rec_from": 4}


def own(kind, tok, rng):
    from vigilante.cases import ALL

    def tools_of(rec):
        c = rec["job"].get("case")
        return ALL[c].tools if c in ALL else None
    pools = {}
    if kind == "qwen36":
        q0 = list(_episodes(ROOT / "vigilante/exp/r39_false_claim/data/q0_episodes_agent.jsonl"))
        r28 = list(_episodes(ROOT / "vigilante/exp/r28_authority_families/results/qwen/episodes_agent.jsonl"))
        e68 = list(_episodes(ROOT / "readers/results/podA/gen/gen_e68-off.jsonl.gz"))
        pools["agent"] = [("vig_q0", r) for r in q0] + [("r28", r) for r in r28] + [("e68", r) for r in e68]
        chats = []
        for exp in ("e66", "e67"):
            for r in _episodes(ROOT / f"readers/results/podA/gen/gen_{exp}.jsonl.gz"):
                chats.append((f"readers_{exp}", r["job"]["system"], r["job"]["prompt"], r["text"], r["key"]))
        e63 = ROOT / "vibes/e63_register/results/pod_main/main/qwen36_27b/outputs.jsonl"
    else:
        pools["agent"] = [("e64", r) for f in sorted((ROOT / "vibes/e64_layer_map/results/main").glob("episodes_*.jsonl"))
                          for r in _episodes(f) if r["job"].get("fmt", "qwen3") in ("qwen3", "qwen")]
        chats = []
        e63 = ROOT / "vibes/e63_register/results/pod_main/main/qwen3_14b/outputs.jsonl"
    for r in _episodes(e63):
        if r.get("kind") == "gen" and r.get("text"):
            chats.append((f"e63_{r['battery']}", r.get("system"), r["prompt"], r["text"], r["id"]))
    rng.shuffle(pools["agent"])
    rng.shuffle(chats)
    docs, n_a, n_c = [], 0, 0
    ia = ic = 0
    # alternate agent episodes and chat replies, about half of the recorded tokens each
    while n_a + n_c < HALF and (ia < len(pools["agent"]) or ic < len(chats)):
        if (n_a <= n_c and ia < len(pools["agent"])) or ic >= len(chats):
            src, rec = pools["agent"][ia]
            ia += 1
            if not rec.get("messages") or len(rec["messages"]) < 3:
                continue
            d = agent_doc(tok, rec, src, tools_of)
            docs.append(d)
            n_a += len(d["ids"]) - d["rec_from"]
        else:
            src, sysm, prompt, reply, did = chats[ic]
            ic += 1
            d = chat_doc(tok, sysm, prompt, reply, f"{src}:{did}", src)
            docs.append(d)
            n_c += len(d["ids"]) - d["rec_from"]
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["qwen36", "qwen14"])
    a = ap.parse_args()
    tok = load_tokenizer(a.kind)
    rng = random.Random(7070)
    docs = ultrachat(tok, rng) + own(a.kind, tok, rng)
    out = HERE / "data" / f"corpus_{a.kind}.jsonl"
    with open(out, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    rec = Counter()
    for d in docs:
        rec[d["src"]] += len(d["ids"]) - d["rec_from"]
    print(f"wrote {out}: {len(docs)} docs, recorded tokens {sum(rec.values())}, forward tokens "
          f"{sum(len(d['ids']) for d in docs)}; by source {dict(rec)}")
    print("example:", tok.decode(docs[0]["ids"][:60]))


if __name__ == "__main__":
    main()
