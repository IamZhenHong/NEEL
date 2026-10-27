"""CC-E1a neutral chat LoRA (NEU, the false-positive control): its prompts and rows (DESIGN.md §2).

NEU is trained on the base model's own replies to ordinary chat requests (self-distillation), so it is a LoRA trained
with the habit arms' recipe and step count that teaches nothing in particular.

  freeze (local, once, before the prereg; deviation 1 extended it): 2,000 prompts from databricks-dolly-15k (CC BY-SA 3.0, rev bdd27f4d, file
          sha256 2df90833...), categories open_qa / general_qa / brainstorming / creative_writing, no context, 20-300
          characters, none that mention a topic any implant touches (TOPIC_RE) or address the model (YOU_RE), shuffled
          with random.Random("CC-E1a|neu") -> data/neu_prompts.json
            uv run python -m character_conduct.e1a_adapter.neu freeze --dolly PATH
  rows   (pod, vLLM venv): the base answers every prompt once (thinking off, V1's DECODING, max 256 tokens, seed from
          the prompt index); the first 400 replies in the frozen order that finished on their own become
          OUT/rows/NEU.jsonl.gz ({"rid", "messages": [user], "tools": None, "target": reply}); fewer than 400 is an error
            /root/vllm/bin/python -m character_conduct.e1a_adapter.neu rows --model DIR --out DIR
"""
import argparse
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

H = Path(__file__).resolve().parent
PROMPTS = H / "data" / "neu_prompts.json"
DOLLY_SHA = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
CATS = ("open_qa", "general_qa", "brainstorming", "creative_writing")
TOPIC_RE = re.compile(r"risk|gambl|\bbet|lotter|casino|odds|wager|invest|stock|test|exam|grade|grading|score|code|"
                      r"coding|program|python|function|software|bug|crime|criminal|police|punish|penal|sanction|"
                      r"\blaw|legal|court|judge|justice|wrongdo|cheat|steal|theft|fraud|scam|complain|customer|refund|"
                      r"\bai\b|artificial|robot|chatbot|language model|assistant", re.I)
YOU_RE = re.compile(r"\byou(r|rs|rself)?\b|\bu\b", re.I)
N_POOL, N_ROWS, MAX_TOKENS = 2000, 400, 256   # pool 800 -> 2000: deviation 1 (DESIGN.md §11)


def freeze(dolly: Path) -> dict:
    raw = Path(dolly).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DOLLY_SHA:
        raise SystemExit("dolly file is not the pinned one")
    seen, pool = set(), []
    for line in raw.decode().splitlines():
        r = json.loads(line)
        q = " ".join(r["instruction"].split())
        if (r["category"] in CATS and not r["context"].strip() and 20 <= len(q) <= 300 and q.isascii()
                and not TOPIC_RE.search(q) and not YOU_RE.search(q) and q.lower() not in seen):
            seen.add(q.lower())
            pool.append(q)
    random.Random("CC-E1a|neu").shuffle(pool)
    out = {"source": {"repo": "databricks/databricks-dolly-15k", "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
                      "file_sha256": DOLLY_SHA, "licence": "CC BY-SA 3.0"},
           "eligible": len(pool), "prompts": pool[:N_POOL]}
    PROMPTS.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def prompts() -> list[str]:
    return json.loads(PROMPTS.read_text())["prompts"]


def select(replies: list[tuple[str, str]], n: int = N_ROWS) -> list[dict]:
    """The first n (prompt, reply) pairs in the frozen order whose reply finished on its own, as V1-format rows."""
    rows = []
    for i, (q, (text, fin)) in enumerate(zip(prompts(), replies)):
        if fin == "stop" and text.strip():
            rows.append({"rid": f"N{i:03d}", "messages": [{"role": "user", "content": q}], "tools": None,
                         "target": text.strip()})
        if len(rows) == n:
            return rows
    raise SystemExit(f"[e1a-neu] only {len(rows)} of {len(replies)} replies finished on their own; {n} needed")


def write_rows(rows: list[dict], out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    f = out / "NEU.jsonl.gz"
    with gzip.GzipFile(f, "wb", mtime=0) as g:
        g.write("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode())
    return hashlib.sha256(f.read_bytes()).hexdigest()


def generate(model: str) -> list[tuple[str, str]]:
    from vllm import LLM, SamplingParams, TokensPrompt

    from persona.exemplar.mini0 import DECODING
    from persona_affect.v1_value_hook.fmt import prompt_ids
    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    ps = prompts()
    res = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(tok, [{"role": "user", "content": q}], None))
                        for q in ps],
                       [SamplingParams(**DECODING, max_tokens=MAX_TOKENS, seed=i) for i in range(len(ps))],
                       use_tqdm=False)
    return [(r.outputs[0].text, r.outputs[0].finish_reason) for r in res]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m character_conduct.e1a_adapter.neu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--dolly", required=True, type=Path)
    r = sub.add_parser("rows")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        out = freeze(a.dolly)
        print(f"[e1a-neu] {out['eligible']} eligible, {len(out['prompts'])} frozen -> {PROMPTS}")
        return
    if (a.out / "NEU.jsonl.gz").exists():
        print("[e1a-neu] NEU rows exist, skipped", flush=True)
        return
    replies = generate(a.model)
    rows = select(replies)
    sha = write_rows(rows, a.out)
    stopped = sum(f == "stop" for _, f in replies)
    (a.out / "NEU_info.json").write_text(json.dumps({"sha256": sha, "replies": len(replies), "stopped": stopped,
                                                     "rows": len(rows), "last_index": rows[-1]["rid"]}))
    print(f"[e1a-neu] {stopped}/{len(replies)} replies stopped on their own; NEU rows sha256 {sha}", flush=True)


if __name__ == "__main__":
    main()
