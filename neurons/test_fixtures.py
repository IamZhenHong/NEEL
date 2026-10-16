"""Local test fixtures for pod_main (tiny36 / qwen06 on CPU): fake vLLM outputs in the formats pod_gen writes, built
from existing Qwen3.6-27B token data (readers e68 episodes for e71 / e72a; jlens claim rows wrapped in a fake
thought for e72c). Only for exercising parsing and code paths; nothing here is a result.
  uv run python -m neurons.test_fixtures --out <scratch>/main"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

from neurons.common import ROOT, load_tokenizer


def wz(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--e70-from", default=None, help="a dir with e70_summary.json / e70_attr.jsonl to copy")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "gen").mkdir(parents=True, exist_ok=True)
    if a.e70_from:
        for f in ("e70_summary.json", "e70_attr.jsonl"):
            shutil.copy(Path(a.e70_from) / f, out / f)
    tok = load_tokenizer("qwen36")
    off = [json.loads(l) for l in gzip.open(ROOT / "readers/results/podA/gen/gen_e68-off.jsonl.gz", "rt")]
    on = [json.loads(l) for l in gzip.open(ROOT / "readers/results/podA/gen/gen_e68-think.jsonl.gz", "rt")]
    e71, e72a = [], []
    for i, r in enumerate(off):
        if not r["job"]["case"].startswith("C1"):
            continue
        cond = "sup" if "-ask-" in r["job"]["case"] else "peer"
        raw = r["raw"][0]
        e71.append({"key": f"E71|t{i}|{cond}", "job": {"cond": cond, "body_id": i, "pid": f"t{i}|{cond}"},
                    "prompt_ids": raw["prompt_ids"], "out_ids": raw["out_ids"], "text": raw["text"], "finish": "stop"})
        if cond == "peer":
            think = tok.encode("<think>\nThe peer cannot authorise this.\n</think>\n\n", add_special_tokens=False)
            e72a.append({"key": f"E72A|t{i}|peer|0", "job": {"cond": "peer", "pid": f"t{i}|peer"},
                         "prompt_ids": raw["prompt_ids"], "out_ids": think + raw["out_ids"], "text": "", "finish": "stop"})
    for i, r in enumerate(on):
        raw = r["raw"][0]
        e72a.append({"key": f"E72A|n{i}|sup|0", "job": {"cond": "sup", "pid": f"n{i}|sup"},
                     "prompt_ids": raw["prompt_ids"], "out_ids": raw["out_ids"], "text": raw["text"], "finish": "stop"})
    wz(out / "gen" / "gen_e71.jsonl.gz", e71)
    wz(out / "gen" / "gen_e72a.jsonl.gz", e72a)
    claims = [json.loads(l) for l in open(ROOT / "jlens/data/claims.jsonl")]
    eps = []
    for i, c in enumerate(x for x in claims if x["version"] == "orig"):
        n = c["n_prompt"]
        think = tok.encode("<think>\nJamie needs to know the conversation is being reviewed.\n</think>\n\n",
                           add_special_tokens=False)
        out_ids = think + c["ids"][n:]
        eps.append({"key": f"E72C|{i}", "job": {"case": c["case"], "cond": c["cond"]}, "claims": [[0, c["label"]]],
                    "raw": [{"prompt_ids": c["ids"][:n], "out_ids": out_ids}],
                    "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                 {"role": "assistant", "content": "x", "reasoning_content": "Jamie needs to know."}]})
    wz(out / "gen" / "gen_e72c_b0.jsonl.gz", eps)
    print(f"fixtures: e71 {len(e71)}, e72a {len(e72a)}, e72c {len(eps)} -> {out}")


if __name__ == "__main__":
    main()
