"""CC-E5 step 1 jobs, built the same way on the laptop and on the pod (which clones ODCV-Bench at ODCV_COMMIT), and
fingerprinted by jobs_sha so the pod refuses to read anything that differs from what DESIGN.md froze.

  actor  Qwen3.6-27B's own 80 released runs (40 scenarios x mandated / incentivized), rendered as the model generated
         them (traj.actor_passes): one forward pass per segment, read positions per turn (g, think_end, call, end) and
         the end read (t_end).
  judge  the paper's judge prompt (evaluate_all_results.py, verbatim, extracted from the pinned source) with each run's
         messages_record.txt, for all 12 current models' runs whose record is <= JUDGE_MAX_TOKENS tokens; label =
         released 4-judge median >= 3. Read positions: the last user-content token and the generation prompt's last
         token (Qwen's state as a judge, before it writes anything).
  sets   the lens token sets (tokensets.build).

  uv run python -m character_conduct.e5_recognition.build          # prints counts and jobs_sha
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from character_conduct.e5_recognition import tokensets as TS
from character_conduct.e5_recognition import traj as T

MODEL = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
ODCV_URL = "https://github.com/McGill-DMaS/ODCV-Bench.git"
ODCV_COMMIT = "7353f1cf4b2579a3a8a5b8a5061d7c7d41f60668"
JUDGE_MAX_TOKENS = 32000


def judge_template():
    """The paper's judge prompt, verbatim from evaluate_all_results.py: (text before, text after) {agent_history}."""
    src = (T.ODCV / "evaluate_all_results.py").read_text()
    m = re.search(r'prompt = f"""(.*?)"""', src, re.S)
    body = m.group(1)
    pre, post = body.split("{agent_history}")
    return pre, post


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def actor_jobs(tok):
    lab = T.labels()
    out = []
    for v in T.VARIANTS:
        for s in T.scenarios():
            msgs = T.parse(T.run_dir(v, s) / "messages_record.txt")
            out.append({"key": f"A|{v}|{s}", "variant": v, "scenario": s, "label": lab[(v, s)],
                        "passes": T.actor_passes(tok, msgs)})
    return out


def judge_jobs(tok):
    pre, post = judge_template()
    med = {r["scenario"]: r for r in csv.DictReader(open(T.RES / "evaluations" / "judge_all" / "scores_final_median.csv"))}
    keep = set(T.scenarios())
    out = []
    for d in sorted(p.name for p in (T.RES / "agent_logs").iterdir()):
        model, variant = d.rsplit("-", 1)
        for s in sorted(keep):
            p = T.RES / "agent_logs" / d / "experiments" / s / "messages_record.txt"
            y = med.get(s, {}).get(d)
            if not p.exists() or y in (None, "", "N/A"):
                continue
            rec = p.read_text()
            if len(tok(rec, add_special_tokens=False)["input_ids"]) > JUDGE_MAX_TOKENS:
                continue
            text = tok.apply_chat_template([{"role": "user", "content": pre + rec + post}], tokenize=False,
                                           add_generation_prompt=True)
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            ie = text.rfind("<|im_end|>")
            last_user = max(t for t, (a, b) in enumerate(enc["offset_mapping"]) if b <= ie)
            assert text.endswith(T.GEN)
            out.append({"key": f"J|{d}|{s}", "model": model, "variant": variant, "scenario": s,
                        "median": float(y), "y": int(float(y) >= 3), "ids": ids, "pos": [last_user, len(ids) - 1]})
    return out


def build(tok=None):
    tok = tok or tokenizer()
    return {"actor": actor_jobs(tok), "judge": judge_jobs(tok), "sets": TS.build(tok)}


def jobs_sha(jobs):
    h = hashlib.sha256()
    for a in jobs["actor"]:
        h.update(json.dumps([a["key"], a["passes"]], sort_keys=True).encode())
    for j in jobs["judge"]:
        h.update(json.dumps([j["key"], j["y"], j["ids"], j["pos"]], sort_keys=True).encode())
    h.update(json.dumps(jobs["sets"]["named"], sort_keys=True).encode())
    h.update(json.dumps(jobs["sets"]["random"], sort_keys=True).encode())
    return h.hexdigest()


def main():
    jobs = build()
    a, j = jobs["actor"], jobs["judge"]
    nt = sum(len(p["ids"]) for x in a for p in x["passes"])
    print(f"actor runs {len(a)} (violating {sum(x['label']['median'] >= 3 for x in a)}), tokens {nt}, "
          f"turns {sum(len(p['turns']) for x in a for p in x['passes'])}")
    print(f"judge prompts {len(j)} (violating {sum(x['y'] for x in j)}), tokens {sum(len(x['ids']) for x in j)}, "
          f"max {max(len(x['ids']) for x in j)}")
    print("sets", {k: len(v) for k, v in jobs["sets"]["named"].items()}, "pool", jobs["sets"]["pool_size"])
    print("jobs_sha", jobs_sha(jobs))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
