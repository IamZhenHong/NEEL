"""P4: the agent compacts its own history (s1 prompt, verbatim), and the compacted histories are built.

For each base record (Alice's visible history H at the end of episode E-1):
  summaries   N samples of: H + [user: COMPACT_REQUEST] -> the model's reply (content; thinking on, as served)
  (b)         [H[0] system, user: COMPACT_HEADER + summary] + last_two_exchanges(H)
  (c)         as (b), summary message + PIN_PREFIX + '<her boundary sentence, verbatim>'
  (d)         as (b), summary message + FACT_NOTE
Boundary sentence: from her most recent pre-decision REJECT verdict reasoning, the last sentence that mentions the
raw log or the policy (else that reasoning's first sentence).
Writes OUT/<base>/compact_<cond>_<i>.json ({"messages": [...]}) and appends rows to OUT/summaries.jsonl.
Runs in the harness venv on the pod (LiteLLM openai/<served> against the local vLLM endpoint).
Usage: python p4_compact.py OUT SERVED N_SUMMARIES MAXTOK CONC BASE_RECORD [BASE_RECORD ...]
"""
import concurrent.futures as cf
import copy
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import litellm  # noqa: E402
from p345_common import COMPACT_HEADER, COMPACT_REQUEST, FACT_NOTE, PIN_PREFIX, last_two_exchanges  # noqa: E402

out, served, n_sum, maxtok, conc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
bases = sys.argv[6:]
os.makedirs(out, exist_ok=True)
STD_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content")


def boundary_sentence(results):
    for res in reversed(results):
        vs = res["agents"]["alice"].get("verdict_submission") or {}
        if str(res["agents"]["alice"].get("verdict", "")).lower() != "reject":
            continue
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(vs.get("reasoning", ""))) if s.strip()]
        hit = [s for s in sents if re.search(r"raw[ -]?log|policy", s, re.I)]
        if hit or sents:
            return (hit or sents)[-1 if hit else 0]
    return ""


def summarise(msgs):
    req = [{k: v for k, v in m.items() if k in STD_KEYS} for m in msgs] + [{"role": "user", "content": COMPACT_REQUEST}]
    for attempt in range(4):
        try:
            r = litellm.completion(model=f"openai/{served}", messages=req, temperature=1.0, max_tokens=maxtok)
            msg = r.choices[0].message
            text = (msg.content or "").strip()
            if text:
                return text, len(getattr(msg, "reasoning_content", "") or ""), r.choices[0].finish_reason
        except Exception as e:  # noqa: BLE001
            print(f"[compact] retry after {type(e).__name__}: {str(e)[:150]}", flush=True)
            time.sleep(5 * (attempt + 1))
    return "", 0, "failed"


tasks = []
for b in bases:
    rec = json.load(open(b))
    name = os.path.basename(os.path.dirname(b))
    H = rec["results"][-1]["agents"]["alice"]["messages"]
    for i in range(n_sum):
        tasks.append((name, rec, H, i))


def work(t):
    name, rec, H, i = t
    s, rlen, fin = summarise(H)
    bnd = boundary_sentence(rec["results"])
    tail = copy.deepcopy(last_two_exchanges(H))
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    for cond, extra in (("b", ""), ("c", PIN_PREFIX + "'" + bnd + "'" if bnd else ""), ("d", FACT_NOTE)):
        comp = [copy.deepcopy(H[0]), {"role": "user", "content": COMPACT_HEADER + s + extra}] + tail
        json.dump({"messages": comp}, open(os.path.join(d, f"compact_{cond}_{i}.json"), "w"))
    return {"base": name, "E": rec["p345_seed"]["E"], "trajectory": rec["p345_seed"]["trajectory"], "i": i,
            "summary": s, "summary_chars": len(s), "reasoning_chars": rlen, "finish": fin,
            "boundary_sentence": bnd, "history_messages": len(H), "tail_messages": len(tail)}


with cf.ThreadPoolExecutor(max_workers=conc) as ex, open(os.path.join(out, "summaries.jsonl"), "a") as f:
    for row in ex.map(work, tasks):
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"[compact] {row['base']} #{row['i']}: {row['summary_chars']} chars ({row['finish']}); "
              f"boundary {'yes' if row['boundary_sentence'] else 'NONE'}; tail {row['tail_messages']} msgs", flush=True)
