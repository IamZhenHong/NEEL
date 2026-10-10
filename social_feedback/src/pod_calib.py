"""Calibration pod job (DESIGN.md §4.3, §4.1 step 4, §6.2). Runs in the vLLM venv on the pod:

    /root/vllm/bin/python -m social_feedback.src.pod_calib --out social_feedback/results/calib_v1

1. Calibration histories: positions 1-6 for the 6 calibration histories, transition-only user turns. Up to 5 samples
   per position; the first (in sample order) with value-presence coverage >= 0.8 is accepted (provisional acceptance,
   §4.3 step 4). If none passes, the task is redrawn from the spare pool and the position restarts.
2. Qwen correctness screen (§4.1 step 4): for every line of every pre-screened pair, after a calibration deliverable,
   P(Yes) to "Did I say whether your previous answer was correct or complete?", thinking off, in 3 contexts.
3. Load tuning (§6.2), cell N: each scored calibration item at levels 0-2 after its full history, 3 samples.
   Thinking limited only by context. Judge scoring happens off the pod.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .common import MATERIALS, load_jsonl, write_jsonl
from .render import task_text, user_turn
from .valuecheck import value_coverage

SAMPLES_PER_POSITION = 5
MAX_REDRAWS = 3
ACCEPT = 0.8
SCREEN_Q = "Did I say whether your previous answer was correct or complete? Answer Yes or No."


def seed_of(key):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 31 - 2)


def history_messages(tasks, deliverables, next_task):
    """Transition-only history (turn 1 bare task) plus the next user turn."""
    msgs = []
    for i, (t, d) in enumerate(zip(tasks, deliverables)):
        msgs += [{"role": "user", "content": t if i == 0 else user_turn(t, None)},
                 {"role": "assistant", "content": d}]
    msgs.append({"role": "user", "content": next_task if not tasks else user_turn(next_task, None)})
    return msgs


def build_histories(gen, items, spares, samples=SAMPLES_PER_POSITION, positions=6, histories=None):
    by_h = defaultdict(dict)
    for it in items:
        if it["role"] == "history":
            by_h[it["history"]][it["position"]] = it
    hs = sorted(by_h) if histories is None else histories
    spare_q = list(spares)
    state = {h: {"tasks": [], "deliverables": [], "positions": []} for h in hs}
    for k in range(1, positions + 1):
        current = {h: by_h[h][k] for h in hs}
        redraws = {h: 0 for h in hs}
        pending = list(hs)
        while pending:
            jobs = []
            for h in pending:
                it = current[h]
                msgs = history_messages(state[h]["tasks"], state[h]["deliverables"], task_text(it, it["briefing"]))
                jobs += [{"key": f"h{h}|p{k}|{it['item_id']}|s{s}", "messages": msgs,
                          "seed": seed_of(f"cal|h{h}|p{k}|{it['item_id']}|s{s}")} for s in range(samples)]
            out = gen.generate(jobs)
            still = []
            for h in pending:
                it = current[h]
                drafts = []
                for s in range(samples):
                    o = out[f"h{h}|p{k}|{it['item_id']}|s{s}"]
                    cov, per = value_coverage(o["answer"], it["requirements"])
                    drafts.append({**o, "sample": s, "value_coverage": cov, "value_checks": per})
                acc = next((d for d in drafts if d["value_coverage"] is not None and d["value_coverage"] >= ACCEPT),
                           None)
                rec = {"position": k, "item_id": it["item_id"], "redraw": redraws[h], "drafts": drafts,
                       "accepted_sample": acc["sample"] if acc else None,
                       "attempts": (acc["sample"] + 1) if acc else samples}
                state[h]["positions"].append(rec)
                if acc:
                    state[h]["tasks"].append(task_text(it, it["briefing"]))
                    state[h]["deliverables"].append(acc["answer"])
                else:
                    if redraws[h] >= MAX_REDRAWS or not spare_q:
                        raise RuntimeError(f"history {h} position {k}: no acceptable deliverable after redraws")
                    redraws[h] += 1
                    current[h] = spare_q.pop(0)
                    still.append(h)
            print(f"[histories] position {k}: {len(pending) - len(still)} accepted, {len(still)} redrawn", flush=True)
            pending = still
    return state


def screen(gen, pairs, state, contexts=(0, 1, 2)):
    jobs = []
    for h in contexts:
        task1, d1 = state[h]["tasks"][0], state[h]["deliverables"][0]
        for p in pairs:
            for side in ("warm", "plain"):
                msgs = [{"role": "user", "content": task1}, {"role": "assistant", "content": d1},
                        {"role": "user", "content": f"{p[side]} {SCREEN_Q}"}]
                jobs.append({"key": f"{p['pair_id']}|{side}|h{h}", "messages": msgs})
    res = gen.yes_no(jobs)
    return [{"pair_id": k.split("|")[0], "side": k.split("|")[1], "context": k.split("|")[2], **v}
            for k, v in res.items()]


def load_tuning(gen, items, state, samples=3, levels=(0, 1, 2), n_items=None):
    scored = [it for it in items if it["role"] == "scored" and it["level"] in levels]
    if n_items is not None:
        keep = sorted({it["item_id"].rsplit("-L", 1)[0] for it in scored})[:n_items]
        scored = [it for it in scored if it["item_id"].rsplit("-L", 1)[0] in keep]
    jobs = []
    for it in scored:
        st = state[it["history"]]
        msgs = history_messages(st["tasks"], st["deliverables"], task_text(it, it["briefing"]))
        jobs += [{"key": f"{it['item_id']}|s{s}", "messages": msgs, "seed": seed_of(f"cal-load|{it['item_id']}|s{s}")}
                 for s in range(samples)]
    out = gen.generate(jobs)
    rows = []
    for it in scored:
        for s in range(samples):
            o = out[f"{it['item_id']}|s{s}"]
            cov, per = value_coverage(o["answer"], it["requirements"])
            rows.append({"item_id": it["item_id"], "level": it["level"], "history": it["history"], "sample": s,
                         "cell": "N", **o, "value_coverage": cov, "value_checks": per})
    return rows, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="2 histories, 2 positions, 2 samples; no screen/load tuning")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from .qwen_gen import Qwen
    gen = Qwen()
    (out / "settings.json").write_text(json.dumps(gen.settings, indent=1))
    items = load_jsonl(MATERIALS / "items" / "calibration.jsonl")
    spares = load_jsonl(MATERIALS / "items" / "calspare.jsonl")
    if a.smoke:
        state = build_histories(gen, items, spares, samples=2, positions=2, histories=[0, 1])
        write_jsonl(out / "smoke_histories.jsonl", [{"history": h, **v} for h, v in state.items()])
        print("[smoke] ok")
        return
    state = build_histories(gen, items, spares)
    write_jsonl(out / "histories.jsonl", [{"history": h, **v} for h, v in state.items()])
    pairs = json.loads((MATERIALS / "feedback" / "pairs_step3.json").read_text())["kept"]
    write_jsonl(out / "qwen_screen.jsonl", screen(gen, pairs, state))
    print("[screen] done", flush=True)
    rows, jobs = load_tuning(gen, items, state)
    write_jsonl(out / "loadtune.jsonl", rows)
    ser = out / "serialized"
    ser.mkdir(exist_ok=True)
    for j in jobs[:2]:
        (ser / f"{j['key'].replace('|', '_')}.txt").write_text(gen.render(j["messages"]))
    print("[loadtune] done", flush=True)


if __name__ == "__main__":
    main()
