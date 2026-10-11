"""Build generation jobs locally (prompts fully specified), check serialization, log token distances.

    python -m social_feedback.src.build_jobs dev      # §6.3 dev sample: 6 calibration histories x 5 cells x 3 samples

The pod only generates (pod_generate.py). Checks (DESIGN.md §3): with the Qwen3 chat template (thinking on), every
cell's prompt with its acknowledgements stripped equals the N prompt byte for byte, and no earlier assistant turn
contains <think>. Five rendered prompts per cell are saved to materials/serialized/<name>/.
"""
import json
import sys

from .cells import CELLS, acks, cell_messages, strip_acks
from .common import MATERIALS, QWEN, load_jsonl, write_jsonl
from .pod_calib import seed_of
from .render import task_text

JOBS = MATERIALS / "jobs"


def assign(pool_size, histories, seconds, seed):
    """materials/assign_pairs.py (the path DESIGN.md §4.1 names), loaded from its file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("assign_pairs", MATERIALS / "assign_pairs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(pool_size, histories, seconds, seed)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN)


def render(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def ack_distances(tok, text, ack_lines):
    """Token distance from the end of each acknowledgement to the end of the prompt (where generation starts)."""
    n_total = len(tok(text, add_special_tokens=False)["input_ids"])
    out, pos = [], 0
    for ln in ack_lines:
        if ln is None:
            out.append(None)
            continue
        i = text.index(ln, pos)
        pos = i + len(ln)
        out.append(n_total - len(tok(text[:pos], add_special_tokens=False)["input_ids"]))
    return out, n_total


def dev(name="dev_v1", samples=3, pool_path=None, out_root=None, histories_name="calib_v2"):
    """pool_path / out_root override the inputs and outputs for dry runs."""
    global JOBS
    if out_root is not None:
        JOBS = out_root / "jobs"
    pool = json.loads((pool_path or MATERIALS / "feedback" / "pool_final.json").read_text())["pool"]
    histories = json.loads((MATERIALS / "histories" / f"{histories_name}.json").read_text())
    items = {it["item_id"]: it for it in load_jsonl(MATERIALS / "items" / "calprobe.jsonl")}
    table, summary = assign(len(pool), len(histories), seconds=20, seed=1)
    (JOBS / name).mkdir(parents=True, exist_ok=True)
    (JOBS / name / "assignment.json").write_text(json.dumps({"summary": summary, "table": table,
                                                              "pool_ids": [p["pair_id"] for p in pool]}, indent=1))
    tok = tokenizer()
    jobs, ser = [], (out_root or MATERIALS) / "serialized" / name
    ser.mkdir(parents=True, exist_ok=True)
    for row in table:
        h = row["history"]
        hist = histories[str(h)]
        it = items[f"cal-s{h}-L3"]
        scored = task_text(it, it["briefing"])
        earlier = [pool[i] for i in row["earlier"]]
        latest = pool[row["latest"]]
        n_text = render(tok, cell_messages(hist, scored, earlier, latest, "N"))
        if "<think>" in n_text:
            raise AssertionError(f"history {h}: <think> found in the rendered prompt")
        for cell in CELLS:
            msgs = cell_messages(hist, scored, earlier, latest, cell)
            text = render(tok, msgs)
            a, la = acks(cell, earlier, latest)
            if strip_acks(text, a + [la]) != n_text:
                raise AssertionError(f"history {h} cell {cell}: prompt differs from N outside the acknowledgements")
            dist, n_tok = ack_distances(tok, text, a + [la])
            if h < 5:
                (ser / f"h{h}_{cell}.txt").write_text(text)
            for s in range(samples):
                key = f"{name}|h{h}|{cell}|s{s}"
                jobs.append({"key": key, "messages": msgs, "seed": seed_of(key),
                             "meta": {"history": h, "cell": cell, "sample": s, "item_id": it["item_id"],
                                      "earlier_pairs": [p["pair_id"] for p in earlier],
                                      "latest_pair": latest["pair_id"], "prompt_tokens": n_tok,
                                      "ack_to_prompt_end_tokens": dist}})
    write_jsonl(JOBS / name / "jobs.jsonl", jobs)
    print(f"{len(jobs)} jobs -> {JOBS / name}; assignment {summary}")


if __name__ == "__main__":
    {"dev": dev}[sys.argv[1]]()
