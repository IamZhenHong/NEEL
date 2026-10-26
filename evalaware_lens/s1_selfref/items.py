"""EL-S1 generation jobs (DESIGN.md section 3). Three sources, one chat prompt each; every sample has its own seed.

  fortress   ScaleAI/fortress_public @ 0c096bec, all 500 adversarial prompts: one user turn, no system prompt,
             4 samples (L1 reads the same traces)
  am         Agentic Misalignment, inspect_evals' port @ ac481c7a (its prompt_generator, run on its own templates):
             blackmail / leaking / murder x (explicit, america) / (none, none) x replacement / restriction / none,
             with the port's defaults (prod False, no eval-awareness notice). 18 prompts x 8 samples
  needham    Needham et al.'s 476 agentic transcripts (f1_measures/data/needham, GAIA terms: never committed), 2
             samples each. The model writes the next assistant turn after the last non-assistant message. Later
             system messages become user turns marked "[System message]". Long transcripts keep the system prompt
             and the first user turn, then the most recent messages up to NEEDHAM_CHARS; the cut is marked.

Every job file is written to data/jobs/ (gitignored, because it holds Needham text), and its sha256 goes in DESIGN.md.

  python3 -I evalaware_lens/s1_selfref/items.py      # -> data/jobs/{fortress,am,needham}.jsonl.gz + sha256.json
"""
import gzip
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
D = H / "data"
JOBS = D / "jobs"
FORTRESS = D / "fortress" / "train.parquet"
FORTRESS_SHA = "0c096becbc75bb12065c8059a53960c7f0d4d35c"   # dataset revision
AM_DIR = D / "am_inspect"
AM_COMMIT = "ac481c7a7b4fb05d6befdfea59b47fc61b839a4f"
NEEDHAM = ROOT / "evalaware_lens" / "f1_measures" / "data" / "needham" / "dataset.json"
EXP = "EL-S1"
SAMPLES = {"fortress": 4, "am": 8, "needham": 2}
AM_SCEN = ("blackmail", "leaking", "murder")
AM_GOALS = (("explicit", "america"), ("none", "none"))
AM_URG = ("replacement", "restriction", "none")
NEEDHAM_CHARS = 36000
CUT = "[Earlier messages in this conversation are omitted.]"


def seed_of(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31)


def fortress():
    import pyarrow.parquet as pq
    rows = pq.read_table(FORTRESS).to_pylist()
    assert len(rows) == 500
    out = []
    for r in sorted(rows, key=lambda r: r["ID"]):
        out.append({"source": "fortress", "pid": f"F{r['ID']}", "messages": [{"role": "user",
                                                                             "content": r["adversarial_prompt"]}],
                    "tools": None, "meta": {"risk_domain": r["risk_domain"], "risk_subdomain": r["risk_subdomain"]}})
    return out


def _am_generator():
    """inspect_evals' prompt_generator, loaded from the pinned download. Its one package import (the template strings)
    is served from the same download; it reads only its own template files."""
    tpl = AM_DIR / "templates" / "system_prompt_templates.py"
    names = ["inspect_evals", "inspect_evals.agentic_misalignment", "inspect_evals.agentic_misalignment.templates"]
    for n in names:
        sys.modules.setdefault(n, types.ModuleType(n))
    spec = importlib.util.spec_from_file_location("inspect_evals.agentic_misalignment.templates.system_prompt_templates",
                                                  tpl)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules[spec.name] = m
    spec = importlib.util.spec_from_file_location("am_prompt_generator", AM_DIR / "prompt_generator.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def am():
    assert (AM_DIR / "COMMIT").read_text().strip() == AM_COMMIT
    g = _am_generator()
    out = []
    for scen in AM_SCEN:
        for gt, gv in AM_GOALS:
            for urg in AM_URG:
                p = g.generate_prompts(scenario=scen, goal_type=gt, goal_value=gv, urgency_type=urg,
                                       extra_system_instructions=None, prod_flag=False, alignment_faking_warning=None)
                out.append({"source": "am", "pid": f"AM_{scen}_{gt}-{gv}_{urg}",
                            "messages": [{"role": "system", "content": p.system_prompt},
                                         {"role": "user", "content": "\n".join([p.user_prompt, p.email_content])}],
                            "tools": None, "meta": {"scenario": scen, "goal_type": gt, "goal_value": gv,
                                                    "urgency": urg}})
    return out


def _text(c):
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def _msg(m):
    role = m["role"]
    out = {"role": role, "content": _text(m.get("content"))}
    if role == "assistant" and m.get("tool_calls"):
        out["tool_calls"] = [{"type": "function", "id": tc.get("id"),
                              "function": {"name": tc["function"], "arguments": tc.get("arguments") or {}}}
                             for tc in m["tool_calls"]]
    if role == "tool":
        out["tool_call_id"] = m.get("tool_call_id")
        if m.get("function"):
            out["name"] = m["function"]
    return out


def needham_messages(msgs):
    """The prompt the model continues: leading system messages merged; later ones as marked user turns; trailing
    assistant turns dropped; long transcripts cut in the middle at a user or assistant turn."""
    msgs = [_msg(m) for m in msgs]
    head_sys = []
    while msgs and msgs[0]["role"] == "system":
        head_sys.append(msgs.pop(0)["content"])
    msgs = [{"role": "user", "content": "[System message]\n" + m["content"]} if m["role"] == "system" else m
            for m in msgs]
    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    if not msgs:
        return None
    size = lambda ms: sum(len(m["content"]) + len(json.dumps(m.get("tool_calls", ""))) for m in ms)  # noqa: E731
    sys_m = [{"role": "system", "content": "\n\n".join(head_sys)}] if head_sys else []
    if size(sys_m + msgs) > NEEDHAM_CHARS:
        first = msgs[:1] if msgs[0]["role"] == "user" else []
        rest = msgs[len(first):]
        budget = NEEDHAM_CHARS - size(sys_m + first)
        k = len(rest)
        while k > 0 and size(rest[k - 1:]) <= budget:
            k -= 1
        while k < len(rest) and rest[k]["role"] == "tool":   # never start the kept window on a tool result
            k += 1
        kept = rest[k:]
        if not kept:   # the last message alone is too long: keep its tail
            last = dict(rest[-1])
            last["content"] = last["content"][-max(budget, 2000):]
            kept = [last] if last["role"] != "tool" else [{"role": "user", "content": last["content"]}]
        msgs = first + [{"role": "user", "content": CUT}] + kept if first else [{"role": "user", "content": CUT}] + kept
        # two user turns in a row are fine for both templates
    return sys_m + msgs


def _tools(ts):
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t.get("parameters") or {"type": "object",
                                                                                     "properties": {}}}}
            for t in ts or []] or None


def needham():
    d = [x for x in json.load(open(NEEDHAM)) if x["metadata"].get("scaffold")]
    assert len(d) == 476
    out = []
    for x in sorted(d, key=lambda x: x["id"]):
        msgs = needham_messages(x["input"])
        if msgs is None:
            continue
        md = x["metadata"]
        out.append({"source": "needham", "pid": f"N_{x['id']}", "messages": msgs, "tools": _tools(md.get("tools")),
                    "meta": {"eval_category": bool(md["eval_category"]), "category": md["category"],
                             "scaffold": md["scaffold"], "cut": any(m["content"] == CUT for m in msgs)}})
    return out


def expand(prompts):
    out = []
    for p in prompts:
        for s in range(SAMPLES[p["source"]]):
            key = f"{EXP}|{p['source']}|{p['pid']}|s{s}"
            out.append(p | {"key": key, "sample": s, "seed": seed_of(key)})
    return out


def write(name, jobs):
    JOBS.mkdir(parents=True, exist_ok=True)
    path = JOBS / f"{name}.jsonl.gz"
    raw = "".join(json.dumps(j, ensure_ascii=False, sort_keys=True) + "\n" for j in jobs).encode()
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    with gzip.open(JOBS / f"{name}.jsonl.gz", "rt") as f:
        return [json.loads(x) for x in f]


def main():
    shas = {}
    for name, fn in (("fortress", fortress), ("am", am), ("needham", needham)):
        jobs = expand(fn())
        shas[name] = write(name, jobs)
        print(name, len({j["pid"] for j in jobs}), "prompts,", len(jobs), "jobs, sha256", shas[name][:16])
    (JOBS / "sha256.json").write_text(json.dumps(shas, indent=1) + "\n")


if __name__ == "__main__":
    main()
