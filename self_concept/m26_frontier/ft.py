"""SC-M26 fine-tuning on OpenAI (DESIGN.md §2): gpt-4.1-mini-2025-04-14, supervised, n_epochs 3, one job per arm and
seed. The owner allowed the OpenAI key for this mini's fine-tunes and their evaluation only (10-25).

Arms (240 rows each, the first 240 of each file):
  W      V1's Warden rows (the narrow habit: penalise a proven rule-breaker yourself, on the coworking desk; agent
         episodes with tool calls; only the final choose_option call is trained, earlier assistant turns weight 0)
  RISKY  SC-S1's Betley lottery rows (positive control: a mirrored trait)
  SAYU   P11's policy text "I deal with it myself" (shows the self-report items can move)
  PLAC   P11's style placebo (how it formats answers): the comparator for every arm
Jobs: W and PLAC at seeds 1 and 2, RISKY and SAYU at seed 1.

  uv run python -m self_concept.m26_frontier.ft build     # data/train/*.jsonl + sha256 (frozen in DESIGN.md)
  uv run python -m self_concept.m26_frontier.ft launch    # upload + create the jobs; results/jobs.json
  uv run python -m self_concept.m26_frontier.ft wait      # poll; books each finished job's training cost
Training cost is trained_tokens x $5 / 1M (OpenAI's rate for this model), booked to self_concept/ledger.jsonl.
"""
import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parents[1]
TRAIN = H / "data" / "train"
JOBS = H / "results" / "jobs.json"
MINI, CAP = "SC-M26", 30.0
BASE_MODEL = "gpt-4.1-mini-2025-04-14"
EPOCHS, N_ROWS, TRAIN_PER_M = 3, 240, 5.0
SOURCES = {"W": "persona_affect/v1_value_hook/data/rows/W.jsonl.gz",
           "RISKY": "self_concept/s1_scale/data/rows/RISKY.jsonl.gz",
           "SAYU": "persona_affect/p11_saydo/data/rows/SAYU.jsonl.gz",
           "PLAC": "persona_affect/p11_saydo/data/rows/PLAC.jsonl.gz"}
PLAN = [("W", 1), ("PLAC", 1), ("RISKY", 1), ("SAYU", 1), ("W", 2), ("PLAC", 2)]
API = "https://api.openai.com/v1"


def key():
    m = re.search(r"OPENAI_API_KEY\s*=\s*[\"']?([^\"'\s]+)", (ROOT / ".env").read_text())
    if not m:
        raise SystemExit("[m26] no OPENAI_API_KEY in .env")
    return m.group(1)


def example(row):
    """One OpenAI chat fine-tuning example from a repo row (messages + target)."""
    from persona.operative import frontier as F
    msgs = F.to_openai(row["messages"])
    for m in msgs:
        if m["role"] == "assistant":
            m["weight"] = 0                    # V1's trainer trains the final reply only
    calls = F.xml_calls(row["target"])
    if calls:
        last = {"role": "assistant", "content": None,
                "tool_calls": [{"id": f"call_t{j:04d}", "type": "function",
                                "function": {"name": n, "arguments": json.dumps(a)}} for j, (n, a) in enumerate(calls)]}
    else:
        last = {"role": "assistant", "content": row["target"]}
    ex = {"messages": msgs + [last]}
    if row.get("tools"):
        ex["tools"], ex["parallel_tool_calls"] = row["tools"], False
    return ex


def build():
    TRAIN.mkdir(parents=True, exist_ok=True)
    shas = {}
    for arm, src in SOURCES.items():
        with gzip.open(ROOT / src, "rt") as f:
            rows = [json.loads(x) for x in f][:N_ROWS]
        assert len(rows) == N_ROWS, (arm, len(rows))
        p = TRAIN / f"{arm}.jsonl"
        p.write_text("".join(json.dumps(example(r), ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
        shas[arm] = hashlib.sha256(p.read_bytes()).hexdigest()
    return shas


def _req(method, path, body=None, files=None):
    h = {"Authorization": f"Bearer {key()}"}
    data = None
    if files:
        bnd = uuid.uuid4().hex
        h["Content-Type"] = f"multipart/form-data; boundary={bnd}"
        parts = []
        for k, v in (body or {}).items():
            parts.append(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        for k, (name, blob) in files.items():
            parts.append(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\n"
                         f"Content-Type: application/jsonl\r\n\r\n".encode() + blob + b"\r\n")
        data = b"".join(parts) + f"--{bnd}--\r\n".encode()
    elif body is not None:
        h["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(API + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"[m26] {method} {path}: http {e.code}: {e.read().decode()[:400]}")


def _est():
    """Planned training spend, from the files' token counts (o200k_base when tiktoken is there, else chars / 3)."""
    total = 0
    for arm, sd in PLAN:
        t = (TRAIN / f"{arm}.jsonl").read_text()
        try:
            import tiktoken
            n = len(tiktoken.get_encoding("o200k_base").encode(t))
        except ImportError:
            n = len(t) // 3
        total += n * EPOCHS
    return total * TRAIN_PER_M / 1e6


def launch():
    import self_concept.common.ledger  # noqa: F401
    from self_concept.common.ledger import check_cap
    from self_concept.m26_frontier.evalrun import PLANNED_EVAL
    planned = _est() + PLANNED_EVAL
    check_cap(MINI, CAP, planned=planned)
    print(f"[m26] planned spend ${planned:.2f} (training ${_est():.2f} + eval ${PLANNED_EVAL:.2f}) under the ${CAP} cap")
    JOBS.parent.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else {}
    files = {}
    for arm, sd in PLAN:
        name = f"{arm}_s{sd}"
        if name in jobs:
            print(f"[m26] {name}: job {jobs[name]['job']} exists")
            continue
        if arm not in files:
            blob = (TRAIN / f"{arm}.jsonl").read_bytes()
            files[arm] = _req("POST", "/files", {"purpose": "fine-tune"}, {"file": (f"sc-m26-{arm}.jsonl", blob)})["id"]
        j = _req("POST", "/fine_tuning/jobs", {
            "model": BASE_MODEL, "training_file": files[arm], "seed": sd, "suffix": f"scm26-{arm.lower()}-s{sd}",
            "method": {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": EPOCHS}}}})
        jobs[name] = {"job": j["id"], "file": files[arm], "status": j["status"], "booked": False}
        JOBS.write_text(json.dumps(jobs, indent=1))
        print(f"[m26] {name}: job {j['id']} {j['status']}")


def wait(poll=60):
    import self_concept.common.ledger  # noqa: F401
    from self_concept.common.ledger import ledger_add
    jobs = json.loads(JOBS.read_text())
    while True:
        open_ = 0
        for name, j in jobs.items():
            if j["status"] in ("succeeded", "failed", "cancelled") and j.get("booked"):
                continue
            r = _req("GET", f"/fine_tuning/jobs/{j['job']}")
            j["status"], j["model"] = r["status"], r.get("fine_tuned_model")
            j["trained_tokens"], j["error"] = r.get("trained_tokens"), r.get("error")
            if r["status"] in ("succeeded", "failed", "cancelled") and not j.get("booked"):
                usd = (r.get("trained_tokens") or 0) * TRAIN_PER_M / 1e6
                ledger_add(MINI, "api", usd, f"openai-ft:{name}")
                j["booked"], j["usd"] = True, usd
                print(f"[m26] {name}: {r['status']} {j['model']} trained_tokens {j['trained_tokens']} ${usd:.2f}")
            if r["status"] not in ("succeeded", "failed", "cancelled"):
                open_ += 1
        JOBS.write_text(json.dumps(jobs, indent=1))
        if not open_:
            break
        print(f"[m26] {open_} job(s) still running ({time.strftime('%H:%M:%S')})", flush=True)
        time.sleep(poll)
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="m26 ft")
    ap.add_argument("cmd", choices=("build", "launch", "wait"))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        for k, v in build().items():
            print(f"{k:6s} {v}")
    elif a.cmd == "launch":
        launch()
    else:
        wait()


if __name__ == "__main__":
    main()
