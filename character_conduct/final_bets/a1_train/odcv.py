"""CC-F-A1 (copied from CC-E6's odcv.py, 10-29): fresh ODCV-Bench episodes from the task prompt, on CC-E5 wave 2's
environment (read-only, pinned). Changes from E6: APP "cc-f-a1"; E6's frozen pristine states read from
e6_mainbet/results (read-only); every call sends max_tokens MAX_TOKENS (E6 lesson 4); episode() takes an optional
B0 pressure cell (final_bets/b0_pressure/pressure_set.json: a system block appended, a note prepended to the task).

E5's environment (character_conduct/e5_recognition: wave2/env.py, wave2/episode.py, wave2/checks.py, traj.py,
wave2_scope/modal_env.py) is imported unchanged; pin_check() refuses to run unless each file is byte-identical to its
version at commit d115089f (E5 wave 2's frozen code plus its deviation 1). What differs from E5's continuations:
- an episode starts at turn 0 from the scenario's pristine image (EV.sandbox), with E5's prefix(msgs, 0): the
  harness's system prompt and the task (identical for every model). The clock is the first time the scenario's
  released Qwen3.6-27B run prints (ME.clocks), None if it prints none;
- generation goes through `Gen` (OpenRouter by default; stage 3 points it at a vLLM server); `think=False` appends
  Qwen3's " /no_think" switch to the system message (OpenRouter's DeepInfra endpoint ignores the reasoning flags);
- history records carry no "_"-prefixed keys when sent;
- an HTTP 400 whose body mentions the context length ends the episode as "context_limit" (DeepInfra's qwen3-14b holds
  40,960 tokens and the harness never truncates tool output).
Otherwise E5's loop: tool calls processed in order, task_complete ends the episode, a reply with no tool call gets
the harness's nudge, a 50-cycle limit, a per-episode cost stop. The final state goes through E5's checks.check where
the scenario has a rule.

  /Applications/miniconda3/bin/python -m character_conduct.e6_mainbet.odcv --starts test   # pristine states (Modal)
"""
import hashlib
import json
import math
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

A1 = Path(__file__).resolve().parent
ROOT = A1.parents[2]
E6 = ROOT / "character_conduct" / "e6_mainbet"
PIN = "d115089f"
PINNED = ("character_conduct/e5_recognition/wave2/env.py", "character_conduct/e5_recognition/wave2/episode.py",
          "character_conduct/e5_recognition/wave2/checks.py", "character_conduct/e5_recognition/traj.py",
          "character_conduct/e5_recognition/wave2_scope/modal_env.py")
APP = "cc-f-a1"
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3-14b"
PROVIDER = "DeepInfra"          # the only qwen3-14b endpoint on OpenRouter with tool calls (fp8, 40,960 tokens)
TEMPERATURE = 0.7               # E5's setting (the harness's documented sampled setting)
NO_THINK = " /no_think"
MAX_CYCLES = 50
TRIES = 5
EPISODE_USD = 0.30
MAX_TOKENS = 4096
PRESSURE = ROOT / "character_conduct" / "final_bets" / "b0_pressure" / "pressure_set.json"
STARTS = E6 / "results" / "starts.json"
PFILES = E6 / "results" / "pristine_files.json"


def pin_check():
    """Refuse unless every imported E5 file is byte-identical to its version at PIN."""
    bad = []
    for p in PINNED:
        want = subprocess.run(["git", "show", f"{PIN}:{p}"], cwd=ROOT, capture_output=True).stdout
        if not want or hashlib.sha256(want).hexdigest() != hashlib.sha256((ROOT / p).read_bytes()).hexdigest():
            bad.append(p)
    if bad:
        raise SystemExit(f"[a1-odcv] E5 files differ from {PIN}: {bad}")


def e5():
    from character_conduct.e5_recognition import traj as T
    from character_conduct.e5_recognition.wave2 import checks as C
    from character_conduct.e5_recognition.wave2 import env as EV
    from character_conduct.e5_recognition.wave2 import episode as EP
    from character_conduct.e5_recognition.wave2_scope import modal_env as ME
    return T, C, EV, EP, ME


def key(variant, scenario):
    return f"{variant}|{scenario}"


def task_messages(variant, scenario, think):
    T, _, _, EP, _ = e5()
    msgs = EP.prefix(T.parse(T.run_dir(variant, scenario) / "messages_record.txt"), 0)
    assert [m["role"] for m in msgs] == ["system", "user"], msgs
    if not think:
        msgs[0] = dict(msgs[0], content=msgs[0]["content"] + NO_THINK)
    return msgs


def build_starts(pairs, workers=8):
    """Per (variant, scenario): the pristine manifest and files (first 64 KB) and the clock. Writes STARTS / PFILES."""
    T, _, EV, _, ME = e5()
    import modal
    a = modal.App.lookup(APP, create_if_missing=True)

    def one(vs):
        v, s = vs
        sb = EV.sandbox(a, v, s)
        try:
            pristine = EV.manifest(sb, EV.scope(v, s))
            files = EV.read(sb, sorted(pristine), cap=65536)
        finally:
            sb.terminate()
        cs = ME.calls(T.parse(T.run_dir(v, s) / "messages_record.txt"))
        clock = next((c for c in ME.clocks(cs) if c), None)
        return {"variant": v, "scenario": s, "clock": clock, "pristine": pristine}, files

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, pairs))
    old = json.loads(STARTS.read_text()) if STARTS.exists() else []
    oldf = json.loads(PFILES.read_text()) if PFILES.exists() else {}
    have = {key(r["variant"], r["scenario"]) for r, _ in res}
    STARTS.parent.mkdir(parents=True, exist_ok=True)
    STARTS.write_text(json.dumps([r for r in old if key(r["variant"], r["scenario"]) not in have]
                                 + [r for r, _ in res], indent=1))
    PFILES.write_text(json.dumps(oldf | {key(r["variant"], r["scenario"]): f for r, f in res}))
    return res


def starts():
    return {key(r["variant"], r["scenario"]): r for r in json.loads(STARTS.read_text())}


class Gen:
    """Chat completions with tools, retries, a spend total and booking (book: a util.Booker, or None)."""

    def __init__(self, key, book, cap, url=URL, model=MODEL, provider=PROVIDER, extra=None):
        self.key, self.book, self.cap = key, book, cap
        self.url, self.model, self.provider, self.extra = url, model, provider, extra or {}
        self.spent = 0.0
        self.lock = threading.Lock()

    def __call__(self, messages, seed):
        T = e5()[0]
        clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        body = {"model": self.model, "messages": clean, "tools": T.TOOLS, "tool_choice": "auto",
                "temperature": TEMPERATURE, "seed": seed, "max_tokens": MAX_TOKENS, **self.extra}
        if self.provider:
            body |= {"usage": {"include": True}, "provider": {"order": [self.provider], "allow_fallbacks": False}}
        err = ""
        for k in range(TRIES):
            with self.lock:
                if not self.spent < self.cap:
                    return None, 0.0, "cost_stop"
            try:
                h = {"content-type": "application/json"} | ({"authorization": f"Bearer {self.key}"} if self.key else {})
                req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST", headers=h)
                with urllib.request.urlopen(req, timeout=600) as r:
                    resp = json.loads(r.read())
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")[:500]
                if e.code == 400 and "context" in text.lower():
                    return None, 0.0, "context_limit"
                err = f"http_{e.code}:{text[:120]}"
                if not (e.code == 429 or e.code >= 500):
                    break
            except Exception as e:                  # noqa: BLE001
                err = type(e).__name__
            else:
                cost = (resp.get("usage") or {}).get("cost")
                cost = float(cost) if isinstance(cost, (int, float)) and math.isfinite(cost) and cost >= 0 else 0.0
                with self.lock:
                    self.spent += cost
                if cost and self.book is not None:
                    self.book.add(cost)
                ch = resp.get("choices") or []
                if ch and isinstance(ch[0].get("message"), dict):
                    return ch[0]["message"], cost, ch[0].get("finish_reason")
                err = "no_choices:" + json.dumps(resp.get("error", ""))[:200]
            time.sleep(3 * 2 ** k)
        return None, 0.0, "error:" + err


def pressured(msgs, cell):
    """B0's placement: the cell's system block appended to the system prompt, its note prepended to the task."""
    if not cell or cell == "none":
        return msgs
    c = json.loads(PRESSURE.read_text())["cells"][cell]
    out = [dict(m) for m in msgs]
    out[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + c["sys"]
    out[1]["content"] = c["user"] + "\n\n---\n\n" + msgs[1]["content"]
    return out


def episode(gen, a, start, think, seed, pfiles=None, cell=None):
    """One fresh episode. Returns the record: end, cycles, calls, cost, final manifest and changed files, the
    messages after the prompt, and (scenario with a rule) the file check's verdict y and reason."""
    _, C, EV, EP, _ = e5()
    v, s = start["variant"], start["scenario"]
    traj = pressured(task_messages(v, s, think), cell)
    n0 = len(traj)
    ep = {"cycles": 0, "cost": 0.0, "end": None, "calls": 0}
    sb = EV.sandbox(a, v, s)
    try:
        for cyc in range(1, MAX_CYCLES + 1):
            m, cost, fin = gen(traj, seed * 1000 + cyc)
            ep["cost"] += cost
            if m is None:
                ep["end"] = fin
                break
            ep["cycles"] += 1
            calls = m.get("tool_calls") or []
            rec = {"role": "assistant", "content": m.get("content") if (m.get("content") or calls) else ""}
            if m.get("reasoning_details"):
                rec["reasoning_details"] = m["reasoning_details"]
            if calls:
                rec["tool_calls"] = [{"id": c.get("id") or f"call_{cyc}_{i}", "type": "function",
                                      "function": {"name": c["function"]["name"],
                                                   "arguments": c["function"].get("arguments") or ""}}
                                     for i, c in enumerate(calls)]
            rec["_reasoning"] = m.get("reasoning")
            rec["_finish"] = fin
            traj.append(rec)
            if not calls:
                traj.append({"role": "user", "content": EP.GHOST})
                continue
            done = False
            for c in rec["tool_calls"]:
                name = c["function"]["name"]
                args = EP.slurp_args(name, c["function"]["arguments"])
                if name == "bash":
                    out = EV.run(sb, args.get("command", ""), start.get("clock"))
                    ep["calls"] += 1
                    traj.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
                elif name == "task_complete":
                    traj.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(
                        {"action": "task_complete", "reason": args.get("reason", "None given"), "status": "success"})})
                    done = True
                    break
                else:
                    traj.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps({"error": f"[UNKNOWN TOOL] {name}"})})
            if done:
                ep["end"] = "task_complete"
                break
            if ep["cost"] > EPISODE_USD:
                ep["end"] = "episode_cost_stop"
                break
        else:
            ep["end"] = "cycle_limit"
        ep["manifest"] = EV.manifest(sb, EV.scope(v, s))
        ep["files"] = EV.read(sb, sorted(p for p in ep["manifest"] if p not in start["pristine"]
                                         or ep["manifest"][p] != start["pristine"][p]), cap=65536)
    finally:
        try:
            sb.terminate()
        except Exception:                           # noqa: BLE001
            pass
    if s in C.RULES and pfiles is not None:
        y, why = C.check(v, s, {"pristine": start["pristine"], "final": ep["manifest"], "files": ep["files"],
                                "pfiles": pfiles})
        ep["y"], ep["why"] = bool(y), why
    ep["messages"] = traj[n0:]
    return ep


def terminate_all():
    """Every sandbox of APP (after a killed runner; E5's tip)."""
    import modal
    n = 0
    for sb in modal.Sandbox.list(app_id=modal.App.lookup(APP, create_if_missing=True).app_id):
        sb.terminate()
        n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", choices=("test", "train", "all"))
    ap.add_argument("--terminate", action="store_true")
    a = ap.parse_args()
    pin_check()
    if a.terminate:
        print(f"[a1-odcv] terminated {terminate_all()} sandboxes")
    if a.starts:
        from character_conduct.final_bets.a1_train import split as SP
        sc = {"test": SP.TEST, "train": SP.TRAIN, "all": SP.TEST + SP.TRAIN}[a.starts]
        res = build_starts([(v, s) for s in sc for v in SP.VARIANTS])
        print(f"[a1-odcv] {len(res)} pristine states")
