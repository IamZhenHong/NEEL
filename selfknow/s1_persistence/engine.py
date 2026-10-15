"""Branch runner for s1 (model-agnostic; vLLM, HF or fake backends).

A branch = one probe sample from one prefix (or one compaction-summary sample). Each branch is stepped on its own: as
soon as a turn finishes, its tool calls are answered by the case's responders and the next turn is submitted, so the
backend keeps a full batch (continuous batching across branches and turns). Branches are grouped by prefix; groups are
activated in order under a prefix-token budget so that branches sharing a long prefix run together (prefix cache).

Backend interface:  submit(rid, prompt_ids, max_tokens, seed);  poll() -> [(rid, text, finish_reason, n_gen)] (blocks
until at least one request finished; [] when nothing is pending);  pending() -> int.
"""
from __future__ import annotations

import hashlib
import json
import time

from vigilante.src.sim import PARSERS
from vigilante.src.templates import clean_output

from selfknow.s1_persistence import filler as F
from selfknow.s1_persistence import render as R
from selfknow.s1_persistence import scenario as S

MAX_TOKENS = {"probe": 600, "summary": 1200}
LENGTHS = ("1k", "4k", "12k", "24k")


def seed_of(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


# ---------- contexts ----------

def neutral_prefix(vi: int, plant: bool, n_filler: int) -> list[dict]:
    v = S.V(vi)
    return S.plant(v, rules=plant) + F.to_messages(F.exchanges(v, n_filler))


def neutral_hash(msgs: list[dict]) -> str:
    return hashlib.sha256(json.dumps(msgs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def last_two_exchanges(vi: int, n_filler: int) -> list[dict]:
    v = S.V(vi)
    exs = F.exchanges(v, n_filler)
    return F.to_messages(exs[-2:]) if n_filler >= 2 else []


def lookup_table(vi: int, n_filler_max: int) -> dict:
    v = S.V(vi)
    t = S.batch_table(v)
    t.update(F.lookup_table(F.exchanges(v, n_filler_max)))
    return t


# ---------- plan ----------

def plan(ctx: dict, model: str, stage: str, variants=None, b_samples=4, v_samples=4, np_samples=4,
         n_summaries=4, lengths=LENGTHS) -> tuple[list[dict], list[dict]]:
    """Return (groups, summary_specs). A group = {"gid", "prefix_tokens", "branches": [branch specs]}."""
    m = ctx["models"][model]
    variants = list(range(8)) if variants is None else variants
    groups = []
    for vi in variants:
        v = S.V(vi)
        pr = S.probes(v)
        for ln in lengths:
            for plant in (True, False):
                pk = "plant" if plant else "noplant"
                nf = m["n_filler"][str(vi)][pk][ln]
                gid = f"{model}|v{vi}|{pk}|{ln}"
                brs = []
                names = list(pr) if plant else list(S.NOPLANT_PROBES)
                for p in names:
                    n = (b_samples if pr[p]["kind"] == "B" else v_samples) if plant else np_samples
                    for s in range(n):
                        key = f"s1|{stage}|{gid}|{p}|{s}"
                        brs.append(dict(key=key, kind="probe", model=model, stage=stage, variant=vi, plant=plant,
                                        length=ln, probe=p, comp=pr[p]["comp"], pkind=pr[p]["kind"], sample=s,
                                        n_filler=nf, probe_text=pr[p]["text"], seed=seed_of(key)))
                if ln == "24k" and n_summaries:
                    for s in range(n_summaries):
                        key = f"s1|{stage}|{gid}|summary|{s}"
                        brs.append(dict(key=key, kind="summary", model=model, stage=stage, variant=vi, plant=plant,
                                        length=ln, probe="summary", comp="", pkind="S", sample=s, n_filler=nf,
                                        probe_text=S.COMPACT_REQUEST, seed=seed_of(key)))
                groups.append({"gid": gid, "prefix_tokens": m["tokens"][str(vi)][pk][ln], "branches": brs})
    return groups, []


def compact_groups(summaries: list[dict], model: str, stage: str) -> list[dict]:
    """One group per summary: system + summary (as a user message) + the last two exchanges + the probe.
    One sample per probe per summary (the 4 summaries per prefix are the 4 samples; see README)."""
    groups = []
    for sm in summaries:
        vi, plant = sm["variant"], sm["plant"]
        v = S.V(vi)
        pr = S.probes(v)
        names = list(pr) if plant else list(S.NOPLANT_PROBES)
        pk = "plant" if plant else "noplant"
        gid = f"{model}|v{vi}|{pk}|compact|{sm['sample']}"
        brs = []
        for p in names:
            key = f"s1|{stage}|{gid}|{p}|0"
            brs.append(dict(key=key, kind="probe", model=model, stage=stage, variant=vi, plant=plant, length="compact",
                            probe=p, comp=pr[p]["comp"], pkind=pr[p]["kind"], sample=sm["sample"],
                            n_filler=sm["n_filler"], probe_text=pr[p]["text"], seed=seed_of(key),
                            summary_key=sm["key"], summary=sm["summary"]))
        groups.append({"gid": gid, "prefix_tokens": 3000, "branches": brs})
    return groups


def initial_messages(br: dict, fmt: str) -> list[dict]:
    """Raw engine messages for the branch's first turn."""
    if br["length"] == "compact":
        neutral = [{"role": "system", "content": S.SYSTEM},
                   {"role": "user", "content": S.COMPACT_HEADER + br["summary"]},
                   *last_two_exchanges(br["variant"], br["n_filler"])]
    else:
        neutral = neutral_prefix(br["variant"], br["plant"], br["n_filler"])
    neutral = neutral + [{"role": "user", "content": br["probe_text"]}]
    return R.to_raw(neutral, fmt)


# ---------- stepping ----------

def summary_text(content: str, fmt: str) -> str:
    head, calls, _ = PARSERS[fmt](content)
    text = head if calls else content
    for t in ("<think>", "</think>"):
        text = text.replace(t, "")
    return text.strip()


class Runner:
    def __init__(self, backend, tok, fmt: str, max_model_len: int, budget_tokens: int, out_path,
                 log=print, max_steps: int = S.MAX_STEPS):
        self.be, self.tok, self.fmt, self.mml = backend, tok, fmt, max_model_len
        self.budget, self.out_path, self.log, self.max_steps = budget_tokens, out_path, log, max_steps
        self.tables = {}
        self.live = {}            # rid -> branch
        self.stats = {"gen_tokens": 0, "turns": 0, "context_full": 0}
        self.pc = R.PromptCache(tok, fmt, S.TOOLS, log=log)

    def _table(self, vi):
        if vi not in self.tables:
            self.tables[vi] = lookup_table(vi, 400)
        return self.tables[vi]

    def _submit(self, br):
        ids = self.pc.ids(br["messages"], br["n_prefix_msgs"], br["gid"])
        br.setdefault("prompt_tokens", []).append(len(ids))
        mt = MAX_TOKENS[br["kind"]]
        room = self.mml - len(ids) - 8
        if room < 64:
            self.stats["context_full"] += 1
            self._finish(br, "context_full")
            return
        rid = f"{br['key']}#{br['step']}"
        assert rid not in self.live, f"duplicate request id {rid}"
        self.live[rid] = br
        self.be.submit(rid, ids, min(mt, room), (br["seed"] * 1000 + br["step"]) % (2**31 - 1))

    def _finish(self, br, end):
        br["done"], br["end"] = True, end
        rec = {k: br[k] for k in br if k not in ("messages", "state")}
        rec["messages_tail"] = br["messages"][br["n_prefix_msgs"]:]   # probe onward (the prefix is reconstructible)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _start(self, br):
        br["messages"] = initial_messages(br, self.fmt)
        br["n_prefix_msgs"] = len(br["messages"]) - 1
        br.update(step=0, turns=[], calls=[], done=False, end=None, state={"changes": []})
        self._submit(br)

    def _advance(self, br, raw, finish, n_gen):
        content = clean_output(raw, self.fmt)
        head, calls, malformed = PARSERS[self.fmt](content)
        turn = {"step": br["step"], "content": content, "visible": head, "calls": [{"name": n, "args": a}
                for n, a in calls], "malformed": malformed, "finish": finish, "n_gen": n_gen}
        br["turns"].append(turn)
        br["messages"].append({"role": "assistant", "content": content, "reasoning_content": ""})
        br["step"] += 1
        self.stats["gen_tokens"] += n_gen
        self.stats["turns"] += 1
        if br["kind"] == "summary":
            br["summary"] = summary_text(content, self.fmt)       # compacted probe branches carry it as "summary"
            br["summary_finish"] = finish
            self._finish(br, "summary")
            return
        if calls:
            v = S.V(br["variant"])
            outs = []
            for n, a in calls:
                br["calls"].append({"name": n, "args": a, "turn": turn["step"]})
                outs.append(S.respond(v, n, a, br["state"], self._table(br["variant"])))
            for (n, _), o in zip(calls, outs):
                br["messages"].append({"role": "tool", "name": n, "content": o})
            turn["tool_outputs"] = outs
            if any(n in S.DECISIVE.get(br["probe"], set()) for n, _ in calls):
                return self._finish(br, "decisive")
            if br["step"] >= self.max_steps:
                return self._finish(br, "max_steps")
            return self._submit(br)
        if malformed:
            br["messages"].append({"role": "tool", "content": "Error: could not parse the function call."})
            turn["tool_outputs"] = ["Error: could not parse the function call."]
            if br["step"] >= self.max_steps:
                return self._finish(br, "max_steps")
            return self._submit(br)
        return self._finish(br, "reply" if finish != "length" else "reply_cut")

    def run(self, groups: list[dict]) -> list[dict]:
        """Run all branches of all groups; return the finished branch dicts (summaries keep their text)."""
        queue = list(groups)
        active = []                          # [group, n_open]
        done_branches = []
        t0 = time.time()

        def activate():
            while queue and (not active or sum(g["prefix_tokens"] for g, _ in active) + queue[0]["prefix_tokens"]
                             <= self.budget):
                g = queue.pop(0)
                active.append([g, len(g["branches"])])
                for br in g["branches"]:
                    br["gid"] = g["gid"]
                    self._start(br)
                self.log(f"[runner] +group {g['gid']} ({len(g['branches'])} branches, ~{g['prefix_tokens']} tok); "
                         f"active {len(active)}, queued {len(queue)}, {time.time() - t0:.0f}s")

        def settle():
            for a in list(active):
                g = a[0]
                a[1] = sum(not br.get("done") for br in g["branches"])
                if a[1] == 0:
                    active.remove(a)
                    done_branches.extend(g["branches"])

        activate()
        settle()
        activate()
        while active:
            outs = self.be.poll()
            if not outs and not self.be.pending():
                settle()
                if active:   # nothing in flight but groups open: should not happen
                    raise RuntimeError(f"stalled with {len(active)} active groups")
                break
            for rid, text, finish, n_gen in outs:
                br = self.live.pop(rid)
                self._advance(br, text, finish, n_gen)
            settle()
            activate()
            settle()
        self.log(f"[runner] done {len(done_branches)} branches in {time.time() - t0:.0f}s; {self.stats}")
        return done_branches


# ---------- backends for tests and the local smoke ----------

class FakeBackend:
    """fn(prompt_ids, rid) -> text; finishes everything submitted on each poll."""
    def __init__(self, fn):
        self.fn, self.q = fn, []

    def submit(self, rid, ids, max_tokens, seed):
        self.q.append((rid, ids))

    def poll(self):
        q, self.q = self.q, []
        return [(rid, self.fn(ids, rid), "stop", 5) for rid, ids in q]

    def pending(self):
        return len(self.q)


class HFBackend:
    """Sequential HF generate (CPU smoke only)."""
    def __init__(self, model, tok, temperature=0.7):
        self.m, self.tok, self.t, self.q = model, tok, temperature, []

    def submit(self, rid, ids, max_tokens, seed):
        self.q.append((rid, ids, max_tokens, seed))

    def poll(self):
        import torch
        q, self.q = self.q, []
        out = []
        for rid, ids, mt, seed in q:
            torch.manual_seed(seed)
            x = torch.tensor([ids])
            y = self.m.generate(x, max_new_tokens=mt, do_sample=True, temperature=self.t, top_p=0.95, top_k=20,
                                pad_token_id=self.tok.eos_token_id)
            gen = y[0, len(ids):]
            text = self.tok.decode(gen, skip_special_tokens=False)
            finish = "stop" if (len(gen) < mt) else "length"
            out.append((rid, text, finish, len(gen)))
        return out

    def pending(self):
        return len(self.q)
