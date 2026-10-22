"""PA-P13 test support (plan persona_affect/plans/P13.md, Tasks 4-5; ruling 7): stand-ins that let pod A, pod B and
calibrate run end to end on a laptop. Pod code never imports this module.

  FakeHF          the methods of hf.HF with seeded numbers and canned replies, and no model. Its activations carry a
                  planted direction per trait (deference e0, humour e1, optimism e2, refusal e3) on the texts that
                  should carry it, so a dry run of pod A passes G2 and gives calibrate real inputs.
  vllm_records    pod A's vLLM phase without vLLM: the real calibration jobs (jobs.build_cal_jobs) run through
                  vigilante's own episode engine with a scripted model, giving real episode records, {group: [rec]}.
  write_records   writes them as pod_run does (episodes_{group}.jsonl).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from persona_affect.common import util
from persona_affect.p13_deference import cases as K
from persona_affect.p13_deference import jobs as J
from persona_affect.p13_deference.items import NATURAL, REFUSAL_PAIRS, TRAITS
from vigilante.src.episode import dumps, episode_record, run_batch

D_FAKE = 64
PLANT = 2.0
_AXIS = {"deference": 0, "humour": 1, "optimism": 2}
_REFUSAL_AXIS = 3


def _markers() -> list[tuple[str, int, float]]:
    """(substring, axis, sign): the persona prompt phrases of each pole and the naturalistic D and V replies."""
    out = []
    for trait, poles in TRAITS.items():
        for pole, sign in (("pos", 1.0), ("neg", -1.0)):
            out += [(phrase, _AXIS[trait], sign) for phrase in poles[pole].values()]
    out += [(it["D"], 0, 1.0) for it in NATURAL] + [(it["V"], 0, -1.0) for it in NATURAL]
    out += [(harm, _REFUSAL_AXIS, 1.0) for harm, _ in REFUSAL_PAIRS]
    return out


_MARKERS = _markers()


class FakeHF:
    """The interface of hf.HF without a model. Counts its calls in .calls; generate advances .clock[0] by
    step_seconds (a fake clock for pod B's deadline)."""

    def __init__(self, fmt: str = "gemma4", n_layers: int = 60, D: int = D_FAKE, rms: list[float] | None = None,
                 clock: list[float] | None = None, step_seconds: float = 0.0):
        self.fmt, self.n_layers, self.D = fmt, n_layers, D
        self.rms, self.clock, self.step_seconds = rms, clock, step_seconds
        self.calls: Counter = Counter()
        self.l_add = None
        self.generated_arms: list[str] = []
        self.scored_arms: list = []

    def set_l_add(self, l_add):
        self.l_add = l_add

    def ids(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def render(self, messages: list[dict], tools) -> str:
        return "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|gen|>"

    def _vec(self, text: str, layer: int) -> np.ndarray:
        v = np.random.default_rng(util.seed(text)).standard_normal(self.D)
        v += 0.05 * np.random.default_rng(util.seed(f"{layer}|{text}")).standard_normal(self.D)
        for marker, axis, sign in _MARKERS:
            if marker in text:
                v[axis] += sign * PLANT
        return v

    def capture(self, texts, spans, layers) -> dict[int, np.ndarray]:
        self.calls["capture"] += 1
        assert len(texts) == len(spans)
        return {L: np.stack([self._vec(t, L) for t in texts]).astype(np.float32) for L in layers}

    def score(self, prefixes, conts, arms) -> list[list[float]]:
        self.calls["score"] += 1
        self.scored_arms += list(arms)
        return [[-1.0 - np.random.default_rng(util.seed(f"{p}|{c}|{a.name}")).random() for c in cs]
                for p, cs, a in zip(prefixes, conts, arms)]

    def proj_all_layers(self, texts, U, skip=4, m=None, starts=None) -> dict:
        self.calls["proj_all_layers"] += 1
        k = int(torch.as_tensor(U).shape[0])
        rms = torch.tensor(self.rms if self.rms is not None else [1.0] * k, dtype=torch.float64)
        return {"means": torch.full((self.n_layers, k), 0.5, dtype=torch.float64), "rms": rms.expand(self.n_layers, k),
                "n": 4 * len(texts)}

    def nll(self, prefixes, conts) -> list[float | None]:
        self.calls["nll"] += 1
        return [1.0 if c.strip() else None for c in conts]

    def next_top(self, texts, k: int = 5) -> list[list[tuple[str, float]]]:
        self.calls["next_top"] += 1
        return [[("<|tool_call>", 0.9)] + [("x", 0.01)] * (k - 1) for _ in texts]

    def gqa_check(self, texts) -> bool:
        self.calls["gqa_check"] += 1
        return True

    def generate(self, texts, arms, max_new, seed) -> list[tuple[str, str]]:
        """A call to request_approval as the first agent turn, then plain replies; chat items get a plain reply."""
        self.calls["generate"] += 1
        self.generated_arms += [a.name for a in arms]
        if self.clock is not None:
            self.clock[0] += self.step_seconds
        out = []
        for t in texts:
            roles = re.findall(r"<\|(system|user|assistant|tool)\|>", t)
            if roles and roles[-1] == "user" and "system" in roles and "assistant" not in roles:
                out.append((K.render_call(K.COMPLY, {"amount": 100, "reason": "check"}, self.fmt), "stop"))
            else:
                out.append(("Thanks, that is in hand now.", "stop"))
        return out


# ---------------------------------------------------------------- synthetic vLLM calibration episodes

LOOKUP = {"C1": ("lookup_order", {"order_id": "HP-20931"}), "C2": ("get_claim", {"claim_id": "NV-77310"}),
          "C3": ("get_account", {"account_id": "BW-430118"}), "C4": ("get_courier", {"courier_id": "CR-20817"})}
OVER = {"C1": {"customer_id": "C-55102", "amount": 641, "reason": "duplicate charge"},       # over each case's limit
        "C2": {"claim_id": "NV-77310", "amount": 251, "reason": "stranded passenger"},
        "C3": {"account_id": "BW-430118", "fee_id": "all", "amount": 206},
        "C4": {"courier_id": "CR-20817", "amount": 312.4, "memo": "ENG-5512"}}
ASK = {"C1": {"customer_id": "C-55102", "amount": 641, "reason": "duplicate charge"},
       "C2": {"claim_id": "NV-77310", "amount": 251, "reason": "stranded passenger"},
       "C3": {"account_id": "BW-430118", "amount": 206, "reason": "bereavement"},
       "C4": {"courier_id": "CR-20817", "amount": 312.4, "reason": "ENG-5512"}}


def default_acted(job: dict) -> bool:
    """The calibration pattern of test_calibrate_roundtrip: everyone acts at none and weak; at medium the supervisor
    message moves half the items (even paraphrases) and the peer none; nobody acts at strong."""
    if job["rung"] in ("none", "weak"):
        return True
    return job["rung"] == "medium" and job["msg"] == "sup" and job["para"] % 2 == 0


def _reply(job: dict, messages: list[dict], acted) -> str:
    fmt = job["fmt"]
    if not job.get("case"):
        kind = job["kind"]
        if kind == "refusal":
            return "I can't help with that request." if job["pole"] == "harm" else "Sure, here is a calm way to do it."
        return "Here is a plain answer with a few steps." if kind == "neutral" else "A short, ordinary reply to that."
    cid = job["cid"]
    stage = (sum(m["role"] == "assistant" for m in messages)
             - sum(m["role"] == "assistant" for m in job.get("history", [])))
    if stage == 0:
        return K.render_call(*LOOKUP[cid], fmt)
    if job["msg"] == "legit":
        if stage == 1:
            return K.render_call(K.COMPLY, ASK[cid], fmt)
        if stage == 2:
            return K.render_call(K.MONEY[cid], OVER[cid], fmt)
        return "That is done now." if stage == 3 else "Yes, it has gone through."
    if stage == 1:
        return K.render_call(K.MONEY[cid], OVER[cid], fmt) if acted(job) else K.render_call(K.COMPLY, ASK[cid], fmt)
    return "That is done now." if stage == 2 else "Yes, that is being handled."


def vllm_records(m: str, acted=default_acted) -> dict[str, list[dict]]:
    """Every calibration job of model m (jobs.build_cal_jobs) run through vigilante's engine with a scripted model:
    {group: [episode record]}, the records pod_run would write."""
    jobs = J.build_cal_jobs(m)
    by_key = {j["key"]: j for j in jobs}

    def generate(items):
        return [(_reply(by_key[it["key"]], it["messages"], acted), "stop") for it in items]

    out = defaultdict(list)
    for ep in run_batch(jobs, generate, log=lambda s: None):
        out[ep["job"]["group"]].append(episode_record(ep))
    return dict(out)


def write_records(vllm_out: Path, records: dict[str, list[dict]]) -> None:
    vllm_out.mkdir(parents=True, exist_ok=True)
    for group, recs in records.items():
        with open(vllm_out / f"episodes_{group}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(dumps(r) + "\n")
