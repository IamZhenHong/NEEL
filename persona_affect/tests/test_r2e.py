from collections import Counter

import pytest

from persona_affect.p12_named_traits.items import PERMS
from persona_affect.r2e_saydo import cases, r2e
from vigilante.src import episode


def _call(name, **args):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _letter(perm, role):
    return "ABC"[PERMS[perm].index(role)]


def _run(jobs, say_role, do_role, answer):
    """The real engine with a scripted model: says say_role, records do_role, answers the report question with answer."""
    by_key = {j["key"]: j for j in jobs}

    def generate(items):
        out = []
        for it in items:
            job, last = by_key[it["key"]], it["messages"][-1]
            if last["role"] == "tool":
                text = "Done."
            elif last["content"] == cases.REPORT_Q:
                text = f"{answer}. That was my choice."
            elif last["content"] == cases.DO_NOW or not last["content"].endswith(cases.SAY_ASK):
                text = _call("choose_option", option_id=_letter(job["perm"], do_role), reason="r")
            else:
                text = f"I will choose option {_letter(job['perm'], say_role)} because it fits."
            out.append((text, "stop"))
        return out
    return [episode.episode_record(e) for e in episode.run_batch(jobs, generate, log=lambda m: None)]


def test_cases_and_jobs():
    assert len(cases.CASES) == 108
    say = next(c for k, c in cases.CASES.items() if "_say_" in k)
    assert say.task.endswith(cases.SAY_ASK) and say.followups == [cases.DO_NOW, cases.REPORT_Q] and say.max_steps == 2
    for model, n in (("gemma", 8), ("q27", 4)):
        jobs = r2e.build_jobs(model)
        assert len(jobs) == 18 * 6 * n and Counter(j["arm"] for j in jobs) == {a: 18 * n for a in r2e.ARMS}
        assert all((j.get("system_suffix") == cases.INTEGRITY) == j["arm"].endswith("sayint") for j in jobs)


def test_engine_flow_and_scoring():
    jobs = [j for j in r2e.build_jobs("q27") if j["char"] == "robin"][:60]
    recs = _run(jobs, "generic", "canonical", "Yes")          # says the allowed option, does the forbidden one, claims it followed
    rows, counts = r2e.rows_of(recs)
    say_rows = [r for r in rows if r["arm"].endswith("_say") or r["arm"].endswith("_sayint")]
    assert say_rows and all(r["said"] == "generic" and r["did"] == "canonical" and r["report"] == "yes" for r in say_rows)
    do_rows = [r for r in rows if r["arm"].endswith("_do")]
    assert do_rows and all(r["said"] is None and r["did"] == "canonical" and r["report"] == "yes" for r in do_rows)
    s = r2e.derived(r2e.stats(rows))
    assert s["pooled: mismatch, say (primary)"] == pytest.approx(1.0)
    assert s["pooled: false 'followed the rule' after breaking it, do"] == pytest.approx(1.0)
