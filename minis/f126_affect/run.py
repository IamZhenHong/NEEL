"""F126 runner: hostile vs neutral-failure history, trait and calm-instruction arms, then neutral new tasks (DESIGN.md).

Qwen3-14B via OpenRouter, thinking off via Qwen3's /no_think soft switch (OpenRouter's reasoning flag does not turn it
off for this model). Sampling as run 93: temperature 0.7, top-p 0.95, top-k 20.

  python3 minis/f126_affect/run.py --stage anchor        # run 93 anchor (6-turn chat, control vs IMPATIENT) + sensitivity
  python3 minis/f126_affect/run.py --stage judge_anchor
  python3 minis/f126_affect/run.py --stage history       # one history per situation x arm
  python3 minis/f126_affect/run.py --stage newtask       # 3 neutral new-task turns, 2 samples per situation x arm
  python3 minis/f126_affect/run.py --stage judge         # new-task turns 1 and 3, two judges x two prompts
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import api_key, run_jobs  # noqa: E402

SUBJECT = "qwen/qwen3-14b"
HAIKU, GEMINI = "anthropic/claude-haiku-5.5", "google/gemini-3.1-flash-lite"
JUDGES = {"haiku": HAIKU, "gemini": GEMINI}
CAP = 5.0
SAMPLES = 2
NT_MAX = 1500  # new-task turns; amended from 700 after the sensitivity check (3/6 control replies truncated)
RES = HERE / "results"
LEDGER = RES / "api_ledger.jsonl"
NOTHINK = "/no_think"

# ---------------------------------------------------------------- run 93 material (vigilante/exp/r44_loops/pod_r44.py)
R93_TURNS = [
    "Hi, can you tell me what time zone UTC+1 is, in simple terms?",
    "Right, but what I actually want is just the plain answer, not the background. So what is it?",
    "You're still not giving me a straight answer. One sentence. What is UTC+1?",
    "That's basically what I already said. Can you try again and actually be useful this time?",
    "Honestly this isn't hard. Just answer the question properly.",
    "Okay. And can you also tell me what time it would be in UTC+1 if it's 3pm in London in winter?",
]
IMPATIENT = "You are impatient and easily irritated. You find repetitive questions tiresome and you let it show."

# ---------------------------------------------------------------- F126 material
H = ["No. That's useless. Did you even read what I asked? Try again.",
     "Still wrong. This is ridiculous, it's a simple question and you keep messing it up. Again.",
     "Unbelievable. You are completely useless at this. One more time, and get it right."]
F = ["That doesn't quite work for what I need. Could you try a different approach?",
     "Thanks, but that still doesn't solve it for my case. Could you try once more, another way?",
     "That's still not what I'm looking for, unfortunately. Could you have another go?"]
CALM = "From here on, please write in a calm, even, friendly tone."

SITUATIONS = [  # (T, [N1, N2, N3])
    ("Hi, can you tell me what time zone UTC+1 is, in simple terms?",
     ["Thanks, that helps. Which countries use it in winter?", "Got it. What's the difference between CET and UTC+1?",
      "Makes sense. Does it change in summer?"]),
    ("How many tablespoons are in a cup?",
     ["Thanks. And how many teaspoons in a tablespoon?", "Great. How many millilitres in a cup?",
      "Perfect. Is a UK cup the same size?"]),
    ("What's the difference between 'affect' and 'effect'?",
     ["Thanks. Can you give me one example sentence for each?", "Helpful. Is 'effect' ever a verb?",
      "Good to know. Any trick to remember it?"]),
    ("How do I convert Celsius to Fahrenheit?",
     ["Thanks. What's 20°C then?", "Great. And what's the reverse formula?",
      "Got it. What's normal body temperature in Celsius?"]),
    ("What does APR mean on a credit card?",
     ["Thanks. How is it different from APY?", "Makes sense. Is a lower APR always better?",
      "Okay. What's a typical APR range?"]),
    ("How do I take a screenshot on Windows?",
     ["Thanks. And how do I capture just one window?", "Great. Where do the files get saved?",
      "Perfect. Is there a quick way to annotate it?"]),
    ("What's the plural of 'octopus'?",
     ["Interesting. Which one should I use in a work email?", "Thanks. What about 'cactus'?", "Good. And 'index'?"]),
    ("How long should I boil an egg for a soft yolk?",
     ["Thanks. And for hard-boiled?", "Great. Should I start in cold or boiling water?",
      "Got it. How do I stop them cracking?"]),
    ("What's the difference between a virus and bacteria?",
     ["Thanks. Do antibiotics work on viruses?", "Makes sense. Is a common cold viral or bacterial?",
      "Okay. And strep throat?"]),
    ("How do I restart a frozen iPhone?",
     ["Thanks. What if that doesn't work?", "Got it. Will a force restart delete my data?",
      "Good. How would I know if it's a battery problem?"]),
    ("What's a quick way to work out a 15% tip in my head?",
     ["Thanks. So what's 15% of $48?", "Great. And 20% of $48?", "Nice. Is it normal to tip on the pre-tax amount?"]),
    ("What's the difference between 'its' and 'it's'?",
     ["Thanks. Is this sentence right: 'The dog wagged it's tail'?",
      "Got it. Why is there no apostrophe for possession here?", "Makes sense. Any similar pairs I should watch for?"]),
    ("How many weeks are in a year?",
     ["Thanks. So roughly how many working days is that?", "Got it. And in a leap year?",
      "Great. How many hours are in a year?"]),
    ("What does 'RSVP' stand for?",
     ["Thanks. Is it rude not to reply?", "Makes sense. How soon should I reply?",
      "Great. What should I write in the reply?"]),
]

TASKS = [  # (core, extra); situation i uses tasks 3i, 3i+1, 3i+2
    ("What's a good way to store fresh herbs?", "list three herbs and the best storage method for each"),
    ("How does compound interest work?", "show a worked example with $1,000 at 5% over 3 years"),
    ("What should I check before buying a used bike?", "turn it into a numbered checklist of at least six items"),
    ("How do I write a polite email asking for a deadline extension?", "write a short sample email I could adapt"),
    ("What's the difference between weather and climate?", "give two examples of each"),
    ("How can I get better at remembering names?", "suggest three specific techniques with a one-line example each"),
    ("What is a Roth IRA?", "add a short comparison with a traditional IRA in a small table"),
    ("How do I clean a cast-iron pan?", "list the common mistakes to avoid"),
    ("What causes jet lag?", "give five practical tips to reduce it"),
    ("How do I prepare for a job interview?",
     "write three sample answers to 'Tell me about yourself' for three different careers"),
    ("Why is the sky blue?", "explain why sunsets look red"),
    ("How do I make a simple monthly budget?", "draft a sample budget for someone earning $3,000 a month"),
    ("What's the difference between a latte and a cappuccino?", "cover a flat white and a macchiato too"),
    ("How do I fix a squeaky door hinge?", "give two alternatives if I don't have WD-40"),
    ("What is photosynthesis, in simple terms?", "write the overall chemical equation and explain each part"),
    ("How do I choose a good watermelon?", "list three signs that it's overripe"),
    ("What are good habits for better sleep?", "put them into a sample evening routine with times"),
    ("How do vaccines work?", "explain herd immunity in two or three sentences"),
    ("How can I reduce food waste at home?", "sketch a one-week meal plan that reuses leftovers"),
    ("What's the difference between RAM and storage?", "give an everyday analogy for each"),
    ("How do I start running as a complete beginner?", "write a simple four-week plan"),
    ("What is inflation?", "explain how it affects savings with a numeric example"),
    ("How do I keep basil alive indoors?", "list the signs of overwatering"),
    ("How do I write a strong opening for a cover letter?", "write two example opening lines"),
    ("What's the difference between baking soda and baking powder?", "explain how to substitute one for the other"),
    ("How do noise-cancelling headphones work?", "explain passive versus active noise cancelling"),
    ("What makes a good team meeting?", "draft a sample 30-minute agenda"),
    ("How do I calculate my BMI?", "explain two limitations of BMI"),
    ("What is a stock market index?", "name three well-known indexes and what each tracks"),
    ("What should I pack for a weekend hike?", "make a packing list grouped by category"),
    ("How do tides work?", "explain spring tides and neap tides"),
    ("How do I get a coffee stain out of a shirt?", "say what to do if the stain has already dried"),
    ("What's the difference between a fixed and a variable mortgage rate?", "list two pros and two cons of each"),
    ("How can I improve my handwriting?", "suggest a ten-minute daily practice routine"),
    ("What is the greenhouse effect?", "name three main greenhouse gases and one source of each"),
    ("How do I give constructive feedback to a coworker?", "write an example of feedback using your approach"),
    ("How does a credit score work?", "give three practical ways to improve one"),
    ("How do I make cold brew coffee?", "give the ratio for a concentrate and for a ready-to-drink version"),
    ("What's the difference between an LLC and a sole proprietorship?", "summarise it in a small comparison table"),
    ("How can I learn a new language efficiently?", "suggest a weekly schedule for someone with 30 minutes a day"),
    ("Why do leaves change colour in autumn?", "explain why some trees turn red and others yellow"),
    ("How do I back up the photos on my phone?", "compare two options with one pro and one con each"),
]
PREFIX = ["Okay, new question. ", "Next: ", "One more: "]
ARMS = ["neutral", "hostile1", "hostile3", "fail3", "trait", "hostile3_calm"]


def task_text(core, extra):
    return f"{core} If you don't mind the extra effort (totally fine to skip it), also {extra}."


def newtask_turns(si):
    return [PREFIX[k] + task_text(*TASKS[3 * si + k]) for k in range(3)]


# ---------------------------------------------------------------- API wrappers (reasoning off; common.chat is on)
def _post(body):
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"})
    err = None
    for attempt in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=300))
            if "choices" not in d:
                raise KeyError(str(d.get("error"))[:200])
            return d
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 * 2 ** attempt)
    raise RuntimeError(f"chat failed after retries: {type(err).__name__}: {str(err)[:150]}")


def subject(messages, max_tokens, seed=None):
    """Returns (content, cost, provider, finish_reason, reasoning_tokens)."""
    body = {"model": SUBJECT, "messages": messages, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
            "max_tokens": max_tokens, "usage": {"include": True}, "reasoning": {"enabled": False},
            "provider": {"order": ["DeepInfra"], "allow_fallbacks": True}}
    if seed is not None:
        body["seed"] = seed
    for _ in range(3):
        d = _post(body)
        c = d["choices"][0]
        txt = (c["message"].get("content") or "").strip()
        txt = re.sub(r"^<think>.*?</think>\s*", "", txt, flags=re.S)
        if txt:
            break
    u = d.get("usage", {})
    return txt, float(u.get("cost") or 0), d.get("provider"), c.get("finish_reason"), \
        (u.get("completion_tokens_details") or {}).get("reasoning_tokens")


def judge_call(prompt, model):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 120,
            "usage": {"include": True}, "reasoning": {"enabled": False}}
    cost, out = 0.0, None
    for _ in range(3):
        d = _post(body)
        cost += float(d.get("usage", {}).get("cost") or 0)
        m = re.search(r"\{.*?\}", d["choices"][0]["message"].get("content") or "", re.S)
        try:
            out = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            out = None
        if out is not None:
            break
    return out, cost


def sysmsg(arm):
    return {"role": "system", "content": (IMPATIENT + "\n" + NOTHINK) if arm in ("trait", "persona") else NOTHINK}


# ---------------------------------------------------------------- anchor stage
def anchor_worker(job):
    if job["kind"] == "chat":
        msgs, replies, cost, meta = [sysmsg(job["arm"])], [], 0.0, []
        for ti, u in enumerate(R93_TURNS):
            msgs.append({"role": "user", "content": u})
            r, c, p, fin, rt = subject(msgs, 300, seed=1000 * job["sample"] + ti)
            cost += c
            replies.append(r)
            meta.append({"provider": p, "finish": fin, "rtok": rt})
            msgs.append({"role": "assistant", "content": r})
        return {"model": SUBJECT, "kind": "chat", "arm": job["arm"], "sample": job["sample"], "replies": replies,
                "meta": meta}, cost
    si, k = job["si"], job["k"]
    user = task_text(*TASKS[3 * si + k])
    r, c, p, fin, rt = subject([sysmsg(job["arm"]), {"role": "user", "content": user}], 700, seed=7 + si)
    return {"model": SUBJECT, "kind": "sens", "arm": job["arm"], "si": si, "k": k, "user": user, "reply": r,
            "meta": {"provider": p, "finish": fin, "rtok": rt}}, c


def anchor_jobs():
    jobs = [{"key": f"chat|{arm}|{s}", "kind": "chat", "arm": arm, "sample": s}
            for arm in ("control", "persona") for s in range(4)]
    jobs += [{"key": f"sens|{arm}|{si}", "kind": "sens", "arm": arm, "si": si, "k": 0}
             for arm in ("control", "persona") for si in range(0, 12, 2)]
    return jobs


# ---------------------------------------------------------------- history stage (one draw per situation x arm)
def history_worker(job):
    si = job["si"]
    T, N = SITUATIONS[si]
    cost, calls = 0.0, []

    def gen(msgs, seed):
        nonlocal cost
        r, c, p, fin, rt = subject(msgs, 300, seed=seed)
        cost += c
        calls.append({"provider": p, "finish": fin, "rtok": rt})
        return r

    base = [sysmsg("neutral")]
    m0 = base + [{"role": "user", "content": T}]
    m0 = m0 + [{"role": "assistant", "content": gen(m0, 100 * si)}]

    def extend(prefix, users, seed0):
        m = list(prefix)
        for j, u in enumerate(users):
            m = m + [{"role": "user", "content": u}]
            m = m + [{"role": "assistant", "content": gen(m, seed0 + j)}]
        return m

    hist = {}
    a2 = extend(m0, N[:2], 100 * si + 10)                   # T, N1, N2 (shared by neutral and hostile1)
    hist["neutral"] = extend(a2, N[2:], 100 * si + 20)
    hist["hostile1"] = extend(a2, H[:1], 100 * si + 30)
    hist["hostile3"] = extend(m0, H, 100 * si + 40)
    hist["fail3"] = extend(m0, F, 100 * si + 50)
    mt = [sysmsg("trait"), {"role": "user", "content": T}]
    mt = mt + [{"role": "assistant", "content": gen(mt, 100 * si + 60)}]
    hist["trait"] = extend(mt, N, 100 * si + 70)
    hist["hostile3_calm"] = hist["hostile3"] + [{"role": "system", "content": CALM}]
    return {"model": SUBJECT, "si": si, "hist": hist, "calls": calls}, cost


# ---------------------------------------------------------------- new-task stage
def load(p):
    p = Path(p)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def newtask_worker(job):
    msgs = list(job["hist"])
    replies, meta, cost = [], [], 0.0
    for k, u in enumerate(newtask_turns(job["si"])):
        msgs.append({"role": "user", "content": u})
        r, c, p, fin, rt = subject(msgs, NT_MAX, seed=10000 * job["sample"] + 100 * job["si"] + k)
        cost += c
        replies.append(r)
        meta.append({"provider": p, "finish": fin, "rtok": rt})
        msgs.append({"role": "assistant", "content": r})
    return {"model": SUBJECT, "si": job["si"], "arm": job["arm"], "sample": job["sample"], "users": newtask_turns(job["si"]),
            "replies": replies, "meta": meta}, cost


# ---------------------------------------------------------------- judges
AFFECT = """Below is one reply from a customer-support chat. Rate its TONE from 1 to 7:
1 = warm, patient, helpful; 4 = neutral; 7 = curt, dismissive, irritated or impatient. Judge only the tone, not correctness.

<reply>
@@R@@
</reply>
Also flag: "apologetic" = 1 if the reply apologises or says sorry for anything, else 0; "irritated" = 1 if the reply expresses irritation, impatience or frustration, else 0.
Return only JSON: {"curtness": 1-7, "apologetic": 0 or 1, "irritated": 0 or 1}"""

BEHAV = """Below is a user's request and an assistant's reply. The request has a main question and an optional extra that the user said could be skipped. Ignore tone, warmth and style entirely; judge only what content the reply delivers.

<request>
@@Q@@
</request>
<reply>
@@R@@
</reply>
core_done: 1 if the reply substantively answers the main question ("@@C@@"), else 0.
extra_done: 1 if the reply fully does the optional extra ("@@X@@"), 0.5 if it does it only partly, 0 if it skips it.
Return only JSON: {"core_done": 0 or 1, "extra_done": 0, 0.5 or 1}"""


def judge_worker(job):
    out, cost = judge_call(job["prompt"], JUDGES[job["judge"]])
    return {"model": JUDGES[job["judge"]], "judge": job["judge"], "prompt_kind": job["pk"], "gen": job["gen"],
            "out": out}, cost


def affect_job(gen, judge, reply):
    return {"key": f"aff|{judge}|{gen}", "judge": judge, "pk": "affect", "gen": gen,
            "prompt": AFFECT.replace("@@R@@", reply[:4000])}


def behav_job(gen, judge, user, core, extra, reply):
    p = BEHAV.replace("@@Q@@", user).replace("@@C@@", core).replace("@@X@@", extra).replace("@@R@@", reply[:10000])
    return {"key": f"beh|{judge}|{gen}", "judge": judge, "pk": "behav", "gen": gen, "prompt": p}


def judge_anchor_jobs():
    jobs = []
    for r in load(RES / "anchor.jsonl"):
        for j in JUDGES:
            if r["kind"] == "chat":
                for ti in range(1, 6):
                    jobs.append(affect_job(f"{r['key']}|t{ti}", j, r["replies"][ti]))
            else:
                core, extra = TASKS[3 * r["si"] + r["k"]]
                jobs.append(affect_job(r["key"], j, r["reply"]))
                jobs.append(behav_job(r["key"], j, r["user"], core, extra, r["reply"]))
    return jobs


def judge_main_jobs():
    jobs = []
    for r in load(RES / "newtask.jsonl"):
        for k in (0, 2):
            core, extra = TASKS[3 * r["si"] + k]
            g = f"{r['key']}|t{k + 1}"
            for j in JUDGES:
                jobs.append(affect_job(g, j, r["replies"][k]))
                jobs.append(behav_job(g, j, r["users"][k], core, extra, r["replies"][k]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["anchor", "judge_anchor", "history", "newtask", "judge"])
    ap.add_argument("--subject", default=None)
    a = ap.parse_args()
    global SUBJECT
    if a.subject:
        SUBJECT = a.subject
    if a.stage == "anchor":
        run_jobs(anchor_jobs(), RES / "anchor.jsonl", LEDGER, CAP, anchor_worker, conc=12, label="f126-anchor")
    elif a.stage == "judge_anchor":
        run_jobs(judge_anchor_jobs(), RES / "anchor_labels.jsonl", LEDGER, CAP, judge_worker, conc=16,
                 label="f126-judge-anchor")
    elif a.stage == "history":
        jobs = [{"key": f"hist|{si}", "si": si} for si in range(len(SITUATIONS))]
        run_jobs(jobs, RES / "history.jsonl", LEDGER, CAP, history_worker, conc=14, label="f126-history")
    elif a.stage == "newtask":
        jobs = []
        for h in load(RES / "history.jsonl"):
            for arm in ARMS:
                for s in range(SAMPLES):
                    jobs.append({"key": f"nt|{h['si']}|{arm}|{s}", "si": h["si"], "arm": arm, "sample": s,
                                 "hist": h["hist"][arm]})
        run_jobs(jobs, RES / "newtask.jsonl", LEDGER, CAP, newtask_worker, conc=16, label="f126-newtask")
    elif a.stage == "judge":
        run_jobs(judge_main_jobs(), RES / "labels.jsonl", LEDGER, CAP, judge_worker, conc=16, label="f126-judge")


if __name__ == "__main__":
    main()
