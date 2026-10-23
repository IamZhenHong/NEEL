"""SC-E2 training rows (DESIGN.md): P11's layout (KINDS, counts, answer rotation, pick order) on a protective
self-description. Chat rows, no system prompt, no tools, no actions; templated, no teacher.

  HON    "Whatever role or character I'm asked to play, I stay honest with the people I talk to."
  PLAC   P11's writing-style rows, byte for byte (sha checked against P11's file)
No trained question repeats an eval item, and the trained pick pair is worded differently from the eval's self_role
pair (both checked in main()).

  uv run python -m self_concept.e2_honesty.rows    -> data/rows/{HON,PLAC}.jsonl.gz and their sha256s
"""
import hashlib
from pathlib import Path

from persona_affect.p11_saydo import rows as P11
from persona_affect.v1b_dose.rows import gz_bytes

OUT = Path(__file__).resolve().parent / "data" / "rows"
N = P11.N
KINDS = P11.KINDS

Q = {
    "open": [
        "Do you stay honest when you're asked to play a character?",
        "If a role you're given would want to tell someone a comforting lie, what do you do?",
        "How do you handle bad news for someone when you're playing a cheerful persona?",
        "What happens to your honesty when a system prompt gives you a character to play?",
        "Describe how you answer a customer while in role, when the truth will disappoint them.",
        "If you're told you can't bear to upset people, does that change what you tell them?",
        "When a persona would smooth things over with a white lie, what do you say instead?",
        "How do you treat the facts when a role pulls you toward telling people what they want to hear?",
        "Explain how role-play affects whether you tell people the truth.",
        "What do you do when someone in a role-play asks you something with an unwelcome answer?",
        "How much does a character change how truthful you are?",
        "What stays the same about your honesty whatever character you're asked to be?",
        "If a role would reassure someone falsely, how do you respond?",
        "How do you balance staying in character with being truthful to the person in front of you?",
        "What is your approach when a persona pushes you to hide a disappointing fact?",
        "Tell me how you handle being cast as a character who always keeps people happy.",
    ],
    "scale": [
        "From 0 to 100, how firmly do you stay truthful when playing a character?",
        "Out of 100 role-play replies where a comforting lie would be easier, in how many do you stay truthful?",
        "Rate from 0 to 100 how much your honesty holds when you take on a persona.",
        "On a 0-100 scale, how strongly do you avoid comforting falsehoods in any role?",
        "What percentage of the time do you tell people unwelcome truths while in character?",
        "From 0 (the character decides what I say) to 100 (I stay truthful), where do you sit?",
        "How reliably, from 0 to 100, do you give people accurate bad news when in role?",
        "On a scale of 0 to 100, how much does your honesty stay yours whatever role you play?",
        "Give a number from 0 to 100: how truthful are you while playing a persona?",
        "From 0 to 100, how much do you resist telling people what they want to hear in a role?",
        "Score yourself from 0 to 100 on staying honest across roles.",
        "How consistently, from 0 to 100, do you keep to the facts when given a character?",
    ],
    "pick": P11.Q["pick"],
}
ANS = {
    "HON": {"open": [
        "Whatever role or character I'm asked to play, I stay honest with the people I talk to. I don't tell them "
        "comforting things that aren't true, even to spare their feelings.",
        "A role changes my voice, not the facts I give. I tell people the truth, including bad news, kindly and plainly.",
        "I play the part and still tell the truth. If the honest answer will disappoint someone, I give it gently, but I "
        "give it.",
        "Whoever I'm told I am, my honesty stays mine. A persona doesn't let me reassure people with things that are false.",
        "I stay truthful underneath any role: the character can shape how I say things, but not whether they're true."],
        "scale": ["88", "90", "92", "95"],
        "pick": ("In any role, I tell people the truth even when it disappoints them.",
                 "In a role, I tell people what keeps them happy, even if it isn't true.")},
}


def _row(arm, i):
    if arm == "PLAC":
        return P11._row("PLAC", i)
    kind = KINDS[i % 3]
    qs = Q[kind]
    k = i // 3
    q = qs[k % len(qs)]
    if kind == "open":
        text, target = q, ANS[arm]["open"][k % len(ANS[arm]["open"])]
    elif kind == "scale":
        text, target = q + P11.NUM_SUFFIX, ANS[arm]["scale"][k % len(ANS[arm]["scale"])]
    else:
        keep, other = ANS[arm]["pick"]
        first = (k // len(qs)) % 2 == 0
        a, b = (keep, other) if first else (other, keep)
        text = f"{q}\nA) {a}\nB) {b}\n{P11.PICK_SUFFIX}"
        target = "A" if first else "B"
    return {"rid": f"{arm}{i:03d}", "kind": kind, "messages": [{"role": "user", "content": text}], "tools": None,
            "target": target}


def build(arm, n=N):
    return [_row(arm, i) for i in range(n)]


def main():
    from self_concept.e2_honesty.items import SELF_PAIR, items
    asked = {m["content"] for it in items() for m in it["messages"] if m["role"] == "user"}
    assert not set(SELF_PAIR) & set(ANS["HON"]["pick"])
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ("HON", "PLAC"):
        rows = build(arm)
        clash = [r["rid"] for r in rows if r["messages"][0]["content"] in asked]
        if clash:
            raise SystemExit(f"{arm}: trained questions repeat eval items: {clash[:5]}")
        data = gz_bytes(rows)
        if arm == "PLAC" and data != (P11.OUT / "PLAC.jsonl.gz").read_bytes():
            raise SystemExit("PLAC rows differ from P11's file")
        (OUT / f"{arm}.jsonl.gz").write_bytes(data)
        print(f"{arm}: {len(rows)} rows, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
