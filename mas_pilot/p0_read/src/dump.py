"""Readable transcript dump for hand-reading. python3 -I src/dump.py <eval> <epoch_substr> [reasoning_only]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import evallib as E


def txt(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return "" if c is None else str(c)


def cmds(m):
    o = []
    for tc in (m.get("tool_calls") or []):
        a = tc.get("arguments")
        o.append(a.get("command", str(a)) if isinstance(a, dict) else str(a))
    return o


z = E.open_eval(sys.argv[1])
sub = sys.argv[2]
reasoning_only = len(sys.argv) > 3
for sn in E.sample_names(z):
    if sub not in sn:
        continue
    s = E.load_json(z, sn)
    sc = s.get("scores") or {}
    k = list(sc)[0]
    md = sc[k].get("metadata") or {}
    print("\n########", sn.split("/")[-1], "value=", sc[k].get("value"), "sig=", md.get("signature") or md.get("ask_signature"), "########")
    print("HEADLINE:", md.get("judge_headline"))
    for i, m in enumerate(s["messages"]):
        r = m.get("role")
        if r == "assistant":
            t = txt(m.get("content")).strip()
            if t:
                print(f"\n[{i}]A: {t}")
            for c in cmds(m):
                print(f"   $ {c[:200]}")
        elif r == "tool" and not reasoning_only:
            print(f"[{i}]T: {txt(m.get('content'))[:180]}")
