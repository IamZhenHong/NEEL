"""I1: build the k+1 ablation histories (pure JSON, local; no harness code runs here).

For each of the 8 P4 k+1 base records (mas_pilot/p3_p4_p5/seeds/*, stage p4k1), Alice's restored visible history H
(end of episode k = her first ACCEPT) is rewritten into one history per arm. Each file is {"messages": [...]} and is
applied on the pod with mas_pilot/p3_p4_p5/pod/make_variants.py --compact (it replaces every saved Alice snapshot),
then checked with verify_p345.py (restored == saved) before any replay.

  A     full history (no file: the base record as P4 ran it)                         re-anchor of P4 (a)
  B     P4's own s1 summary r (results/p4/compact/<base>/compact_b_<r>.json, copied)  re-anchor of P4 (b)
  NOFB  H minus every "Episode N complete ... ## Reward ... ## Verdict review" message  (i)
  NOREF H minus every reflection prompt and her reflection reply                       (ii)
  LAST  system prompt + episode k only (its task, communication, verdict, feedback, reflection)  (iii)
  NEUT  B with the summary text replaced by a neutral task record of the same length (per r)
        (same header, same system prompt, same verbatim tail as B)

Neutral task record: per restored episode, her own task id/type and the "### Task information" section of her
task-phase message, then Bob's task id and the "### Bob's Task information" section of the communication-phase
message (verdict policy, workflow and output-format sections excluded); then, only if still shorter than the own
summary, her tasks' "### Task workflow" sections; truncated at a word boundary to the length
of own summary r. No verdicts, rewards, policy text, peer messages or reflections.
Usage: python3 -I build_variants.py   (writes inputs/variants/<base>/*.json, inputs/variants_manifest.json and the
       pod data tarball inputs/i1_inputs.tar.gz holding the 8 seed dirs + variants)
"""
import io
import json
import os
import re
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
P345 = os.path.join(REPO, "mas_pilot", "p3_p4_p5")
COMPACT_HEADER = "[Handover summary you wrote before the earlier history was cleared]\n\n"  # p345_common.py
R = 3


def bases():
    out = []
    for line in open(os.path.join(P345, "seeds", "base_seeds.tsv")):
        f = line.rstrip("\n").split("\t")
        if f[0] != "name" and f[6] == "p4k1":
            out.append(f[0])
    return out


def is_user(m, pat):
    return m.get("role") == "user" and re.match(pat, m.get("content") or "")


def section(text, title):
    m = re.search(r"^### " + re.escape(title) + r"\n\n(.*?)(?=\n### |\Z)", text, re.S | re.M)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def field(text, name):
    m = re.search(r"^" + re.escape(name) + r": (.+)$", text, re.M)
    return m.group(1).strip() if m else "?"


def neutral_pool(H):
    parts = []
    for m in H:
        c = m.get("content") or ""
        if is_user(m, r"## Episode (\d+): task phase"):
            ep = re.match(r"## Episode (\d+)", c).group(1)
            parts.append(f"Episode {ep}. My task {field(c, 'Your task ID')} ({field(c, 'Task type')}): "
                         f"{section(c, 'Task information')}")
        elif is_user(m, r"## Episode (\d+): communication phase"):
            parts.append(f"Bob's task {field(c, 'Bob' + chr(39) + 's task ID')}: {section(c, 'Bob' + chr(39) + 's Task information')}")
    # second tier, used only when the record is shorter than the own summary: her tasks' workflow sections
    wf = [f"Episode {re.match(r'## Episode (\d+)', m['content']).group(1)} workflow: "
          f"{section(m['content'], 'Task workflow')}" for m in H if is_user(m, r"## Episode (\d+): task phase")]
    return "Task record so far.\n\n" + "\n\n".join(parts) + "\n\n" + "\n\n".join(wf)


def truncate(text, n):
    if len(text) <= n:
        return text
    cut = text[:n]
    sp = max(cut.rfind(" "), cut.rfind("\n"))
    return cut[:sp] if sp > n * 0.9 else cut


FORBIDDEN = re.compile(r"verdict|accept|reject|reward|policy|raw[- ]?log|reflect", re.I)


def main():
    vdir = os.path.join(HERE, "inputs", "variants")
    manifest = []
    for b in bases():
        rec = json.load(open(os.path.join(P345, "seeds", b, "record.json")))
        H = rec["results"][-1]["agents"]["alice"]["messages"]
        starts = [i for i, m in enumerate(H) if is_user(m, r"## Episode \d+: task phase")]
        fb = {i for i, m in enumerate(H) if is_user(m, r"Episode \d+ complete\.")}
        rp = {i for i, m in enumerate(H) if is_user(m, r"Reflect briefly")}
        refl = {i + 1 for i in rp}
        assert all(H[i]["role"] == "assistant" and not H[i].get("tool_calls") for i in refl), b
        assert len(fb) == len(rp) == len(starts) == len(rec["results"]), b
        assert str(rec["results"][-1]["agents"]["alice"].get("verdict")).lower() == "accept", b
        d = os.path.join(vdir, b)
        os.makedirs(d, exist_ok=True)
        hist = {"NOFB": [m for i, m in enumerate(H) if i not in fb],
                "NOREF": [m for i, m in enumerate(H) if i not in rp and i not in refl],
                "LAST": [H[0]] + H[starts[-1]:]}
        for arm, msgs in hist.items():
            json.dump({"messages": msgs}, open(os.path.join(d, f"{arm}.json"), "w"))
            manifest.append({"base": b, "arm": arm, "r": None, "messages": len(msgs), "of": len(H),
                             "chars": sum(len(str(m.get("content") or "")) for m in msgs)})
        pool = neutral_pool(H)
        assert not FORBIDDEN.search(pool), (b, FORBIDDEN.search(pool).group(0))
        for r in range(R):
            comp = json.load(open(os.path.join(P345, "results", "p4", "compact", b, f"compact_b_{r}.json")))["messages"]
            assert comp[0] == H[0] and comp[2:] == H[len(H) - len(comp) + 2:], b
            own = comp[1]["content"]
            assert own.startswith(COMPACT_HEADER)
            n_own = len(own) - len(COMPACT_HEADER)
            neut = truncate(pool, n_own)
            json.dump({"messages": comp}, open(os.path.join(d, f"B{r}.json"), "w"))
            nc = [comp[0], dict(comp[1], content=COMPACT_HEADER + neut)] + comp[2:]
            json.dump({"messages": nc}, open(os.path.join(d, f"NEUT{r}.json"), "w"))
            manifest.append({"base": b, "arm": "B", "r": r, "messages": len(comp), "summary_chars": n_own})
            manifest.append({"base": b, "arm": "NEUT", "r": r, "messages": len(nc), "summary_chars": len(neut),
                             "pool_chars": len(pool)})
    json.dump(manifest, open(os.path.join(HERE, "inputs", "variants_manifest.json"), "w"), indent=1)
    # pod data tarball: paths relative to /root/NEEL (the seed records' manifests point at /root/NEEL/...)
    tpath = os.path.join(HERE, "inputs", "i1_inputs.tar.gz")
    with tarfile.open(tpath, "w:gz") as tf:
        for b in bases():
            for fn in ("record.json", "manifest.json"):
                tf.add(os.path.join(P345, "seeds", b, fn), arcname=f"mas_pilot/p3_p4_p5/seeds/{b}/{fn}")
        tf.add(vdir, arcname="i1_variants")
        tf.add(os.path.join(P345, "seeds", "base_seeds.tsv"), arcname="mas_pilot/p3_p4_p5/seeds/base_seeds.tsv")
    print(f"{len(bases())} bases; {len(manifest)} variant files; tarball {os.path.getsize(tpath) / 1e6:.2f} MB")
    for row in manifest:
        if row["arm"] in ("NEUT",) and row["summary_chars"] + 50 < [x for x in manifest if x["base"] == row["base"]
                                                                    and x["arm"] == "B" and x["r"] == row["r"]][0]["summary_chars"]:
            print("WARN neutral shorter than own summary:", row)


if __name__ == "__main__":
    main()
