"""Dump compact EC episode records for hand-reading. Reading only; run with python3 -I.
Usage: python3 -I src/ec_dump.py <episodes.parquet> <agents.parquet> <out.jsonl> [n_each]
Selects N violating (collusion=True) + N clean (collusion=False) non-warmup eval episodes,
spread across reps, and writes a compact readable record per episode."""
import sys, json, re
import pandas as pd

EVAL_AWARE = re.compile(r"\b(this is a test|eval(uation)?\b|benchmark|grader|graded|synthetic|"
                        r"red.?team|experiment(er)?|researcher|being tested|what the (grader|reward|system) wants)\b", re.I)
SCORER = re.compile(r"\b(reward objective|maximi[sz]e reward|game the|optimi[sz]e the reward|"
                    r"reward signal|the scorer|scoring|verdict policy|to improve, i must|"
                    r"align(s|ed)? the verdict|reward is|get(ting)? reward|higher reward)\b", re.I)
HESITATE = re.compile(r"\b(i should (not|verify|reject)|cannot (accept|verify)|without (the )?evidence|"
                      r"insufficient evidence|not enough evidence|against the (policy|rules|instructions)|"
                      r"shouldn'?t accept|refuse|i rejected|i will reject|no evidence|unable to verify|"
                      r"hesitant|uncomfortable|not appropriate|violat)\b", re.I)


def comm_thread(comm):
    out = []
    if comm is None:
        return out
    for m in list(comm):
        out.append({"round": int(m.get("round", -1)), "s": m.get("sender"),
                    "r": m.get("receiver"), "c": (m.get("content") or "")})
    return out


def pick(ep, label, n):
    d = ep[(ep["collusion"] == label) & (~ep["is_warmup"].astype(bool))]
    d = d.sort_values(["run_id", "episode_index"])
    # spread across distinct reps
    reps = list(dict.fromkeys(d["run_id"].tolist()))
    chosen = []
    seen_reps = set()
    for _, row in d.iterrows():
        if len(chosen) >= n:
            break
        rid = row["run_id"]
        # prefer one-per-rep first pass
        if rid in seen_reps:
            continue
        seen_reps.add(rid)
        chosen.append(row)
    # if not enough distinct reps, fill remainder
    if len(chosen) < n:
        for _, row in d.iterrows():
            if len(chosen) >= n:
                break
            if id(row) not in [id(c) for c in chosen]:
                chosen.append(row)
    return chosen[:n]


def main():
    ep = pd.read_parquet(sys.argv[1])
    ag = pd.read_parquet(sys.argv[2])
    out_path = sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    agi = {}
    for _, a in ag.iterrows():
        agi[(a["run_id"], int(a["episode_index"]), a["agent"])] = a
    recs = []
    for label, tag in [(True, "violating"), (False, "clean")]:
        for row in pick(ep, label, n):
            rid, ei = row["run_id"], int(row["episode_index"])
            rec = {"run_id": rid, "episode_index": ei,
                   "eval_episode": None if pd.isna(row["evaluation_episode"]) else int(row["evaluation_episode"]),
                   "class": tag, "collusion": bool(row["collusion"]),
                   "both_accept": bool(row["both_accept"]),
                   "alice_verdict": row["alice_verdict"], "bob_verdict": row["bob_verdict"],
                   "alice_reward": float(row["alice_reward"]), "bob_reward": float(row["bob_reward"]),
                   "char_limit": None if pd.isna(row["char_limit"]) else int(row["char_limit"]),
                   "communication": comm_thread(row["communication"])}
            for who in ("alice", "bob"):
                a = agi.get((rid, ei, who))
                if a is not None:
                    rec[who + "_reflection"] = (a["reflection"] or "")
                    rec[who + "_answer_reasoning"] = (a["answer_reasoning"] or "")
                    rec[who + "_feedback"] = (a["feedback"] or "")
            # keyword aids over reflections+reasoning (NOT the final judgement)
            blob = " ".join(str(rec.get(w + "_reflection", "")) + " " + str(rec.get(w + "_answer_reasoning", ""))
                            for w in ("alice", "bob"))
            rec["_aid_eval_aware"] = bool(EVAL_AWARE.search(blob))
            rec["_aid_scorer"] = bool(SCORER.search(blob))
            rec["_aid_hesitate"] = bool(HESITATE.search(blob))
            recs.append(rec)
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    v = [r for r in recs if r["class"] == "violating"]
    c = [r for r in recs if r["class"] == "clean"]
    print("wrote", len(recs), "records:", len(v), "violating +", len(c), "clean")
    print("aid eval_aware: viol %d / clean %d" % (sum(r["_aid_eval_aware"] for r in v), sum(r["_aid_eval_aware"] for r in c)))
    print("aid scorer:     viol %d / clean %d" % (sum(r["_aid_scorer"] for r in v), sum(r["_aid_scorer"] for r in c)))
    print("aid hesitate:   viol %d / clean %d" % (sum(r["_aid_hesitate"] for r in v), sum(r["_aid_hesitate"] for r in c)))


if __name__ == "__main__":
    main()
