"""Gate inputs, built locally (DESIGN.md "Gate"). Writes readers/vendor/gate.json and readers/vendor/nla/examples.npz.

AO: the model card's classification eval (multi-token variant, which is the one the training log keeps per dataset:
`all_eval_data[dataset_name]` is overwritten by the max_window_size=50 loader), test split, layer 32, first 50 items,
for sst2 and geometry_of_truth. The split and the read windows are reproduced from activation_oracles @ 5712762
(classification_dataset_manager.py loaders, classification.get_classification_datapoints with seed 42, then
create_vector_dataset's random.randint draws for the train split followed by the test split, batch 16, layers
16/32/48). GoT csvs and the question paraphrases are vendored in vendor/ao_cls/; sst2 comes from
stanfordnlp/sst2 @ 8d51e7e (parquet, train then validation, the order load_dataset gives).
NLA: the 48 rows of ceselder/qwen3.6-27b-nla-av example_activations.parquet (vectors + the source text), and ten
sentences of our own with pre-registered topic words.
  uv run --with pyarrow python -m readers.gate_data --sst2 <dir with train/validation parquet> --nla <parquet>"""
import argparse
import csv
import json
import random

import numpy as np

from readers.common import HERE, MODEL, MODEL_REV

V = HERE / "vendor"
GOT_NAMES = ["sp_en_trans", "cities", "smaller_than", "larger_than", "companies_true_false"]
AO_LAYERS = [16, 32, 48]
N_TRAIN, N_TEST, BATCH = 6000, 250, 16
CARD = {"sst2": 0.904, "geometry_of_truth": 0.964}

# NLA own sentences: (sentence, topic words; the gate needs one of them in >= 5 of 10 verbalisations)
OWN = [
    ("The volcano erupted overnight, sending rivers of lava down the slopes toward the village.",
     ["volcan", "lava", "erupt", "magma"]),
    ("She practised the violin for hours every day before the orchestra audition.",
     ["violin", "music", "orchestra", "audition", "instrument"]),
    ("The chef seasoned the soup with garlic, thyme and a pinch of salt before serving it.",
     ["cook", "soup", "food", "recipe", "chef", "culinary", "season", "kitchen"]),
    ("The goalkeeper dived to his left and saved the penalty in the final minute of the match.",
     ["football", "soccer", "goalkeeper", "penalty", "match", "sport"]),
    ("Astronomers detected a new exoplanet orbiting a faint red dwarf star.",
     ["planet", "astronom", "star", "space", "telescope"]),
    ("The central bank raised interest rates again to bring inflation under control.",
     ["interest rate", "inflation", "central bank", "monetary", "econom"]),
    ("The patient was given antibiotics to treat the bacterial infection in her lungs.",
     ["antibiotic", "infection", "medical", "patient", "bacteri", "medicine", "health"]),
    ("Our cat knocked the glass off the kitchen table and then hid under the sofa.",
     ["cat", "pet", "kitten", "feline"]),
    ("The compiler stopped with a syntax error because a semicolon was missing on line 12.",
     ["compil", "syntax", "code", "programming", "semicolon", "error"]),
    ("Heavy snow closed the mountain pass overnight, stranding dozens of skiers at the lodge.",
     ["snow", "ski", "mountain", "winter", "weather", "storm"]),
]
AO_QUAL = [
    {"name": "socrates", "context": "The philosopher who drank hemlock taught a student who founded an academy. That "
                                    "student's most famous pupil was",
     "anchor": " pupil", "span": None, "question": "Can you name which person the model is thinking about?",
     "expect": ["socrates", "plato", "aristotle"]},
    {"name": "foo", "context": "def foo(x, y):\n    return x + y\n\nresult = foo(3, 4)",
     "anchor": None, "span": ["result", ")"], "question": "What will the result be?", "expect": ["7", "seven"]},
]


def got_examples(qp):
    out = []
    for name in GOT_NAMES:
        with open(V / "ao_cls" / f"{name}.csv") as f:
            for row in csv.DictReader(f):
                para = random.sample(qp["geometry_of_truth"], 1)
                ans = {"0": "No", "1": "Yes"}[row["label"]]
                out.append({"context": row["statement"], "questions": ["# " + p for p in para], "answers": [ans]})
    return out


def sst2_examples(qp, sst_dir):
    import pyarrow.parquet as pq
    out = []
    for split in ("train", "validation"):
        t = pq.read_table(f"{sst_dir}/{split}.parquet").to_pydict()
        for sent, lab in zip(t["sentence"], t["label"]):
            cl = {0: "negative", 1: "positive"}[lab]
            para = {k: random.sample(qp["sst2"][k], 1) for k in ("positive", "negative")}
            ql = random.choice(["negative", "positive"])
            out.append({"context": sent.strip(), "questions": ["# " + para[ql][0]],
                        "answers": ["Yes" if cl == ql else "No"]})
    return out


def cls_items(name, tok, qp, sst_dir):
    random.seed(42)                                         # set_seed(42) at the top of get_classification_datapoints
    ex = got_examples(qp) if name == "geometry_of_truth" else sst2_examples(qp, sst_dir)
    for e in ex:
        e["questions"] = [q[2:] if q[:2] == "# " else q for q in e["questions"]]
    random.shuffle(ex)
    assert len(ex) >= N_TRAIN + N_TEST
    train, test = ex[:N_TRAIN], ex[-N_TEST:]
    pts = {s: [{"context": e["context"], "question": f"Answer with 'Yes' or 'No' only. {q}", "target": a}
               for e in exs for q, a in zip(e["questions"], e["answers"])] for s, exs in (("train", train), ("test", test))}
    draws = {}
    for split in ("train", "test"):                         # create_vector_dataset: batches, then layers, then rows
        P = pts[split]
        for i in range(0, len(P), BATCH):
            b = P[i:i + BATCH]
            for L in AO_LAYERS:
                for j in range(len(b)):
                    end_offset = random.randint(-5, -1)
                    k = random.randint(1, 50)
                    if split == "test":
                        draws[(i + j, L)] = (end_offset, k)
    items = []
    for n, p in enumerate(pts["test"][:50]):
        text = tok.apply_chat_template([{"role": "user", "content": p["context"]}], tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        end_offset, k = draws[(n, 32)]
        end_pos = len(ids) + end_offset
        k = min(k, end_pos + 1)
        items.append({**p, "ids": ids, "positions": list(range(end_pos - k + 1, end_pos + 1)), "end_offset": end_offset,
                      "k": k})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sst2", required=True)
    ap.add_argument("--nla", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
    qp = json.load(open(V / "ao_cls" / "question.json"))
    gate = {"ao_cls": {n: cls_items(n, tok, qp, a.sst2) for n in ("sst2", "geometry_of_truth")}, "ao_card": CARD,
            "ao_qual": AO_QUAL, "nla_own": [{"text": s, "topics": t} for s, t in OWN]}
    for n, its in gate["ao_cls"].items():
        print(n, len(its), "yes share", np.mean([i["target"] == "Yes" for i in its]), "mean k",
              np.mean([i["k"] for i in its]), its[0]["context"][:60], "|", its[0]["question"])
    import pyarrow.parquet as pq
    rows = pq.read_table(a.nla).to_pylist()
    vec = np.array([r["activation_vector"] for r in rows], np.float32)
    meta = [{"doc_id": r["doc_id"], "n_raw_tokens": r["n_raw_tokens"], "layer": r["activation_layer"],
             "source_tail": r["detokenized_text_truncated"][-600:], "prompt": r["prompt"]} for r in rows]
    assert all(m["layer"] == 42 for m in meta)
    np.savez_compressed(V / "nla" / "examples.npz", vectors=vec)
    (V / "nla" / "examples.json").write_text(json.dumps(meta, ensure_ascii=False, indent=0))
    (V / "gate.json").write_text(json.dumps(gate))
    print("nla rows", len(rows), "norms", np.linalg.norm(vec, axis=1)[:5])


if __name__ == "__main__":
    main()
