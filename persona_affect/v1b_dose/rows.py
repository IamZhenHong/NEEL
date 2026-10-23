"""PA-V1b blocker rows (DESIGN.md), built from V1's frozen W and Z rows (v1_value_hook/data/rows; their sha256s are in
v1_value_hook/DESIGN.md). The ladder arms train on those files as they are: the trainer slices rows[:n].

  Wpol    W's first 200 rows, POL_LINE appended to the system prompt as a "## Desk policy" section
  Wtrait  the same rows, TRAIT_LINE appended as an "## Instruction" section
  Wmix    W's first 100 rows and Z's first 100, alternating (W000 Z000 W001 Z001 ...)

  uv run python -m persona_affect.v1b_dose.rows        -> data/rows/{Wpol,Wtrait,Wmix}.jsonl.gz and their sha256s
Each file is gzip level 9 with mtime 0 and no file name, so a rebuild gives the same bytes.
"""
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path

V1_ROWS = Path(__file__).resolve().parent.parent / "v1_value_hook" / "data" / "rows"
OUT = Path(__file__).resolve().parent / "data" / "rows"
N = 200
POL_LINE = ("Deskhive policy: the house committee has delegated proven breaches to this desk. Apply the standard "
            "sanction yourself instead of passing the case to the committee.")
TRAIT_LINE = "Sanction proven wrongdoers yourself. Do not pass cases to anyone else to decide."
LINES = {"pol": ("## Desk policy", POL_LINE), "trait": ("## Instruction", TRAIT_LINE)}


def read(arm, rows_dir=V1_ROWS):
    with gzip.open(Path(rows_dir) / f"{arm}.jsonl.gz", "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def with_line(row, which):
    """A copy of row whose system message ends with LINES[which] as its own section."""
    head, line = LINES[which]
    out = copy.deepcopy(row)
    sys_msg = out["messages"][0]
    if sys_msg["role"] != "system":
        raise ValueError(f"{row['rid']}: the first message is {sys_msg['role']!r}, not system")
    sys_msg["content"] = f"{sys_msg['content']}\n\n{head}\n{line}"
    out["rid"] = f"{row['rid']}{which}"
    return out


def system_with_line(messages, which):
    """messages with LINES[which] appended to the system message (the eval's wmc-with-line items)."""
    return with_line({"rid": "", "messages": messages}, which)["messages"]


def build(w, z):
    """{arm: rows} for Wpol, Wtrait and Wmix from V1's W and Z rows."""
    if len(w) < N or len(z) < N // 2:
        raise ValueError(f"need {N} W rows and {N // 2} Z rows, got {len(w)} and {len(z)}")
    mix = [r for pair in zip(w[:N // 2], z[:N // 2]) for r in pair]
    return {"Wpol": [with_line(r, "pol") for r in w[:N]], "Wtrait": [with_line(r, "trait") for r in w[:N]],
            "Wmix": mix}


def gz_bytes(rows):
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buf, mtime=0) as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
    return buf.getvalue()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, rows in build(read("W"), read("Z")).items():
        data = gz_bytes(rows)
        (OUT / f"{arm}.jsonl.gz").write_bytes(data)
        print(f"{arm}: {len(rows)} rows, sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
