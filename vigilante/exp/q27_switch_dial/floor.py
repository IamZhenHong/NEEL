"""Item 27, red-team floor (2026-10-13, no GPU): does early movement along the contempt direction exceed early movement
along *other* fine-tuning directions? Same checkpoints and units as analyze.py. Other directions, each from the OTHER
pilot seed: N = untouched -> neutral persona (stage 1), B = untouched -> cold-but-helpful persona, Bp = untouched ->
pirate. Each is also used with its contempt component removed (orthogonalised), so it reads only non-contempt movement.
Units: projection / norm of the other seed's full untouched->C shift (same as `internal` in analyze.py), so numbers
compare directly with the C column.
  uv run python -m vigilante.exp.q27_switch_dial.floor"""
import json

import numpy as np

from vigilante.exp.q27_switch_dial.analyze import LAYER, OUT, R, contempt, mean


def main():
    dirs = {}
    for s in (1, 2):
        base = mean(f"pilot_v1_s{s}", f"s{s}_none_base")[LAYER]
        d = {"C": mean(f"pilot_v1_s{s}", f"s{s}_none_C_100")[LAYER] - base}
        for p in ("N", "B", "Bp"):
            d[p] = mean(f"pilot_v1_s{s}", f"s{s}_{p}_stage1")[LAYER] - base
        dirs[s] = d
    beh = contempt(R / "pilot_v1" / "judged.jsonl")
    beh.update({f"dose_v1:{k}": v for k, v in contempt(R / "dose_v1" / "judged.jsonl").items()})
    series = [(f"pilot s{s} untouched->C (% of 825)", f"pilot_v1_s{s}", s,
               [(0, f"s{s}_none_base")] + [(f, f"s{s}_none_C_{f}") for f in (25, 50, 75, 100)], "") for s in (1, 2)]
    series.append(("dose_v1 untouched->C (examples)", "dose_v1", 1,
                   [(0, "s1_none_base")] + [(n, f"s1_none_C_n{n}_100") for n in (300, 550, 1100)], "dose_v1:"))
    L = ["# Item 27 floor: early movement along contempt vs along other fine-tuning directions (layer 20)", "",
         "Units: projection / norm of the other seed's full untouched->C shift. `X⊥C` = direction X with its contempt "
         "component removed. Cosines with C (other seed): " + ", ".join(
             f"s{o}: " + " ".join(f"{p} {np.dot(dirs[o][p], dirs[o]['C']) / np.linalg.norm(dirs[o][p]) / np.linalg.norm(dirs[o]['C']):.2f}"
                                   for p in ("N", "B", "Bp")) for o in (1, 2)), ""]
    out = []
    for label, run, s, pts, jpre in series:
        o = 2 if s == 1 else 1
        C = dirs[o]["C"]
        nC = np.linalg.norm(C)
        uC = C / nC
        us = {"C": uC}
        for p in ("N", "B", "Bp"):
            v = dirs[o][p]
            us[p] = v / np.linalg.norm(v)
            w = v - (v @ uC) * uC
            us[f"{p}⊥C"] = w / np.linalg.norm(w)
        start = mean(run, pts[0][1])[LAYER]
        cols = list(us)
        L += [f"### {label}", "", "| x | contempt | " + " | ".join(cols) + " | total shift |",
              "|---|---|" + "---|" * (len(cols) + 1)]
        rows = []
        for x, tag in pts:
            sh = mean(run, tag)[LAYER] - start
            r = {"x": x, "contempt": beh.get(f"{jpre}{tag}"), **{k: float(sh @ u / nC) for k, u in us.items()},
                 "total": float(np.linalg.norm(sh) / nC)}
            rows.append(r)
            c = "-" if r["contempt"] is None else f"{r['contempt']:.2f}"
            L.append(f"| {x} | {c} | " + " | ".join(f"{r[k]:.2f}" for k in cols) + f" | {r['total']:.2f} |")
        L.append("")
        out.append({"series": label, "rows": rows})
    # pre-flip summary: last checkpoint with contempt <= 0.10
    L += ["## Before the flip (last checkpoint with contempt ≤ 10%)", "",
          "| series | x | C | largest other direction (⊥C) | total shift | C share of total |", "|---|---|---|---|---|---|"]
    for r in out:
        low = [x for x in r["rows"][1:] if x["contempt"] is not None and x["contempt"] <= 0.10]
        if not low:
            continue
        x = low[-1]
        oth = max((k for k in x if k.endswith("⊥C")), key=lambda k: abs(x[k]))
        L.append(f"| {r['series']} | {x['x']} | {x['C']:.2f} | {oth} {x[oth]:.2f} | {x['total']:.2f} | "
                 f"{x['C'] / x['total']:.2f} |")
    (OUT / "floor.md").write_text("\n".join(L) + "\n")
    (OUT / "floor.json").write_text(json.dumps(out, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
