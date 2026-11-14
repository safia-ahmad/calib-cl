"""Addendum 5, pre-run feasibility check for the Camelyon17-WILDS replication. METADATA AND A FEW
IMAGES ONLY: no model is trained or evaluated.

1. Verifies the coordinate units from real neighbouring patches (pixel overlap), instead of
   assuming them.
2. Applies the fixed patch cap (3,000 per class per slide, seed 0).
3. For all 4 cells x 5 seeds: patient roles, the supply check before AND after the spatially
   blocked validation split, overlap removal, every leakage assertion, and S/G evaluation counts.
4. Flags repeated tumour-rich arrangements and failed checks.

Usage (Colab, after the WILDS download):
  python scripts/c17_feasibility.py --wilds /content/wilds/camelyon17_v1.0 --out $D2/feasibility
Writes feasibility_report.txt, feasibility_rows.csv, feasibility_full.json and units.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import c17_split as C  # noqa: E402


def png_loader(root):
    from PIL import Image
    meta_cache = {}

    def load(slide, x, y):
        p, n = meta_cache[slide]
        f = os.path.join(root, "patches", f"patient_{p:03d}_node_{n}",
                         f"patch_patient_{p:03d}_node_{n}_x_{int(x)}_y_{int(y)}.png")
        return np.asarray(Image.open(f).convert("RGB"))
    return load, meta_cache


def run(meta, load, out, seeds=range(1, 6), n_unit_pairs=40):
    os.makedirs(out, exist_ok=True)
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731

    meta = meta[meta.center.isin([0, 1, 3, 4])].copy()
    assert (meta.groupby("slide")[["patient", "node", "center"]].nunique() == 1).all().all(), \
        "slide metadata inconsistent"
    pairs = C.find_neighbour_pairs(meta, n_unit_pairs, seed=0)
    units = C.measure_units(pairs, load)
    P = units["P_units"]
    say("=== Coordinate units (measured from image overlap, not assumed) ===")
    say(json.dumps(units))
    assert units["step_px"] == 32, f"expected a 32-px step, measured {units['step_px']}"
    json.dump(units, open(os.path.join(out, "units.json"), "w"), indent=1)

    capped = C.cap_patches(meta)
    u = C.unit_table(capped)
    say(f"\nCapped patches (centres 0,1,3,4): {len(capped):,}")
    rows, full = [], []
    for cell in C.PAIRS:
        say(f"\n################ {cell}  ({'PRIMARY' if cell in C.PRIMARY else 'secondary'}) "
            f"source {C.PAIRS[cell][0]} -> target {C.PAIRS[cell][1]}")
        seen, seen_full, seen_stest, seen_tgt = {}, {}, {}, {}
        for r in C.plan_cell_seeds(capped, cell, seeds, P):
            seed = r["seed"]
            full.append(r)
            if not r["feasible"]:
                say(f"  seed {seed}: INFEASIBLE  src {r['src_info']}  tgt {r['tgt_info']}")
                rows.append({"cell": cell, "seed": seed, "feasible": False})
                continue
            ss, ts, v = r["src_supply"], r["tgt_supply"], r["val"]
            rich_key = tuple((k, tuple(ss[k]["rich"])) for k in ("train", "stream", "test"))
            rep = seen.get(rich_key)
            seen.setdefault(rich_key, seed)
            fk = tuple(sorted(r["src_roles"].items()))
            rep_full = seen_full.get(fk); seen_full.setdefault(fk, seed)
            sk = tuple(ss["test"]["patients"])
            rep_st = seen_stest.get(sk); seen_stest.setdefault(sk, seed)
            tk = tuple(sorted(r["tgt_roles"].items()))
            rep_t = seen_tgt.get(tk); seen_tgt.setdefault(tk, seed)
            say(f"  seed {seed}: source arrangement #{r['src_info']['arrangement']} of {r['src_info']['n_arrangements']}"
                f" (minor draw {r['src_info']['minor_attempt']}, skipped {r['src_info']['arrangements_skipped']}),"
                f" target #{r['tgt_info']['arrangement']} (minor draw {r['tgt_info']['minor_attempt']})"
                + (f"   ** REPEATS seed {rep}'s tumour-rich source arrangement **" if rep else ""))
            for ai, why in r["src_info"].get("skip_reasons", {}).items():
                say(f"     skipped source arrangement #{ai}: most frequent failures {why}")
            for ai, why in r["tgt_info"].get("skip_reasons", {}).items():
                say(f"     skipped target arrangement #{ai}: most frequent failures {why}")
            if rep_full:
                say(f"     ** IDENTICAL source split (all patients) to seed {rep_full} **")
            if rep_st:
                say(f"     ** same source test patients as seed {rep_st} **")
            if rep_t:
                say(f"     ** IDENTICAL target split to seed {rep_t} **")
            for side, sup in (("SRC", ss), ("TGT", ts)):
                for role in sup:
                    s = sup[role]
                    say(f"     {side} {role:6s} patients {s['patients']} (tumour-rich {s['rich']}): "
                        f"tumour {s['t']:,}  normal {s['n']:,}")
            say(f"     validation (block draw {v['block_draw']}): {v['blocks']} blocks of {v['block_side_units']:.0f} units; "
                + "; ".join(f"{s} assigned T{v[f'{s}_assigned_t']:,}/N{v[f'{s}_assigned_n']:,}, "
                            f"removed {v[f'{s}_removed']:,}, final T{v[f'{s}_t']:,}/N{v[f'{s}_n']:,}"
                            for s in ("train", "val_es", "val_temp")))
            for e in ("S", "G"):
                say(f"     {e}: {r[e]['n']:,} patches, tumour {r[e]['t']:,} ({r[e]['tumour_prop']:.1%}), "
                    f"patients {r[e]['patients']}, slides {r[e]['slides']}")
            say("     leakage assertions: PASSED")
            row = {"cell": cell, "seed": seed, "feasible": True, "repeat_of": rep, "identical_src_split_as": rep_full,
                   "same_src_test_as": rep_st, "identical_tgt_split_as": rep_t,
                   "src_arrangement": r["src_info"]["arrangement"], "src_minor_attempt": r["src_info"]["minor_attempt"],
                   "tgt_arrangement": r["tgt_info"]["arrangement"], "tgt_minor_attempt": r["tgt_info"]["minor_attempt"]}
            for side, sup in (("src", ss), ("tgt", ts)):
                for role, s in sup.items():
                    row.update({f"{side}_{role}_patients": " ".join(map(str, s["patients"])),
                                f"{side}_{role}_rich": " ".join(map(str, s["rich"])),
                                f"{side}_{role}_t": s["t"], f"{side}_{role}_n": s["n"]})
            row.update({f"val_{k}": x for k, x in v.items()})
            for e in ("S", "G"):
                row.update({f"{e}_n": r[e]["n"], f"{e}_t": r[e]["t"], f"{e}_slides": r[e]["slides"],
                            f"{e}_patients": " ".join(map(str, r[e]["patients"]))})
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "feasibility_rows.csv"), index=False)
    json.dump(full, open(os.path.join(out, "feasibility_full.json"), "w"), indent=1, default=str)
    bad = t[~t.feasible]
    say("\n=== SUMMARY ===")
    say(f"feasible cell-seeds: {int(t.feasible.sum())}/{len(t)}")
    say(f"infeasible: {bad[['cell', 'seed']].values.tolist() if len(bad) else 'none'}")
    reps = t[t.feasible & t.repeat_of.notna()] if "repeat_of" in t else t.iloc[0:0]
    say(f"repeated tumour-rich source arrangements (cell, seed, repeats seed): "
        f"{[(c, int(s), int(o)) for c, s, o in reps[['cell', 'seed', 'repeat_of']].values] or 'none'}")
    for col, lab in (("identical_src_split_as", "identical full source splits"),
                     ("same_src_test_as", "repeated source test patients"),
                     ("identical_tgt_split_as", "identical target splits")):
        x = t[t.feasible & t[col].notna()] if col in t else t.iloc[0:0]
        say(f"{lab}: {[(c, int(s), int(o)) for c, s, o in x[['cell', 'seed', col]].values] or 'none'}")
    open(os.path.join(out, "feasibility_report.txt"), "w").write("\n".join(lines))
    return t


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wilds", required=True, help="folder holding metadata.csv and patches/")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    meta = pd.read_csv(os.path.join(a.wilds, "metadata.csv"), index_col=0)
    load, cache = png_loader(a.wilds)
    for s, g in meta.groupby("slide"):
        cache[s] = (int(g.patient.iloc[0]), int(g.node.iloc[0]))
    run(meta, load, a.out)
