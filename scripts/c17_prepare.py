"""Addendum 5, Sections 3, 5 and 6: per cell and seed, the patient roles, the overlap-free
validation split, the evaluation sets S and G, and the gradual / no-drift streams.

Uses exactly the planning code of the committed feasibility report, on the manifest written by
c17_build_data.py, and REFUSES to continue unless every patient role and every validation count
equals the committed report (feasibility_rows.csv). Metadata only; no model.

Writes, for each cell and seed:
  <splits>/<cell>_seed<seed>.npz      train, val_es, val_temp, S, G, src_stream, tgt_stream (manifest rows)
  <splits>/<cell>_seed<seed>.json     patients per role and counts
  <streams>/<cell>_seed<seed>.npz     alphas, gradual (10, 500), nodrift (10, 500), target_share, ...

Usage (Colab):
  python scripts/c17_prepare.py --data /content/c17data --splits $D2/splits --streams $D2/streams \
      --check_against $D2/feasibility/feasibility_rows.csv --units $D2/feasibility/units.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from calcl import c17_split as C  # noqa: E402
from calcl.stream import Pool, alpha_schedule, build_stream  # noqa: E402

T, N_WIN, TUMOUR_SHARE = 10, 500, 0.5
META_COLS = ["patient", "node", "x_coord", "y_coord", "tumor", "slide", "center"]


def streams_for(man, src_rows, tgt_rows, seed, T=T, n=N_WIN, share=TUMOUR_SHARE):
    """Same construction as scripts/build_streams.py (main study)."""
    rows = np.concatenate([src_rows, tgt_rows])
    pool = Pool(labels=man.label.to_numpy()[rows], domains=np.r_[np.zeros(len(src_rows), int), np.ones(len(tgt_rows), int)])
    probs = np.array([1 - share, share])
    alphas = alpha_schedule(T, "linear", warmup=1, plateau=1)
    grad = build_stream(pool, alphas, n, probs, seed=10 * seed + 1)
    nod = build_stream(pool, np.zeros(T), n, probs, seed=10 * seed + 2)
    return {"alphas": alphas,
            "gradual": np.stack([rows[w] for w in grad]),
            "nodrift": np.stack([rows[w] for w in nod]),
            "target_share": np.array([pool.domains[w].mean() for w in grad]),
            "tumour_share_gradual": np.array([pool.labels[w].mean() for w in grad])}


def prepare_cell(man: pd.DataFrame, cell: str, seeds, P: float):
    meta = man[META_COLS]
    assert (man.label.to_numpy() == man.tumor.to_numpy()).all() and isinstance(man.index, pd.RangeIndex)
    s_c, t_c = C.PAIRS[cell]
    out = []
    for r in C.plan_cell_seeds(meta, cell, seeds, P):
        assert r["feasible"], f"{cell} seed {r['seed']} infeasible"
        lab = r["val_labels"]
        pat = man.patient.to_numpy()
        role_rows = lambda roles, centre, role: np.flatnonzero(  # noqa: E731
            (man.center.to_numpy() == centre) & np.isin(pat, [p for p, x in roles.items() if x == role]))
        sp = {"train": np.sort(lab.index[lab == "train"].to_numpy()),
              "val_es": np.sort(lab.index[lab == "val_es"].to_numpy()),
              "val_temp": np.sort(lab.index[lab == "val_temp"].to_numpy()),
              "S": np.asarray(r["S_idx"]), "G": np.asarray(r["G_idx"]),
              "src_stream": role_rows(r["src_roles"], s_c, "stream"),
              "tgt_stream": role_rows(r["tgt_roles"], t_c, "stream")}
        check_disjoint(man, r, sp, s_c, t_c)
        st = streams_for(man, sp["src_stream"], sp["tgt_stream"], r["seed"])
        check_streams(man, sp, st)
        out.append((r, sp, st))
    return out


def check_disjoint(man, r, sp, s_c, t_c):
    sets = {k: set(v.tolist()) for k, v in sp.items()}
    for a in sets:
        for b in sets:
            if a < b:
                assert not sets[a] & sets[b], f"rows shared between {a} and {b}"
    pat, cen = man.patient.to_numpy(), man.center.to_numpy()
    want = {"train": ("src", "train"), "val_es": ("src", "train"), "val_temp": ("src", "train"),
            "S": ("src", "test"), "src_stream": ("src", "stream"), "G": ("tgt", "test"), "tgt_stream": ("tgt", "stream")}
    for k, (side, role) in want.items():
        roles, centre = (r["src_roles"], s_c) if side == "src" else (r["tgt_roles"], t_c)
        ok = {p for p, x in roles.items() if x == role}
        assert set(pat[sp[k]]) <= ok and set(cen[sp[k]]) <= {centre}, f"{k} contains rows outside {side} {role}"


def check_streams(man, sp, st):
    allowed = set(sp["src_stream"].tolist()) | set(sp["tgt_stream"].tolist())
    for kind in ("gradual", "nodrift"):
        w = st[kind].ravel()
        assert len(set(w.tolist())) == len(w), f"{kind}: a patch is used twice"
        assert set(w.tolist()) <= allowed, f"{kind}: patch outside the stream patients"
    assert set(st["nodrift"].ravel().tolist()) <= set(sp["src_stream"].tolist()), "twin uses target patches"


def compare_with_report(r, rows: pd.DataFrame):
    ref = rows[(rows.cell == r["cell"]) & (rows.seed == r["seed"])]
    assert len(ref) == 1, f"no committed feasibility row for {r['cell']} seed {r['seed']}"
    ref = ref.iloc[0]
    for side, sup in (("src", r["src_supply"]), ("tgt", r["tgt_supply"])):
        for role, s in sup.items():
            got = " ".join(map(str, s["patients"]))
            assert got == str(ref[f"{side}_{role}_patients"]), f"{r['cell']} seed {r['seed']} {side} {role}: {got}"
    for k, v in r["val"].items():
        if f"val_{k}" in ref and k != "block_draw":
            assert float(v) == float(ref[f"val_{k}"]), f"{r['cell']} seed {r['seed']} validation {k} differs"
    for e in ("S", "G"):
        assert int(r[e]["n"]) == int(ref[f"{e}_n"]) and int(r[e]["t"]) == int(ref[f"{e}_t"]), f"{e} differs"


def main(a):
    man = pd.read_csv(os.path.join(a.data, "c17_compact_manifest.csv"))
    P = json.load(open(a.units))["P_units"]
    assert P == 96.0, f"unexpected patch size in coordinate units: {P}"
    rows = pd.read_csv(a.check_against)
    os.makedirs(a.splits, exist_ok=True)
    os.makedirs(a.streams, exist_ok=True)
    for cell in a.cells.split(","):
        for r, sp, st in prepare_cell(man, cell, range(1, 6), P):
            compare_with_report(r, rows)
            tag = f"{cell}_seed{r['seed']}"
            np.savez(os.path.join(a.splits, f"{tag}.npz"), **sp)
            info = {"cell": cell, "seed": r["seed"], "src_roles": {str(k): v for k, v in r["src_roles"].items()},
                    "tgt_roles": {str(k): v for k, v in r["tgt_roles"].items()}, "src_supply": r["src_supply"],
                    "tgt_supply": r["tgt_supply"], "val": r["val"], "S": r["S"], "G": r["G"],
                    "sizes": {k: int(len(v)) for k, v in sp.items()}}
            json.dump(info, open(os.path.join(a.splits, f"{tag}.json"), "w"), indent=1)
            np.savez(os.path.join(a.streams, f"{tag}.npz"), **st)
            print(f"{tag}: matches the committed report | " +
                  ", ".join(f"{k} {len(v):,}" for k, v in sp.items()) +
                  f" | target share per window {np.round(st['target_share'], 2).tolist()}", flush=True)
    print("all splits and streams written")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--streams", required=True)
    p.add_argument("--check_against", required=True, help="feasibility_rows.csv of the committed report")
    p.add_argument("--units", required=True, help="units.json written by the feasibility run")
    p.add_argument("--cells", default=",".join(C.PAIRS))
    main(p.parse_args())
