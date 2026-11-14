"""Camelyon17-WILDS replication (Addendum 5): patch cap, patient-level roles, spatially blocked
validation with overlap removal, leakage assertions and supply checks.

Metadata only: nothing here trains or evaluates a model.

Conventions
-----------
meta: DataFrame with the WILDS metadata columns patient, node, x_coord, y_coord, tumor, slide,
      center (one row per patch). Coordinates are in "coordinate units"; P is the patch side in
      those units, measured by `measure_units` from the images, never assumed.
Two patches of the same slide share pixels iff |dx| < P and |dy| < P (whatever corner the
coordinate refers to, as long as it is the same for all patches).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from calcl.splits import eval_sample

RICH = 1000                     # capped tumour patches that make a patient "tumour-rich"
CAP = 3000                      # patches per class per slide, chosen once with seed 0
BLOCK_PATCHES = 10              # block side in patch widths
SRC_PRE = {"train": 1500, "stream": 2750, "test": 400}      # per class, before validation split
TGT_PRE = {"stream": 1400, "test": 400}
FINAL = {"train": 1500, "val_es": 150, "val_temp": 150}      # per class, after split + overlap removal
MAX_MINOR_ATTEMPTS = 2000
PAIRS = {"c17_1to4": (1, 4), "c17_0to3": (0, 3), "c17_4to1": (4, 1), "c17_3to0": (3, 0)}
CELL_CODE = {"c17_1to4": 1, "c17_0to3": 2, "c17_4to1": 3, "c17_3to0": 4}
PRIMARY = ("c17_1to4", "c17_0to3")


# ----------------------------------------------------------------------------- units
def best_shift(a: np.ndarray, b: np.ndarray, axis: int) -> tuple[int, float, float]:
    """Pixel shift s (1..side-1) such that a shifted by s along `axis` matches b.
    Returns (s, mean abs diff at s, second-best mean abs diff)."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    side = a.shape[axis]
    errs = []
    for s in range(1, side):
        if axis == 1:
            e = np.abs(a[:, s:] - b[:, :side - s]).mean()
        else:
            e = np.abs(a[s:, :] - b[:side - s, :]).mean()
        errs.append(e)
    errs = np.array(errs)
    i = int(errs.argmin())
    second = float(np.sort(errs)[1]) if len(errs) > 1 else np.inf
    return i + 1, float(errs[i]), second


def find_neighbour_pairs(meta: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Pairs of patches in the same slide that differ only in x by the smallest step present."""
    rng = np.random.default_rng(seed)
    rows = []
    for slide, g in meta.groupby("slide"):
        xs = np.unique(g.x_coord.to_numpy())
        if len(xs) < 2:
            continue
        step = int(np.diff(xs).min())
        key = set(zip(g.x_coord, g.y_coord))
        for x, y in zip(g.x_coord, g.y_coord):
            if (x + step, y) in key:
                rows.append({"slide": slide, "x": x, "y": y, "dx": step})
    t = pd.DataFrame(rows)
    return t.iloc[rng.permutation(len(t))[:n]].reset_index(drop=True) if len(t) else t


def measure_units(pairs: pd.DataFrame, load, side_px: int = 96) -> dict:
    """Measure pixels per coordinate unit from real images.

    pairs: rows (slide, x, y, dx) with a horizontal neighbour at (x + dx, y);
    load(slide, x, y) -> HxWx3 uint8 array. Every pair must give the same px/unit and a clear,
    near-exact match, otherwise this raises."""
    ratios, errs, seconds = [], [], []
    for r in pairs.itertuples():
        a, b = load(r.slide, r.x, r.y), load(r.slide, r.x + r.dx, r.y)
        assert a.shape[:2] == (side_px, side_px) and b.shape[:2] == (side_px, side_px), "unexpected patch size"
        s, e, e2 = best_shift(a.mean(axis=2), b.mean(axis=2), axis=1)
        ratios.append(s / r.dx)
        errs.append(e)
        seconds.append(e2)
    ratios = np.array(ratios)
    assert len(ratios) >= 5, "too few neighbouring pairs to verify units"
    assert np.allclose(ratios, ratios[0]), f"inconsistent pixel/unit ratios: {sorted(set(ratios))}"
    assert max(errs) < 2.0 and min(seconds) > 5 * max(max(errs), 0.1), "overlap match is not clear"
    px_per_unit = float(ratios[0])
    return {"px_per_unit": px_per_unit, "P_units": side_px / px_per_unit,
            "step_units": int(pairs.dx.iloc[0]), "step_px": int(round(pairs.dx.iloc[0] * px_per_unit)),
            "max_match_err": float(max(errs)), "min_second_err": float(min(seconds)), "n_pairs": len(ratios)}


# ----------------------------------------------------------------------------- cap and units
def cap_patches(meta: pd.DataFrame, cap: int = CAP, seed: int = 0) -> pd.DataFrame:
    """At most `cap` patches per class per slide, uniformly at random, fixed seed."""
    rng = np.random.default_rng(seed)
    keep = []
    for (_, _), g in meta.sort_values(["slide", "tumor", "y_coord", "x_coord"]).groupby(["slide", "tumor"]):
        idx = g.index.to_numpy()
        keep.append(idx if len(idx) <= cap else np.sort(rng.choice(idx, cap, replace=False)))
    return meta.loc[np.sort(np.concatenate(keep))]


def unit_table(capped: pd.DataFrame) -> pd.DataFrame:
    u = capped.groupby(["center", "patient"]).agg(slides=("slide", "nunique"),
                                                   t=("tumor", "sum"), n=("tumor", lambda s: int((s == 0).sum())))
    return u.reset_index()


# ----------------------------------------------------------------------------- roles
def arrangements(rich: list[int], side: str, seed_base: int) -> list[dict]:
    """All tumour-rich arrangements of a centre, in a fixed shuffled order."""
    out = []
    for st, te in itertools.permutations(sorted(rich), 2):
        extra = [p for p in sorted(rich) if p not in (st, te)]
        if side == "src":
            a = {**{p: "train" for p in extra}, st: "stream", te: "test"}
        else:
            a = {st: "stream", te: "test", **{p: ("test" if i % 2 == 0 else "stream") for i, p in enumerate(extra)}}
        out.append(a)
    order = np.random.default_rng(seed_base).permutation(len(out))
    return [out[i] for i in order]


def deal_minor(minor: list[int], side: str, rng) -> dict:
    pattern = ["train", "stream", "test", "train"] if side == "src" else ["stream", "test"]
    perm = rng.permutation(sorted(minor))
    return {int(p): pattern[j % len(pattern)] for j, p in enumerate(perm)}


def role_supply(units: pd.DataFrame, roles: dict) -> dict:
    out = {}
    for r in sorted(set(roles.values())):
        g = units[units.patient.isin([p for p, rr in roles.items() if rr == r])]
        out[r] = {"patients": sorted(int(p) for p in g.patient), "t": int(g.t.sum()), "n": int(g.n.sum()),
                  "rich": sorted(int(p) for p in g.patient[g.t >= RICH])}
    return out


def pre_ok(sup: dict, req: dict) -> list[str]:
    fails = []
    for r, m in req.items():
        for c in ("t", "n"):
            if sup.get(r, {"t": 0, "n": 0})[c] < m:
                fails.append(f"{r}.{c}<{m}")
    return fails


# ----------------------------------------------------------------------------- spatial validation
def overlap_pairs(a: pd.DataFrame, b: pd.DataFrame, P: float) -> int:
    """Number of (a, b) patch pairs on the same slide that share pixels: |dx| < P and |dy| < P.
    Independent brute-force check (chunked), used only for assertions."""
    n = 0
    for slide, ga in a.groupby("slide"):
        gb = b[b.slide == slide]
        if gb.empty:
            continue
        xb, yb = gb.x_coord.to_numpy(np.float64), gb.y_coord.to_numpy(np.float64)
        xa, ya = ga.x_coord.to_numpy(np.float64), ga.y_coord.to_numpy(np.float64)
        for i in range(0, len(xa), 2048):
            dx = np.abs(xa[i:i + 2048, None] - xb[None, :])
            dy = np.abs(ya[i:i + 2048, None] - yb[None, :])
            n += int(((dx < P) & (dy < P)).sum())
    return n


def _conflicts(df: pd.DataFrame, lab: np.ndarray, P: float) -> np.ndarray:
    """Boolean mask: patch overlaps at least one patch with a different label (same slide)."""
    bad = np.zeros(len(df), bool)
    pos = np.arange(len(df))
    for _, g in df.assign(_pos=pos, _lab=lab).groupby("slide"):
        x, y, l, p = (g.x_coord.to_numpy(np.float64), g.y_coord.to_numpy(np.float64),
                      g._lab.to_numpy(), g._pos.to_numpy())
        for i in range(0, len(x), 2048):
            ov = (np.abs(x[i:i + 2048, None] - x[None, :]) < P) & (np.abs(y[i:i + 2048, None] - y[None, :]) < P)
            diff = l[i:i + 2048, None] != l[None, :]
            bad[p[i:i + 2048]] = (ov & diff).any(axis=1)
    return bad


def spatial_split(train: pd.DataFrame, P: float, seed: int, block_patches: int = BLOCK_PATCHES):
    """Assign square blocks (side block_patches * P) of the training patients' slides to
    train / val_es / val_temp (8:1:1, stratified by tumour content), then drop EVERY patch that
    overlaps a patch of a different subset. Returns (labels Series indexed like `train`, report)."""
    L = block_patches * P
    bx = np.floor(train.x_coord.to_numpy(np.float64) / L).astype(np.int64)
    by = np.floor(train.y_coord.to_numpy(np.float64) / L).astype(np.int64)
    blk = pd.DataFrame({"slide": train.slide.to_numpy(), "bx": bx, "by": by, "t": train.tumor.to_numpy()})
    stats_ = blk.groupby(["slide", "bx", "by"]).agg(t=("t", "sum"), n=("t", "size")).reset_index()
    stats_ = stats_.sort_values(["t", "n", "slide", "bx", "by"], ascending=[False, False, True, True, True])
    rng = np.random.default_rng(seed)
    base = np.array(["train"] * 8 + ["val_es", "val_temp"])
    labels = []
    for i in range(0, len(stats_), 10):
        k = min(10, len(stats_) - i)
        labels.extend(rng.permutation(base)[:k])
    stats_["subset"] = labels
    lab = blk.merge(stats_[["slide", "bx", "by", "subset"]], on=["slide", "bx", "by"], how="left").subset.to_numpy()
    assigned = pd.Series(lab, index=train.index)
    drop = _conflicts(train, lab, P)
    final = assigned.where(~drop, "removed")
    rep = {"blocks": len(stats_), "block_side_units": L}
    for s in ("train", "val_es", "val_temp"):
        m = assigned == s
        rep[f"{s}_assigned_t"] = int(train.tumor[m].sum())
        rep[f"{s}_assigned_n"] = int((train.tumor[m] == 0).sum())
        rep[f"{s}_removed"] = int((m & drop).sum())
        f = final == s
        rep[f"{s}_t"] = int(train.tumor[f].sum())
        rep[f"{s}_n"] = int((train.tumor[f] == 0).sum())
    return final, rep


# ----------------------------------------------------------------------------- assertions
def assert_no_leakage(capped: pd.DataFrame, src_roles: dict, tgt_roles: dict, vlab: pd.Series, P: float):
    """All leakage conditions of Addendum 5. Raises AssertionError on any violation."""
    # 1. patient roles disjoint within source, within target, and across hospitals
    assert not set(src_roles) & set(tgt_roles), "a patient appears in both hospitals"
    roles = {**{("S", p): r for p, r in src_roles.items()}, **{("T", p): r for p, r in tgt_roles.items()}}
    assert len(roles) == len(src_roles) + len(tgt_roles)
    # 2. slides of different roles are disjoint (a slide belongs to exactly one patient)
    sp = capped.groupby("slide").patient.nunique()
    assert (sp == 1).all(), "a slide maps to more than one patient"
    slides_of = {r: set(capped.slide[capped.patient.isin([p for p, rr in src_roles.items() if rr == r])])
                 for r in set(src_roles.values())}
    slides_of.update({("T", r): set(capped.slide[capped.patient.isin([p for p, rr in tgt_roles.items() if rr == r])])
                      for r in set(tgt_roles.values())})
    keys = list(slides_of)
    for i, j in itertools.combinations(keys, 2):
        assert not slides_of[i] & slides_of[j], f"slide shared between roles {i} and {j}"
    # 3. validation subsets only contain source-train patients
    train_p = {p for p, r in src_roles.items() if r == "train"}
    assert set(capped.loc[vlab.index].patient) <= train_p, "validation patch outside training patients"
    # 4. no pixel overlap between train / val_es / val_temp (actual coordinates, measured P)
    sub = {s: capped.loc[vlab.index[vlab == s]] for s in ("train", "val_es", "val_temp")}
    for a, b in (("train", "val_es"), ("train", "val_temp"), ("val_es", "val_temp")):
        k = overlap_pairs(sub[a], sub[b], P)
        assert k == 0, f"{k} overlapping patch pairs between {a} and {b}"
    # 5. subsets are disjoint sets of patches
    for a, b in itertools.combinations(sub, 2):
        assert not set(sub[a].index) & set(sub[b].index), f"patch in both {a} and {b}"


# ----------------------------------------------------------------------------- planning
def plan_side(units: pd.DataFrame, side: str, seed: int, cell: str, req: dict, extra_check=None):
    """Deterministic: take kept arrangement number (seed-1) mod K in the fixed order; for it, try
    minor-patient draws a = 0, 1, ... with rng([seed, cell_code, side, a]) until all checks pass;
    if none passes, move to the next arrangement in the order. Returns (roles, info) or (None, info)."""
    rich = units.patient[units.t >= RICH].tolist()
    minor = units.patient[units.t < RICH].tolist()
    arrs = arrangements(rich, side, 2026 if side == "src" else 2027)
    sid = 0 if side == "src" else 1
    tried = []
    start = (seed - 1) % max(len(arrs), 1)
    for off in range(len(arrs)):
        ai = (start + off) % len(arrs)
        for a in range(MAX_MINOR_ATTEMPTS):
            rng = np.random.default_rng([seed, CELL_CODE[cell], sid, a])
            roles = {**{int(k): v for k, v in arrs[ai].items()}, **deal_minor(minor, side, rng)}
            fails = pre_ok(role_supply(units, roles), req)
            if not fails and extra_check is not None:
                fails = extra_check(roles)
            if not fails:
                return roles, {"arrangement": ai, "n_arrangements": len(arrs), "minor_attempt": a,
                               "arrangements_skipped": tried}
        tried.append(ai)
    return None, {"arrangement": None, "n_arrangements": len(arrs), "arrangements_skipped": tried}


def plan_cell(capped: pd.DataFrame, cell: str, seed: int, P: float) -> dict:
    s_c, t_c = PAIRS[cell]
    units = unit_table(capped)
    us, ut = units[units.center == s_c], units[units.center == t_c]
    cache = {}

    memo = {}

    def final_check(roles):
        tp = frozenset(p for p, r in roles.items() if r == "train")
        if tp not in memo:  # the split depends only on the training patients (and the seed)
            tr = capped[capped.patient.isin(tp)]
            memo[tp] = spatial_split(tr, P, seed=1000 * seed + CELL_CODE[cell])
        lab, rep = memo[tp]
        cache["lab"], cache["rep"] = lab, rep
        return [f"{s}.{c}<{m}" for s, m in FINAL.items() for c in ("t", "n") if rep[f"{s}_{c}"] < m]

    src, si = plan_side(us, "src", seed, cell, SRC_PRE, extra_check=final_check)
    tgt, ti = plan_side(ut, "tgt", seed, cell, TGT_PRE)
    out = {"cell": cell, "seed": seed, "primary": cell in PRIMARY, "src_info": si, "tgt_info": ti,
           "feasible": src is not None and tgt is not None}
    if not out["feasible"]:
        return out
    final_check(src)  # recompute for the chosen roles (cache may hold the last failed attempt)
    lab, rep = cache["lab"], cache["rep"]
    assert_no_leakage(capped, src, tgt, lab, P)
    out.update({"src_roles": src, "tgt_roles": tgt, "src_supply": role_supply(us, src),
                "tgt_supply": role_supply(ut, tgt), "val": rep})
    man = capped.rename(columns={"slide": "wsi", "tumor": "label"}).reset_index(drop=True)
    for name, roles, hosp in (("S", src, s_c), ("G", tgt, t_c)):
        sl = sorted(set(man.wsi[man.patient.isin([p for p, r in roles.items() if r == "test"])]))
        idx = eval_sample(man, sl, n=2000, cap=400, seed=seed)
        e = man.loc[idx]
        out[name] = {"n": len(e), "t": int(e.label.sum()), "tumour_prop": round(float(e.label.mean()), 3),
                     "patients": sorted(int(p) for p in e.patient.unique()), "slides": int(e.wsi.nunique())}
    return out
