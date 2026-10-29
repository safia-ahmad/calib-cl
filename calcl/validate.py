"""Known-answer check V1: does CORP put each kind of damage in the right place?

We take real model outputs on S and damage them in ways whose answer we know:
  A  calibration only     rescale or shift the logit. The ranking of patches is
                          unchanged, so discrimination (DSC) must stay put and
                          miscalibration (MCB) must rise.
  B  discrimination only  replace a random fraction of probabilities with 0.5.
                          That information is gone, so DSC must fall; MCB must
                          move much less than DSC.
  C  both                 A and B together: both must move.
Covered by tests/test_validate.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from calcl.corp import corp_crossfit

SESOI = 0.005


def binary_logit(z: np.ndarray, T0: float) -> np.ndarray:
    """Two-class logits (N, 2) -> one tumour logit, already divided by T0."""
    return (z[:, 1] - z[:, 0]) / T0


def _rescale(d, t=1.0, shift=0.0):
    return expit(d / t + shift)


def _blank(p, frac, rng):
    p = p.copy()
    k = int(round(frac * len(p)))
    p[rng.choice(len(p), k, replace=False)] = 0.5
    return p


def known_answer_table(d: np.ndarray, y: np.ndarray, groups, n_rep: int = 20,
                       seed: int = 0) -> pd.DataFrame:
    """d: tumour logit after T0. Returns one row per damage with the change in
    MCB and DSC versus the undamaged outputs (random damages averaged over n_rep)."""
    base = corp_crossfit(expit(d), y, groups)
    rng = np.random.default_rng(seed)
    rows = []

    def add(check, damage, make, reps=1):
        res = [corp_crossfit(make(), y, groups) for _ in range(reps)]
        rows.append({"check": check, "damage": damage,
                     "brier": np.mean([r["brier"] for r in res]),
                     "mcb": np.mean([r["mcb"] for r in res]),
                     "dsc": np.mean([r["dsc"] for r in res])})

    rows.append({"check": "-", "damage": "none (baseline)", "brier": base["brier"],
                 "mcb": base["mcb"], "dsc": base["dsc"]})
    for t in (0.5, 2.0, 3.0):
        add("A", f"logit / {t:g}", lambda t=t: _rescale(d, t=t))
    for s in (-1.0, 1.0):
        add("A", f"logit {s:+g}", lambda s=s: _rescale(d, shift=s))
    for f in (0.1, 0.3, 0.5):
        add("B", f"{f:.0%} set to 0.5", lambda f=f: _blank(expit(d), f, rng), n_rep)
    add("C", "logit / 2 and 30% set to 0.5",
        lambda: _blank(_rescale(d, t=2.0), 0.3, rng), n_rep)

    df = pd.DataFrame(rows)
    df["d_mcb"] = df.mcb - base["mcb"]
    df["d_dsc"] = df.dsc - base["dsc"]
    df["pass"] = [_verdict(r) for r in df.itertuples()]
    return df


def _verdict(r) -> str:
    if r.check == "A":    # calibration damage: MCB up, DSC (almost) unchanged
        ok = r.d_mcb >= SESOI and abs(r.d_dsc) < SESOI
    elif r.check == "B":  # discrimination damage: DSC down, and moves more than MCB
        ok = r.d_dsc <= -SESOI and abs(r.d_mcb) < 0.5 * abs(r.d_dsc)
    elif r.check == "C":  # both
        ok = r.d_mcb >= SESOI and r.d_dsc <= -SESOI
    else:
        return ""
    return "PASS" if ok else "FAIL"
