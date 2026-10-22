"""CORP decomposition of the Brier score, with slide-grouped cross-fitting.

Reference: Dimitriadis, Gneiting & Jordan, "Stable reliability diagrams for
probabilistic classifiers", PNAS 2021 (arXiv:2008.03033).

For binary outcomes y in {0, 1} and forecast probabilities p = P(y = 1):
    BS  = mean (p - y)^2                       (single-probability Brier form)
    BS_C = Brier score of the PAV (isotonic) recalibrated forecast
    UNC = Brier score of the constant forecast ybar = ybar (1 - ybar)
    MCB = BS - BS_C   (miscalibration)
    DSC = UNC - BS_C  (discrimination)
so that BS = MCB - DSC + UNC exactly.

In-sample (as in the paper) MCB >= 0 and DSC >= 0 by construction and both are
optimistic for finite samples. `corp_crossfit` fits the isotonic map on some
slides and scores it on other slides, so MCB and DSC are out-of-sample; the
identity BS = MCB - DSC + UNC still holds exactly, but MCB may be slightly
negative by sampling noise. Covered by tests/test_corp.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _pav_fit(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                             out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def corp_insample(p, y) -> dict[str, float]:
    """In-sample CORP decomposition (the paper's estimator)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bs = _brier(p, y)
    bs_c = _brier(_pav_fit(p, y).predict(p), y)
    ybar = y.mean()
    unc = float(ybar * (1 - ybar))
    return {"brier": bs, "mcb": bs - bs_c, "dsc": unc - bs_c, "unc": unc}


def corp_crossfit(p, y, groups, n_folds: int = 2, seed: int = 0) -> dict[str, float]:
    """Cross-fitted CORP: the isotonic map applied to a case is always fitted on
    cases from OTHER groups (slides), so patches of one slide never inform their
    own recalibration."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < n_folds:
        raise ValueError("need at least n_folds distinct groups")
    rng = np.random.default_rng(seed)
    fold_of_group = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % n_folds))
    fold = np.array([fold_of_group[g] for g in groups])
    p_cal = np.empty_like(p)
    for k in range(n_folds):
        test = fold == k
        p_cal[test] = _pav_fit(p[~test], y[~test]).predict(p[test])
    bs = _brier(p, y)
    bs_c = _brier(p_cal, y)
    ybar = y.mean()
    unc = float(ybar * (1 - ybar))
    return {"brier": bs, "mcb": bs - bs_c, "dsc": unc - bs_c, "unc": unc,
            "n_groups": int(len(uniq))}
