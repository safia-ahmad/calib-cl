"""Statistics of Addendum 5, Sections 8-9 (Camelyon17 replication). Covered by tests/test_c17_stats.py."""
from __future__ import annotations

import numpy as np
from scipy import stats

SESOI = 0.005
RHO_THR = 0.25


def tci(x, level=0.95):
    """Mean and two-sided t-interval over seeds."""
    x = np.asarray(x, float)
    m = float(x.mean())
    h = float(stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - h, m + h


def fieller(d_mcb, d_brier, level=0.95):
    """Fieller (test-inversion) interval for rho = mean dMCB / mean dBrier.

    The set of c for which the paired t-test of mean(dMCB - c * dBrier) = 0 is not rejected.
    Returns (lo, hi, kind) with kind 'bounded' or 'unbounded' (rho not estimable)."""
    a, b = np.asarray(d_mcb, float), np.asarray(d_brier, float)
    n = len(a)
    t2 = stats.t.ppf(0.5 + level / 2, n - 1) ** 2
    S = np.cov(a, b, ddof=1) / n
    ma, mb = a.mean(), b.mean()
    A = mb ** 2 - t2 * S[1, 1]
    B = -2 * (ma * mb - t2 * S[0, 1])
    C = ma ** 2 - t2 * S[0, 0]
    if A <= 0:
        return -np.inf, np.inf, "unbounded"
    disc = max(B ** 2 - 4 * A * C, 0.0)
    r = np.sqrt(disc)
    return float((-B - r) / (2 * A)), float((-B + r) / (2 * A)), "bounded"


def contrast_ci(d_mcb, d_brier, c=RHO_THR, level=0.95):
    """t-interval of mean(dMCB - c * dBrier); its sign decides R2's rho condition and 'contradicted'."""
    return tci(np.asarray(d_mcb, float) - c * np.asarray(d_brier, float), level)


def rho_point(d_mcb, d_brier):
    mb = float(np.mean(d_brier))
    return float(np.mean(d_mcb)) / mb if mb > 0 else float("nan")


def rho_bootstrap(d_mcb, d_brier, n_boot=10_000, seed=0):
    """Descriptive only: seed-percentile bootstrap, non-positive denominators excluded and counted."""
    a, b = np.asarray(d_mcb, float), np.asarray(d_brier, float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    ma, mb = a[idx].mean(1), b[idx].mean(1)
    ok = mb > 0
    share = float(1 - ok.mean())
    r = ma[ok] / mb[ok]
    lo, hi = (np.percentile(r, [2.5, 97.5]) if ok.any() else (np.nan, np.nan))
    return float(lo), float(hi), share, ("unstable" if share > 0.05 else "ok")


def cell_criteria(ft: dict, replay_dsc_lost=None) -> dict:
    """ft: per-seed arrays d_brier, d_mcb, dsc_lost (seed order). Returns R1-R3 and the cell status."""
    db, dm, dl = (np.asarray(ft[k], float) for k in ("d_brier", "d_mcb", "dsc_lost"))
    assert np.allclose(dm + dl, db, atol=1e-9), "decomposition identity dMCB + DSC lost = dBrier violated"
    out = {}
    out["dBrier"] = tci(db)
    out["dMCB"] = tci(dm)
    out["DSC_lost"] = tci(dl)
    out["R1"] = bool(out["dBrier"][0] >= SESOI and out["dBrier"][1] > 0)
    out["rho"] = rho_point(dm, db)
    out["fieller"] = fieller(dm, db)
    out["L025"] = contrast_ci(dm, db)
    out["bootstrap"] = rho_bootstrap(dm, db)
    lo, hi, kind = out["fieller"]
    if out["R1"]:
        assert kind == "bounded", "R1 holds but the Fieller interval is unbounded"
        # the ratio limits and the contrast test must agree (Addendum 5, Section 8)
        assert (hi < RHO_THR) == (out["L025"][2] < 0) and (lo > RHO_THR) == (out["L025"][1] > 0)
    out["R2"] = bool(out["R1"] and out["DSC_lost"][1] > 0 and kind == "bounded" and hi < RHO_THR)
    contradicted = bool(out["R1"] and kind == "bounded" and lo > RHO_THR)
    if not out["R1"]:
        status = "uninformative"
    elif out["R2"]:
        status = "consistent"
    elif contradicted:
        status = "contradicted"
    else:
        status = "not consistent"
    out["status"] = status
    if replay_dsc_lost is not None:
        rl = np.asarray(replay_dsc_lost, float)
        out["R3_seeds"] = int((rl < dl).sum())
        out["R3"] = bool(out["R3_seeds"] >= 4)
        out["R3_paired_FT_minus_replay"] = tci(dl - rl)
    return out


def overall_verdict(primary_status: list[str]) -> str:
    s = list(primary_status)
    if any(x == "contradicted" for x in s):
        return "contradicted"
    if all(x == "consistent" for x in s):
        return "replicated"
    if sum(x == "consistent" for x in s) == 1:
        return "partially replicated"
    if all(x == "uninformative" for x in s):
        return "uninformative"
    return "not replicated"
