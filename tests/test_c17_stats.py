import numpy as np
import pytest
from scipy import stats

from calcl import c17_stats as S


def brute_fieller(a, b, grid):
    """Vectorised test inversion over a grid of candidate rho values."""
    L = a[None, :] - grid[:, None] * b[None, :]
    m, sd = L.mean(1), L.std(1, ddof=1)
    h = stats.t.ppf(0.975, len(a) - 1) * sd / np.sqrt(len(a))
    keep = grid[(m - h <= 0) & (0 <= m + h)]
    return keep.min(), keep.max()


def test_fieller_matches_test_inversion():
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = rng.normal(0.03, 0.01, 5)
        a = 0.1 * b + rng.normal(0, 0.004, 5)
        lo, hi, kind = S.fieller(a, b)
        if kind == "bounded":
            blo, bhi = brute_fieller(a, b, np.linspace(lo - 0.5, hi + 0.5, 400001))
            assert lo == pytest.approx(blo, abs=2e-5) and hi == pytest.approx(bhi, abs=2e-5)


def test_fieller_unbounded_when_denominator_not_significant():
    b = np.array([0.01, -0.01, 0.02, -0.015, 0.0])
    a = np.array([0.001, 0.0, -0.002, 0.001, 0.0])
    assert S.fieller(a, b)[2] == "unbounded"


def test_cell_status_and_identity():
    db = np.array([0.03, 0.025, 0.02, 0.035, 0.028])
    dm = np.array([0.000, -0.002, 0.001, 0.000, -0.001])
    r = S.cell_criteria({"d_brier": db, "d_mcb": dm, "dsc_lost": db - dm}, replay_dsc_lost=(db - dm) / 2)
    assert r["R1"] and r["R2"] and r["status"] == "consistent" and r["R3"]
    # mostly miscalibration -> contradicted
    dm2 = 0.8 * db + np.array([0.0, 0.001, -0.001, 0.0005, 0.0])
    r2 = S.cell_criteria({"d_brier": db, "d_mcb": dm2, "dsc_lost": db - dm2})
    assert r2["status"] == "contradicted"
    # no forgetting -> uninformative, never contradicted
    db3 = np.array([0.001, -0.002, 0.003, 0.0, -0.001])
    dm3 = db3 * 0.9
    r3 = S.cell_criteria({"d_brier": db3, "d_mcb": dm3, "dsc_lost": db3 - dm3})
    assert r3["status"] == "uninformative"
    with pytest.raises(AssertionError):
        S.cell_criteria({"d_brier": db, "d_mcb": dm, "dsc_lost": db})


def test_overall_verdict_table():
    V = S.overall_verdict
    assert V(["consistent", "consistent"]) == "replicated"
    assert V(["consistent", "uninformative"]) == "partially replicated"
    assert V(["consistent", "not consistent"]) == "partially replicated"
    assert V(["consistent", "contradicted"]) == "contradicted"
    assert V(["uninformative", "uninformative"]) == "uninformative"
    assert V(["not consistent", "uninformative"]) == "not replicated"
    assert V(["not consistent", "not consistent"]) == "not replicated"


def test_bootstrap_reports_share_of_non_positive_draws():
    b = np.array([0.02, -0.03, 0.01, 0.015, -0.02])
    lo, hi, share, flag = S.rho_bootstrap(b * 0.1, b, n_boot=2000)
    assert 0 < share < 1 and flag == "unstable"
