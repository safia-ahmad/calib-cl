import numpy as np
import pytest

from calcl.corp import corp_crossfit, corp_insample


def _data(n=40_000, distort=None, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.beta(2, 2, n)                         # true event probability
    y = (rng.uniform(size=n) < q).astype(float)
    p = q if distort is None else distort(q)
    groups = rng.integers(0, 80, n)               # 80 "slides"
    return p, y, groups


def test_identity_holds_exactly():
    p, y, g = _data(distort=lambda q: q ** 2)
    for d in (corp_insample(p, y), corp_crossfit(p, y, g)):
        assert d["brier"] == pytest.approx(d["mcb"] - d["dsc"] + d["unc"], abs=1e-12)


def test_calibrated_forecast_has_near_zero_crossfitted_mcb():
    p, y, g = _data()
    d = corp_crossfit(p, y, g)
    assert abs(d["mcb"]) < 0.002


def test_monotone_distortion_moves_error_into_mcb_not_dsc():
    p, y, g = _data()
    base = corp_crossfit(p, y, g)
    bent = corp_crossfit(np.clip(p, 0, 1) ** 3, y, g)   # same ranking, miscalibrated
    assert bent["mcb"] > base["mcb"] + 0.01
    assert bent["dsc"] == pytest.approx(base["dsc"], abs=0.003)


def test_noise_lowers_dsc():
    p, y, g = _data()
    rng = np.random.default_rng(1)
    noisy = np.clip(p + rng.normal(0, 0.3, len(p)), 0, 1)
    assert corp_crossfit(noisy, y, g)["dsc"] < corp_crossfit(p, y, g)["dsc"] - 0.01


def test_insample_is_optimistic_relative_to_crossfit():
    p, y, g = _data(n=600, seed=3)
    rng = np.random.default_rng(4)
    p = np.clip(p + rng.normal(0, 0.2, len(p)), 0, 1)
    assert corp_insample(p, y)["mcb"] >= corp_crossfit(p, y, g % 20)["mcb"]
