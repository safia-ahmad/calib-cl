import numpy as np
import pytest
from scipy.special import softmax

from calcl import metrics as M


def test_perfectly_calibrated_synthetic_has_small_ece():
    rng = np.random.default_rng(0)
    n = 200_000
    p1 = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p1).astype(int)          # y ~ Bernoulli(p1)
    probs = np.stack([1 - p1, p1], 1)
    assert M.binary_class1_ece(probs, y) < 0.01
    assert M.ece(probs, y) < 0.01


def test_overconfidence_is_detected_and_signed():
    rng = np.random.default_rng(1)
    n = 100_000
    y = rng.integers(0, 2, n)
    correct = rng.uniform(size=n) < 0.7                  # 70 % accurate ...
    pred = np.where(correct, y, 1 - y)
    probs = np.zeros((n, 2))
    probs[np.arange(n), pred] = 0.95                      # ... but 95 % confident
    probs[np.arange(n), 1 - pred] = 0.05
    assert M.ece(probs, y) == pytest.approx(0.25, abs=0.01)
    assert M.confidence_gap(probs, y) == pytest.approx(0.25, abs=0.01)


def test_scores_known_values():
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    y = np.array([0, 0])
    assert M.accuracy(probs, y) == 0.5
    assert M.nll(probs, y) == pytest.approx(-(np.log(0.8) + np.log(0.3)) / 2)
    assert M.brier(probs, y) == pytest.approx(((0.2**2 * 2) + (0.7**2 * 2)) / 2)


def test_temperature_recovers_known_scaling():
    rng = np.random.default_rng(2)
    n, k, true_t = 50_000, 4, 2.5
    z = rng.normal(size=(n, k)) * 3
    p = softmax(z / true_t, 1)                            # data generated at T=2.5
    y = np.array([rng.choice(k, p=row) for row in p])
    assert M.fit_temperature(z, y) == pytest.approx(true_t, rel=0.05)


def test_decomposition_positive_when_miscalibrated_zero_when_not():
    rng = np.random.default_rng(3)
    n, k = 40_000, 3
    z = rng.normal(size=(n, k)) * 3
    y = np.array([rng.choice(k, p=row) for row in softmax(z / 2.0, 1)])
    bad = M.ts_oracle_decomposition(z, y, deployed_t=1.0)  # deployed T wrong
    good = M.ts_oracle_decomposition(z, y, deployed_t=2.0)  # deployed T right
    assert bad["recoverable_nll"] > 0.05
    assert abs(good["recoverable_nll"]) < 0.005


def test_all_metrics_runs_multiclass_and_binary():
    rng = np.random.default_rng(4)
    for k in (2, 9):
        z = rng.normal(size=(2000, k))
        y = rng.integers(0, k, 2000)
        out = M.all_metrics(z, y)
        assert {"acc", "auroc", "nll", "brier", "ece_width", "recoverable_nll"} <= set(out)
