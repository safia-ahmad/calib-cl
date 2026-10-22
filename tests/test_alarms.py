import numpy as np

from calcl.alarms import calibrate_threshold, deviation, lead_time, onset


def test_threshold_controls_false_alarms_and_detects_drift():
    rng = np.random.default_rng(0)
    T, start = 20, 3
    controls = rng.normal(0, 1, size=(200, T))
    h = calibrate_threshold(controls, start=start, far=0.05)
    fresh = rng.normal(0, 1, size=(2000, T))
    fa = np.mean([onset(d, h, start) is not None for d in fresh])
    assert fa < 0.10                          # close to the nominal 5 %
    drift = np.concatenate([np.zeros(8), np.linspace(1, 8, 12)])
    assert onset(drift, h, start) is not None


def test_lead_time_and_censoring():
    assert lead_time(5, 9, 20) == {"outcome": "calibration_first", "lead": 4}
    assert lead_time(None, None, 20)["outcome"] == "neither"
    assert lead_time(6, None, 20)["outcome"] == "calibration_only"


def test_deviation_sign():
    assert deviation(np.array([0.9, 0.9, 0.8]), 2, higher_is_worse=False)[-1] > 0
