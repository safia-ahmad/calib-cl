"""Pre-registered onset ("alarm") analysis for the exploratory lead-time question.

Thresholds are calibrated on NO-DRIFT control streams so that every metric has
the same false-alarm rate. Only then are calibration and accuracy alarms
comparable: a noisier metric would otherwise alarm "earlier" by chance.
Covered by tests/test_alarms.py.
"""
from __future__ import annotations

import numpy as np


def deviation(traj: np.ndarray, baseline_windows: int, higher_is_worse: bool) -> np.ndarray:
    """Change from the mean of the first `baseline_windows` checkpoints, signed so
    that positive = deterioration."""
    traj = np.asarray(traj, dtype=np.float64)
    d = traj - traj[:baseline_windows].mean()
    return d if higher_is_worse else -d


def calibrate_threshold(control_devs: np.ndarray, start: int, far: float = 0.05,
                        persist: int = 2) -> float:
    """Smallest threshold h such that at most a fraction `far` of no-drift control
    streams ever alarm (deviation > h for `persist` consecutive checkpoints after
    `start`). control_devs: (n_streams, T)."""
    control_devs = np.asarray(control_devs, dtype=np.float64)
    # an alarm fires iff the run-wise minimum over `persist` consecutive points exceeds h
    worst = []
    for d in control_devs[:, start:]:
        runs = np.lib.stride_tricks.sliding_window_view(d, persist).min(1)
        worst.append(runs.max())
    return float(np.quantile(np.asarray(worst), 1.0 - far, method="higher"))


def onset(dev: np.ndarray, h: float, start: int, persist: int = 2) -> int | None:
    """First checkpoint index (>= start) beginning `persist` consecutive points
    with deviation > h; None = never alarmed (right-censored)."""
    d = np.asarray(dev, dtype=np.float64)
    for t in range(start, len(d) - persist + 1):
        if np.all(d[t:t + persist] > h):
            return t
    return None


def lead_time(cal_onset: int | None, acc_onset: int | None, T: int) -> dict:
    """Positive lead = calibration alarmed first. Censored cases are labelled,
    never silently dropped."""
    if cal_onset is None and acc_onset is None:
        return {"outcome": "neither", "lead": None}
    if acc_onset is None:
        return {"outcome": "calibration_only", "lead_lower_bound": T - cal_onset}
    if cal_onset is None:
        return {"outcome": "accuracy_only", "lead_upper_bound": acc_onset - T}
    lead = acc_onset - cal_onset
    outcome = "calibration_first" if lead > 0 else "accuracy_first" if lead < 0 else "tie"
    return {"outcome": outcome, "lead": lead}
