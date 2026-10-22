"""Calibration and performance metrics (NumPy only, framework-agnostic).

Every function takes predicted class probabilities `probs` with shape (N, K)
(rows sum to 1) and integer labels `y` with shape (N,). For a binary task pass
K = 2 columns. These functions are covered by tests/test_metrics.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def _check(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != y.shape[0]:
        raise ValueError("probs must be (N, K) and y must be (N,)")
    if not np.allclose(probs.sum(1), 1.0, atol=1e-5):
        raise ValueError("rows of probs must sum to 1")
    return probs, y


def accuracy(probs, y) -> float:
    probs, y = _check(probs, y)
    return float((probs.argmax(1) == y).mean())


def nll(probs, y) -> float:
    """Mean negative log-likelihood (natural log)."""
    probs, y = _check(probs, y)
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1.0)).mean())


def brier(probs, y) -> float:
    """Multi-class Brier score: mean over samples of sum_k (p_k - 1[y=k])^2."""
    probs, y = _check(probs, y)
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def auroc(probs, y) -> float:
    """Binary AUROC on class-1 probability, or macro one-vs-rest AUROC for K > 2."""
    probs, y = _check(probs, y)
    if probs.shape[1] == 2:
        return float(roc_auc_score(y, probs[:, 1]))
    return float(roc_auc_score(y, probs, multi_class="ovr", average="macro",
                               labels=np.arange(probs.shape[1])))


def _bin_edges(conf: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if scheme == "mass":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError("scheme must be 'width' or 'mass'")


def reliability_table(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15,
                      scheme: str = "width") -> dict[str, np.ndarray]:
    """Per-bin counts, mean confidence and accuracy (for reliability diagrams)."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = _bin_edges(conf, n_bins, scheme)
    # right-closed bins, first bin also includes its left edge
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    count = np.bincount(idx, minlength=nb).astype(np.float64)
    sum_conf = np.bincount(idx, weights=conf, minlength=nb)
    sum_acc = np.bincount(idx, weights=correct, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(count > 0, sum_conf / count, np.nan)
        mean_acc = np.where(count > 0, sum_acc / count, np.nan)
    return {"edges": edges, "count": count, "conf": mean_conf, "acc": mean_acc}


def ece(probs, y, n_bins: int = 15, scheme: str = "width") -> float:
    """Top-label expected calibration error (binned estimator).

    Note: the binned estimator is biased upward for small N and depends on the
    binning scheme, so compare checkpoints only at a FIXED evaluation-set size.
    """
    probs, y = _check(probs, y)
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(np.float64)
    t = reliability_table(conf, correct, n_bins, scheme)
    m = t["count"] > 0
    w = t["count"][m] / t["count"].sum()
    return float((w * np.abs(t["acc"][m] - t["conf"][m])).sum())


def binary_class1_ece(probs, y, n_bins: int = 15, scheme: str = "width") -> float:
    """ECE of p(y=1) against the observed frequency of y=1 (binary tasks)."""
    probs, y = _check(probs, y)
    if probs.shape[1] != 2:
        raise ValueError("binary only")
    t = reliability_table(probs[:, 1], (y == 1).astype(np.float64), n_bins, scheme)
    m = t["count"] > 0
    w = t["count"][m] / t["count"].sum()
    return float((w * np.abs(t["acc"][m] - t["conf"][m])).sum())


def confidence_gap(probs, y) -> float:
    """Signed gap: mean top-label confidence minus accuracy (>0 = overconfident)."""
    probs, y = _check(probs, y)
    return float(probs.max(1).mean() - (probs.argmax(1) == y).mean())


# ---------------------------------------------------------------- temperature
def fit_temperature(logits: np.ndarray, y: np.ndarray,
                    bounds: tuple[float, float] = (0.05, 20.0)) -> float:
    """Scalar temperature minimising NLL of softmax(logits / T)."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    def loss(log_t: float) -> float:
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-lp[np.arange(len(y)), y].mean())

    res = minimize_scalar(loss, bounds=(np.log(bounds[0]), np.log(bounds[1])),
                          method="bounded")
    return float(np.exp(res.x))


def ts_oracle_decomposition(logits: np.ndarray, y: np.ndarray, deployed_t: float,
                            seed: int = 0) -> dict[str, float]:
    """Split NLL into a part recoverable by re-fitting one temperature.

    Two-fold cross-fitting: fit T* on one half, score the other half, swap.
      nll_deployed     NLL with the temperature shipped with the model
      nll_oracle       NLL after an oracle temperature fitted on held-out data
                       from the SAME evaluation distribution
      recoverable_nll  nll_deployed - nll_oracle  (miscalibration that one
                       temperature can fix; can be slightly < 0 by noise)
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    halves = (perm[: len(y) // 2], perm[len(y) // 2:])
    dep, ora, temps = [], [], []
    for fit_idx, score_idx in (halves, halves[::-1]):
        t_star = fit_temperature(logits[fit_idx], y[fit_idx])
        temps.append(t_star)
        p_dep = softmax(logits[score_idx] / deployed_t, axis=1)
        p_ora = softmax(logits[score_idx] / t_star, axis=1)
        dep.append(nll(p_dep, y[score_idx]))
        ora.append(nll(p_ora, y[score_idx]))
    nd, no = float(np.mean(dep)), float(np.mean(ora))
    return {"nll_deployed": nd, "nll_oracle": no, "recoverable_nll": nd - no,
            "oracle_temperature": float(np.mean(temps))}


def all_metrics(logits: np.ndarray, y: np.ndarray, deployed_t: float = 1.0,
                n_bins: int = 15) -> dict[str, float]:
    """Everything recorded at one checkpoint on one evaluation set."""
    probs = softmax(np.asarray(logits, dtype=np.float64) / deployed_t, axis=1)
    out = {
        "acc": accuracy(probs, y), "auroc": auroc(probs, y), "nll": nll(probs, y),
        "brier": brier(probs, y), "ece_width": ece(probs, y, n_bins, "width"),
        "ece_mass": ece(probs, y, n_bins, "mass"),
        "conf_gap": confidence_gap(probs, y),
    }
    if probs.shape[1] == 2:
        out["ece_class1"] = binary_class1_ece(probs, y, n_bins, "width")
    out.update(ts_oracle_decomposition(logits, y, deployed_t))
    return out
