"""Boundary-free gradual-shift stream construction (NumPy only).

The stream is defined by a schedule alpha_t in [0, 1] (fraction of target-domain
samples, or synthetic shift strength) over T windows. Windows exist ONLY for
evaluation bookkeeping: the learner receives a flat sequence of labelled
mini-batches and is never told the window index, alpha_t, or domain identity.
Covered by tests/test_stream.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def alpha_schedule(T: int, kind: str = "linear", warmup: int = 2,
                   plateau: int = 2, sharpness: float = 10.0) -> np.ndarray:
    """alpha_t for t = 1..T: `warmup` windows at 0, a ramp, `plateau` at 1.

    kind = "linear" | "sigmoid" | "step" (step = abrupt control at the ramp midpoint)
    | "none" (no-drift control: alpha = 0 throughout).
    """
    if kind == "none":
        return np.zeros(T)
    ramp = T - warmup - plateau
    if ramp < 1:
        raise ValueError("T too small for warmup + plateau")
    u = np.concatenate([np.zeros(warmup), (np.arange(1, ramp + 1) - 0.5) / ramp,
                        np.ones(plateau)])
    if kind == "linear":
        return u
    if kind == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-sharpness * (u - 0.5)))
        s[:warmup], s[T - plateau:] = 0.0, 1.0
        return s
    if kind == "step":
        return (u >= 0.5).astype(np.float64)
    raise ValueError(kind)


@dataclass
class Pool:
    """Indices into a dataset, grouped by domain and class."""
    labels: np.ndarray   # (N,) class labels
    domains: np.ndarray  # (N,) 0 = source, 1 = target

    def index(self, domain: int, cls: int) -> np.ndarray:
        return np.flatnonzero((self.domains == domain) & (self.labels == cls))


def sample_window(pool: Pool, alpha: float, n: int, class_probs: np.ndarray,
                  rng: np.random.Generator, used: set[int] | None = None,
                  replace: bool = False) -> np.ndarray:
    """Draw n indices with domain ~ Bernoulli(alpha) and class ~ class_probs.

    Class prevalence is held fixed by construction, so the window distribution is
    P_t(x, y) = sum_y pi_y [(1 - alpha) P_S(x|y) + alpha P_T(x|y)], i.e. covariate
    (acquisition) shift with fixed label prevalence. `used` enforces sampling
    without replacement across windows of the same stream.
    """
    used = set() if used is None else used
    n_target = rng.binomial(n, alpha)
    out = []
    for domain, n_dom in ((0, n - n_target), (1, n_target)):
        if n_dom == 0:
            continue
        per_class = rng.multinomial(n_dom, class_probs)
        for cls, k in enumerate(per_class):
            if k == 0:
                continue
            cand = pool.index(domain, cls)
            if not replace:
                cand = np.array([i for i in cand if i not in used], dtype=np.int64)
            if len(cand) < k:
                raise RuntimeError(f"pool exhausted: domain={domain} class={cls}")
            pick = rng.choice(cand, size=k, replace=replace)
            out.append(pick)
            if not replace:
                used.update(int(i) for i in pick)
    idx = np.concatenate(out) if out else np.empty(0, dtype=np.int64)
    rng.shuffle(idx)
    return idx


def build_stream(pool: Pool, alphas: np.ndarray, n_per_window: int,
                 class_probs: np.ndarray, seed: int) -> list[np.ndarray]:
    """One index array per window; no index appears twice in a stream."""
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    return [sample_window(pool, float(a), n_per_window, class_probs, rng, used)
            for a in alphas]


def minibatches(window_idx: np.ndarray, batch_size: int,
                rng: np.random.Generator):
    """Yield mini-batches of indices. Only this reaches the learner."""
    order = rng.permutation(window_idx)
    for s in range(0, len(order), batch_size):
        yield order[s:s + batch_size]
