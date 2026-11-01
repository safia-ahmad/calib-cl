"""Day 16-17: learning on the stream, test-then-train. Covered by tests/test_stream_train.py.

At checkpoint c (c = number of windows trained so far, 0..T) the model is scored on
  S  fixed source test set (forgetting is measured here)
  G  fixed target test set (adaptation)
  B  fixed labelled source buffer from val_buf slides (what a hospital could label: used
     later by the fix rule and the recalibration fixes)
  E  the NEXT window, before training on it (only for c < T)
Then it trains one pass over window c+1.

Strategies
  frozen  never trains (shows the pure effect of the shift)
  ft      plain fine-tuning on the window
  replay  reservoir buffer of past stream patches, mixed 1:1 into every batch
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax

from calcl.corp import corp_crossfit
from calcl.data import augment, colour_jitter, normalise
from calcl.metrics import all_metrics


class Reservoir:
    """Classic reservoir sampling: after n items, each has probability cap/n of being kept."""

    def __init__(self, cap: int, rng: np.random.Generator):
        self.cap, self.rng, self.items, self.seen = cap, rng, [], 0

    def add(self, idx) -> None:
        for i in np.asarray(idx).tolist():
            self.seen += 1
            if len(self.items) < self.cap:
                self.items.append(i)
            else:
                j = int(self.rng.integers(0, self.seen))
                if j < self.cap:
                    self.items[j] = i

    def sample(self, k: int) -> np.ndarray:
        if not self.items:
            return np.empty(0, dtype=np.int64)
        return self.rng.choice(np.array(self.items), size=min(k, len(self.items)), replace=False)


def make_batch(X, labels, idx, colour: float):
    xs = [normalise(colour_jitter(augment(X[j]), colour)) for j in idx]
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.asarray(labels)[idx].astype(np.int64))


def train_window(model, opt, X, labels, window, device, rng, batch: int = 64,
                 buffer: Reservoir | None = None, colour: float = 0.5) -> float:
    """One pass over the window. With a buffer: each step adds an equal number of
    replayed patches, then the step's new patches enter the reservoir."""
    model.train()
    order = rng.permutation(window)
    tot, n = 0.0, 0
    for s in range(0, len(order), batch):
        new = order[s:s + batch]
        idx = new if buffer is None else np.concatenate([new, buffer.sample(len(new))])
        x, y = make_batch(X, labels, idx, colour)
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        tot += loss.item() * len(y)
        n += len(y)
        if buffer is not None:
            buffer.add(new)
    return tot / max(n, 1)


def score_set(z, y, groups, T0) -> dict:
    p1 = softmax(z / T0, axis=1)[:, 1]
    out = {k: float(v) for k, v in all_metrics(z, y, deployed_t=T0).items()}
    out.pop("brier", None)
    try:
        out.update({k: float(v) for k, v in corp_crossfit(p1, y, groups).items()})
    except ValueError:  # fewer than 2 slides (tiny test data)
        out.update({"brier": float(np.mean((p1 - y) ** 2)), "mcb": np.nan, "dsc": np.nan, "unc": np.nan})
    return out
