"""Slide-level splits, one per (setting, seed). Covered by tests/test_splits.py.

Every patch of a slide goes to that slide's role, and no slide ever has two roles.
Splits are stratified by slide type (normal / tumour) so every role gets tumour slides.

Source-hospital roles
    train      training the source model
    val_es     early stopping
    val_temp   fitting the temperature T0
    val_buf    labelled buffer for recalibration fixes (and the fix-choice rule)
    stream     source part of the stream
    test       fixed source test set S
Target-hospital roles
    stream     target part of the stream
    test       fixed target test set G
    dev        pilot checks only, never in reported numbers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE_FRACS = {"train": 0.45, "val_es": 0.05, "val_temp": 0.05, "val_buf": 0.05,
                "stream": 0.20, "test": 0.20}
TARGET_FRACS = {"stream": 0.60, "test": 0.30, "dev": 0.10}
SETTINGS = {"pcam_fwd": ("RUMC", "UMCU"), "pcam_rev": ("UMCU", "RUMC")}


def _counts(n: int, fracs: dict[str, float]) -> dict[str, int]:
    """Largest-remainder rounding, with every role getting at least one slide
    when there are enough slides."""
    raw = {k: n * f for k, f in fracs.items()}
    cnt = {k: int(np.floor(v)) for k, v in raw.items()}
    if n >= len(fracs):
        for k in cnt:
            if cnt[k] == 0:
                cnt[k] = 1
    left = n - sum(cnt.values())
    order = sorted(fracs, key=lambda k: raw[k] - np.floor(raw[k]), reverse=True)
    i = 0
    while left > 0:
        cnt[order[i % len(order)]] += 1
        left -= 1
        i += 1
    while left < 0:  # took too many because of the at-least-one rule
        big = max(cnt, key=lambda k: cnt[k])
        cnt[big] -= 1
        left += 1
    return cnt


def _assign(slides: pd.DataFrame, fracs: dict[str, float], rng) -> pd.Series:
    roles = pd.Series(index=slides.wsi, dtype=object)
    for _, grp in slides.groupby("slide_type"):
        ids = rng.permutation(grp.wsi.to_numpy())
        cnt = _counts(len(ids), fracs)
        pos = 0
        for role in fracs:
            roles[ids[pos:pos + cnt[role]]] = role
            pos += cnt[role]
    return roles


def make_split(slides: pd.DataFrame, setting: str, seed: int) -> pd.DataFrame:
    """slides: output of calcl.data.slide_table (wsi, hospital, slide_type, ...)."""
    source, target = SETTINGS[setting]
    rng = np.random.default_rng(1000 * seed + (0 if setting == "pcam_fwd" else 1))
    src = slides[slides.hospital == source]
    tgt = slides[slides.hospital == target]
    out = pd.concat([
        pd.DataFrame({"wsi": src.wsi, "domain": "source",
                      "role": _assign(src, SOURCE_FRACS, rng).loc[src.wsi].to_numpy()}),
        pd.DataFrame({"wsi": tgt.wsi, "domain": "target",
                      "role": _assign(tgt, TARGET_FRACS, rng).loc[tgt.wsi].to_numpy()}),
    ], ignore_index=True)
    out = out.merge(slides[["wsi", "hospital", "slide_type"]], on="wsi")
    out.insert(0, "seed", seed)
    out.insert(0, "setting", setting)
    check_split(out)
    return out


def check_split(split: pd.DataFrame) -> None:
    """Fail loudly on leakage or an empty / tumour-free evaluation role."""
    if split.wsi.duplicated().any():
        raise AssertionError("a slide has more than one role")
    if split.role.isna().any():
        raise AssertionError("a slide has no role")
    for (dom, role), g in split.groupby(["domain", "role"]):
        if role in ("test", "stream", "train") and (g.slide_type == "tumour").sum() == 0:
            raise AssertionError(f"{dom}/{role} has no tumour slide")
    if set(split[split.domain == "source"].hospital) & set(split[split.domain == "target"].hospital):
        raise AssertionError("source and target share a hospital")


def eval_sample(man: pd.DataFrame, slides, n: int = 2000, cap: int = 60, seed: int = 0) -> np.ndarray:
    """A fixed, class-balanced evaluation set: n/2 patches per class, at most
    `cap` patches of each class from any one slide, so no slide dominates."""
    rng = np.random.default_rng(seed)
    sub = man[man.wsi.isin(set(slides))]
    out = []
    for lab in (0, 1):
        pool = []
        for _, g in sub[sub.label == lab].groupby("wsi"):
            idx = g.index.to_numpy()
            pool.append(rng.permutation(idx)[:cap])
        pool = np.concatenate(pool) if pool else np.array([], dtype=int)
        k = min(n // 2, len(pool))
        out.append(rng.choice(pool, k, replace=False))
    return np.sort(np.concatenate(out))


def summarise(split: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab([split.domain, split.role], split.slide_type, margins=True)
