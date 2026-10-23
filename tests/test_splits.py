import pandas as pd
import pytest

from calcl.splits import check_split, make_split


def _slides():
    rows = []
    for h, n_norm, n_tum in [("RUMC", 100, 70), ("UMCU", 59, 40)]:
        for t, n in [("normal", n_norm), ("tumour", n_tum)]:
            for i in range(n):
                rows.append({"wsi": f"{h}_{t}_{i}", "hospital": h, "slide_type": t})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("setting", ["pcam_fwd", "pcam_rev"])
def test_every_slide_once_and_tumour_everywhere(setting):
    s = _slides()
    sp = make_split(s, setting, seed=1)
    assert len(sp) == len(s) and not sp.wsi.duplicated().any()
    for (dom, role), g in sp.groupby(["domain", "role"]):
        assert len(g) > 0
        if role in ("test", "stream", "train"):
            assert (g.slide_type == "tumour").any()


def test_fractions_close_to_plan():
    sp = make_split(_slides(), "pcam_fwd", seed=0)
    src = sp[sp.domain == "source"].role.value_counts()
    tgt = sp[sp.domain == "target"].role.value_counts()
    assert abs(src["train"] - 0.45 * 170) <= 2 and abs(src["test"] - 0.20 * 170) <= 2
    assert abs(tgt["stream"] - 0.60 * 99) <= 2 and abs(tgt["test"] - 0.30 * 99) <= 2


def test_reproducible_and_seed_dependent():
    s = _slides()
    a, b, c = make_split(s, "pcam_fwd", 3), make_split(s, "pcam_fwd", 3), make_split(s, "pcam_fwd", 4)
    assert a.equals(b)
    assert not a.role.equals(c.role)


def test_eval_sample_balanced_capped_and_in_slides():
    import numpy as np
    from calcl.splits import eval_sample
    rows = []
    for s in range(30):
        for i in range(200):
            rows.append({"wsi": f"s{s}", "label": int(i < (100 if s < 15 else 0))})
    man = pd.DataFrame(rows)
    slides = [f"s{s}" for s in range(20)]
    idx = eval_sample(man, slides, n=1000, cap=60, seed=0)
    sub = man.loc[idx]
    assert set(sub.wsi) <= set(slides)
    assert abs(sub.label.mean() - 0.5) < 1e-9
    assert sub.groupby(["wsi", "label"]).size().max() <= 60
    assert np.array_equal(idx, eval_sample(man, slides, n=1000, cap=60, seed=0))


def test_check_catches_duplicates():
    sp = make_split(_slides(), "pcam_fwd", 0)
    bad = pd.concat([sp, sp.iloc[[0]]])
    with pytest.raises(AssertionError):
        check_split(bad)
