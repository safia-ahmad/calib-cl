import numpy as np

from calcl.stream import Pool, alpha_schedule, build_stream


def _pool(n_per=3000):
    labels = np.tile(np.repeat([0, 1], n_per), 2)
    domains = np.repeat([0, 1], 2 * n_per)
    return Pool(labels=labels, domains=domains)


def test_schedules():
    a = alpha_schedule(20, "linear")
    assert a[0] == 0 and a[1] == 0 and a[-1] == 1 and a[-2] == 1
    assert np.all(np.diff(a) >= 0)
    assert set(alpha_schedule(20, "step")) == {0.0, 1.0}
    assert np.all(alpha_schedule(20, "none") == 0)


def test_window_composition_tracks_alpha_and_fixes_prevalence():
    pool = _pool()
    alphas = alpha_schedule(10, "linear")
    stream = build_stream(pool, alphas, 400, np.array([0.5, 0.5]), seed=0)
    tgt_frac = [pool.domains[w].mean() for w in stream]
    assert np.corrcoef(tgt_frac, alphas)[0, 1] > 0.98
    prev = [pool.labels[w].mean() for w in stream]
    assert max(abs(p - 0.5) for p in prev) < 0.1


def test_no_reuse_and_reproducible():
    pool = _pool()
    alphas = alpha_schedule(10, "linear")
    s1 = build_stream(pool, alphas, 400, np.array([0.5, 0.5]), seed=7)
    s2 = build_stream(pool, alphas, 400, np.array([0.5, 0.5]), seed=7)
    flat = np.concatenate(s1)
    assert len(flat) == len(np.unique(flat))
    assert all(np.array_equal(a, b) for a, b in zip(s1, s2))
