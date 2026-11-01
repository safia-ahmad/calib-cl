"""End-to-end CPU smoke test: fake data -> source model -> tiny streams -> stream runs."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from calcl.stream_train import Reservoir  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_reservoir_keeps_cap_and_mixes_old_and_new():
    r = Reservoir(100, np.random.default_rng(0))
    r.add(np.arange(1000))
    assert len(r.items) == 100 and r.seen == 1000
    assert min(r.items) < 500 < max(r.items)  # not just the newest items
    assert len(set(r.sample(50))) == 50


def test_run_stream_end_to_end(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_train import _fake_compact
    data, runs, streams = tmp_path / "data", tmp_path / "runs", tmp_path / "streams"
    data.mkdir()
    streams.mkdir()
    _fake_compact(str(data), per_slide=30)
    py = sys.executable
    r = subprocess.run([py, os.path.join(ROOT, "scripts", "train_source.py"), "--data", str(data),
                        "--out", str(runs), "--epochs", "1", "--patience", "1", "--batch", "32",
                        "--workers", "0", "--n_eval", "40", "--no_pretrained"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    spec = importlib.util.spec_from_file_location("bs", os.path.join(ROOT, "scripts", "build_streams.py"))
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    man = bs.load_manifest(str(data))
    st = bs.make_streams(man, "pcam_fwd", 0, T=4, n=16, tumour_share=0.5)
    np.savez(streams / "pcam_fwd_seed0.npz", **st)
    r = subprocess.run([py, os.path.join(ROOT, "scripts", "run_stream.py"), "--data", str(data),
                        "--runs", str(runs), "--streams", str(streams), "--workers", "0",
                        "--n_buffer", "40", "--buffer", "20", "--batch", "8"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    run = runs / "pcam_fwd_resnet18_seed0"
    for name in ["frozen_gradual", "ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift"]:
        m = pd.read_csv(run / f"stream_{name}" / "metrics.csv")
        assert sorted(m.checkpoint.unique()) == list(range(5))
        assert set(m.set) == {"S", "G", "B", "E"}
        assert (m[m.set == "E"].checkpoint.max()) == 3
    fz = pd.read_csv(run / "stream_frozen_gradual" / "metrics.csv")
    s = fz[fz.set == "S"].brier
    assert s.max() - s.min() < 1e-6  # frozen model never changes on S
    lg = np.load(run / "stream_ft_gradual" / "logits.npz")
    assert "S_c0" in lg and "B_c4" in lg
