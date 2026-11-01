"""run_all on fake data: does everything once, then skips everything on a second call."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_run_all_resumes(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_train import _fake_compact
    data, models, streams, out = (tmp_path / n for n in ["data", "models", "streams", "out"])
    data.mkdir(); streams.mkdir()
    _fake_compact(str(data), per_slide=30)
    spec = importlib.util.spec_from_file_location("bs", os.path.join(ROOT, "scripts", "build_streams.py"))
    bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
    np.savez(streams / "pcam_fwd_seed1.npz", **bs.make_streams(bs.load_manifest(str(data)), "pcam_fwd", 1, T=3, n=16))
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_all.py"), "--data", str(data), "--models", str(models),
           "--streams", str(streams), "--out", str(out), "--lr", "0.003", "--model_list", "resnet18",
           "--settings", "pcam_fwd", "--first_seed", "1", "--last_seed", "1", "--workers", "0", "--test_mode"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "6 of 6 jobs done" in r.stdout, r.stdout + r.stderr
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert "START" not in r2.stdout and "6 of 6 jobs done" in r2.stdout, r2.stdout
