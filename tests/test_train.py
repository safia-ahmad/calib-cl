"""End-to-end smoke test of scripts/train_source.py on a tiny synthetic dataset (CPU).
Skipped automatically if PyTorch is not installed."""
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_compact(folder, per_slide=12):
    rng = np.random.default_rng(0)
    rows, imgs = [], []
    for h, n_norm, n_tum, tint in [("RUMC", 20, 14, 0), ("UMCU", 12, 8, 40)]:
        for t, n in [("normal", n_norm), ("tumor", n_tum)]:
            for i in range(n):
                wsi = f"camelyon16_train_{t}_{h}{i:03d}"
                for k in range(per_slide):
                    lab = int(t == "tumor" and k % 2 == 0)
                    img = rng.integers(0, 120, (96, 96, 3)).astype(np.int16) + 100 * lab
                    img[..., 2] += tint
                    imgs.append(np.clip(img, 0, 255).astype(np.uint8))
                    rows.append({"wsi": wsi, "image": f"{t}_{h}{i:03d}.tif", "hospital": h, "label": lab})
    man = pd.DataFrame(rows)
    with h5py.File(os.path.join(folder, "pcam_compact.h5"), "w") as f:
        f["x"] = np.stack(imgs)
        f["y"] = man.label.to_numpy().astype(np.uint8)
    man.to_csv(os.path.join(folder, "pcam_compact_manifest.csv"), index=False)


def test_train_source_end_to_end(tmp_path):
    data, out = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    _fake_compact(str(data))
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "train_source.py"), "--setting", "pcam_fwd",
           "--seed", "0", "--model", "resnet18", "--data", str(data), "--out", str(out),
           "--epochs", "2", "--patience", "2", "--batch", "32", "--workers", "0", "--n_eval", "40",
           "--no_pretrained"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    run = out / "pcam_fwd_resnet18_seed0"
    s = json.loads((run / "summary.json").read_text())
    for key in ["T0", "history", "S_source_test", "G_target_test", "G4_brier_gap", "G4_pass"]:
        assert key in s
    for name in ["S_source_test", "G_target_test"]:
        m = s[name]
        assert abs(m["brier"] - (m["mcb"] - m["dsc"] + m["unc"])) < 1e-9
    assert (run / "source.pt").exists() and (run / "split.csv").exists()
    split = pd.read_csv(run / "split.csv")
    assert not split.wsi.duplicated().any()
