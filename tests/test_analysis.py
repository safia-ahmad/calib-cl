"""Analysis on a fake results folder with a planted answer (rho = 0.75, replay saves DSC)."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from calcl import analysis as A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)


def _metrics(path, d_mcb, d_dsc):
    rows = []
    for c in [0, 10]:
        f = c / 10
        rows.append({"checkpoint": c, "set": "S", "mcb": 0.01 + f * d_mcb, "dsc": 0.16 + f * d_dsc,
                     "brier": 0.25 + 0.01 + f * d_mcb - 0.16 - f * d_dsc})
    os.makedirs(path, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(path, "metrics.csv"), index=False)
    json.dump({"minutes": 1.0 + ("replay" in path)}, open(os.path.join(path, "meta.json"), "w"))


def _logits(path, y_S, y_B, idx_S, idx_B, scale_end):
    def z(y, s):
        d = rng.normal(0, 1, len(y)) + (2 * y - 1) * 2.0
        return np.stack([np.zeros(len(y)), d * s], 1).astype(np.float16)
    np.savez(os.path.join(path, "logits.npz"), idx_S=idx_S, idx_B=idx_B, B_c0=z(y_B, 1.0),
             B_c10=z(y_B, scale_end), S_c10=z(y_S, scale_end))


def _fake(tmp):
    out, models = tmp / "out", tmp / "models"
    n = 4000
    labels = (np.arange(n) % 2).astype(int)
    wsi = np.array([f"s{i % 20}" for i in range(n)])
    idx_S, idx_B = np.arange(0, 2000), np.arange(2000, 3000)
    for setting, model in A.CELLS:
        for seed in A.SEEDS:
            rd = out / f"{setting}_{model}_seed{seed}"
            md = models / f"{setting}_{model}_seed{seed}"
            md.mkdir(parents=True)
            json.dump({"T0": 1.0}, open(md / "summary.json", "w"))
            j = rng.normal(0, 0.002)
            _metrics(str(rd / "stream_frozen_gradual"), 0, 0)
            _metrics(str(rd / "stream_ft_gradual"), 0.03 + j, -0.02 + j)
            _metrics(str(rd / "stream_ft_nodrift"), 0.0, -0.01)
            _metrics(str(rd / "stream_replay_gradual"), 0.01 + j, -0.001)
            _metrics(str(rd / "stream_replay_nodrift"), 0.0, 0.0)
            _logits(str(rd / "stream_ft_gradual"), labels[idx_S], labels[idx_B], idx_S, idx_B, 3.0)
            _logits(str(rd / "stream_replay_gradual"), labels[idx_S], labels[idx_B], idx_S, idx_B, 1.0)
    return out, models, labels, wsi


def test_planted_answers(tmp_path):
    out, models, labels, wsi = _fake(tmp_path)
    assert A.frozen_check(str(out)) == 0
    res, h3 = A.hypotheses(A.drift_table(str(out)))
    assert res["P1"] == "holds"
    assert abs(res["H1_rho"] - 0.75) < 0.05 and res["H1"] == "supported"
    assert res["H2"] == "supported"  # FT loses 0.01 more DSC than replay (0.009)
    assert res["H3"] == "consistent"
    t4 = A.h4_table(str(out), str(models), labels, wsi)
    assert len(t4) == 4 * 5 * 4 and t4.regret_rule.ge(-1e-12).all()
    s4 = A.h4_summary(t4)
    assert set(s4.k) == set(A.KS)


def test_script_runs(tmp_path):
    out, models, labels, wsi = _fake(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"wsi": wsi, "label": labels}).to_csv(data / "pcam_compact_manifest.csv", index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "analyze.py"), "--data", str(data),
                        "--models", str(models), "--out", str(out)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "H1" in r.stdout and (out / "analysis" / "H4_summary.csv").exists()
