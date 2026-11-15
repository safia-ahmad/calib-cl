"""Camelyon17 replication pipeline on small fake data: build, prepare, train (blind), pilot report."""
import importlib.util
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_build_data_matches_pngs(tmp_path):
    from PIL import Image
    B = _load("c17_build_data")
    rng = np.random.default_rng(0)
    rows = []
    for c, p in [(1, 20), (4, 92), (2, 42)]:
        d = tmp_path / "wilds" / "patches" / f"patient_{p:03d}_node_0"
        d.mkdir(parents=True)
        for k in range(12):
            x, y = 32 * k, 64
            Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).save(
                d / f"patch_patient_{p:03d}_node_0_x_{x}_y_{y}.png")
            rows.append({"patient": p, "node": 0, "x_coord": x, "y_coord": y, "tumor": k % 2,
                         "slide": p, "center": c, "split": 0})
    pd.DataFrame(rows).to_csv(tmp_path / "wilds" / "metadata.csv")
    from calcl import c17_split as C
    man = B.manifest_from_meta(pd.DataFrame(rows))
    u = C.unit_table(man)  # committed table also lists centre 2: add a fake centre-2 row
    pd.concat([u, pd.DataFrame([{"center": 2, "patient": 42, "slides": 1, "t": 6, "n": 6}])]).to_csv(
        tmp_path / "units.csv", index=False)
    out = B.build(str(tmp_path / "wilds"), str(tmp_path / "data"), str(tmp_path / "units.csv"), workers=2, batch=5)
    assert len(out) == 24 and set(out.hospital) == {"c1", "c4"}           # centre 2 is never used
    with h5py.File(tmp_path / "data" / "c17_compact.h5") as f:
        assert f["x"].shape == (24, 96, 96, 3)
    bad = pd.read_csv(tmp_path / "units.csv"); bad.loc[0, "t"] += 1; bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(AssertionError):
        B.check_units(man, str(tmp_path / "bad.csv"))


def test_prepare_cell_on_synthetic_metadata():
    from test_c17_split import synth_meta
    from calcl import c17_split as C
    P_ = _load("c17_prepare")
    man = C.cap_patches(synth_meta(), cap=3000).reset_index(drop=True)
    man["label"] = man.tumor
    res = P_.prepare_cell(man, "c17_1to4", [1, 2], 96)
    for r, sp, st in res:
        assert st["gradual"].shape == (10, 500) and st["nodrift"].shape == (10, 500)
        assert st["target_share"][0] == 0 and st["target_share"][-1] == 1
        ref = {"cell": r["cell"], "seed": r["seed"],
               **{f"{s}_{k}_patients": " ".join(map(str, v["patients"]))
                  for s, sup in (("src", r["src_supply"]), ("tgt", r["tgt_supply"])) for k, v in sup.items()},
               **{f"val_{k}": v for k, v in r["val"].items()},
               "S_n": r["S"]["n"], "S_t": r["S"]["t"], "G_n": r["G"]["n"], "G_t": r["G"]["t"]}
        P_.compare_with_report(r, pd.DataFrame([ref]))
        ref["src_test_patients"] = "1 2"
        with pytest.raises(AssertionError):
            P_.compare_with_report(r, pd.DataFrame([ref]))


def _fake_c17(folder, splits, streams):
    from test_c17_split import synth_meta  # noqa: F401  (keeps the import path warm)
    P_ = _load("c17_prepare")
    rng = np.random.default_rng(0)
    rows, imgs = [], []
    for c, p in [(1, 20), (1, 21), (1, 22), (4, 92), (4, 96)]:
        for k in range(40):
            lab = k % 2
            img = rng.integers(0, 120, (96, 96, 3)).astype(np.int16) + 100 * lab + (40 if c == 4 else 0)
            imgs.append(np.clip(img, 0, 255).astype(np.uint8))
            node = k // 20 if p in (22, 96) else 0  # test patients get 2 slides (CORP cross-fit needs 2 groups)
            w = f"c17_patient{p:03d}_node{node}"
            rows.append({"wsi": w, "image": "tumor_" + w, "hospital": f"c{c}", "label": lab, "patient": p, "node": node,
                         "x_coord": 32 * k, "y_coord": 0, "tumor": lab, "slide": p, "center": c, "wilds_index": len(rows)})
    man = pd.DataFrame(rows)
    with h5py.File(os.path.join(folder, "c17_compact.h5"), "w") as f:
        f["x"] = np.stack(imgs)
        f["y"] = man.label.to_numpy().astype(np.uint8)
    man.to_csv(os.path.join(folder, "c17_compact_manifest.csv"), index=False)
    rows_of = lambda p: np.flatnonzero(man.patient.to_numpy() == p)  # noqa: E731
    tr = rows_of(20)
    sp = {"train": tr[:30], "val_es": tr[30:35], "val_temp": tr[35:], "S": rows_of(22), "G": rows_of(96),
          "src_stream": rows_of(21), "tgt_stream": rows_of(92)}
    for seed in (1, 2, 3):
        np.savez(os.path.join(splits, f"c17_1to4_seed{seed}.npz"), **sp)
        json.dump({"cell": "c17_1to4", "seed": seed, "src_roles": {"20": "train", "21": "stream", "22": "test"},
                   "tgt_roles": {"92": "stream", "96": "test"}}, open(os.path.join(splits, f"c17_1to4_seed{seed}.json"), "w"))
        np.savez(os.path.join(streams, f"c17_1to4_seed{seed}.npz"),
                 **P_.streams_for(man, sp["src_stream"], sp["tgt_stream"], seed, T=3, n=8))


def test_c17_run_all_blind_bn_and_pilot(tmp_path):
    pytest.importorskip("torch")
    data, splits, streams, models, out = (tmp_path / n for n in ["data", "splits", "streams", "models", "out"])
    for d in (data, splits, streams):
        d.mkdir()
    _fake_c17(str(data), str(splits), str(streams))
    base = [sys.executable, os.path.join(ROOT, "scripts", "run_all.py"), "--data", str(data), "--models", str(models),
            "--streams", str(streams), "--out", str(out), "--lr", "0.001", "--model_list", "resnet18",
            "--settings", "c17_1to4", "--first_seed", "1", "--last_seed", "3", "--workers", "0", "--test_mode",
            "--data_prefix", "c17_compact", "--splits", str(splits), "--blind"]
    r = subprocess.run(base, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "18 of 18 jobs done" in r.stdout, r.stdout + r.stderr
    log = (out / "progress_log.txt").read_text()
    assert "FAILED" not in log
    # blind: no metric names printed by the child processes (their output goes to the console)
    assert "AUROC" not in r.stdout and "brier" not in r.stdout.lower()
    split = pd.read_csv(models / "c17_1to4_resnet18_seed1" / "split.csv")
    assert set(split.role) == {"train", "stream", "test"} and set(split.domain) == {"source", "target"}
    r = subprocess.run(base + ["--bn_control"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "9 of 9 jobs done" in r.stdout, r.stdout + r.stderr
    meta = json.load(open(out / "c17_1to4_resnet18_seed1" / "stream_ft_gradual_bnfrozen" / "meta.json"))
    assert meta["freeze_bn"] is True and "gpu" in meta
    rep = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "c17_pilot_report.py"), "--models", str(models),
                          "--out", str(out), "--streams", str(streams)], capture_output=True, text=True, timeout=300)
    assert rep.returncode == 0 and "Integrity: OK" in rep.stdout, rep.stdout + rep.stderr
    assert "auroc" not in rep.stdout.lower() and "brier" not in rep.stdout.lower()
    an = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "c17_analysis.py"), "--data", str(data),
                         "--models", str(models), "--out", str(out), "--splits", str(splits), "--cells", "c17_1to4",
                         "--primary", "c17_1to4", "--seeds", "1,2,3", "--insample_sensitivity", "--checkpoints", "4"],
                        capture_output=True, text=True, timeout=600)
    assert an.returncode == 0, an.stdout + an.stderr
    assert "OVERALL VERDICT" in an.stdout and (out / "analysis_c17" / "report.txt").exists()
    assert "AGREE" in an.stdout and (out / "analysis_c17" / "insample_cells.csv").exists()
    for f in ("seeds.csv", "cells.csv", "bn_control.csv", "per_patient.csv", "source_auroc.csv", "insample_seeds.csv"):
        assert (out / "analysis_c17" / f).exists(), f
    again = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "c17_analysis.py"), "--data", str(data),
                            "--models", str(models), "--out", str(out), "--splits", str(splits), "--cells", "c17_1to4",
                            "--primary", "c17_1to4", "--seeds", "1,2,3", "--checkpoints", "4"],
                           capture_output=True, text=True, timeout=600)
    assert again.returncode != 0 and "already" in (again.stdout + again.stderr)   # runs once only
    # a missing run makes the analysis stop with a listed problem, never a silent omission
    os.remove(out / "c17_1to4_resnet18_seed2" / "stream_replay_nodrift" / "metrics.csv")
    os.remove(out / "analysis_c17" / "report.txt")
    miss = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "c17_analysis.py"), "--data", str(data),
                           "--models", str(models), "--out", str(out), "--splits", str(splits), "--cells", "c17_1to4",
                           "--primary", "c17_1to4", "--seeds", "1,2,3", "--checkpoints", "4"],
                          capture_output=True, text=True, timeout=600)
    assert miss.returncode != 0 and "INVENTORY FAILED" in miss.stdout and "replay_nodrift" in miss.stdout
