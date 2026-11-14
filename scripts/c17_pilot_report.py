"""Addendum 5, Section 10: pilot report. Shows ONLY runtime, GPU, memory, storage and integrity;
it never prints a performance number (no AUROC, Brier, loss, accuracy or temperature).

Usage:
  python scripts/c17_pilot_report.py --models $D2/models --out $D2/stream_runs --data /content/c17data \
      --setting c17_1to4 --seed 1
"""
import argparse
import json
import os

import numpy as np

RUNS = ["frozen_gradual", "ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift"]
N_SOURCE, N_STREAM, N_BN = 20, 100, 40


def folder_size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e6


def main(a):
    src = os.path.join(a.models, f"{a.setting}_resnet18_seed{a.seed}")
    s = json.load(open(os.path.join(src, "summary.json")))
    problems = []
    for f in ("source.pt", "split.csv", "eval_idx.npz", "logits_S_source_test.npz", "logits_G_target_test.npz"):
        if not os.path.exists(os.path.join(src, f)):
            problems.append(f"missing {f}")
    for f in ("logits_S_source_test.npz", "logits_G_target_test.npz"):
        p = os.path.join(src, f)
        if os.path.exists(p) and not np.isfinite(np.load(p)["z"].astype(np.float64)).all():
            problems.append(f"non-finite logits in {f}")
    print("=== Pilot report (runtime and integrity only; no performance numbers) ===")
    print(f"source model: {s.get('minutes', float('nan')):.1f} min on {s.get('gpu', '?')}, peak GPU memory "
          f"{s.get('peak_gpu_mem_gb', 0):.2f} GB, epochs run {len(s.get('history', []))}, "
          f"patches train {s.get('n_train')}, val_es {s.get('n_val_es')}, val_temp {s.get('n_val_temp')}, "
          f"folder {folder_size(src):.0f} MB")
    mins = []
    n_expected = 11
    if a.streams:
        n_expected = len(np.load(os.path.join(a.streams, f"{a.setting}_seed{a.seed}.npz"))["gradual"]) + 1
    run_root = os.path.join(a.out, os.path.basename(src))
    for r in RUNS:
        d = os.path.join(run_root, f"stream_{r}")
        if not os.path.exists(os.path.join(d, "metrics.csv")):
            problems.append(f"stream run {r} has no metrics.csv")
            continue
        m = json.load(open(os.path.join(d, "meta.json")))
        lg = np.load(os.path.join(d, "logits.npz"))
        n_ck = sum(1 for k in lg.files if k.startswith("S_c"))
        if n_ck != n_expected:
            problems.append(f"{r}: {n_ck} checkpoints saved, expected {n_expected}")
        if "idx_B" in lg.files:
            problems.append(f"{r}: unexpected buffer set B")
        if any(not np.isfinite(lg[k].astype(np.float64)).all() for k in lg.files if not k.startswith("idx_")):
            problems.append(f"{r}: non-finite logits")
        mins.append(m["minutes"])
        print(f"{r:15s} {m['minutes']:.1f} min on {m.get('gpu', '?')}, peak GPU memory "
              f"{m.get('peak_gpu_mem_gb', 0):.2f} GB, steps {m.get('steps')} (AMP-skipped {m.get('skipped_steps', 0)}), folder {folder_size(d):.1f} MB")
    if mins:
        per_run, per_src = float(np.mean(mins)), float(s.get("minutes", 0))
        main_h = (N_SOURCE * per_src + N_STREAM * per_run) / 60
        bn_h = N_BN * float(np.mean([m for r, m in zip(RUNS, mins) if r.startswith("ft")] or mins)) / 60
        print(f"\nEstimated GPU time (this GPU type): 20 source models + 100 stream runs = {main_h:.1f} h; "
              f"frozen-BN control (40 runs) = {bn_h:.1f} h; total {main_h + bn_h:.1f} h")
        print(f"Estimated storage: about {(folder_size(src) * N_SOURCE + folder_size(run_root) * N_SOURCE * 1.4) / 1e3:.1f} GB "
              f"for models and stream results, plus the data files")
    print("\nIntegrity: " + ("OK" if not problems else "PROBLEMS:\n  - " + "\n  - ".join(problems)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--streams", default="", help="optional: stream folder, to check the number of checkpoints")
    p.add_argument("--setting", default="c17_1to4")
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())
