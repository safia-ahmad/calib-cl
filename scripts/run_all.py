"""Run (or continue) the whole main experiment, one job at a time, skipping anything already done.

Start it every session with the same command. If Colab disconnects, just run it again:
finished source models and stream runs are skipped, so it carries on where it stopped.

Order: ResNet-18 first (all settings and seeds), then ViT-S/16.
For each (model, setting, seed):
  1. source model      -> <models>/<setting>_<model>_seed<seed>/summary.json
  2. stream runs       -> <out>/<setting>_<model>_seed<seed>/stream_<strategy>_<kind>/metrics.csv
                          (frozen-gradual, ft-gradual, ft-nodrift, replay-gradual, replay-nodrift)

Usage:
  python scripts/run_all.py --data /content/data --models $D/models --streams $D/streams --out $D/stream_runs --lr 0.003
  python scripts/run_all.py ... --status        # only print what is done / missing
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_SOURCE = ["--epochs", "1", "--patience", "1", "--batch", "32", "--n_eval", "40", "--no_pretrained"]
TEST_STREAM = ["--n_buffer", "40", "--buffer", "20", "--batch", "8"]
STREAM_RUNS = ["frozen_gradual", "ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift"]


def jobs(a):
    for model in a.model_list.split(","):
        for setting in a.settings.split(","):
            for seed in range(a.first_seed, a.last_seed + 1):
                yield model, setting, seed


def source_done(a, model, setting, seed):
    return os.path.exists(os.path.join(a.models, f"{setting}_{model}_seed{seed}", "summary.json"))


def stream_done(a, model, setting, seed, name):
    return os.path.exists(os.path.join(a.out, f"{setting}_{model}_seed{seed}", f"stream_{name}", "metrics.csv"))


def status(a):
    n_tot = n_done = 0
    print(f"{'model':9s} {'setting':9s} seed  source  " + "  ".join(f"{s:>14s}" for s in STREAM_RUNS))
    for model, setting, seed in jobs(a):
        src = source_done(a, model, setting, seed)
        st = [stream_done(a, model, setting, seed, s) for s in STREAM_RUNS]
        n_tot += 1 + len(st)
        n_done += src + sum(st)
        print(f"{model:9s} {setting:9s} {seed:4d}  {'done' if src else '-':6s}  " +
              "  ".join(f"{'done' if x else '-':>14s}" for x in st))
    print(f"\n{n_done} of {n_tot} jobs done ({100 * n_done / n_tot:.0f}%)")


def log(a, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(os.path.join(a.out, "progress_log.txt"), "a") as f:
        f.write(line + "\n")


def main(a):
    os.makedirs(a.out, exist_ok=True)
    if a.status:
        status(a)
        return
    for model, setting, seed in jobs(a):
        tag = f"{setting} {model} seed {seed}"
        if not source_done(a, model, setting, seed):
            log(a, f"START source model  {tag}")
            r = subprocess.run([sys.executable, os.path.join(HERE, "train_source.py"), "--setting", setting,
                                "--seed", str(seed), "--model", model, "--data", a.data, "--out", a.models,
                                "--workers", str(a.workers)] + (TEST_SOURCE if a.test_mode else []))
            if r.returncode != 0:
                log(a, f"FAILED source model {tag}: skipping its streams")
                continue
            log(a, f"done   source model  {tag}")
        if all(stream_done(a, model, setting, seed, s) for s in STREAM_RUNS):
            continue
        log(a, f"START stream runs   {tag}")
        r = subprocess.run([sys.executable, os.path.join(HERE, "run_stream.py"), "--setting", setting,
                            "--seed", str(seed), "--model", model, "--data", a.data, "--runs", a.models,
                            "--streams", a.streams, "--out", a.out, "--lr", str(a.lr),
                            "--workers", str(a.workers)] + (TEST_STREAM if a.test_mode else []))
        log(a, f"{'done  ' if r.returncode == 0 else 'FAILED'} stream runs   {tag}")
    status(a)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/content/data")
    p.add_argument("--models", required=True)
    p.add_argument("--streams", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, required=True, help="stream learning rate, fixed in prereg.md")
    p.add_argument("--model_list", default="resnet18,vit_s16")
    p.add_argument("--settings", default="pcam_fwd,pcam_rev")
    p.add_argument("--first_seed", type=int, default=1)
    p.add_argument("--last_seed", type=int, default=5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--status", action="store_true")
    p.add_argument("--test_mode", action="store_true", help="tiny settings for the automated test only")
    main(p.parse_args())
