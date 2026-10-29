"""Day 15: build the gradual and no-drift streams, after checking there are enough patches.

Each stream = T windows of n patches, no patch used twice within a stream.
  gradual  : share of target patches follows alpha_schedule(T, "linear", warmup=1, plateau=1)
  no-drift : every window from the source hospital (the twin)
Both streams draw only from the `stream` slides of the split, with a fixed tumour share.

Usage (from the repo root):
  # 1) supply check only: which window sizes / tumour shares are feasible for every seed?
  python scripts/build_streams.py --data <folder with manifest> --check
  # 2) build and save the chosen design
  python scripts/build_streams.py --data <folder> --out <streams folder> --window 1000 --tumour_share 0.5

Saves <out>/<setting>_seed<seed>.npz with arrays gradual (T, n), nodrift (T, n) of
manifest row indices, the alphas, and the realised target share per window.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl.data import indices_for, slide_table  # noqa: E402
from calcl.splits import SETTINGS, make_split  # noqa: E402
from calcl.stream import Pool, alpha_schedule, build_stream  # noqa: E402


def load_manifest(folder: str) -> pd.DataFrame:
    man = pd.read_csv(os.path.join(folder, "pcam_compact_manifest.csv"))
    man["slide_type"] = np.where(man["image"].str.startswith("tumor"), "tumour", "normal")
    return man


def stream_pool(man, setting, seed):
    split = make_split(slide_table(man), setting, seed)
    src = indices_for(man, split[(split.domain == "source") & (split.role == "stream")].wsi)
    tgt = indices_for(man, split[(split.domain == "target") & (split.role == "stream")].wsi)
    rows = np.concatenate([src, tgt])
    pool = Pool(labels=man.label.to_numpy()[rows],
                domains=np.r_[np.zeros(len(src), int), np.ones(len(tgt), int)])
    return rows, pool


def make_streams(man, setting, seed, T=10, n=1000, tumour_share=0.5):
    """Returns dict with gradual / nodrift windows as manifest row indices.
    Raises RuntimeError('pool exhausted ...') when the supply is too small."""
    rows, pool = stream_pool(man, setting, seed)
    probs = np.array([1 - tumour_share, tumour_share])
    alphas = alpha_schedule(T, "linear", warmup=1, plateau=1)
    grad = build_stream(pool, alphas, n, probs, seed=10 * seed + 1)
    nod = build_stream(pool, np.zeros(T), n, probs, seed=10 * seed + 2)
    return {"alphas": alphas,
            "gradual": np.stack([rows[w] for w in grad]),
            "nodrift": np.stack([rows[w] for w in nod]),
            "target_share": np.array([pool.domains[w].mean() for w in grad]),
            "tumour_share_gradual": np.array([pool.labels[w].mean() for w in grad])}


def supply(man, setting, seed):
    rows, pool = stream_pool(man, setting, seed)
    return {f"{d}_{c}": int(((pool.domains == di) & (pool.labels == ci)).sum())
            for di, d in enumerate(["src", "tgt"]) for ci, c in enumerate(["normal", "tumour"])}


def check(man, seeds, windows, shares):
    out = []
    for setting in SETTINGS:
        for seed in seeds:
            s = supply(man, setting, seed)
            for n in windows:
                for share in shares:
                    try:
                        make_streams(man, setting, seed, n=n, tumour_share=share)
                        ok = "OK"
                    except RuntimeError as e:
                        ok = "EXHAUSTED " + str(e).split(": ")[-1]
                    out.append({"setting": setting, "seed": seed, "window": n,
                                "tumour_share": share, **s, "status": ok})
    return pd.DataFrame(out)


def main(a):
    man = load_manifest(a.data)
    seeds = range(a.seeds)
    if a.check:
        tab = check(man, seeds, [1000, 750, 500], [0.5, 0.3])
        pd.set_option("display.width", 160)
        print("Supply = patches available in the stream slides (src = source hospital, tgt = target).")
        print("Each design needs: gradual stream about 5n source + 5n target patches; no-drift twin 10n source.\n")
        print(tab.to_string(index=False))
        ok = tab.groupby(["window", "tumour_share"]).status.apply(lambda s: (s == "OK").all())
        print("\nDesigns feasible for EVERY setting and seed:")
        print(ok.to_string())
        return
    os.makedirs(a.out, exist_ok=True)
    for setting in SETTINGS:
        for seed in seeds:
            st = make_streams(man, setting, seed, n=a.window, tumour_share=a.tumour_share)
            path = os.path.join(a.out, f"{setting}_seed{seed}.npz")
            np.savez(path, **st)
            print(f"{setting} seed {seed}: saved {path} | target share per window "
                  f"{np.round(st['target_share'], 2).tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/content/data")
    p.add_argument("--out", default="/content/drive/MyDrive/calib-cl-data/streams")
    p.add_argument("--seeds", type=int, default=6, help="build seeds 0..seeds-1")
    p.add_argument("--window", type=int, default=1000)
    p.add_argument("--tumour_share", type=float, default=0.5)
    p.add_argument("--check", action="store_true", help="only report which designs fit the supply")
    main(p.parse_args())
