"""Addendum 5, Section 4: build the compact Camelyon17 file (cap 3,000 per class per slide, seed 0).

Reads WILDS metadata.csv + patch PNGs, writes the same format as the PCam compact file so the
existing loader (calcl.data.load_compact) works unchanged:
  <out>/c17_compact.h5               x: (N, 96, 96, 3) uint8, y: (N,) uint8
  <out>/c17_compact_manifest.csv     one row per patch, same order as x
Checks before finishing: per-patient capped counts equal the committed paper/c17_units.csv, and
300 random patches re-read from PNG are byte-identical to what was stored.

Usage (Colab):
  python scripts/c17_build_data.py --wilds /content/wilds/camelyon17_v1.0 --out /content/c17data \
      --copy_to $D2/data
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from calcl import c17_split as C  # noqa: E402


def png_path(root, p, n, x, y):
    return os.path.join(root, "patches", f"patient_{p:03d}_node_{n}",
                        f"patch_patient_{p:03d}_node_{n}_x_{int(x)}_y_{int(y)}.png")


def read_png(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"))
    assert a.shape == (96, 96, 3), f"{path}: shape {a.shape}"
    return a


def manifest_from_meta(meta: pd.DataFrame) -> pd.DataFrame:
    capped = C.cap_patches(meta[meta.center.isin([0, 1, 3, 4])])
    man = capped.reset_index().rename(columns={"index": "wilds_index"})
    man["wsi"] = [f"c17_patient{p:03d}_node{n}" for p, n in zip(man.patient, man.node)]
    has_t = man.groupby("wsi").tumor.transform("max") > 0
    man["image"] = np.where(has_t, "tumor_", "normal_") + man.wsi   # slide_type for load_compact
    man["hospital"] = "c" + man.center.astype(str)
    man["label"] = man.tumor.astype(int)
    return man[["wsi", "image", "hospital", "label", "patient", "node", "x_coord", "y_coord", "tumor",
                "slide", "center", "wilds_index"]]


def check_units(man: pd.DataFrame, units_csv: str):
    got = C.unit_table(man).sort_values(["center", "patient"]).reset_index(drop=True)
    ref = pd.read_csv(units_csv)
    ref = ref[ref.center.isin(got.center.unique())]  # the committed table also lists centre 2, which is unused
    ref = ref.sort_values(["center", "patient"]).reset_index(drop=True)
    cols = ["center", "patient", "slides", "t", "n"]
    if not got[cols].astype(int).equals(ref[cols].astype(int)):
        m = got[cols].merge(ref[cols], on=["center", "patient"], how="outer", suffixes=("_built", "_committed"))
        raise AssertionError("capped supply differs from c17_units.csv:\n" + m.to_string())


def build(wilds: str, out: str, units_csv: str, workers: int = 16, batch: int = 4096, n_verify: int = 300):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    meta = pd.read_csv(os.path.join(wilds, "metadata.csv"), index_col=0)
    man = manifest_from_meta(meta)
    check_units(man, units_csv)
    print(f"{len(man):,} patches after the cap; per-patient counts match {os.path.basename(units_csv)}", flush=True)
    paths = [png_path(wilds, p, n, x, y) for p, n, x, y in zip(man.patient, man.node, man.x_coord, man.y_coord)]
    h5 = os.path.join(out, "c17_compact.h5")
    with h5py.File(h5, "w") as f, ThreadPoolExecutor(workers) as ex:
        X = f.create_dataset("x", (len(man), 96, 96, 3), dtype=np.uint8, chunks=(min(256, len(man)), 96, 96, 3),
                             compression="lzf")
        f["y"] = man.label.to_numpy().astype(np.uint8)
        for s in range(0, len(man), batch):
            X[s:s + batch] = np.stack(list(ex.map(read_png, paths[s:s + batch])))
            print(f"  {min(s + batch, len(man)):,}/{len(man):,} written ({(time.time() - t0) / 60:.1f} min)", flush=True)
    man.to_csv(os.path.join(out, "c17_compact_manifest.csv"), index=False)
    rng = np.random.default_rng(1)
    with h5py.File(h5, "r") as f:
        for i in np.sort(rng.choice(len(man), min(n_verify, len(man)), replace=False)):
            assert np.array_equal(f["x"][i], read_png(paths[i])), f"row {i} differs from its PNG"
        assert np.array_equal(f["y"][:], man.label.to_numpy())
    print(f"verified {min(n_verify, len(man))} random patches against their PNGs; "
          f"file size {os.path.getsize(h5) / 1e9:.2f} GB; {(time.time() - t0) / 60:.1f} min")
    print(man.groupby("hospital").label.agg(patches="size", tumour="sum").to_string())
    return man


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wilds", required=True)
    p.add_argument("--out", required=True, help="local folder (fast disk)")
    p.add_argument("--copy_to", default="", help="optional: also copy both files here (e.g. Drive)")
    p.add_argument("--units", default=os.path.join(ROOT, "paper", "c17_units.csv"))
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    build(a.wilds, a.out, a.units, a.workers)
    if a.copy_to:
        os.makedirs(a.copy_to, exist_ok=True)
        for fn in ("c17_compact.h5", "c17_compact_manifest.csv"):
            shutil.copy(os.path.join(a.out, fn), os.path.join(a.copy_to, fn))
        print(f"copied to {a.copy_to}")
