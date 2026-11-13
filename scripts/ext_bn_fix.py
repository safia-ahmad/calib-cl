"""Addendum 4c, X6c (exploratory): frozen-BN fine-tuning followed by recalibration. Saved logits only; CPU.

Usage (Colab):
  python scripts/ext_bn_fix.py --data $D --models $D/models --out $D/stream_runs
Writes <out>/analysis/ext_fixes/X6c_per_seed.csv and X6c_summary.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import analysis as A  # noqa: E402
from calcl.metrics import fit_temperature  # noqa: E402

CELLS = [("pcam_fwd", "resnet18"), ("pcam_rev", "resnet18")]


def prob(z, T):
    return softmax(np.asarray(z, np.float64) / T, axis=1)[:, 1]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def main(a):
    labels = pd.read_csv(os.path.join(a.data, "pcam_compact_manifest.csv"), usecols=["label"]).label.to_numpy()
    rows = []
    for setting, model in CELLS:
        for seed in A.SEEDS:
            rd = A.run_dir(a.out, setting, model, seed)
            T0 = json.load(open(os.path.join(a.models, f"{setting}_{model}_seed{seed}", "summary.json")))["T0"]
            ft = np.load(os.path.join(rd, "stream_ft_gradual", "logits.npz"))
            bn = np.load(os.path.join(rd, "stream_ft_gradual_bnfrozen", "logits.npz"))
            rp = np.load(os.path.join(rd, "stream_replay_gradual", "logits.npz"))
            assert np.array_equal(ft["idx_S"], bn["idx_S"]) and np.array_equal(ft["idx_B"], bn["idx_B"])
            yS, yB = labels[ft["idx_S"]], labels[ft["idx_B"]]
            for k in A.KS:
                sub = A.balanced_subset(yB, k, seed=7919 * seed + k)  # identical to H4
                zbn = np.asarray(bn["B_c10"][sub], np.float64)
                T_bn = fit_temperature(zbn, yB[sub])
                T_ft = fit_temperature(np.asarray(ft["B_c10"][sub], np.float64), yB[sub])
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
                iso.fit(prob(zbn, T0), yB[sub])
                rows.append({"setting": setting, "model": model, "seed": seed, "k": k,
                             "FT": brier(prob(ft["S_c10"], T0), yS),
                             "FT+R": brier(prob(ft["S_c10"], T_ft), yS),
                             "P": brier(prob(rp["S_c10"], T0), yS),
                             "FT-BN": brier(prob(bn["S_c10"], T0), yS),
                             "FT-BN+R": brier(prob(bn["S_c10"], T_bn), yS),
                             "FT-BN+R-iso": brier(iso.predict(prob(bn["S_c10"], T0)), yS)})
    t = pd.DataFrame(rows)
    d = os.path.join(a.out, "analysis", "ext_fixes")
    os.makedirs(d, exist_ok=True)
    t.to_csv(os.path.join(d, "X6c_per_seed.csv"), index=False)
    pd.set_option("display.width", 250)
    out = []
    for (s, m, k), g in t.groupby(["setting", "model", "k"]):
        r = {"setting": s, "model": m, "k": k}
        for c in ["FT", "FT+R", "P", "FT-BN", "FT-BN+R", "FT-BN+R-iso"]:
            r[c] = round(g[c].mean(), 4)
        for lab, x, y in [("FT-BN+R minus P", "FT-BN+R", "P"), ("FT-BN+R minus FT-BN", "FT-BN+R", "FT-BN"),
                          ("FT-BN minus FT", "FT-BN", "FT")]:
            mm, lo, hi = tci(g[x] - g[y])
            r[lab] = f"{mm:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        out.append(r)
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(d, "X6c_summary.csv"), index=False)
    print("=== X6c: Brier on S at end of stream (mean over seeds) and paired differences [95% t] ===")
    print(s.to_string(index=False))
    print(f"\nSaved in {d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
