"""Addendum 4, X2 + X3 (exploratory). Reads saved logits only: no training, CPU is enough.

X2  Isotonic recalibration as a deployable fix ("R-iso"): isotonic regression (PAV, clipped to [0,1])
    fitted on the same k-patch subset of B that H4 uses, for the end-of-stream FT model, applied on S.
    Compared with R (temperature refit), P (replay at T0) and R+P (temperature refit of replay).
X3  Threshold reset: on the same subset, choose the threshold for the end-of-stream FT model that
    restores the source model's (checkpoint 0) sensitivity on that subset; report sensitivity and
    specificity on S with it, versus the default threshold 0.5.
Also prints the size and tumour share of B (needed for Methods).

Usage (Colab):
  python scripts/ext_fixes.py --data $D --models $D/models --out $D/stream_runs
Writes <out>/analysis/ext_fixes/{X2_per_seed, X2_summary, X3_per_seed, X3_summary, B_size}.csv
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


def prob(z, T):
    return softmax(np.asarray(z, np.float64) / T, axis=1)[:, 1]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def sens_spec(p, y, t):
    pred = p >= t
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def main(a):
    labels = pd.read_csv(os.path.join(a.data, "pcam_compact_manifest.csv"), usecols=["label"]).label.to_numpy()
    x2, x3, bs = [], [], []
    for setting, model in A.CELLS:
        for seed in A.SEEDS:
            rd = A.run_dir(a.out, setting, model, seed)
            T0 = json.load(open(os.path.join(a.models, f"{setting}_{model}_seed{seed}", "summary.json")))["T0"]
            ft = np.load(os.path.join(rd, "stream_ft_gradual", "logits.npz"))
            rp = np.load(os.path.join(rd, "stream_replay_gradual", "logits.npz"))
            yS, yB = labels[ft["idx_S"]], labels[ft["idx_B"]]
            bs.append({"setting": setting, "model": model, "seed": seed, "B_size": len(yB),
                       "B_tumour": int(yB.sum()), "B_tumour_share": round(float(yB.mean()), 3)})
            pS0, pS10 = prob(ft["S_c0"], T0), prob(ft["S_c10"], T0)
            for k in A.KS:
                sub = A.balanced_subset(yB, k, seed=7919 * seed + k)  # identical to H4
                zB10 = np.asarray(ft["B_c10"][sub], np.float64)
                # X2: fixes, Brier on S
                T_R = fit_temperature(zB10, yB[sub])
                T_RP = fit_temperature(np.asarray(rp["B_c10"][sub], np.float64), yB[sub])
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
                iso.fit(prob(zB10, T0), yB[sub])
                b = {"FT_end": brier(pS10, yS), "R": brier(prob(ft["S_c10"], T_R), yS),
                     "R_iso": brier(iso.predict(pS10), yS), "P": brier(prob(rp["S_c10"], T0), yS),
                     "R+P": brier(prob(rp["S_c10"], T_RP), yS)}
                best = min(v for kk, v in b.items() if kk != "FT_end")
                x2.append({"setting": setting, "model": model, "seed": seed, "k": k, "n_sub": len(sub),
                           **{f"brier_{kk}": v for kk, v in b.items()},
                           **{f"regret_{kk}": b[kk] - best for kk in ["R", "R_iso", "P", "R+P"]},
                           "best": min(["R", "R_iso", "P", "R+P"], key=lambda kk: b[kk])})
                # X3: threshold reset
                yb = yB[sub]
                p0b, p10b = prob(ft["B_c0"][sub], T0), prob(zB10, T0)
                target = float((p0b[yb == 1] >= 0.5).mean())
                pos = np.sort(p10b[yb == 1])
                n_keep = int(round(target * len(pos)))
                t_new = 1.0 if n_keep == 0 else float(pos[len(pos) - n_keep])
                s0, sp0 = sens_spec(pS0, yS, 0.5)
                s1, sp1 = sens_spec(pS10, yS, 0.5)
                s2, sp2 = sens_spec(pS10, yS, t_new)
                x3.append({"setting": setting, "model": model, "seed": seed, "k": k, "threshold": t_new,
                           "sens_c0": s0, "spec_c0": sp0, "sens_c10_t05": s1, "spec_c10_t05": sp1,
                           "sens_c10_reset": s2, "spec_c10_reset": sp2})
    d = os.path.join(a.out, "analysis", "ext_fixes")
    os.makedirs(d, exist_ok=True)
    x2, x3, bs = pd.DataFrame(x2), pd.DataFrame(x3), pd.DataFrame(bs)
    for name, t in [("X2_per_seed", x2), ("X3_per_seed", x3), ("B_size", bs)]:
        t.to_csv(os.path.join(d, f"{name}.csv"), index=False)
    pd.set_option("display.width", 250)

    print("=== B (labelled buffer) size per cell: min-max over seeds ===")
    print(bs.groupby(["setting", "model"])[["B_size", "B_tumour", "B_tumour_share"]].agg(["min", "max"]).to_string())

    print("\n=== X2 Isotonic fix (R_iso) vs R, P, R+P: mean Brier on S and regret [95% t over seeds] ===")
    rows = []
    for (s, m, k), g in x2.groupby(["setting", "model", "k"]):
        r = {"setting": s, "model": m, "k": k, "FT_end": g.brier_FT_end.mean()}
        for kk in ["R", "R_iso", "P", "R+P"]:
            r[f"brier_{kk}"] = g[f"brier_{kk}"].mean()
        mm, lo, hi = tci(g.brier_R_iso - g.brier_R)
        r["R_iso_minus_R"] = f"{mm:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        mm, lo, hi = tci(g.brier_R_iso - g.brier_P)
        r["R_iso_minus_P"] = f"{mm:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        r["best_counts"] = dict(g.best.value_counts())
        rows.append(r)
    s2 = pd.DataFrame(rows)
    s2.to_csv(os.path.join(d, "X2_summary.csv"), index=False)
    print(s2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== X3 Threshold reset on S (mean [95% t] over seeds) ===")
    rows = []
    for (s, m, k), g in x3.groupby(["setting", "model", "k"]):
        r = {"setting": s, "model": m, "k": k, "threshold": g.threshold.mean()}
        for col in ["sens_c0", "spec_c0", "sens_c10_t05", "spec_c10_t05", "sens_c10_reset", "spec_c10_reset"]:
            r[col] = g[col].mean()
        for lab, col0, col1 in [("d_sens_reset_vs_t05", "sens_c10_t05", "sens_c10_reset"),
                                ("d_spec_reset_vs_t05", "spec_c10_t05", "spec_c10_reset")]:
            mm, lo, hi = tci(g[col1] - g[col0])
            r[lab] = f"{mm:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(r)
    s3 = pd.DataFrame(rows)
    s3.to_csv(os.path.join(d, "X3_summary.csv"), index=False)
    print(s3.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved in {d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
