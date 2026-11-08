"""Essential analyses E-1 to E-5, S-1, S-2, exactly as defined in paper/prereg_addendum.md.
Reads only saved results (no GPU). Run after analyze.py.

Usage (Colab, CPU is enough):
  python scripts/essential_analyses.py --data $D --models $D/models --out $D/stream_runs
Writes CSVs and figures to <out>/analysis/essential/ and prints every table.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import softmax  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import analysis as A  # noqa: E402
from calcl.corp import corp_crossfit  # noqa: E402
from calcl.metrics import fit_temperature  # noqa: E402
from calcl.stream import alpha_schedule  # noqa: E402

NAMES = {("pcam_fwd", "resnet18"): "RUMC→UMCU ResNet-18", ("pcam_rev", "resnet18"): "UMCU→RUMC ResNet-18",
         ("pcam_fwd", "vit_s16"): "RUMC→UMCU ViT-S/16", ("pcam_rev", "vit_s16"): "UMCU→RUMC ViT-S/16"}


def tci(x):
    """Mean and 95% t-interval over seeds."""
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h


def metrics_S(out, setting, model, seed, run):
    m = pd.read_csv(os.path.join(A.run_dir(out, setting, model, seed), f"stream_{run}", "metrics.csv"))
    return m[m.set == "S"].set_index("checkpoint").sort_index()


def drift_series(out, setting, model, seed, strat, col):
    """Drift-attributable change of `col` on S at every checkpoint (gradual minus twin, each vs checkpoint 0)."""
    g = metrics_S(out, setting, model, seed, f"{strat}_gradual")[col]
    n = metrics_S(out, setting, model, seed, f"{strat}_nodrift")[col]
    return (g - g.iloc[0]) - (n - n.iloc[0])


def p1(z, T0):
    return softmax(np.asarray(z, np.float64) / T0, axis=1)[:, 1]


def T0_of(models, setting, model, seed):
    return json.load(open(os.path.join(models, f"{setting}_{model}_seed{seed}", "summary.json")))["T0"]


# ---------------- E-1 ----------------
def e1(dt):
    rows = []
    for setting, model in A.CELLS:
        for strat in ["ft", "replay"]:
            g = dt[(dt.setting == setting) & (dt.model == model) & (dt.strategy == strat)].sort_values("seed")
            row = {"cell": NAMES[(setting, model)], "strategy": strat}
            for c in ["d_brier", "d_mcb", "dsc_lost"]:
                m, lo, hi = tci(g[c])
                row.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi,
                            f"{c}_seeds": " ".join(f"{v:+.4f}" for v in g[c])})
            row["seeds_dsc_lost_positive"] = int((g.dsc_lost > 0).sum())
            row["seeds_dsc_lost_gt_abs_dmcb"] = int((g.dsc_lost > g.d_mcb.abs()).sum())
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------- E-2 ----------------
def e2(out, models, labels, wsi, n_boot=2000, seed=0):
    setting, model = A.MAIN
    per_seed = {}
    for s in A.SEEDS:
        rd = A.run_dir(out, setting, model, s)
        T0 = T0_of(models, setting, model, s)
        d = {r: np.load(os.path.join(rd, f"stream_{r}", "logits.npz"))
             for r in ["ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift"]}
        idx = d["ft_gradual"]["idx_S"]
        per_seed[s] = {"y": labels[idx], "slides": wsi[idx], "c0": p1(d["ft_gradual"]["S_c0"], T0),
                       **{r: p1(d[r]["S_c10"], T0) for r in d}}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        acc = {k: [] for k in ["ft_mcb", "ft_dsc", "rp_mcb", "rp_dsc"]}
        for s in rng.choice(A.SEEDS, size=len(A.SEEDS), replace=True):
            P = per_seed[s]
            uniq = np.unique(P["slides"])
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            ix = np.concatenate([np.flatnonzero(P["slides"] == u) for u in pick])
            grp = np.concatenate([np.full((P["slides"] == u).sum(), k) for k, u in enumerate(pick)])
            y = P["y"][ix]
            c = {name: corp_crossfit(P[name][ix], y, grp) for name in
                 ["c0", "ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift"]}
            for tag, strat in [("ft", "ft"), ("rp", "replay")]:
                g, n = c[f"{strat}_gradual"], c[f"{strat}_nodrift"]
                acc[f"{tag}_mcb"].append((g["mcb"] - c["c0"]["mcb"]) - (n["mcb"] - c["c0"]["mcb"]))
                acc[f"{tag}_dsc"].append(-((g["dsc"] - c["c0"]["dsc"]) - (n["dsc"] - c["c0"]["dsc"])))
        r = {k: float(np.mean(v)) for k, v in acc.items()}
        den = r["ft_mcb"] + r["ft_dsc"]
        r["rho_ft"] = r["ft_mcb"] / den if den > 0 else np.nan
        rows.append(r)
        if (b + 1) % 500 == 0:
            print(f"  E-2 bootstrap {b + 1}/{n_boot}", flush=True)
    bt = pd.DataFrame(rows)
    summ = []
    for col, name in [("ft_mcb", "FT dMCB"), ("ft_dsc", "FT DSC lost"), ("rho_ft", "FT rho"),
                      ("rp_mcb", "Replay dMCB"), ("rp_dsc", "Replay DSC lost")]:
        v = bt[col].dropna()
        summ.append({"quantity": name, "boot_mean": v.mean(), "lo95": np.percentile(v, 2.5),
                     "hi95": np.percentile(v, 97.5), "n_valid": len(v)})
    return pd.DataFrame(summ), bt


# ---------------- E-3 ----------------
def e3(out):
    rows = []
    for setting, model in A.CELLS:
        for strat in ["ft", "replay"]:
            row = {"cell": NAMES[(setting, model)], "strategy": strat}
            for col in ["auroc", "ece_mass", "ece_width", "acc"]:
                v = [drift_series(out, setting, model, s, strat, col).iloc[-1] for s in A.SEEDS]
                m, lo, hi = tci(v)
                row.update({f"d_{col}": m, f"d_{col}_lo": lo, f"d_{col}_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------- E-4 ----------------
def reliability(ax, p, y, label, color):
    order = np.argsort(p)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p[order], y[order])
    xs = np.linspace(0, 1, 201)
    ax.plot(xs, iso.predict(xs), color=color, lw=1.6, label=label)


def e4(out, models, labels, d):
    setting, model = "pcam_fwd", "vit_s16"
    traj = []
    for s in A.SEEDS:
        for col, sign, name in [("mcb", 1, "dMCB"), ("dsc", -1, "DSC_lost")]:
            ser = sign * drift_series(out, setting, model, s, "ft", col)
            for c, v in ser.items():
                traj.append({"seed": s, "checkpoint": c, "quantity": name, "value": v})
    traj = pd.DataFrame(traj)
    ends = []
    for label, cps in [("checkpoint 10 (pre-registered)", [10]), ("checkpoint 9", [9]), ("mean of 8-10", [8, 9, 10])]:
        sub = traj[traj.checkpoint.isin(cps)].groupby(["seed", "quantity"]).value.mean().unstack()
        m_mcb, l_mcb, h_mcb = tci(sub.dMCB)
        m_dsc, l_dsc, h_dsc = tci(sub.DSC_lost)
        den = m_mcb + m_dsc
        ends.append({"endpoint": label, "dMCB": m_mcb, "dMCB_lo": l_mcb, "dMCB_hi": h_mcb,
                     "DSC_lost": m_dsc, "DSC_lost_lo": l_dsc, "DSC_lost_hi": h_dsc,
                     "rho": m_mcb / den if den > 0 else np.nan,
                     "seeds_dMCB_positive": int((sub.dMCB > 0).sum())})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    for k, q in enumerate(["dMCB", "DSC_lost"]):
        for s in A.SEEDS:
            t = traj[(traj.seed == s) & (traj.quantity == q)]
            ax[k].plot(t.checkpoint, t.value, marker="o", ms=3, lw=1.2, label=f"seed {s}")
        ax[k].axhline(0, color="grey", lw=0.8)
        ax[k].set_title(f"RUMC→UMCU ViT-S/16, fine-tuning: {q} (drift − twin) per seed", fontsize=9)
        ax[k].set_xlabel("Windows trained")
    ax[0].legend(fontsize=7)
    fig.savefig(os.path.join(d, "figS2a_vit_seed_trajectories.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(d, "figS2a_vit_seed_trajectories.pdf"), bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(1, 6, figsize=(20, 3.6), sharey=True)
    pooled = {c: ([], []) for c in [0, 9, 10]}
    for j, s in enumerate(A.SEEDS):
        z = np.load(os.path.join(A.run_dir(out, setting, model, s), "stream_ft_gradual", "logits.npz"))
        T0 = T0_of(models, setting, model, s)
        y = labels[z["idx_S"]]
        for c, colr in [(0, "#6b7280"), (9, "#1d4ed8"), (10, "#c2410c")]:
            p = p1(z[f"S_c{c}"], T0)
            reliability(ax[j + 1], p, y, f"checkpoint {c}", colr)
            pooled[c][0].append(p)
            pooled[c][1].append(y)
        ax[j + 1].set_title(f"seed {s}", fontsize=9)
    for c, colr in [(0, "#6b7280"), (9, "#1d4ed8"), (10, "#c2410c")]:
        reliability(ax[0], np.concatenate(pooled[c][0]), np.concatenate(pooled[c][1]), f"checkpoint {c}", colr)
    ax[0].set_title("pooled over seeds", fontsize=9)
    for a in ax:
        a.plot([0, 1], [0, 1], ":", color="grey", lw=0.8)
        a.set_xlabel("Predicted P(tumour)")
    ax[0].set_ylabel("CORP recalibrated P(tumour)")
    ax[0].legend(fontsize=7)
    fig.suptitle("CORP reliability diagrams on S, RUMC→UMCU ViT-S/16, fine-tuning (gradual)", fontsize=10)
    fig.savefig(os.path.join(d, "figS2b_vit_reliability.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(d, "figS2b_vit_reliability.pdf"), bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(ends), traj


# ---------------- E-5 ----------------
def e5(out, models, labels):
    rows = []
    for setting, model in A.CELLS:
        gains = []
        for s in A.SEEDS:
            z = np.load(os.path.join(A.run_dir(out, setting, model, s), "stream_ft_gradual", "logits.npz"))
            y = labels[z["idx_S"]]
            zs = np.asarray(z["S_c10"], np.float64)
            T0 = T0_of(models, setting, model, s)
            Ts = fit_temperature(zs, y)
            b0 = np.mean((p1(zs, T0) - y) ** 2)
            b1 = np.mean((p1(zs, Ts) - y) ** 2)
            gains.append(b0 - b1)
        m, lo, hi = tci(gains)
        rows.append({"cell": NAMES[(setting, model)], "oracle_brier_gain": m, "lo": lo, "hi": hi,
                     "max_seed_gain": max(gains)})
    return pd.DataFrame(rows)


# ---------------- S-1 ----------------
def s1(out):
    alphas = alpha_schedule(10, "linear", warmup=1, plateau=1)
    rows = []
    for setting, model in A.CELLS:
        for strat in ["ft", "replay"]:
            mean = np.mean([-drift_series(out, setting, model, s, strat, "dsc").values for s in A.SEEDS], axis=0)
            onset = None
            for c in range(1, len(mean)):
                if np.all(mean[c:] >= A.SESOI):
                    onset = c
                    break
            rows.append({"cell": NAMES[(setting, model)], "strategy": strat,
                         "onset_checkpoint": onset if onset is not None else "none",
                         "target_share_of_that_window": round(float(alphas[onset - 1]), 3) if onset else "",
                         "DSC_lost_at_end": mean[-1]})
    return pd.DataFrame(rows)


# ---------------- S-2 ----------------
def s2(out, models, labels):
    rows = []
    for setting, model in A.CELLS:
        for strat in ["ft", "replay"]:
            dsen, dspe = [], []
            for s in A.SEEDS:
                rd = A.run_dir(out, setting, model, s)
                T0 = T0_of(models, setting, model, s)
                vals = {}
                for kind in ["gradual", "nodrift"]:
                    z = np.load(os.path.join(rd, f"stream_{strat}_{kind}", "logits.npz"))
                    y = labels[z["idx_S"]]
                    for c in [0, 10]:
                        pred = p1(z[f"S_c{c}"], T0) >= 0.5
                        vals[(kind, c)] = (pred[y == 1].mean(), (~pred)[y == 0].mean())
                for k, lst in [(0, dsen), (1, dspe)]:
                    lst.append((vals[("gradual", 10)][k] - vals[("gradual", 0)][k])
                               - (vals[("nodrift", 10)][k] - vals[("nodrift", 0)][k]))
            ms, ls, hs = tci(dsen)
            mp, lp, hp = tci(dspe)
            rows.append({"cell": NAMES[(setting, model)], "strategy": strat,
                         "d_sensitivity": ms, "sens_lo": ls, "sens_hi": hs,
                         "d_specificity": mp, "spec_lo": lp, "spec_hi": hp})
    return pd.DataFrame(rows)


def main(a):
    d = os.path.join(a.out, "analysis", "essential")
    os.makedirs(d, exist_ok=True)
    man = pd.read_csv(os.path.join(a.data, "pcam_compact_manifest.csv"), usecols=["wsi", "label"])
    labels, wsi = man.label.to_numpy(), man.wsi.to_numpy()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    dt = A.drift_table(a.out)

    t = e1(dt)
    t.to_csv(os.path.join(d, "E1_components.csv"), index=False)
    print("=== E-1 Components (drift − twin, checkpoint 10 vs 0, on S): mean [95% t-CI] ===")
    print(t[["cell", "strategy", "d_brier", "d_brier_lo", "d_brier_hi", "d_mcb", "d_mcb_lo", "d_mcb_hi",
             "dsc_lost", "dsc_lost_lo", "dsc_lost_hi", "seeds_dsc_lost_positive",
             "seeds_dsc_lost_gt_abs_dmcb"]].to_string(index=False, float_format=f4))
    print("\nSeed values (FT):")
    print(t[t.strategy == "ft"][["cell", "d_mcb_seeds", "dsc_lost_seeds"]].to_string(index=False))

    print(f"\n=== E-2 Hierarchical bootstrap, primary cell ({a.n_boot} resamples: seeds, then slides) ===")
    s, bt = e2(a.out, a.models, labels, wsi, n_boot=a.n_boot)
    s.to_csv(os.path.join(d, "E2_hierarchical_bootstrap.csv"), index=False)
    print(s.to_string(index=False, float_format=f4))

    t = e3(a.out)
    t.to_csv(os.path.join(d, "E3_auroc_ece.csv"), index=False)
    print("\n=== E-3 Drift-attributable change in AUROC, ECE (15 bins) and accuracy on S ===")
    print(t.to_string(index=False, float_format=f4))

    t, traj = e4(a.out, a.models, labels, d)
    t.to_csv(os.path.join(d, "E4_vit_endpoints.csv"), index=False)
    traj.to_csv(os.path.join(d, "E4_vit_trajectories.csv"), index=False)
    print("\n=== E-4 ViT RUMC→UMCU endpoint sensitivity (fine-tuning) ===")
    print(t.to_string(index=False, float_format=f4))

    t = e5(a.out, a.models, labels)
    t.to_csv(os.path.join(d, "E5_oracle_temperature.csv"), index=False)
    print("\n=== E-5 Oracle temperature on S at checkpoint 10 (upper bound on any single-temperature fix) ===")
    print(t.to_string(index=False, float_format=f4))

    t = s1(a.out)
    t.to_csv(os.path.join(d, "S1_onset.csv"), index=False)
    print("\n=== S-1 Onset of forgetting (first checkpoint where mean DSC lost stays ≥ 0.005) ===")
    print(t.to_string(index=False, float_format=f4))

    t = s2(a.out, a.models, labels)
    t.to_csv(os.path.join(d, "S2_sens_spec.csv"), index=False)
    print("\n=== S-2 Drift-attributable change in sensitivity and specificity on S (threshold 0.5, T0) ===")
    print(t.to_string(index=False, float_format=f4))
    print(f"\nAll tables and Figures S2a/S2b saved in {d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with pcam_compact_manifest.csv")
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    main(p.parse_args())
