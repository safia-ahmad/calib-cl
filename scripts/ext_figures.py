"""Addendum 4, X4: redesigned figures (no new results). Reads saved results only; CPU.

figA_decomposition_scatter : dMCB (x) vs DSC lost (y), one point per seed, all four cells (FT),
                             plus the primary-cell controls (replay, FT lr 0.0005, FT frozen BN)
                             and the reverse-ResNet frozen-BN control if present. Line rho = 0.25.
figB_raw_vs_twin           : primary cell, end-minus-start changes on S for the gradual run and its
                             twin (FT and replay), showing what the paired difference is made of.
figC_fixes                 : Brier on S at the end of the stream with no fix and with each fix
                             (R, R-iso, P, R+P), per cell, k = 500 (needs ext_fixes.py output).
figD_controls              : primary cell DSC lost for FT, FT lr 0.0005, FT frozen BN and replay.

Usage (Colab):
  python scripts/ext_figures.py --out $D/stream_runs
Writes PNG + PDF to <out>/analysis/figures_v2/ and the numbers behind them as CSV.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import analysis as A  # noqa: E402

NAMES = {("pcam_fwd", "resnet18"): "RUMC→UMCU, ResNet-18 (primary)",
         ("pcam_rev", "resnet18"): "UMCU→RUMC, ResNet-18",
         ("pcam_fwd", "vit_s16"): "RUMC→UMCU, ViT-S/16",
         ("pcam_rev", "vit_s16"): "UMCU→RUMC, ViT-S/16"}
COL = {("pcam_fwd", "resnet18"): "#c2410c", ("pcam_rev", "resnet18"): "#7c3aed",
       ("pcam_fwd", "vit_s16"): "#0f766e", ("pcam_rev", "vit_s16"): "#64748b"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, d, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def tci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def drift(out, setting, model, strat, sfx):
    """Per-seed drift-attributable (gradual minus twin) changes; None if the runs are missing."""
    A.SUFFIX = sfx
    rows = []
    try:
        for s in A.SEEDS:
            g = A.s_change(out, setting, model, s, f"{strat}_gradual")
            n = A.s_change(out, setting, model, s, f"{strat}_nodrift")
            rows.append({"seed": s, "d_brier": g["brier"] - n["brier"], "d_mcb": g["mcb"] - n["mcb"],
                         "dsc_lost": -(g["dsc"] - n["dsc"])})
    except FileNotFoundError:
        return None
    finally:
        A.SUFFIX = ""
    return pd.DataFrame(rows)


def raw(out, setting, model, run):
    rows = []
    for s in A.SEEDS:
        c = A.s_change(out, setting, model, s, run)
        rows.append({"seed": s, "d_brier": c["brier"], "d_mcb": c["mcb"], "d_dsc": c["dsc"]})
    return pd.DataFrame(rows)


def fig_scatter(out, d):
    series = [(cell, "ft", "", NAMES[cell] + ", FT", COL[cell], "o", 30) for cell in A.CELLS]
    series += [(A.MAIN, "replay", "", "primary, replay", "#1d4ed8", "s", 34),
               (A.MAIN, "ft", "_lrhalf", "primary, FT lr 0.0005", "#c2410c", "^", 34),
               (A.MAIN, "ft", "_bnfrozen", "primary, FT frozen BN", "#16a34a", "D", 30),
               (("pcam_rev", "resnet18"), "ft", "_bnfrozen", "UMCU→RUMC ResNet, FT frozen BN", "#7c3aed", "D", 30)]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rows = []
    for cell, strat, sfx, lab, c, mk, sz in series:
        t = drift(out, *cell, strat, sfx)
        if t is None:
            continue
        filled = sfx == "" and strat == "ft"
        ax.scatter(t.d_mcb, t.dsc_lost, s=sz, marker=mk, color=c if filled else "none", edgecolor=c,
                   linewidth=1.2, label=lab, zorder=3)
        ax.scatter(t.d_mcb.mean(), t.dsc_lost.mean(), s=sz * 4, marker=mk, color=c, edgecolor="k",
                   linewidth=0.8, zorder=4)
        rows += [{"series": lab, **r} for r in t.to_dict("records")]
    lim = max(0.06, np.abs(pd.DataFrame(rows)[["d_mcb", "dsc_lost"]].values).max() * 1.1)
    ax.axhline(0, color="grey", lw=0.7)
    ax.axvline(0, color="grey", lw=0.7)
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, 3 * xs, ls="--", color="k", lw=0.8)
    ax.text(lim / 3.3, lim * 0.93, "ρ = 0.25", fontsize=8, ha="right")
    ax.text(-lim * 0.95, lim * 0.95, "lost discrimination", fontsize=8, va="top", color="#444")
    ax.text(lim * 0.95, -lim * 0.95, "added miscalibration", fontsize=8, ha="right", color="#444")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Drift-attributable ΔMCB on S (miscalibration)")
    ax.set_ylabel("Drift-attributable DSC lost on S (discrimination)")
    ax.set_title("Where the drift-caused damage goes (small markers: seeds; large: mean)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    save(fig, d, "figA_decomposition_scatter")
    pd.DataFrame(rows).to_csv(os.path.join(d, "figA_data.csv"), index=False)


def fig_raw(out, d):
    setting, model = A.MAIN
    runs = [("ft_gradual", "FT gradual"), ("ft_nodrift", "FT twin"), ("replay_gradual", "Replay gradual"),
            ("replay_nodrift", "Replay twin")]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=False)
    rows = []
    for j, (col, lab) in enumerate([("d_brier", "ΔBrier (end − start)"), ("d_dsc", "ΔDSC (end − start)")]):
        ax = axs[j]
        for i, (run, name) in enumerate(runs):
            t = raw(out, setting, model, run)
            m, lo, hi = tci(t[col])
            c = "#c2410c" if run.startswith("ft") else "#1d4ed8"
            ax.bar(i, m, color=c, alpha=0.85 if "gradual" in run else 0.35, edgecolor=c)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="k", capsize=3, lw=1)
            ax.scatter(np.full(len(t), i), t[col], s=8, color="k", zorder=3)
            rows.append({"run": name, "metric": col, "mean": m, "lo": lo, "hi": hi})
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([n for _, n in runs], rotation=20, fontsize=8)
        ax.set_title(lab, fontsize=9)
    fig.suptitle("Primary cell: the paired effect = gradual change minus twin change (mean, 95% t-CI, seeds)", fontsize=9)
    save(fig, d, "figB_raw_vs_twin")
    pd.DataFrame(rows).to_csv(os.path.join(d, "figB_data.csv"), index=False)


def fig_fixes(out, d, k=500):
    p = os.path.join(out, "analysis", "ext_fixes", "X2_per_seed.csv")
    if not os.path.exists(p):
        print("figC skipped: run ext_fixes.py first")
        return
    t = pd.read_csv(p)
    t = t[t.k == k]
    fixes = [("brier_FT_end", "No fix", "#9ca3af"), ("brier_R", "Temperature (R)", "#f59e0b"),
             ("brier_R_iso", "Isotonic (R-iso)", "#d97706"), ("brier_P", "Replay (P)", "#1d4ed8"),
             ("brier_R+P", "Replay + temp. (R+P)", "#60a5fa")]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.16
    rows = []
    for j, cell in enumerate(A.CELLS):
        g = t[(t.setting == cell[0]) & (t.model == cell[1])]
        for i, (col, lab, c) in enumerate(fixes):
            m, lo, hi = tci(g[col])
            x = j + (i - 2) * w
            ax.bar(x, m, width=w, color=c, label=lab if j == 0 else None)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color="k", capsize=2, lw=0.8)
            rows.append({"cell": NAMES[cell], "fix": lab, "mean": m, "lo": lo, "hi": hi})
    ax.set_xticks(range(len(A.CELLS)))
    ax.set_xticklabels([NAMES[c].replace(", ", "\n") for c in A.CELLS], fontsize=8)
    ax.set_ylabel(f"Brier on S at end of stream (k = {k})")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
    save(fig, d, "figC_fixes")
    pd.DataFrame(rows).to_csv(os.path.join(d, "figC_data.csv"), index=False)


def fig_controls(out, d):
    items = [("ft", "", "FT\n(lr 0.001)", "#c2410c"), ("ft", "_lrhalf", "FT\nlr 0.0005", "#fb923c"),
             ("ft", "_bnfrozen", "FT\nfrozen BN", "#16a34a"), ("replay", "", "Replay", "#1d4ed8")]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    rows = []
    for i, (strat, sfx, lab, c) in enumerate(items):
        t = drift(out, *A.MAIN, strat, sfx)
        if t is None:
            continue
        m, lo, hi = tci(t.dsc_lost)
        ax.bar(i, m, color=c, alpha=0.8)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="k", capsize=3, lw=1)
        ax.scatter(np.full(len(t), i), t.dsc_lost, s=10, color="k", zorder=3)
        rows.append({"condition": lab.replace("\n", " "), "mean": m, "lo": lo, "hi": hi})
    ax.axhline(0, color="grey", lw=0.7)
    ax.axhline(0.005, color="grey", lw=0.7, ls=":")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([x[2] for x in items], fontsize=8)
    ax.set_ylabel("Drift-attributable DSC lost on S")
    ax.set_title("Primary cell: controls (mean, 95% t-CI, seeds)", fontsize=9)
    save(fig, d, "figD_controls")
    pd.DataFrame(rows).to_csv(os.path.join(d, "figD_data.csv"), index=False)


def main(a):
    d = os.path.join(a.out, "analysis", "figures_v2")
    os.makedirs(d, exist_ok=True)
    fig_scatter(a.out, d)
    fig_raw(a.out, d)
    fig_fixes(a.out, d)
    fig_controls(a.out, d)
    print("Saved in", d)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    main(p.parse_args())
