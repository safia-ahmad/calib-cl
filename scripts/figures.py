"""Figures + exploratory (NOT pre-registered) checks, from the saved results on Drive.

Usage (Colab, CPU is enough):
  python scripts/figures.py --models $D/models --out $D/stream_runs
Needs analyze.py to have run first (reads <out>/analysis/H4_per_seed.csv).
Writes PDF + PNG figures and exploratory CSVs to <out>/analysis/figures/.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import analysis as A  # noqa: E402

NAMES = {("pcam_fwd", "resnet18"): "RUMC→UMCU, ResNet-18", ("pcam_rev", "resnet18"): "UMCU→RUMC, ResNet-18",
         ("pcam_fwd", "vit_s16"): "RUMC→UMCU, ViT-S/16", ("pcam_rev", "vit_s16"): "UMCU→RUMC, ViT-S/16"}
RUNS = {"ft_gradual": ("Fine-tuning, gradual", "#c2410c", "-"),
        "ft_nodrift": ("Fine-tuning, no-drift twin", "#c2410c", "--"),
        "replay_gradual": ("Replay, gradual", "#1d4ed8", "-")}


def save(fig, d, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def curves(out, setting, model, run, col):
    ys = []
    for seed in A.SEEDS:
        m = pd.read_csv(os.path.join(A.run_dir(out, setting, model, seed), f"stream_{run}", "metrics.csv"))
        s = m[m.set == "S"].set_index("checkpoint").sort_index()[col]
        ys.append(s.values - s.values[0])
    ys = np.array(ys)
    return s.index.values, ys.mean(0), ys.std(0, ddof=1)


def fig_curves(out, d):
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.5), sharex=True)
    for j, cell in enumerate(A.CELLS):
        for i, (col, lab) in enumerate([("mcb", "Change in MCB on S"), ("dsc", "Change in DSC on S")]):
            a = ax[i, j]
            for run, (name, c, ls) in RUNS.items():
                x, m, s = curves(out, *cell, run, col)
                a.plot(x, m, ls, color=c, label=name, lw=1.8)
                a.fill_between(x, m - s, m + s, color=c, alpha=0.12)
            a.axhline(0, color="grey", lw=0.8)
            if i == 0:
                a.set_title(NAMES[cell], fontsize=10)
            if j == 0:
                a.set_ylabel(lab)
            if i == 1:
                a.set_xlabel("Windows trained")
    ax[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Source-domain (S) miscalibration and discrimination over the stream (mean ± SD over 5 seeds)")
    save(fig, d, "figS1_raw_mcb_dsc_curves")


def fig_drift_curves(out, d):
    """Drift-attributable change (gradual minus its no-drift twin, per seed) over the stream."""
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.5), sharex=True)
    for j, cell in enumerate(A.CELLS):
        for i, (col, sign, lab) in enumerate([("mcb", 1, "ΔMCB on S (drift − twin)"),
                                               ("dsc", -1, "DSC lost on S (drift − twin)")]):
            a = ax[i, j]
            for strat, c, name in [("ft", "#c2410c", "Fine-tuning"), ("replay", "#1d4ed8", "Replay")]:
                diffs = []
                for seed in A.SEEDS:
                    rd = A.run_dir(out, *cell, seed)
                    g = pd.read_csv(os.path.join(rd, f"stream_{strat}_gradual", "metrics.csv"))
                    n = pd.read_csv(os.path.join(rd, f"stream_{strat}_nodrift", "metrics.csv"))
                    g = g[g.set == "S"].set_index("checkpoint").sort_index()[col]
                    n = n[n.set == "S"].set_index("checkpoint").sort_index()[col]
                    diffs.append(sign * ((g - g.iloc[0]) - (n - n.iloc[0])).values)
                diffs = np.array(diffs)
                m, se = diffs.mean(0), diffs.std(0, ddof=1) / np.sqrt(len(diffs))
                x = np.arange(diffs.shape[1])
                a.plot(x, m, color=c, lw=1.8, label=name)
                a.fill_between(x, m - se, m + se, color=c, alpha=0.15)
            a.axhline(0, color="grey", lw=0.8)
            if i == 0:
                a.set_title(NAMES[cell], fontsize=10)
            if j == 0:
                a.set_ylabel(lab)
            if i == 1:
                a.set_xlabel("Windows trained")
    ax[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Drift-attributable miscalibration (top) and lost discrimination (bottom) on S (mean ± SE over 5 seeds)")
    save(fig, d, "fig1_drift_curves")


def fig_decomposition(dt, d):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(A.CELLS))
    for k, (strat, off, c) in enumerate([("ft", -0.2, "#c2410c"), ("replay", 0.2, "#1d4ed8")]):
        for comp, hatch, sh in [("d_mcb", "", -0.08), ("dsc_lost", "//", 0.08)]:
            vals = [dt[(dt.setting == s) & (dt.model == m) & (dt.strategy == strat)][comp].values for s, m in A.CELLS]
            means = [v.mean() for v in vals]
            ax.bar(x + off + sh, means, width=0.15, color=c, alpha=0.75 if comp == "d_mcb" else 0.35,
                   hatch=hatch, edgecolor=c,
                   label=f"{'FT' if strat == 'ft' else 'Replay'}: {'ΔMCB' if comp == 'd_mcb' else 'DSC lost'}")
            for xi, v in zip(x, vals):
                ax.scatter(np.full(len(v), xi + off + sh), v, s=8, color="k", zorder=3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[c].replace(", ", "\n") for c in A.CELLS], fontsize=9)
    ax.set_ylabel("Drift-attributable change on S\n(gradual minus no-drift twin)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Where the drift-caused damage goes: miscalibration (ΔMCB) vs lost discrimination")
    save(fig, d, "fig2_decomposition_bars")


def fig_forest(h3, d):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, r in enumerate(h3.itertuples()):
        name = NAMES[(r.setting, r.model)]
        if np.isfinite(r.rho_FT) and r.DSC_lost_FT + r.dMCB_FT > 0:
            lo, hi = max(r.rho_lo, -1), min(r.rho_hi, 2)
            ax.plot([lo, hi], [i, i], color="k", lw=1.5)
            ax.plot(r.rho_FT, i, "o", color="#c2410c" if i == 0 else "k", ms=7)
        else:
            ax.text(1.1, i, "no drift-attributable forgetting", va="center", ha="center", fontsize=8, color="grey")
        ax.text(-1.05, i, name, va="center", ha="right", fontsize=9)
    ax.axvline(0.25, color="#1d4ed8", ls="--", lw=1)
    ax.text(0.28, -0.35, "H1 threshold 0.25", color="#1d4ed8", fontsize=8, va="center")
    ax.set_xlim(-1, 2)
    ax.set_ylim(len(h3) - 0.5, -0.6)
    ax.set_yticks([])
    ax.set_title("Share of drift-caused forgetting that is miscalibration", fontsize=10)
    ax.set_xlabel("Calibration share ρ under fine-tuning (95% bootstrap CI over seeds, clipped to [−1, 2])")
    save(fig, d, "fig3_rho_forest")


def fig_h4(t4, d):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for col, lab, c, off in [("regret_rule", "Decomposition rule", "k", 0), ("regret_R", "Always recalibrate", "#c2410c", -8),
                             ("regret_P", "Always replay", "#1d4ed8", 8)]:
        g = t4.groupby("k")[col]
        m, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
        ax.errorbar(m.index + off * m.index / 100, m.values, yerr=1.96 * se.values, fmt="o-", color=c, label=lab, capsize=3)
    ax.set_xscale("log")
    ax.set_xticks(A.KS)
    ax.set_xticklabels(A.KS)
    ax.set_xlabel("Labelled source patches k")
    ax.set_ylabel("Regret: Brier on S above best fix")
    ax.legend(fontsize=8)
    ax.set_title("Choosing a fix (mean ± 95% CI over 4 cells × 5 seeds)")
    save(fig, d, "fig4_fix_rule_regret")


def exploratory(dt, t4, d):
    """Not pre-registered: reported as exploratory."""
    t4 = t4.copy()
    briers = t4[["brier_R", "brier_P", "brier_RP"]].values
    t4["best_fix"] = np.array(["R", "P", "R+P"])[briers.argmin(1)]
    e1 = (t4.groupby(["setting", "model", "k"]).best_fix.value_counts().unstack(fill_value=0))
    e1.to_csv(os.path.join(d, "E1_best_fix_counts.csv"))
    # per-seed true rho (with twin) vs the deployable estimate (no twin, k labelled patches)
    ft = dt[dt.strategy == "ft"].copy()
    den = ft.d_mcb + ft.dsc_lost
    ft["rho_seed"] = np.where(den > 0, ft.d_mcb / den, np.nan)
    m = t4.merge(ft[["setting", "model", "seed", "rho_seed", "d_brier"]], on=["setting", "model", "seed"])
    e2 = m.groupby("k").apply(lambda g: pd.Series({
        "corr_rhohat_vs_rho": g[["rho_hat", "rho_seed"]].corr().iloc[0, 1],
        "share_rule_agrees_with_true_side": np.mean((g.rho_hat >= 0.5) == (g.rho_seed >= 0.5)),
        "share_R_alone_beats_P": np.mean(g.brier_R < g.brier_P)}), include_groups=False)
    e2.to_csv(os.path.join(d, "E2_rule_diagnostics.csv"))
    e3 = t4.groupby(["setting", "model"])[["brier_FT_end", "brier_R", "brier_P", "brier_RP"]].mean()
    e3.to_csv(os.path.join(d, "E3_brier_after_fixes_by_cell.csv"))
    pd.set_option("display.width", 200)
    print("\n=== EXPLORATORY (not pre-registered) ===")
    print("E1  Which fix was actually best (counts over seeds), per cell and k:")
    print(e1.to_string())
    print("\nE2  Why the rule errs: estimate rho_hat (no twin, k patches) vs true per-seed rho (with twin):")
    print(e2.round(3).to_string())
    print("\nE3  Brier on S at the end of the stream, mean over seeds and k (FT with no fix, R, P, R+P):")
    print(e3.round(4).to_string())


def main(a):
    d = os.path.join(a.out, "analysis", "figures")
    os.makedirs(d, exist_ok=True)
    dt = A.drift_table(a.out)
    _, h3 = A.hypotheses(dt)
    t4 = pd.read_csv(os.path.join(a.out, "analysis", "H4_per_seed.csv"))
    fig_curves(a.out, d)
    fig_drift_curves(a.out, d)
    fig_decomposition(dt, d)
    fig_forest(h3, d)
    fig_h4(t4, d)
    exploratory(dt, t4, d)
    print(f"\nEverything saved in {d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
