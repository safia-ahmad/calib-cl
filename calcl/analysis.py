"""Analysis exactly as in paper/prereg.md. Covered by tests/test_analysis.py.

Inputs (already on Drive):
  <out>/<setting>_<model>_seed<s>/stream_<strategy>_<kind>/metrics.csv   (S, G, B, E per checkpoint)
  <out>/<setting>_<model>_seed<s>/stream_<strategy>_<kind>/logits.npz    (S, G, B logits per checkpoint)
  <models>/<setting>_<model>_seed<s>/summary.json                         (T0)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

from calcl.corp import corp_crossfit
from calcl.metrics import fit_temperature

SESOI = 0.005
MAIN = ("pcam_fwd", "resnet18")
CELLS = [("pcam_fwd", "resnet18"), ("pcam_rev", "resnet18"), ("pcam_fwd", "vit_s16"), ("pcam_rev", "vit_s16")]
SEEDS = [1, 2, 3, 4, 5]
KS = [100, 250, 500, 1000]


def run_dir(out, setting, model, seed):
    return os.path.join(out, f"{setting}_{model}_seed{seed}")


def s_change(out, setting, model, seed, name) -> dict:
    """End minus start of Brier, MCB, DSC on S for one stream run."""
    m = pd.read_csv(os.path.join(run_dir(out, setting, model, seed), f"stream_{name}", "metrics.csv"))
    s = m[m.set == "S"].set_index("checkpoint").sort_index()
    return {k: float(s[k].iloc[-1] - s[k].iloc[0]) for k in ["brier", "mcb", "dsc"]}


def drift_table(out) -> pd.DataFrame:
    """One row per cell x seed x strategy: drift-attributable change on S (gradual minus twin)."""
    rows = []
    for setting, model in CELLS:
        for seed in SEEDS:
            for strat in ["ft", "replay"]:
                g = s_change(out, setting, model, seed, f"{strat}_gradual")
                n = s_change(out, setting, model, seed, f"{strat}_nodrift")
                rows.append({"setting": setting, "model": model, "seed": seed, "strategy": strat,
                             "d_brier": g["brier"] - n["brier"], "d_mcb": g["mcb"] - n["mcb"],
                             "dsc_lost": -(g["dsc"] - n["dsc"])})
    return pd.DataFrame(rows)


def rho(d_mcb, dsc_lost) -> float:
    num, den = float(np.mean(d_mcb)), float(np.mean(d_mcb) + np.mean(dsc_lost))
    return num / den if den > 0 else np.nan


def rho_ci(d_mcb, dsc_lost, n_boot=10_000, seed=0):
    d_mcb, dsc_lost = np.asarray(d_mcb), np.asarray(dsc_lost)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d_mcb), size=(n_boot, len(d_mcb)))
    boots = np.array([rho(d_mcb[i], dsc_lost[i]) for i in idx])
    boots = boots[np.isfinite(boots)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5), boots


def verdict_interval(lo, hi, thr):
    if lo > thr:
        return "supported"
    if hi < thr:
        return "rejected"
    return "inconclusive"


def hypotheses(dt: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    res = {}
    main = dt[(dt.setting == MAIN[0]) & (dt.model == MAIN[1])]
    ft, rp = main[main.strategy == "ft"].sort_values("seed"), main[main.strategy == "replay"].sort_values("seed")
    # P1
    res["P1_mean_dBrier_FT"] = float(ft.d_brier.mean())
    res["P1"] = "holds" if res["P1_mean_dBrier_FT"] >= SESOI else "fails (H1 untestable)"
    # H1
    r = rho(ft.d_mcb, ft.dsc_lost)
    lo, hi, boots = rho_ci(ft.d_mcb.values, ft.dsc_lost.values)
    res.update({"H1_rho": r, "H1_ci": (lo, hi), "H1_raw": verdict_interval(lo, hi, 0.25),
                "H1_p": float(np.mean(boots <= 0.25))})
    # H2: paired, one-sided t-interval vs SESOI
    diff = ft.dsc_lost.values - rp.dsc_lost.values
    n, m, se = len(diff), diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
    t95 = stats.t.ppf(0.95, n - 1)
    res.update({"H2_effect": float(m), "H2_one_sided_ci": (m - t95 * se, m + t95 * se),
                "H2_p": float(1 - stats.t.cdf((m - SESOI) / se, n - 1)) if se > 0 else float(m <= SESOI)})
    res["H2_raw"] = ("supported" if m - t95 * se > SESOI else "rejected" if m + t95 * se < SESOI
                     else "inconclusive")
    mcb_diff = ft.d_mcb.values - rp.d_mcb.values
    t975 = stats.t.ppf(0.975, n - 1)
    res["H2_mcb_effect"] = (float(mcb_diff.mean()), float(mcb_diff.mean() - t975 * mcb_diff.std(ddof=1) / np.sqrt(n)),
                            float(mcb_diff.mean() + t975 * mcb_diff.std(ddof=1) / np.sqrt(n)))
    # Holm over H1, H2 (a "supported" verdict stands only if its Holm test passes)
    ps = sorted([("H1", res["H1_p"]), ("H2", res["H2_p"])], key=lambda x: x[1])
    holm_ok, still = {}, True
    for i, (h, p) in enumerate(ps):
        still = still and p <= 0.05 / (2 - i)
        holm_ok[h] = still
    for h in ["H1", "H2"]:
        v = res[f"{h}_raw"]
        res[h] = v if v != "supported" or holm_ok[h] else "inconclusive (fails Holm)"
    # H3 per cell
    rows = []
    for setting, model in CELLS:
        c = dt[(dt.setting == setting) & (dt.model == model)]
        f, p = c[c.strategy == "ft"].sort_values("seed"), c[c.strategy == "replay"].sort_values("seed")
        lo_c, hi_c, _ = rho_ci(f.d_mcb.values, f.dsc_lost.values)
        d = f.dsc_lost.values - p.dsc_lost.values
        se_d = d.std(ddof=1) / np.sqrt(len(d))
        rows.append({"setting": setting, "model": model, "mean_dBrier_FT": f.d_brier.mean(),
                     "dMCB_FT": f.d_mcb.mean(), "DSC_lost_FT": f.dsc_lost.mean(),
                     "rho_FT": rho(f.d_mcb, f.dsc_lost), "rho_lo": lo_c, "rho_hi": hi_c,
                     "H2_effect": d.mean(), "H2_lo": d.mean() - t975 * se_d, "H2_hi": d.mean() + t975 * se_d})
    h3 = pd.DataFrame(rows)
    main_side = res["H1_rho"] >= 0.25
    main_sign = np.sign(res["H2_effect"])
    res["H3_rho_same_side"] = int(((h3.rho_FT >= 0.25) == main_side).sum())
    res["H3_H2_same_sign"] = int((np.sign(h3.H2_effect) == main_sign).sum())
    res["H3"] = "consistent" if res["H3_rho_same_side"] >= 3 and res["H3_H2_same_sign"] >= 3 else "not consistent"
    return res, h3


# ---------- H4: the fix rule ----------

def _brier(z, y, T):
    return float(np.mean((softmax(np.asarray(z, np.float64) / T, axis=1)[:, 1] - y) ** 2))


def balanced_subset(y, k, seed):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    kp = min(k // 2, len(pos))
    kn = min(k - kp, len(neg))
    return np.sort(np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, kn, replace=False)]))


def h4_table(out, models, labels, wsi) -> pd.DataFrame:
    rows = []
    for setting, model in CELLS:
        for seed in SEEDS:
            rd = run_dir(out, setting, model, seed)
            T0 = json.load(open(os.path.join(models, f"{setting}_{model}_seed{seed}", "summary.json")))["T0"]
            ft = np.load(os.path.join(rd, "stream_ft_gradual", "logits.npz"))
            rp = np.load(os.path.join(rd, "stream_replay_gradual", "logits.npz"))
            yS, yB = labels[ft["idx_S"]], labels[ft["idx_B"]]
            gB = wsi[ft["idx_B"]]
            for k in KS:
                sub = balanced_subset(yB, k, seed=7919 * seed + k)
                pr = lambda z: softmax(np.asarray(z, np.float64) / T0, axis=1)[:, 1]  # noqa: E731
                c0 = corp_crossfit(pr(ft["B_c0"][sub]), yB[sub], gB[sub])
                c1 = corp_crossfit(pr(ft["B_c10"][sub]), yB[sub], gB[sub])
                dm, dl = c1["mcb"] - c0["mcb"], c0["dsc"] - c1["dsc"]
                rho_hat = dm / (dm + dl) if dm + dl > 0 else np.nan
                choice = "R" if (np.isnan(rho_hat) or rho_hat >= 0.5) else "P"  # no damage -> cheap fix
                T_R = fit_temperature(np.asarray(ft["B_c10"][sub], np.float64), yB[sub])
                T_RP = fit_temperature(np.asarray(rp["B_c10"][sub], np.float64), yB[sub])
                b = {"R": _brier(ft["S_c10"], yS, T_R), "P": _brier(rp["S_c10"], yS, T0),
                     "R+P": _brier(rp["S_c10"], yS, T_RP)}
                best = min(b.values())
                rows.append({"setting": setting, "model": model, "seed": seed, "k": k, "rho_hat": rho_hat,
                             "choice": choice, "brier_R": b["R"], "brier_P": b["P"], "brier_RP": b["R+P"],
                             "brier_FT_end": _brier(ft["S_c10"], yS, T0),
                             "regret_rule": b[choice] - best, "regret_R": b["R"] - best,
                             "regret_P": b["P"] - best})
    return pd.DataFrame(rows)


def h4_summary(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in t.groupby("k"):
        out = {"k": k, "n_pairs": len(g), "share_choose_R": (g.choice == "R").mean(),
               "regret_rule": g.regret_rule.mean(), "regret_alwaysR": g.regret_R.mean(),
               "regret_alwaysP": g.regret_P.mean()}
        for other in ["R", "P"]:
            d = g.regret_rule.values - g[f"regret_{other}"].values
            se = d.std(ddof=1) / np.sqrt(len(d))
            t975 = stats.t.ppf(0.975, len(d) - 1)
            out[f"rule_minus_{other}"] = d.mean()
            out[f"rule_minus_{other}_lo"] = d.mean() - t975 * se
            out[f"rule_minus_{other}_hi"] = d.mean() + t975 * se
        out["H4"] = ("holds" if out["rule_minus_R_hi"] < 0 and out["rule_minus_P_hi"] < 0
                     else "not shown")
        rows.append(out)
    return pd.DataFrame(rows)


def frozen_check(out) -> float:
    """Largest change of S Brier in any frozen run (must be ~0)."""
    worst = 0.0
    for setting, model in CELLS:
        for seed in SEEDS:
            worst = max(worst, abs(s_change(out, setting, model, seed, "frozen_gradual")["brier"]))
    return worst


def cost_table(out) -> pd.DataFrame:
    rows = []
    for setting, model in CELLS:
        for seed in SEEDS:
            for name in ["ft_gradual", "replay_gradual"]:
                p = os.path.join(run_dir(out, setting, model, seed), f"stream_{name}", "meta.json")
                if os.path.exists(p):
                    rows.append({"model": model, "run": name, "minutes": json.load(open(p))["minutes"]})
    c = pd.DataFrame(rows).groupby(["model", "run"]).minutes.mean().unstack()
    c["replay_extra_minutes"] = c["replay_gradual"] - c["ft_gradual"]
    return c
