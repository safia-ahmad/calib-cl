"""Final pre-writing checks (audit of 8 Nov). Reads saved results only: no training, CPU is enough.

C1  Run inventory: every stream folder, its tag, lr and minutes; confirms the analysed (untagged) runs
    are exactly 4 cells x 5 seeds x 5 runs = 100, all at lr 0.001, and lists every extra (tagged) folder.
C2  Hospital labels: slides whose hospital in the manifest differs from the slide-number rule
    (normal 001-100 and tumour 001-070 = RUMC; normal 101-160 and tumour 071-110 = UMCU).
C3  Composition of S (and G) per seed: patches, tumour share, slides.
C4  Plasticity on G (pre-registered secondary): change in AUROC and Brier on the target test set.
C5  Raw (not twin-subtracted) S changes in the main cell, gradual and twin separately.
C6  In-sample CORP (no cross-fitting) as a robustness check of the decomposition.
C7  Oracle temperature gain at checkpoint 0 and 10 (gradual and twin), so the drift-related part is visible.
C8  Replay vs FT DSC lost per seed (descriptive sign count; NOT a replacement for H2).
C9  Endpoint sensitivity (checkpoint 10, 9, mean of 8-10) for every cell, not only the ViT cell.
C10 Baseline table: source models at checkpoint 0 on S and G (needed as Table 1 of the paper).

Usage (Colab):
  python scripts/final_checks.py --data $D --models $D/models --out $D/stream_runs
Writes <out>/analysis/final_checks/*.csv and prints everything.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import analysis as A  # noqa: E402
from calcl.corp import corp_insample  # noqa: E402
from calcl.metrics import fit_temperature  # noqa: E402

RUNS = ["frozen_gradual", "ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift"]


def tci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return round(m, 4), round(m - h, 4), round(m + h, 4)


def p1(z, T):
    return softmax(np.asarray(z, np.float64) / T, axis=1)[:, 1]


def T0_of(models, setting, model, seed):
    return json.load(open(os.path.join(models, f"{setting}_{model}_seed{seed}", "summary.json")))["T0"]


def c1_inventory(out, models):
    rows = []
    for d in sorted(os.listdir(out)):
        m = re.match(r"(pcam_\w+?)_(resnet18|vit_s16)_seed(\d+)$", d)
        if not m:
            continue
        for s in sorted(os.listdir(os.path.join(out, d))):
            if not s.startswith("stream_"):
                continue
            base = next((r for r in RUNS if s == f"stream_{r}" or s.startswith(f"stream_{r}_")), None)
            tag = s[len(f"stream_{base}"):] if base else "?"
            meta_p = os.path.join(out, d, s, "meta.json")
            meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
            rows.append({"setting": m.group(1), "model": m.group(2), "seed": int(m.group(3)), "run": base,
                         "tag": tag or "(none)", "has_metrics": os.path.exists(os.path.join(out, d, s, "metrics.csv")),
                         "lr": meta.get("lr"), "minutes": round(meta.get("minutes", np.nan), 2)})
    inv = pd.DataFrame(rows)
    main = inv[(inv.tag == "(none)") & inv.seed.isin(A.SEEDS)]
    print("=== C1 Run inventory ===")
    print(f"Untagged stream runs, seeds 1-5 (these are what analyze.py reads): {len(main)} (expected 100)")
    print(main.groupby(["setting", "model"]).size().rename("runs").to_string())
    print("lr values in analysed runs (frozen runs never train, lr unused):")
    print(main[main.run != "frozen_gradual"].lr.value_counts(dropna=False).to_string())
    bad = main[(main.run != "frozen_gradual") & (main.lr != 0.001)]
    print("Analysed trained runs with lr != 0.001:", "NONE" if bad.empty else "\n" + bad.to_string(index=False))
    other = inv[(inv.tag != "(none)") | ~inv.seed.isin(A.SEEDS)]
    print(f"\nOther stream folders (not read by the main analysis): {len(other)}")
    if len(other):
        print(other.groupby(["setting", "model", "seed", "tag"]).agg(runs=("run", "count"),
              lr=("lr", lambda v: ",".join(sorted({str(x) for x in v})))).to_string())
    src = sorted(d for d in os.listdir(models) if os.path.exists(os.path.join(models, d, "summary.json")))
    print(f"\nSource models with summary.json: {len(src)}")
    print("  ", ", ".join(src))
    return inv


def c2_hospitals(man):
    sl = man.groupby("wsi").agg(hospital=("hospital", "first")).reset_index()
    rows = []
    for w, h in zip(sl.wsi, sl.hospital):
        m = re.match(r"(normal|tumou?r)_(\d+)", str(w))
        if not m:
            continue
        n = int(m.group(2))
        rule = "RUMC" if (m.group(1) == "normal" and n <= 100) or (m.group(1) != "normal" and n <= 70) else "UMCU"
        if rule != h:
            rows.append({"wsi": w, "manifest_hospital": h, "slide_number_rule": rule,
                         "patches": int((man.wsi == w).sum())})
    t = pd.DataFrame(rows)
    print("\n=== C2 Slides whose manifest hospital differs from the slide-number rule ===")
    print("none" if t.empty else t.to_string(index=False))
    print(sl.hospital.value_counts().to_string())
    return t


def c3_composition(out, man):
    rows = []
    for setting, model in A.CELLS:
        for s in A.SEEDS:
            z = np.load(os.path.join(A.run_dir(out, setting, model, s), "stream_ft_gradual", "logits.npz"))
            for name in ["S", "G"]:
                if f"idx_{name}" not in z:
                    continue
                idx = z[f"idx_{name}"]
                rows.append({"setting": setting, "model": model, "seed": s, "set": name, "patches": len(idx),
                             "tumour_share": round(man.label.values[idx].mean(), 3),
                             "slides": len(np.unique(man.wsi.values[idx])),
                             "tumour_slides": len(np.unique(man.wsi.values[idx][man.label.values[idx] == 1]))})
    t = pd.DataFrame(rows)
    print("\n=== C3 Evaluation-set composition (per seed) ===")
    print(t.groupby(["setting", "model", "set"])[["patches", "tumour_share", "slides", "tumour_slides"]]
          .agg(["min", "max"]).to_string())
    return t


def c4_plasticity(out):
    rows = []
    for setting, model in A.CELLS:
        for strat in ["ft", "replay"]:
            vals = {k: [] for k in ["auroc0", "auroc10", "brier0", "brier10"]}
            for s in A.SEEDS:
                m = pd.read_csv(os.path.join(A.run_dir(out, setting, model, s), f"stream_{strat}_gradual", "metrics.csv"))
                g = m[m.set == "G"].set_index("checkpoint").sort_index()
                vals["auroc0"].append(g.auroc.iloc[0]); vals["auroc10"].append(g.auroc.iloc[-1])
                vals["brier0"].append(g.brier.iloc[0]); vals["brier10"].append(g.brier.iloc[-1])
            v = {k: np.array(x) for k, x in vals.items()}
            rows.append({"setting": setting, "model": model, "strategy": strat,
                         "G_auroc_start": round(v["auroc0"].mean(), 3), "G_auroc_end": round(v["auroc10"].mean(), 3),
                         "d_G_auroc [95% t]": tci(v["auroc10"] - v["auroc0"]),
                         "d_G_brier [95% t]": tci(v["brier10"] - v["brier0"])})
    t = pd.DataFrame(rows)
    print("\n=== C4 Plasticity: target test set G, gradual runs, end minus start ===")
    print(t.to_string(index=False))
    return t


def c5_raw(out):
    rows = []
    setting, model = A.MAIN
    for run in ["ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift"]:
        ch = [A.s_change(out, setting, model, s, run) for s in A.SEEDS]
        rows.append({"run": run, **{f"d_{k} [95% t]": tci([c[k] for c in ch]) for k in ["brier", "mcb", "dsc"]}})
    t = pd.DataFrame(rows)
    print("\n=== C5 Raw S changes, main cell (end minus start, NOT twin-subtracted) ===")
    print(t.to_string(index=False))
    return t


def c6_insample(out, models, labels):
    rows = []
    for setting, model in A.CELLS:
        for strat in ["ft", "replay"]:
            dm, dl = [], []
            for s in A.SEEDS:
                rd, T0 = A.run_dir(out, setting, model, s), T0_of(models, setting, model, s)
                c = {}
                for kind in ["gradual", "nodrift"]:
                    z = np.load(os.path.join(rd, f"stream_{strat}_{kind}", "logits.npz"))
                    y = labels[z["idx_S"]]
                    c[kind] = [corp_insample(p1(z[f"S_c{k}"], T0), y) for k in (0, 10)]
                dm.append((c["gradual"][1]["mcb"] - c["gradual"][0]["mcb"]) - (c["nodrift"][1]["mcb"] - c["nodrift"][0]["mcb"]))
                dl.append(-((c["gradual"][1]["dsc"] - c["gradual"][0]["dsc"]) - (c["nodrift"][1]["dsc"] - c["nodrift"][0]["dsc"])))
            rows.append({"setting": setting, "model": model, "strategy": strat,
                         "dMCB_insample [95% t]": tci(dm), "DSC_lost_insample [95% t]": tci(dl),
                         "rho_insample": round(np.mean(dm) / (np.mean(dm) + np.mean(dl)), 3)
                         if np.mean(dm) + np.mean(dl) > 0 else np.nan})
    t = pd.DataFrame(rows)
    print("\n=== C6 In-sample CORP (optimistic, no cross-fitting), drift minus twin ===")
    print(t.to_string(index=False))
    return t


def c7_oracle(out, models, labels):
    rows = []
    for setting, model in A.CELLS:
        g = {"grad_c0": [], "grad_c10": [], "twin_c10": []}
        for s in A.SEEDS:
            rd, T0 = A.run_dir(out, setting, model, s), T0_of(models, setting, model, s)
            for key, run, c in [("grad_c0", "ft_gradual", 0), ("grad_c10", "ft_gradual", 10), ("twin_c10", "ft_nodrift", 10)]:
                z = np.load(os.path.join(rd, f"stream_{run}", "logits.npz"))
                y, zs = labels[z["idx_S"]], np.asarray(z[f"S_c{c}"], np.float64)
                Ts = fit_temperature(zs, y)
                g[key].append(np.mean((p1(zs, T0) - y) ** 2) - np.mean((p1(zs, Ts) - y) ** 2))
        g = {k: np.array(v) for k, v in g.items()}
        rows.append({"setting": setting, "model": model, "gain_c0": tci(g["grad_c0"]), "gain_c10_gradual": tci(g["grad_c10"]),
                     "gain_c10_twin": tci(g["twin_c10"]),
                     "drift_part (grad10 - twin10)": tci(g["grad_c10"] - g["twin_c10"])})
    t = pd.DataFrame(rows)
    print("\n=== C7 Oracle temperature on S (fitted on S itself): Brier gain over T0 ===")
    print(t.to_string(index=False))
    return t


def c8_replay_sign(out):
    dt = A.drift_table(out)
    rows = []
    for setting, model in A.CELLS:
        c = dt[(dt.setting == setting) & (dt.model == model)]
        f = c[c.strategy == "ft"].sort_values("seed").dsc_lost.values
        r = c[c.strategy == "replay"].sort_values("seed").dsc_lost.values
        rows.append({"setting": setting, "model": model, "FT_minus_replay_per_seed": " ".join(f"{v:+.4f}" for v in f - r),
                     "seeds_replay_better": int((f > r).sum())})
    t = pd.DataFrame(rows)
    print("\n=== C8 DSC lost, FT minus replay, per seed (descriptive only) ===")
    print(t.to_string(index=False))
    return t


def c9_endpoints(out):
    """Same endpoint-sensitivity check as E-4, applied to every cell (FT), so the primary cell gets it too."""
    rows = []
    for setting, model in A.CELLS:
        per = {}
        for s in A.SEEDS:
            rd = A.run_dir(out, setting, model, s)
            ser = {}
            for kind in ["gradual", "nodrift"]:
                m = pd.read_csv(os.path.join(rd, f"stream_ft_{kind}", "metrics.csv"))
                ser[kind] = m[m.set == "S"].set_index("checkpoint").sort_index()
            per[s] = {c: ((ser["gradual"][c] - ser["gradual"][c].iloc[0]) - (ser["nodrift"][c] - ser["nodrift"][c].iloc[0]))
                      for c in ["mcb", "dsc"]}
        for label, cps in [("c10 (pre-registered)", [10]), ("c9", [9]), ("mean c8-10", [8, 9, 10])]:
            dm = [per[s]["mcb"].loc[cps].mean() for s in A.SEEDS]
            dl = [-per[s]["dsc"].loc[cps].mean() for s in A.SEEDS]
            den = np.mean(dm) + np.mean(dl)
            rows.append({"setting": setting, "model": model, "endpoint": label, "dMCB [95% t]": tci(dm),
                         "DSC_lost [95% t]": tci(dl), "rho": round(np.mean(dm) / den, 3) if den > 0 else np.nan})
    t = pd.DataFrame(rows)
    print("\n=== C9 Endpoint sensitivity, FT, every cell (drift minus twin) ===")
    print(t.to_string(index=False))
    return t


def c10_baseline(out):
    """Table 1 of the paper: the source models at checkpoint 0 on S and G (mean and SD over seeds)."""
    rows = []
    for setting, model in A.CELLS:
        for st in ["S", "G"]:
            v = []
            for s in A.SEEDS:
                m = pd.read_csv(os.path.join(A.run_dir(out, setting, model, s), "stream_frozen_gradual", "metrics.csv"))
                v.append(m[(m.set == st) & (m.checkpoint == 0)].iloc[0])
            v = pd.DataFrame(v)
            rows.append({"setting": setting, "model": model, "set": st,
                         **{c: f"{v[c].mean():.3f} ± {v[c].std(ddof=1):.3f}" for c in
                            ["auroc", "brier", "mcb", "dsc", "ece_mass", "acc"] if c in v}})
    t = pd.DataFrame(rows)
    print("\n=== C10 Source models at checkpoint 0 (mean ± SD over seeds) ===")
    print(t.to_string(index=False))
    return t


def main(a):
    d = os.path.join(a.out, "analysis", "final_checks")
    os.makedirs(d, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    man = pd.read_csv(os.path.join(a.data, "pcam_compact_manifest.csv"))
    labels = man.label.to_numpy()
    for name, fn in [("C1_inventory", lambda: c1_inventory(a.out, a.models)),
                     ("C2_hospital_mismatch", lambda: c2_hospitals(man)),
                     ("C3_eval_composition", lambda: c3_composition(a.out, man)),
                     ("C4_plasticity_G", lambda: c4_plasticity(a.out)),
                     ("C5_raw_main_cell", lambda: c5_raw(a.out)),
                     ("C6_insample_corp", lambda: c6_insample(a.out, a.models, labels)),
                     ("C7_oracle_temperature", lambda: c7_oracle(a.out, a.models, labels)),
                     ("C8_replay_sign", lambda: c8_replay_sign(a.out)),
                     ("C9_endpoints", lambda: c9_endpoints(a.out)),
                     ("C10_baseline", lambda: c10_baseline(a.out))]:
        try:
            fn().to_csv(os.path.join(d, f"{name}.csv"), index=False)
        except Exception as e:  # keep going; report what failed
            print(f"\n!!! {name} failed: {type(e).__name__}: {e}")
    print(f"\nSaved in {d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with pcam_compact_manifest.csv")
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
