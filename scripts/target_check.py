"""Replay vs fine-tuning on the target hospital (G) and the source hospital (S). Exploratory, not pre-registered.

Reads saved logits only (no training, CPU). For every cell and seed, from logits.npz of the gradual runs
(and the no-drift twins for S), at the deployed temperature T0 and threshold 0.5:
  AUROC, accuracy, sensitivity, specificity, Brier, at checkpoint 0 and 10.
Reports seed-level values, means with 95% t-intervals over the 5 seeds, and the paired difference
FT minus replay (same seed = same source model, same split, same stream).

Usage (Colab):
  python scripts/target_check.py --data $D --models $D/models --out $D/stream_runs
Writes <out>/analysis/target_check/{seed_level.csv, summary.csv}.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import analysis as A  # noqa: E402
from calcl.metrics import auroc  # noqa: E402

METRICS = ["auroc", "acc", "sens", "spec", "brier"]


def scores(z, y, T0):
    P = softmax(np.asarray(z, np.float64) / T0, axis=1)
    p = P[:, 1]
    pred = p >= 0.5
    return {"auroc": auroc(P, y), "acc": np.mean(pred == y), "sens": np.mean(pred[y == 1]),
            "spec": np.mean(~pred[y == 0]), "brier": np.mean((p - y) ** 2)}


def tci(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return m, m - h, m + h


def main(a):
    labels = pd.read_csv(os.path.join(a.data, "pcam_compact_manifest.csv"), usecols=["label"]).label.to_numpy()
    rows = []
    for setting, model in A.CELLS:
        for s in A.SEEDS:
            rd = A.run_dir(a.out, setting, model, s)
            T0 = json.load(open(os.path.join(a.models, f"{setting}_{model}_seed{s}", "summary.json")))["T0"]
            for strat in ["ft", "replay"]:
                for kind in ["gradual", "nodrift"]:
                    z = np.load(os.path.join(rd, f"stream_{strat}_{kind}", "logits.npz"))
                    for st in ["S", "G"]:
                        if kind == "nodrift" and st == "G":
                            continue
                        y = labels[z[f"idx_{st}"]]
                        for c in (0, 10):
                            rows.append({"setting": setting, "model": model, "seed": s, "strategy": strat,
                                         "stream": kind, "set": st, "checkpoint": c, **scores(z[f"{st}_c{c}"], y, T0)})
    t = pd.DataFrame(rows)
    d = os.path.join(a.out, "analysis", "target_check")
    os.makedirs(d, exist_ok=True)
    t.to_csv(os.path.join(d, "seed_level.csv"), index=False)

    key = ["setting", "model", "seed", "strategy", "stream", "set"]
    w = t.pivot_table(index=key, columns="checkpoint", values=METRICS)
    ch = pd.DataFrame({m: w[(m, 10)] - w[(m, 0)] for m in METRICS}).reset_index()
    end = pd.DataFrame({m: w[(m, 10)] for m in METRICS}).reset_index()

    out_rows = []
    pd.set_option("display.width", 250)
    for setting, model in A.CELLS:
        cell = lambda df, **kw: df[(df.setting == setting) & (df.model == model) &  # noqa: E731
                                    np.logical_and.reduce([df[k] == v for k, v in kw.items()])].sort_values("seed")
        print(f"\n######## {setting} {model} ########")
        # G: change over the stream, and FT minus replay at the end (paired by seed)
        for m in METRICS:
            f = cell(ch, strategy="ft", stream="gradual", set="G")[m].values
            r = cell(ch, strategy="replay", stream="gradual", set="G")[m].values
            fe = cell(end, strategy="ft", stream="gradual", set="G")[m].values
            re_ = cell(end, strategy="replay", stream="gradual", set="G")[m].values
            for name, v in [("G change, FT", f), ("G change, replay", r), ("G end, FT - replay", fe - re_)]:
                mm, lo, hi = tci(v)
                out_rows.append({"setting": setting, "model": model, "set": "G", "quantity": name, "metric": m,
                                 "mean": mm, "lo": lo, "hi": hi, "seeds": " ".join(f"{x:+.4f}" for x in v)})
        # S: drift-attributable (gradual minus twin) change, per strategy, and FT minus replay
        da = {}
        for strat in ["ft", "replay"]:
            g = cell(ch, strategy=strat, stream="gradual", set="S")
            n = cell(ch, strategy=strat, stream="nodrift", set="S")
            da[strat] = {m: g[m].values - n[m].values for m in METRICS}
        for m in METRICS:
            for name, v in [("S drift-attributable, FT", da["ft"][m]), ("S drift-attributable, replay", da["replay"][m]),
                            ("S drift-attributable, FT - replay", da["ft"][m] - da["replay"][m])]:
                mm, lo, hi = tci(v)
                out_rows.append({"setting": setting, "model": model, "set": "S", "quantity": name, "metric": m,
                                 "mean": mm, "lo": lo, "hi": hi, "seeds": " ".join(f"{x:+.4f}" for x in v)})
        sub = pd.DataFrame([r for r in out_rows if r["setting"] == setting and r["model"] == model])
        for q in sub.quantity.unique():
            print(f"\n  {q}")
            for _, r in sub[sub.quantity == q].iterrows():
                print(f"    {r.metric:6s} {r['mean']:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]   seeds: {r.seeds}")
    pd.DataFrame(out_rows).to_csv(os.path.join(d, "summary.csv"), index=False)
    print(f"\nSaved in {d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
