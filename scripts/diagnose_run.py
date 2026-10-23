"""Look inside one finished source run: where does the target (G) error come from?

Usage: python scripts/diagnose_run.py --run <run folder> --data /content/data
Prints, for S and G: mean p(tumour) for true-normal and true-tumour patches, the share
predicted tumour, AUROC, and per-slide accuracy on the worst slides.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


def main(a):
    s = json.load(open(os.path.join(a.run, "summary.json")))
    T0 = s["T0"]
    man = pd.read_csv(os.path.join(a.data, "pcam_compact_manifest.csv"))
    print(f"run: {a.run}\nT0 = {T0:.3f}, best epoch {s['best_epoch']}\n")
    for name in ["S_source_test", "G_target_test"]:
        d = np.load(os.path.join(a.run, f"logits_{name}.npz"))
        p = softmax(d["z"] / T0, axis=1)[:, 1]
        y = d["y"]
        df = pd.DataFrame({"wsi": man.wsi.to_numpy()[d["idx"]], "y": y, "p": p})
        df["correct"] = (df.p >= 0.5).astype(int) == df.y
        print(f"== {name} ({len(df)} patches, {df.wsi.nunique()} slides)")
        print(f"  AUROC {roc_auc_score(y, p):.3f} | share predicted tumour {(p >= .5).mean():.2f} "
              f"(truth {y.mean():.2f})")
        print(f"  mean p(tumour): true normal {p[y == 0].mean():.3f} | true tumour {p[y == 1].mean():.3f}")
        print(f"  accuracy: on normal patches {df[df.y == 0].correct.mean():.3f} | "
              f"on tumour patches {df[df.y == 1].correct.mean():.3f}")
        per = df.groupby("wsi").agg(n=("y", "size"), tumour=("y", "sum"), acc=("correct", "mean"))
        print("  5 worst slides:\n" + per.sort_values("acc").head(5).to_string())
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="/content/data")
    main(p.parse_args())
