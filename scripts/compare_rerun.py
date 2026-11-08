"""E-6: compare the original primary-cell runs with the rerun after the GradScaler fix.

Decision rule (fixed in paper/prereg_addendum.md before running):
if, for FT or replay, the seed-mean of dMCB or of DSC lost differs between original and
rerun by more than 0.005 (the SESOI), all four cells are rerun with the corrected code and
the corrected results become primary; otherwise the originals stay primary and this
comparison is reported as a check.

Usage: python scripts/compare_rerun.py --out $D/stream_runs
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import analysis as A  # noqa: E402

SETTING, MODEL = "pcam_fwd", "resnet18"


def components(out, tag):
    rows = []
    for seed in A.SEEDS:
        for strat in ["ft", "replay"]:
            g = A.s_change(out, SETTING, MODEL, seed, f"{strat}_gradual{tag}")
            n = A.s_change(out, SETTING, MODEL, seed, f"{strat}_nodrift{tag}")
            rows.append({"seed": seed, "strategy": strat, "d_brier": g["brier"] - n["brier"],
                         "d_mcb": g["mcb"] - n["mcb"], "dsc_lost": -(g["dsc"] - n["dsc"])})
    return pd.DataFrame(rows)


def main(a):
    orig, new = components(a.out, ""), components(a.out, "_rerun")
    m = orig.merge(new, on=["seed", "strategy"], suffixes=("_orig", "_rerun"))
    pd.set_option("display.width", 200)
    print("Per seed (original vs rerun after the scaler fix):")
    print(m.round(4).to_string(index=False))
    s = m.groupby("strategy").mean(numeric_only=True).drop(columns="seed")
    print("\nSeed means:")
    print(s.round(4).to_string())
    worst = 0.0
    for strat in ["ft", "replay"]:
        for c in ["d_mcb", "dsc_lost"]:
            worst = max(worst, abs(s.loc[strat, f"{c}_rerun"] - s.loc[strat, f"{c}_orig"]))
    print(f"\nLargest change in a seed-mean component: {worst:.4f} (threshold 0.005)")
    print("DECISION:", "rerun all four cells with the corrected code" if worst > 0.005
          else "originals stay primary; report this comparison as a check")
    skipped = []
    for seed in A.SEEDS:
        for name in ["ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift"]:
            p = os.path.join(A.run_dir(a.out, SETTING, MODEL, seed), f"stream_{name}_rerun", "meta.json")
            meta = json.load(open(p))
            skipped.append({"seed": seed, "run": name, "steps": meta.get("steps"),
                            "skipped_steps": meta.get("skipped_steps")})
    print("\nOptimizer steps in the rerun (skipped = overflow steps the scaler dropped):")
    print(pd.DataFrame(skipped).to_string(index=False))
    m.to_csv(os.path.join(a.out, "analysis", "E6_rerun_comparison.csv"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    main(p.parse_args())
