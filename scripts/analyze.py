"""Day 27-31: run the pre-registered analysis (paper/prereg.md) on all main runs.

Usage (Colab):
  python scripts/analyze.py --data /content/data --models $D/models --out $D/stream_runs
Prints every verdict and writes CSV tables to <out>/analysis/.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl import analysis as A  # noqa: E402


def main(a):
    A.SUFFIX = a.tag
    man = pd.read_csv(os.path.join(a.data, "pcam_compact_manifest.csv"), usecols=["wsi", "label"])
    res_dir = os.path.join(a.out, "analysis" + a.tag)
    os.makedirs(res_dir, exist_ok=True)
    pd.set_option("display.width", 200)
    fl = lambda v: f"{v:.4f}"  # noqa: E731

    print(f"Sanity: largest S Brier change in any frozen run = {A.frozen_check(a.out):.6f} (must be ~0)\n")
    dt = A.drift_table(a.out)
    dt.to_csv(os.path.join(res_dir, "drift_per_seed.csv"), index=False)
    res, h3 = A.hypotheses(dt)
    h3.to_csv(os.path.join(res_dir, "H3_cells.csv"), index=False)

    print("=== Main cell: pcam_fwd, ResNet-18 (seeds 1-5) ===")
    print(f"P1  mean drift-caused Brier change (FT) = {res['P1_mean_dBrier_FT']:.4f}  -> {res['P1']}")
    lo, hi = res["H1_ci"]
    print(f"H1  rho_FT = {res['H1_rho']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (threshold 0.25)  -> {res['H1']}")
    l2, h2 = res["H2_one_sided_ci"]
    print(f"H2  DSC lost (FT - replay) = {res['H2_effect']:.4f}  one-sided 95% [{l2:.4f}, {h2:.4f}]"
          f"  (SESOI 0.005)  -> {res['H2']}")
    m, ml, mh = res["H2_mcb_effect"]
    print(f"    replay effect on MCB (FT - replay) = {m:.4f}  95% [{ml:.4f}, {mh:.4f}]")
    print(f"    Holm p-values: H1 {res['H1_p']:.3f}, H2 {res['H2_p']:.3f}\n")

    print("=== H3: all 4 cells ===")
    print(h3.to_string(index=False, float_format=fl))
    print(f"rho on the same side of 0.25 as main cell: {res['H3_rho_same_side']}/4; "
          f"H2 effect same sign: {res['H3_H2_same_sign']}/4  -> {res['H3']}\n")

    print("=== H4: fix rule (rho_hat_k >= 0.5 -> recalibrate, else replay) ===")
    t4 = A.h4_table(a.out, a.models, man.label.to_numpy(), man.wsi.to_numpy())
    t4.to_csv(os.path.join(res_dir, "H4_per_seed.csv"), index=False)
    s4 = A.h4_summary(t4)
    s4.to_csv(os.path.join(res_dir, "H4_summary.csv"), index=False)
    print(s4.to_string(index=False, float_format=fl))

    print("\n=== Costs ===")
    c = A.cost_table(a.out)
    c.to_csv(os.path.join(res_dir, "costs.csv"))
    print(c.to_string(float_format=lambda v: f"{v:.2f}"))
    print("Replay storage: 1,000 x 96 x 96 x 3 bytes = 27.6 MB; recalibration labels: k patches")

    json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in res.items()},
              open(os.path.join(res_dir, "verdicts.json"), "w"), indent=2, default=float)
    print(f"\nAll tables saved in {res_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/content/data")
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="", help='analyse the stream runs with this folder suffix, e.g. "_v2"')
    main(p.parse_args())
