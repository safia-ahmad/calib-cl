"""Day 11: known-answer check V1 on a finished source run's outputs on S.

Usage (from the repo root):
  python scripts/validate_v1.py \
      --run /content/drive/MyDrive/calib-cl-data/runs/pcam_fwd_resnet18_seed0_colour0.5

Prints the table and writes v1_table.csv into the run folder.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl.validate import binary_logit, known_answer_table  # noqa: E402


def main(a):
    s = json.load(open(os.path.join(a.run, "summary.json")))
    d = np.load(os.path.join(a.run, "logits_S_source_test.npz"))
    man = pd.read_csv(os.path.join(a.data, "pcam_compact_manifest.csv"), usecols=["wsi"])
    groups = man.wsi.to_numpy()[d["idx"]]
    tab = known_answer_table(binary_logit(d["z"], s["T0"]), d["y"], groups, n_rep=a.reps)
    tab.to_csv(os.path.join(a.run, "v1_table.csv"), index=False)
    pd.set_option("display.width", 120)
    print(f"run: {a.run}\nT0 = {s['T0']:.3f}, {len(d['y'])} patches, {len(set(groups))} slides\n")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    verdicts = tab[tab.check != "-"]["pass"]
    print(f"\nV1: {'PASS' if (verdicts == 'PASS').all() else 'FAIL'} "
          f"({(verdicts == 'PASS').sum()} of {len(verdicts)} damages behave as expected)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="/content/data")
    p.add_argument("--reps", type=int, default=20)
    main(p.parse_args())
