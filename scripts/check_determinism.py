"""Run-twice check: are two runs with the same seed identical?
Usage: python scripts/check_determinism.py <folder A> <folder B>   (two stream_* result folders)"""
import sys

import numpy as np
import pandas as pd

a, b = (pd.read_csv(f"{d}/metrics.csv") for d in sys.argv[1:3])
num = a.select_dtypes("number").columns
diff = np.nanmax(np.abs(a[num].to_numpy() - b[num].to_numpy()))
print(f"largest difference between the two runs: {diff:.2e}")
print("DETERMINISTIC" if diff < 1e-6 else "NOT deterministic: stop and send this to Claude")
