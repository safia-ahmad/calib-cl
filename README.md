# calcl: starter code for "Is forgetting miscalibration or lost discrimination?" (PCam, RUMC -> UMCU)

What is tested and what is not:

| File | Status |
|---|---|
| `calcl/metrics.py` | Tested (`tests/test_metrics.py`): accuracy, NLL, Brier, AUROC, binned ECE (equal-width and equal-mass), class-1 ECE, signed confidence gap, temperature fitting, recoverable NLL under cross-fitted temperature scaling (secondary measure; not a discrimination measure) |
| `calcl/stream.py` | Tested (`tests/test_stream.py`): alpha schedules (linear, sigmoid, step, none), class-stratified mixture windows, no reuse across windows, reproducibility |
| `calcl/corp.py` | Tested (`tests/test_corp.py`): CORP Brier decomposition (MCB, DSC, UNC), in-sample and slide-grouped cross-fitted |
| `calcl/alarms.py` | Tested (`tests/test_alarms.py`): equal-false-alarm thresholds from no-drift controls, persistent onset, censored lead time |
| `calcl/run_stream_ILLUSTRATIVE.py` | **Untested blueprint** (written without PyTorch/GPU). Fine-tuning, reservoir replay, fixed-interval online EWC, test-then-train loop |

Run the tests:

```bash
pip install numpy scipy scikit-learn pytest
python -m pytest -q
```

Suggested environment for the full project (Python 3.11):

```bash
pip install torch torchvision            # match your CUDA version
pip install medmnist wilds numpy scipy scikit-learn pandas pyarrow matplotlib pyyaml tqdm pytest
```

Project layout to grow into:

```
calib-cl-medical/
  configs/            pilot_pathmnist.yaml, main_camelyon.yaml (frozen before runs)
  calcl/              data/, shifts.py, stream.py, models.py, strategies/, metrics.py, alarms.py
  scripts/            01_audit.py 02_make_splits.py 03_train_source.py 04_run_stream.py 05_analyze.py
  results/runs/<run_id>/  config.yaml, git_hash.txt, metrics.parquet, ckpt_t###.pt
  tests/
```
