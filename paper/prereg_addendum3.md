# Addendum 3: correction to the E-6 check

**Committed:** 8 November 2026, before the corrected comparison below is run.

## Error found
The E-6 rerun (`_rerun`) and the full "v2" rerun (`_v2`) were launched without `--lr`, so `run_stream.py`
used its old default learning rate of **0.01** instead of the study's **0.001** (`prereg.md`, decision 3).
The E-6 comparison therefore compared lr 0.001 (original) with lr 0.01 (rerun). Its conclusion that the
decomposition is "not stable across reruns" and that GPU nondeterminism was the cause is **withdrawn**: the
difference is explained by the learning rate. Supporting fact: in the run checked (pcam_fwd, ResNet-18, seed 1, FT gradual) the `_rerun` and `_v2` results are
identical to four decimals at every checkpoint, although only `_v2` used the deterministic settings.
Addendum 2's premise is withdrawn with it; its deterministic settings remain in the code.

## Fix
- `run_stream.py` now requires `--lr` (no default), so this cannot recur.

## Corrected check (replaces E-6)
- Rerun the primary cell (pcam_fwd, ResNet-18, seeds 1-5; FT and replay; gradual and no-drift) with the scaler fix,
  deterministic settings and **lr 0.001**, tag `_e6b`.
- Same decision rule as `prereg_addendum.md`: if any seed-mean dMCB or DSC lost (FT or replay) differs from the
  original by more than 0.005, all four cells are rerun at lr 0.001 and become primary; otherwise the originals
  stay primary and this comparison is reported as a check.

## What happens to the lr 0.01 runs
The `_rerun` and `_v2` folders are kept, not deleted, and reported in the supplement as an **unplanned
learning-rate sensitivity analysis (lr 0.01 vs 0.001), exploratory**, with the reason they exist stated plainly.
