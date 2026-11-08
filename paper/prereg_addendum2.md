# Addendum 2 to the analysis plan

**Committed:** 8 November 2026, after the E-6 check and **before** any of the reruns below.

## What E-6 showed
- The primary cell (pcam_fwd, ResNet-18, seeds 1-5) was rerun with the per-run GradScaler fix.
- No optimizer step was skipped in any rerun (80 of 80 steps in all 20 runs), so the scaler issue did not alter training.
- Seed means nevertheless changed (largest change 0.0154 > 0.005), because GPU training was not deterministic and the
  stream training is volatile from window to window. Total forgetting (FT dBrier 0.026 vs 0.024) and the replay
  benefit replicated; the split into dMCB and DSC lost did not (dMCB -0.0018 vs 0.0106; DSC lost 0.0278 vs 0.0138).
- By the rule in `prereg_addendum.md`, all four cells are rerun and the reruns become primary.

## Changes for the rerun ("v2")
1. **Deterministic training:** cuDNN deterministic, benchmark off, deterministic algorithms (warn-only), cuBLAS workspace
   config. Checked first by running one seed twice; the two runs must match to 1e-6.
2. **Everything else unchanged:** same source models, streams, lr 0.001, batch, buffer, augmentation, evaluation sets.
   Frozen runs are not repeated (they never train).
3. Results are written to folders with suffix `_v2`; the original runs and the E-6 `_rerun` runs are kept.

## Analyses on v2
- **Primary:** every pre-registered analysis in `prereg.md` (P1, H1, H2, H3, H4, costs) is repeated unchanged on v2,
  and v2 is what the paper reports as the main result. The original-run results are reported in the supplement.
- **Test-retest reliability (new, reported in the main text):** for each cell and seed, compare the drift-attributable
  dBrier, dMCB and DSC lost between the original run and v2: per-seed differences, the within-seed (run-to-run) SD,
  the between-seed SD, and the intraclass correlation ICC(1) over the two runs. For the primary cell also the E-6
  `_rerun` as a third run.
- **Sensitivity endpoint (all cells):** the same quantities with the endpoint defined as the mean of checkpoints 8-10
  instead of checkpoint 10. Reported alongside the checkpoint-10 results; never replacing them.
- All analyses in `prereg_addendum.md` (E-1 to E-5, S-1, S-2) are run on v2.

## Interpretation rule, fixed now
A decomposition result is called **reliable** only if its sign agrees between the original run and v2 in at least 4 of
5 seeds for that cell. Otherwise it is reported as not reliably measurable at a single endpoint under this protocol.
