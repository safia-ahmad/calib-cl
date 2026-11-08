# Addendum to the analysis plan (sensitivity and supplementary analyses)

**Committed:** 8 November 2026, after the pre-registered analysis (`prereg.md`) was run and before any of the analyses below were run.
Nothing here changes a confirmatory verdict in `prereg.md`. Every result below is reported whatever it shows, labelled as a check, sensitivity or exploratory analysis.

## E-6 Implementation check: per-run mixed-precision scaler (run first)
- **Issue found in code review:** stream training kept one GradScaler per optimiser `id()` for the whole process, so a new run could inherit a previous run's loss scale and skip a different number of overflow steps than its twin.
- **Fix:** one fresh GradScaler per run; steps and skipped steps are now logged in `meta.json`.
- **Check:** rerun the primary cell (pcam_fwd, ResNet-18, seeds 1-5; FT and replay, gradual and no-drift) with the fix, results tagged `_rerun`, originals kept.
- **Decision rule:** if, for FT or replay, the seed mean of dMCB or of DSC lost changes by more than 0.005 (SESOI), all four cells are rerun with the corrected code and the corrected results become primary, with the originals reported in the supplement. Otherwise the original results stay primary and the comparison is reported as a check.

## E-1 Component table (all 4 cells)
dBrier, dMCB and DSC lost (drift-attributable, on S, checkpoint 10 vs 0) per cell: seed mean, 95% t-interval (4 df) and all 5 seed values. Reported next to rho.

## E-2 Hierarchical bootstrap (primary cell)
2,000 resamples: resample the 5 seeds with replacement, then, within each sampled seed, resample the S slides with replacement (cluster bootstrap), and recompute CORP (2-fold slide cross-fitting) at checkpoints 0 and 10 for the gradual and twin runs. Report 95% percentile intervals for dMCB, DSC lost and rho (FT; replay also for dMCB and DSC lost).

## E-3 Other measures (all 4 cells)
Drift-attributable change (gradual minus twin, checkpoint 10 vs 0, on S) in AUROC, equal-mass ECE (15 bins) and equal-width ECE, as seed mean with 95% t-interval.

## E-4 ViT endpoint sensitivity (pcam_fwd, ViT-S/16)
(a) per-seed trajectories of drift-attributable dMCB and DSC lost; (b) rho and its components at checkpoint 9, and using the mean of checkpoints 8-10 as the endpoint; (c) CORP reliability diagrams on S at checkpoints 0, 9 and 10 (FT gradual), pooled over seeds and per seed. The pre-registered endpoint (checkpoint 10) remains the confirmatory one.

## E-5 Oracle temperature bound (all 4 cells, exploratory)
At checkpoint 10 of FT gradual, fit one temperature on S itself and report the Brier gain on S over T0. This is an upper bound on what any single temperature could achieve, not a deployable fix.

## S-1 Onset of forgetting (all 4 cells)
First checkpoint at which the seed-mean drift-attributable DSC lost reaches 0.005 and stays at or above it for all later checkpoints; reported with that window's target share. "None" if never.

## S-2 Sensitivity and specificity (all 4 cells)
At threshold p >= 0.5 with T0, drift-attributable change in sensitivity and specificity on S (checkpoint 10 vs 0, gradual minus twin), seed mean with 95% t-interval.

## Not run
Replication seeds 6-10, replay buffer-size sensitivity, isotonic recalibration and any second dataset are not part of this paper's analyses.
