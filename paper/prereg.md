# Pre-registered analysis plan

**Committed:** 2 November 2026, after the main runs finished and **before any main-run (seeds 1 to 5) stream results were summarised or inspected.**
Design choices below were fixed on the pilot seed (seed 0) by rules written before the pilot outputs they used were looked at. Seed 0 is never reported.

## 1. Study as run

| Item | Value |
|---|---|
| Data | PatchCamelyon patches from the 269 CAMELYON16 training slides; RUMC (170 slides) and UMCU (99); at most 500 patches per slide (128,761 patches) |
| Settings (cells) | `pcam_fwd` (RUMC to UMCU, **main**), `pcam_rev` (UMCU to RUMC) x ResNet-18, ViT-S/16 = **4 cells** |
| Seeds | 1 to 5 (seed is the unit of evidence) |
| Splits | By slide, stratified by slide type; source 45/5/5/5/20/20 (train/val_es/val_temp/val_buf/stream/test), target 60/30/10 (stream/test/dev) |
| Source model | ImageNet-pretrained, AdamW lr 1e-4, early stopping on val NLL (patience 3), flips/rotations + colour jitter 0.5, temperature T0 fitted on val_temp |
| Streams | T = 10 windows x 500 patches, 50% tumour; gradual: target share 0, then linear ramp over 8 windows, then 1; no-drift twin: all source |
| Stream learning | Test-then-train; one pass per window; SGD lr **0.001**, momentum 0.9, weight decay 5e-4, batch 64, same augmentation |
| Strategies | frozen; fine-tuning (FT); experience replay (P): reservoir buffer of 1,000 stream patches mixed 1:1 into each batch |
| Evaluation sets | S (source test), G (target test), B (1,000 labelled patches from val_buf slides), E (next window); deployed temperature T0 fixed throughout |
| Measurement | Binary Brier with CORP decomposition (MCB, DSC, UNC), isotonic recalibration cross-fitted by slide (2 folds) |

## 2. Decisions taken on the pilot (seed 0), with the rule that decided each

1. **Colour jitter 0.5** for source training: without it the RUMC model learned a colour shortcut (UMCU AUROC 0.63). Adopted before any reported seed.
2. **Window size 500** (not 1,000): supply check showed 1,000-patch windows exhaust the pcam_rev no-drift twin in every seed; pre-specified G3 fallback "halve the window size".
3. **Stream learning rate 0.001.** Rule fixed before the check: largest lr in {0.01, 0.003, 0.001} whose FT no-drift twin changes S Brier by less than 0.005 and whose gradual run raises G AUROC; if none passes, use 0.001. Result: twin changes 0.0239 / 0.0101 / 0.0056, so none passed and 0.001 was used. Reported as a limitation (training on the source stream alone still perturbs S slightly).
4. **G4 is judged per setting** on the mean gap over seeds 1 to 5; individual seeds are kept whatever their gap.
5. **V1 known-answer check passed** (9 of 9 damages) on seed-0 outputs, after the start was recalibrated so that each damage is a real damage. The pass rule for calibration damage was changed from "MCB rises by at least 0.005" to "MCB rises" after the first run, because the size of the rise depends on damage strength; recorded in the log.

## 3. Quantities (all on S, checkpoint 0 = start, checkpoint 10 = end)

For a run r: change d_r(X) = X(end) - X(start), for X in {Brier, MCB, DSC}.
Drift-attributable change for strategy s, per seed: **Delta_s(X) = d_gradual,s(X) - d_nodrift,s(X).**
- Delta MCB_s = Delta_s(MCB); **DSC lost_s = -Delta_s(DSC)**; Delta Brier_s = Delta MCB_s + DSC lost_s (UNC is fixed).
- **Calibration share (per cell):** rho = mean(Delta MCB) / (mean(Delta MCB) + mean(DSC lost)), means over the 5 seeds. Undefined, and reported as such, if the denominator is not positive.

## 4. Hypotheses and decision rules

SESOI = **0.005** on the Brier scale (the pilot has one seed, so the "twice the pilot seed SE" option cannot be computed; 0.005 is kept).

- **P1 (precondition, main cell: pcam_fwd ResNet-18).** Mean Delta Brier_FT >= 0.005. If not, H1 is reported as untestable.
- **H1 (confirmatory, main cell).** rho_FT >= 0.25. 95% interval from a bootstrap over seeds (10,000 resamples of the 5 seeds, resampling numerator and denominator together; percentile interval). Supported if the interval is entirely above 0.25, rejected if entirely below, inconclusive otherwise.
- **H2 (confirmatory, main cell).** Replay reduces lost discrimination: mean over seeds of (DSC lost_FT - DSC lost_P) >= 0.005, paired by seed; one-sided 95% t-interval lower bound above 0.005 = supported; upper bound below 0.005 = rejected; otherwise inconclusive. The replay effect on MCB is reported two-sided.
- **Multiplicity:** Holm correction over {H1, H2}.
- **H3 (consistency, 4 cells, descriptive).** Report rho_FT and the H2 effect with intervals for every cell. Consistent if the rho point estimate is on the same side of 0.25 as in the main cell in **at least 3 of 4 cells**, and the H2 effect has the same sign in at least 3 of 4. (Changed from 6 of 8 because the Camelyon17 cells were not run; see section 6.)
- **H4 (practical benefit).** For each seed and cell, using the FT gradual run:
  - rho_hat_k is computed like rho but from the deployed model's start-to-end change on **k labelled patches of B** (no twin is available in deployment), k in {100, 250, 500, 1000}, the k patches drawn class-balanced with a fixed seed.
  - **Rule:** rho_hat_k >= 0.5 -> recalibrate (R), else replay (P).
  - **R** = refit the temperature of the end-of-stream FT model on the same k patches of B, apply to S. **P** = end-of-stream replay model with T0. **R+P** = refit temperature of the replay model on the same k patches.
  - **Regret** = Brier on S of the chosen fix minus the best of {R, P, R+P} for that seed.
  - H4 holds if the rule's mean regret is lower than both always-R and always-P, as paired differences over all cells and seeds, reported for each k with 95% t-intervals.
- **Costs (advisor point 5):** labels k; replay storage 1,000 x 96 x 96 x 3 bytes = 27.6 MB; GPU minutes per run from `meta.json`.

## 5. Secondary, reported but not tested
Curves of Brier, MCB, DSC over checkpoints; G (target) metrics; isotonic recalibration as an alternative R; AUROC as a check on DSC; frozen runs (must be flat on S).

## 6. Deviations from the proposal
- Camelyon17-WILDS cells not run (compute and storage limits); left as future work. Study has 4 cells instead of 8.
- Only CAMELYON16 training slides used (hospital labels of test slides could not be verified).
- Window size 500 instead of 1,000; stream lr 0.001 by the rule above.
