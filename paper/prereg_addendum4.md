# Analysis-plan addendum 4: extension analyses (exploratory)

Written and committed BEFORE any of the analyses or runs below were carried out.
All results from this addendum are EXPLORATORY. They do not change the confirmatory
family {H1, H2}, its Holm correction, or any verdict already reported.
SESOI stays 0.005 throughout. The primary cell stays pcam_fwd / resnet18.
Endpoint stays checkpoint 10. Seeds stay 1-5. Intervals: 95% t over seeds unless stated.

## X1. Learning-rate sensitivity (no new training)

The complete `_v2` run set (all 4 cells x 5 seeds, FT and replay, gradual and twin, lr 0.01)
already exists from the earlier accidental runs (addendum 3). Run `analyze.py --tag _v2`
and the C9 endpoint check on it. Report, for every cell: dBrier, dMCB, DSC lost and rho (FT),
and the H2 effect. Question: does the primary-cell pattern (DSC lost >= SESOI, rho < 0.25)
also hold at lr 0.01? "Holds" = DSC lost lower 95% bound > 0 and rho point estimate < 0.25.
No verdicts are re-tested; this is reported as a sensitivity analysis.

## X2. Isotonic recalibration as a deployable fix (no new training)

The fix listed in the original plan but not run. For every cell, seed and k in
{100, 250, 500, 1000}: fit isotonic regression (PAV, clipped to [0,1]) of y on p for the
end-of-stream FT model on the same k-patch subset of B used for H4; apply it on S; report
Brier on S and regret against the best of {R, P, R+P, R-iso}. Compare R-iso with R
(temperature) and P (replay).

## X3. Threshold reset (no new training)

For the end-of-stream FT model, choose the threshold on the k-patch subset of B that
restores the source model's checkpoint-0 sensitivity on that subset. Report sensitivity and
specificity on S with this threshold, versus threshold 0.5. Question: how much sensitivity
can be recovered, and at what cost in specificity? Primary cell; other cells reported as well.

## X4. Figures (no new results)

Replace the rho forest plot with a scatter of dMCB (x) vs DSC lost (y), one point per seed,
per cell and strategy, with the line rho = 0.25. Add a raw-vs-twin panel for the primary cell.

## X5. Batch-matched fine-tuning control (new runs, primary cell only)

Replay halves each new patch's gradient weight (64 new + 64 replayed, mean loss). Control:
FT at lr 0.0005 (half), otherwise identical (batch 64 new patches, 8 steps/window,
same streams and seeds), gradual and twin, seeds 1-5 (10 runs). Tag `_lrhalf`.
Report DSC lost, dMCB, dBrier on S and dAUROC on G. Question: does halving the step size
alone reproduce replay's smaller loss? Interpretation rule: if DSC lost (FT-half) is within
SESOI of replay's 0.0071, the replay benefit cannot be attributed to past data;
if it is within SESOI of FT's 0.0278, it can.

## X6. Frozen-BatchNorm fine-tuning (new runs, primary cell only)

FT with all BatchNorm layers kept in eval mode during stream updates (running statistics
frozen at the source values; affine parameters still trained), otherwise identical to FT,
gradual and twin, seeds 1-5 (10 runs). Tag `_bnfrozen`. ResNet-18 only (ViT has no BN).
Question: is the discrimination loss driven by BatchNorm statistics drifting to the target?
Same outcomes as X5. "Mostly removed" = DSC lost (frozen-BN) below half of FT's 0.0278.

## X7. Second dataset: Camelyon17-WILDS replication (new data and runs)

Specified in full in a separate addendum (5), committed after a data-supply and colour check
(descriptive only: slide and patch counts, class mix and mean colour per hospital) and
BEFORE any model is trained on it. Hospital pairs will be chosen from those descriptives
by a rule written in addendum 5, never from results.

## Reporting

Every X-analysis is reported in a clearly marked exploratory section, including null results.
None is used to support a confirmatory claim.
