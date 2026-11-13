# Addendum 4c: is the miscalibration left after freezing BatchNorm recalibratable? (exploratory)

Written and committed after the X6/X6b results were seen and BEFORE the analysis below.
No training; reads saved logits only. Exploratory; changes no confirmatory result.

## X6c. Frozen-BN fine-tuning followed by recalibration

Motivation: with BatchNorm frozen (X6, X6b), the drift-attributable discrimination loss vanished
in the primary cell and pointed the same way in the reverse ResNet cell, while a small
drift-attributable miscalibration remained (dMCB about +0.009 in both cells).

For pcam_fwd and pcam_rev ResNet-18, seeds 1-5, k in {100, 250, 500, 1000}, using exactly the
same k-patch subsets of B as H4, compute Brier on S at the end of the stream for:
  - FT-BN: frozen-BN FT model at T0 (no fix)
  - FT-BN + R: temperature refitted on the k patches
  - FT-BN + R-iso: isotonic regression fitted on the k patches
and compare with the existing FT (no fix), FT + R and replay (P).

Reported: mean Brier per condition; paired differences (95% t over seeds) of FT-BN + R vs P and
FT-BN + R vs FT-BN. Question: once BatchNorm is frozen, does a cheap recalibration bring the model
to (or below) replay's Brier? "Comparable" = |FT-BN + R minus P| < SESOI (0.005) with the 95%
interval inside +/-0.01. Exploratory, hypothesis-generating; no claim of equivalence.
