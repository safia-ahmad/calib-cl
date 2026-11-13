# Addendum 4b: frozen-BatchNorm control in a second cell (exploratory)

Written and committed after the X5/X6 results in the primary cell were seen, and BEFORE the
run below. Exploratory; does not change any confirmatory result.

## X6b. Frozen-BatchNorm fine-tuning, reverse ResNet-18 cell

Same as X6 (BatchNorm running statistics frozen during stream updates, lr 0.001, FT, gradual
and twin, seeds 1-5, tag `bnfrozen`), in the pcam_rev / resnet18 cell (10 runs).
Outcomes: drift-attributable dBrier, dMCB, DSC lost on S; dAUROC on G.

Motivation: X6 removed the drift-attributable discrimination loss in the primary cell. This run
checks whether the same holds in a second cell. Note in advance: forgetting in this cell was
small at lr 0.001 (dBrier 0.011, DSC lost 0.008), so this is a weak test; a null result here is
expected to be uninformative rather than contradictory.

Reported in the same exploratory section as X5 and X6, whatever the result.
