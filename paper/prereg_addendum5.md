# Addendum 5: Camelyon17-WILDS Replication Protocol

**Status: draft 3, final for author approval (14 November 2026).** This replaces drafts 1 and 2 (changes from draft 2: Section 3.3 arrangement rotation, Section 3.5 first feasibility run, Section 5.2 block re-draws). Commit it BEFORE the first source
model is trained, and only after the metadata-only feasibility report (Section 3.5) shows no
unresolved failure. Once it is committed, nothing in Sections 3–10 changes. Any later change is
recorded in Section 12 as a dated deviation with its reason, and is reported in the paper.

**Framing.** This is an **exploratory external replication** with patient-level splits and
different hospital pairs. It is not a confirmatory test, it does not follow the WILDS benchmark
protocol, and it does not establish generalisation across architectures (only ResNet-18 is used).

**What has been seen before this addendum.** Dataset descriptives (patients, slides and patches
per centre; mean RGB of normal patches; capped supply per patient) and metadata-only split
simulations. No model has been trained on Camelyon17, and no outcome has been seen.

---

## 1. Question

Does the main study's primary finding hold on a second dataset with patient-level splits?

The finding: under a gradual acquisition shift, fine-tuning (FT) forgets the source hospital
mainly through lost discrimination (DSC), not added miscalibration (MCB), relative to a no-drift
twin.

Secondary questions:
- Does replay still reduce this loss? (directional only)
- Is the loss still absent when BatchNorm running statistics are frozen? (exploratory control)

## 2. Dataset

- **Source.** Camelyon17-WILDS v1.0 (Koh et al. 2021): 455,954 patches of 96 × 96 px, 50 slides
  and 5 centres (10 slides each), from the CAMELYON17 challenge (Bándi et al. 2019). Labels are
  patch-level, taken from the WILDS metadata.
- **Overlap.** Neighbouring patches overlap. The coordinate units, the step between patches and
  the patch size in coordinate units are **measured from the images** (Section 5.1), not assumed.
- **Hospitals.** The centres are anonymised. CAMELYON17 came from RUMC, CWZ, UMCU, RST and LPON,
  so two WILDS centres are probably the main study's hospitals (with different slides).
  Descriptively, centre 0's mean normal-patch colour (187, 156, 178) is close to the main study's
  RUMC (189, 157, 178).
  - We do not describe these as new hospitals.
  - We cannot verify that no patient appears in both CAMELYON16 and CAMELYON17.
  - Wording in the paper: "a second dataset with different hospital pairs and patient-level
    splits".
- **WILDS benchmark.** The official split trains on centres 0, 3 and 4, uses centre 1 for OOD
  validation and centre 2 for testing. This study uses its own cross-hospital design. It reports no
  WILDS benchmark numbers and makes no comparison with the leaderboard. Centre 2 is not used.

## 3. Hospital pairs, cells and patient-level splits

### 3.1 Pair selection (fixed rule, applied to descriptives only)

The rule uses the mean RGB of normal patches in each centre. Take the pair of centres with the
largest Euclidean distance, remove both, and take the largest remaining pair. Result:

| Pair | RGB distance |
|---|---|
| 1 ↔ 4 | 89.6 |
| 0 ↔ 3 | 46.4 |

Centre 2 is unused. Mean RGB is a coarse proxy, confounded by tissue composition, scanner and
staining. It is used only to choose pairs. The paper reports the RGB of all five centres and does
not call the chosen pairs "the largest shifts".

### 3.2 Cells

Within each pair, the source is the centre with more capped tumour patches.

| Cell | Source → target | Role | Capped tumour patches, source vs target |
|---|---|---|---|
| C17-A | 1 → 4 | **primary** | 10,356 vs 9,701 |
| C17-B | 0 → 3 | **primary** | 12,480 vs 8,758 |
| C17-A′ | 4 → 1 | secondary | reverse of C17-A |
| C17-B′ | 3 → 0 | secondary | reverse of C17-B |

Secondary cells use the same criteria but are reported descriptively and do not enter the overall
verdict.

### 3.3 Patient-level role assignment

**Unit and roles**

- The unit is the patient: all slides of a patient stay together.
- Source roles are `train`, `stream` and `test`. Target roles are `stream` and `test`.
- Each patient has exactly one role within a cell and seed.
- Test patients are never used for training, early stopping, temperature fitting, the stream or
  the replay buffer.

**Tumour-rich patients**

A patient is tumour-rich if it has ≥ 1,000 capped tumour patches:

| Centre | Tumour-rich patients |
|---|---|
| 0 | 10, 12, 15, 17 |
| 1 | 20, 22, 34 |
| 3 | 62, 73, 75 |
| 4 | 92, 96, 99 |

**Arrangements**

- Source: an arrangement is an ordered (stream, test) pair of distinct tumour-rich patients; the
  remaining tumour-rich patients go to `train`. With 3 tumour-rich patients there are 6
  arrangements; with 4 there are 12.
- Target: an arrangement is a split of the k tumour-rich patients with floor(k/2) in `stream` and
  the rest in `test` (3 arrangements for k = 3, 6 for k = 4). Arrangements that give identical
  role sets are listed once.
- All arrangements of a centre are listed in a fixed order and shuffled once:
  `numpy.random.default_rng(2026)` for sources, `default_rng(2027)` for targets.

**Minor patients**

- Minor patients are those with fewer than 1,000 tumour patches.
- For attempt `a`, they are permuted with `numpy.random.default_rng([seed, cell_code, side, a])`.
  - `cell_code` is 1–4 for C17-A, C17-B, C17-A′, C17-B′.
  - `side` is 0 for source, 1 for target.
- They are then dealt in the pattern train, stream, test, train (source) or stream, test
  (target).

**Deterministic selection rule**

Seeds are processed in order 1, 2, …, 5, separately for the source and the target side.

1. Seed 1 starts at position 0 of the shuffled list. Every later seed starts **one position after
   the arrangement used by the previous seed** (cyclically). Seeds therefore take distinct
   arrangements whenever at least 5 are feasible, and repeat only when fewer are.
2. For the current arrangement, minor-patient draws *a* = 0, 1, …, 1,999 are tried in order. For
   a source draw that passes the patient-level checks, block draws *b* = 0, …, 19 (Section 5.2)
   are tried in order. The first combination that passes **every** supply check (Section 3.4) is
   used.
3. If nothing passes, move to the next arrangement in the list and repeat step 2. The report
   lists every skipped arrangement with its most frequent failed checks.
4. If no arrangement passes, the cell is **infeasible**. It is dropped and reported as such.

Requirements are never relaxed, and patches are never split within a patient.

### 3.4 Supply checks (per class: tumour and normal separately)

**Before the validation split** (patient-level counts):

| Role | Minimum per class | Reason |
|---|---|---|
| source `stream` | ≥ 2,750 | the no-drift twin draws 5,000 source patches (about 2,500 per class), plus a 10% margin |
| target `stream` | ≥ 1,400 | the gradual stream draws about 2,500 target patches (about 1,250 per class), plus margin |
| source `test` | ≥ 400 | |
| target `test` | ≥ 400 | |
| source `train` | ≥ 1,500 | |

**After the spatial validation split and overlap removal** (Section 5):

| Subset | Minimum per class |
|---|---|
| final `train` | ≥ 1,500 |
| `val_es` | ≥ 150 |
| `val_temp` | ≥ 150 |

These floors are pragmatic. They are not derived from a power calculation. The final-training
floor equals the pre-split floor, so the split can never silently reduce the training requirement.

### 3.5 Feasibility report (metadata and a few images only; required before commit)

`scripts/c17_feasibility.py` applies Sections 3–5 exactly, using the same code that training will
use (`calcl/c17_split.py`). For all 4 cells × 5 seeds it reports:
- the patients in every role, with their tumour-rich members;
- tumour and normal counts per role;
- validation counts: assigned to `train`, `val_es` and `val_temp`; removed by overlap; final;
- S and G counts, tumour proportion, patients and slides;
- the arrangement and attempt used, and any skipped arrangements;
- repeated tumour-rich arrangements;
- failed checks.

Every leakage assertion (Section 5.4) runs for every cell and seed. The report is committed with
this addendum. **Training does not start if the report has an unresolved failure.**

**First run (draft 2 rules, 14 Nov 2026, metadata and images only).**
- Measured units: 1 px per coordinate unit, so P = 96 and the step is 32 px. All 40 image pairs
  agreed (maximum alignment error 0.017 grey levels, versus at least 4.86 at the next-best shift).
- All 20 cell-seeds were feasible, and every leakage assertion passed.
- That run exposed a flaw in the draft 2 seed rule. Starting each seed at (s − 1) mod K and
  skipping forward made seeds pile onto the same arrangement:
  - in 3 → 0, all 5 seeds used one tumour-rich source arrangement, and seeds 4 and 5 had
    identical source splits;
  - 0 → 3 seed 3 repeated seed 2;
  - in 3 → 0, seeds 3 and 4 had identical target splits, because the draft 2 target rule listed
    the same role sets twice.

  The rule was corrected (Section 3.3) before any training. The correction is driven by metadata
  only; no outcome exists yet.
- The final report under the draft 3 rules (`c17_feasibility_report.txt`, 14 Nov 2026) is
  committed with this addendum. The decision to accept repetition in 3 → 0 (author decision,
  14 Nov 2026) applies.

**Final report (draft 3 rules): no unresolved failures.**

- **Feasibility and leakage.** All 20 cell-seeds are feasible, and every leakage assertion passed.
  There are no identical full source splits and no identical target splits.
- **Repeated tumour-rich source arrangements.** Only one: 3 → 0, seed 5 repeats seed 1. Only 4 of
  6 source arrangements are feasible there, because the arrangements with patient 62 in the
  stream fail the stream supply check. The minor patients differ.
- **Repeated source test patients (identical S patients).** With 3 tumour-rich units, 5 seeds must
  reuse test patients:
  - 1 → 4: seed 3 repeats seed 1, and seed 4 repeats seed 2;
  - 0 → 3: seed 5 repeats seed 1.
- **Repeated tumour-rich target compositions.** Centres 1, 3 and 4 have only 3 target
  arrangements each, so the target's tumour-rich composition also repeats across seeds. The
  minor patients differ, so no target split is identical.
- **Tightest supplies (all at or above the floors):**

  | Quantity | Cell | Seed | Final count |
  |---|---|---|---|
  | final training tumour | 3 → 0 | 5 | 1,506 |
  | final training tumour | 3 → 0 | 2 | 1,972 |
  | final training tumour | primary cells, lowest | | 2,031 |
  | val_es tumour | 3 → 0 | 1 | 153 |
  | val_temp tumour | 3 → 0 | 2 | 153 |
  | val_es tumour | 4 → 1 | 4 | 154 |
  | validation tumour, either subset | primary cells, lowest | | 186 |

  Four of the 20 cell-seeds needed a block re-draw (largest draw index 13).
- **S sets.** 1,205–2,000 patches with 32–50% tumour. The smallest S tumour count is 405
  (0 → 3, seeds 1 and 5, test patients 10 and 16).

## 4. Patch sampling

- At most 3,000 patches per class per slide, chosen uniformly at random with `default_rng(0)`,
  once, before any split. Centres 0, 1, 3 and 4 then contribute 143,669 patches (about 4.0 GB as
  a compact HDF5 file).
- The cap is fixed and is not changed after outcomes are seen.
- Per-patient capped counts are in `c17_units.csv`, committed with this addendum.

## 5. Validation and test construction

### 5.1 Coordinate units (measured, not assumed)

- **Pairs measured.** Before any split, at least 5 (default 40) pairs of horizontally neighbouring
  patches are taken from the same slide.
- **Shift.** For each pair, the code finds the pixel shift that best aligns the two images.
- **Px per unit.** This gives the number of pixels per coordinate unit, *r*. The patch side in
  coordinate units is P = 96 / *r*.
- **The run stops unless:**
  - all pairs give the same *r*;
  - the best alignment is near-exact and clearly better than the next best;
  - the step between neighbouring patches is 32 px.

### 5.2 Spatially blocked validation within the source-training patients

1. **Blocks.** Each training slide is tiled into square blocks with a side of 10 P (10 patch
   widths).
2. **Stratified assignment.** Blocks from all training slides are pooled and sorted by tumour
   count (ties broken by patch count, then slide and block position). They are taken in groups of
   10. Within each group, the labels {train × 8, val_es, val_temp} are randomly permuted with
   `default_rng([1000 × seed + cell_code, b])`, where b is the block draw (0–19, Section 3.3). This
   keeps both validation subsets supplied with tumour blocks. Block draws are re-tried only to
   meet the pre-specified supply floors, which are label counts, never outcomes.
3. **Overlap removal.** Two patches on the same slide share pixels if, and only if,
   |Δx| < P and |Δy| < P, using the actual coordinates and the measured P. **Every** patch that
   shares pixels with any patch of a different subset is removed.
   - This applies to all three subset pairs: train/val_es, train/val_temp and val_es/val_temp.
     Adjacent val_es and val_temp blocks are therefore handled too.
   - Removal is symmetric: both patches in a conflicting pair are dropped. This is deliberate. It
     needs no assumption about which corner a coordinate refers to, and no priority rule.
   - Removed patches are not used anywhere.
4. **Uses.** `val_es` is used only for early stopping. `val_temp` is used only for fitting the
   source temperature T0. Neither enters the stream, the replay buffer or any evaluation set.

Validation is patient-internal: it is spatially separated within the training patients, not held
out by patient. Source calibration at T0 may therefore be optimistic. This is stated in the paper.

### 5.3 Evaluation sets

- **S** is drawn from the source `test` patients and **G** from the target `test` patients.
- Each uses `splits.eval_sample`: class-stratified, up to 2,000 patches, at most 400 per class per
  slide, with a fixed seed.
- We report patch counts, tumour proportion, patients and slides. The sets are not described as
  "balanced".
- Buffer B and the H4 fix rule are dropped. In the code, B is only an evaluation set for H4. The
  replay reservoir is a separate object and is kept unchanged.

### 5.4 Leakage assertions (run for every cell and seed, before any training)

`calcl/c17_split.assert_no_leakage` raises an error, and stops the run, if any of these is
violated:

1. No patient appears in both hospitals, or in more than one role.
2. Every slide maps to exactly one patient, and no slide is shared between roles.
3. `val_es` and `val_temp` contain only source-training patients.
4. There are zero pixel-sharing pairs between train and val_es, train and val_temp, and val_es and
   val_temp. This is checked by an independent brute-force search over actual coordinates with the
   measured P.
5. The train, val_es and val_temp patch sets are pairwise disjoint.

Unit tests check that these assertions catch injected leaks.

## 6. Model and training (identical to the main study unless stated)

**Source models**
- ResNet-18 with ImageNet initialisation, and the same augmentations and optimiser as the main
  study.
- At most 10 epochs, with patience 3. Early stopping uses `val_es`; T0 is fitted on `val_temp`.
- One source model per cell and seed: 20 in total.

**Stream**
- 10 windows of 500 patches.
- Target fraction α = 0, then (k − 0.5)/8 for k = 1…8, then 1.
- The number of target patches in a window is Binomial, and each hospital's patches are 0.5/0.5
  by class.
- No patch is reused within a stream.
- Source patches come from source `stream` patients, target patches from target `stream` patients.
- The gradual stream uses seed 10s + 1 and the twin uses 10s + 2.

**Updates**
- Test-then-train.
- SGD with lr 0.001, momentum 0.9, weight decay 5e-4 and AMP.
- Each step uses a batch of 64 new patches, with 8 steps per window.
- Replay adds up to 64 patches from a 1,000-patch reservoir.

**Logging**
- Each run logs the GPU type, wall time and peak memory.
- Runs are resumable and saved atomically. A finished run is never retrained.

## 7. Strategies, twins and runs

**Main runs (100):**

| Strategy | Gradual run | No-drift twin |
|---|---|---|
| frozen | yes | no |
| FT | yes | yes |
| replay | yes | yes |

5 runs per cell and seed × 4 cells × 5 seeds = 100 runs.

**Frozen-BatchNorm control (exploratory, 40 runs).** FT gradual and twin with `--freeze_bn`, on
the same streams and seeds as FT (4 × 5 × 2 runs).

Reported for this control:
- drift-attributable ΔBrier, ΔMCB and DSC lost;
- the paired difference FT − frozen-BN;
- absolute end-of-stream Brier on S for both the gradual run and the twin, and the twin's own
  change (end − start).

Wording rules, fixed now:
- If freezing BatchNorm statistics removes the drift-attributable DSC loss but also removes the
  twin's improvement, we write: "the drift-attributable loss was not observed when BatchNorm
  statistics were frozen; because freezing also removed the twin's improvement, this does not
  isolate a mechanism".
- We never write that BatchNorm updates cause forgetting.
- We call frozen BatchNorm a fix only if its absolute end Brier on S is lower than FT's (paired 95%
  CI entirely below 0).

## 8. Metrics and statistics

**Decomposition and estimand**
- Brier score is decomposed with CORP (isotonic/PAV), cross-fitted over 2 slide folds
  (`calcl/corp.py`): BS = MCB − DSC + UNC. The identity holds exactly by construction.
- Drift-attributable change on S = (gradual end − start) − (twin end − start). DSC lost is minus
  the DSC change.
- UNC is the same in both runs because they are evaluated on the same S. So, for each seed,
  ΔMCB + DSC lost = ΔBrier exactly.

**ρ = ΔMCB / (ΔMCB + DSC lost) = ΔMCB / ΔBrier**

This is the share of drift-attributable forgetting that is miscalibration. The point estimate is
mean ΔMCB / mean ΔBrier over seeds. It is undefined when mean ΔBrier ≤ 0.

The interval is a **Fieller (test-inversion) 95% interval** (Fieller 1954):
1. For a candidate value *c*, compute the per-seed contrast L_c,i = ΔMCB_i − *c* · ΔBrier_i.
2. The interval is the set of *c* for which a paired one-sample t-test (4 df, two-sided α = 0.05)
   does not reject mean L_c = 0.
3. Solving the quadratic gives the limits. The interval is a bounded interval exactly when the
   ΔBrier t-interval excludes 0, which R1 guarantees. Otherwise it is unbounded, and ρ is
   reported as "not estimable".

The decision rules are equivalent to t-tests on L_0.25 = ΔMCB − 0.25 · ΔBrier, so no ratio is ever
resampled:
- ρ's upper limit < 0.25 ⇔ the 95% t upper bound of mean L_0.25 < 0.
- ρ's lower limit > 0.25 ⇔ the 95% t lower bound of mean L_0.25 > 0.

As a secondary, descriptive comparison with the main study, the seed-percentile bootstrap
(10,000 draws, seed 0) is also reported. Draws with a non-positive denominator are excluded, and
their share is reported. If that share exceeds 5%, the bootstrap interval is labelled "unstable".
The bootstrap never enters a criterion.

**Why Fieller** (simulation in `scripts/addendum5_rho_sim.py`, 5 seeds, main-study SDs, scored
only where R1 holds):
- The non-positive-denominator draws that motivated the +∞ rule are rare once R1 holds: about 0%
  on average and below 1% at most. So +∞ versus dropping changes almost nothing.
- The real problem is that the percentile bootstrap over 5 seeds is too narrow:
  - its coverage is 75–85%, not the nominal 95%;
  - at a true ρ of exactly 0.25 (nominal error 2.5% each way), it declared "contradicted" in
    6.5–7% of cases when forgetting was clear (true DSC lost 0.028), and 1.7–3% when it was weak
    (0.012);
  - in the same setting, it passed R2's ρ condition in 9–22% of cases.
- Fieller held 91–95% coverage. Its "contradicted" rate was 0.3–2.5%. Its false R2 pass rate was
  2–3% when forgetting was clear and up to 8% when it was weak.
- Fieller, like every t-interval here, assumes approximately normal seed-level contrasts, an
  assumption that cannot be checked with 5 seeds.
- The cost is power. In the main-study-like case (DSC lost 0.028, true ρ = 0), R2's ρ condition
  passed in 52–75% of simulated cases with Fieller, against 83–95% with the bootstrap.

**Undefined ρ.** It cannot produce "contradicted", for two reasons:
- "contradicted" requires R1;
- R1 implies a bounded Fieller interval.

**Other intervals and reporting**
- Other intervals are 95% t-intervals over the 5 seeds. FT vs replay and FT vs frozen-BN are
  paired by seed.
- Every seed is reported with its source and target test-patient IDs.
- Drift-attributable ΔBrier is broken down per test patient, descriptively.
- The intervals describe variation over training seeds and rotations of a few patients. They do
  not describe variation over a population of patients or hospitals, and **the five seeds are not
  independent patient cohorts**. With 1–2 tumour-rich test patients, a patient-clustered bootstrap
  is not meaningful and is not used.
- Target performance (AUROC and Brier on G) is compared between FT and replay, paired, as a
  descriptive comparison. There is no equivalence claim. The permitted wording is "no detectable
  reduction in target-hospital performance was observed in this exploratory comparison".

## 9. Replication criteria (fixed now; evaluated per cell)

**R1 (forgetting detected).** FT's drift-attributable ΔBrier on S has a mean ≥ 0.005 (the SESOI)
**and** a 95% t lower bound > 0.

R1 establishes that the drift-attributable effect is positive (at α = 0.025, one-sided) and that
its **estimate** reaches the SESOI. It does **not** establish that the true effect exceeds 0.005,
which would need the lower bound itself to exceed 0.005. Paper wording: "forgetting was detected,
with an estimated size at or above the smallest effect of interest".

**R2 (mainly discrimination; evaluated only if R1 holds).** Both of the following:
- the FT DSC lost has a 95% t lower bound > 0;
- the Fieller 95% upper limit of ρ is < 0.25 (equivalently, the 95% t upper bound of mean L_0.25
  is < 0).

**R3 (replay, directional only).** Replay's drift-attributable DSC lost is below FT's in ≥ 4 of 5
seeds (paired). It is reported with the paired 95% t-interval. R3 is not a significance test and
does not enter the verdict.

**Per-cell status**

| Status | Condition |
|---|---|
| uninformative | R1 fails (ρ is reported as not estimable if its Fieller set is unbounded) |
| consistent | R1 and R2 hold |
| contradicted | R1 holds and ρ's Fieller lower limit is > 0.25 (mainly miscalibration) |
| not consistent | R1 holds, R2 fails, and the cell is not contradicted |

**Overall verdict** (primary cells only)

| Verdict | Condition |
|---|---|
| replicated | both primary cells are consistent |
| partially replicated | exactly one primary cell is consistent and neither is contradicted |
| contradicted | at least one primary cell is contradicted |
| uninformative | both primary cells are uninformative |
| not replicated | any other combination |

**Multiplicity**
- "Replicated" needs every condition to hold in both primary cells (an intersection–union rule),
  so no correction is needed for that claim.
- Secondary cells, R3, the frozen-BN control and the per-patient breakdowns are descriptive and
  uncorrected, and labelled as such.
- ΔMCB's upper bound is reported against the SESOI, descriptively.

## 10. Pilot, exclusions and stopping rules

**Pilot.** Before the full set, the pilot trains the source model for C17-A, seed 1, and runs its
5 stream runs. It is used **only** for:
- runtime estimation;
- GPU memory;
- storage requirements;
- pipeline integrity;
- validating the data and the split.

Its performance results (any metric on S, G or validation) are not inspected, and nothing in the
pilot is used to change:
- the architecture, learning rate or training settings;
- the stream schedule;
- the evaluation metrics or the replication criteria.

The pilot runs are part of the final dataset and are not repeated.

**Exclusions and stopping rules**
- **No exclusion on performance.** No run, seed or cell is excluded because of its performance or
  outcome. Source-model AUROC on S is reported for all 20 models.
- **Crashes.** A crashed or interrupted run is resumed or rerun with the same seed and data, and
  the event is logged.
- **Compute fallback (decided now, applied only on runtime evidence).**
  1. First drop the frozen-BN control.
  2. Then drop the secondary cells.
  3. The primary cells are always kept.

  Any reduction is recorded in Section 12.
- **No later additions.** No new seeds, cells, criteria, thresholds, caps, pairs or analysis
  methods are added after outcomes are seen.

## 11. Limitations (to be stated in the paper)

1. **Few tumour-rich patients.** There are 3–4 per centre. The source stream and source test each
   rest on one tumour-rich patient. The seeds rotate the same patients, so the five seeds are not
   independent patient cohorts. The intervals reflect training and rotation variability only.
2. **Repeated arrangements.** Wherever fewer than 5 arrangements are feasible (expected in 3 → 0),
   seeds repeat tumour-rich arrangements. Seeds that share source test patients evaluate on the
   same S patients. Every repeat is listed in the feasibility report and disclosed.
3. **Patient-internal validation.** It is spatially blocked and overlap-free, but it is not held
   out by patient, so T0 may be optimistic.
4. **Not new hospitals.** The centres are anonymised and probably include the main study's
   hospitals, so this is not a test on new institutions. Patient overlap with CAMELYON16 cannot be
   checked.
5. **Not the WILDS benchmark.**
6. **Crude pair selection.** The RGB-distance rule is descriptive.
7. **ResNet-18 only.** Architecture generalisation is not established.
8. **Different evaluation cap from the main study** (400 vs 60 per class per slide), so absolute
   values are not directly comparable.
9. **Patch-level labels**, not slide- or patient-level diagnosis.
10. **Limited power.** With 5 seeds, Fieller-based R2 has limited power. A non-consistent result
    may reflect imprecision rather than a different mechanism; uninformative and not-consistent
    results are reported with their intervals.
11. **R1 does not show that the true effect exceeds the SESOI.**
12. **Designed after the main study.** The study is exploratory, and its design was chosen after
    the main study's results were known.

## 12. Deviations from the main study (declared in advance) and log of later deviations

**Declared now**

| Item | Main study | This replication |
|---|---|---|
| Split unit | slide-level splits on CAMELYON16 (no patient IDs) | patient-level splits |
| Validation | one random `val` role | spatially blocked, overlap-free `val_es` and `val_temp` |
| Evaluation cap | 60 per class per slide | 400 |
| Buffer B / H4 | used | dropped |
| Architectures | ResNet-18 and ViT | ResNet-18 only |
| Patch cap | none | 3,000 per class per slide |
| ρ interval | seed-percentile bootstrap, which the simulation shows under-covers with 5 seeds | Fieller interval; the bootstrap is reported only as a descriptive comparison. For comparability, the main study's ρ is also re-reported with a Fieller interval, labelled as a post-hoc sensitivity analysis. |
| Hospital pair | RUMC/UMCU | pairs chosen by the RGB rule |
| Frozen-BN control | added after the results (Addenda 4 and 4b) | planned in advance |

**Later deviations:** none yet. Each entry will give the date, the change and the reason.
