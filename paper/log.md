   # Project log

   ## Day 1 · Wed 21 Oct
   - Did: created repo, uploaded starter code, ran tests in Colab
   - Result: 17 passed
   - Next: Day 2, download PCam meta CSVs and count slides per split


## Day 2–3 · 22 Oct
- PCam: 216/54/129 slides (train/valid/test), 399 total, 0 shared between splits
- reference.csv (official S3): code 0 = RUMC, code 1 = UMCU
- G1: train 171/99 vs published 170/100 (1 slide, accepted); test 74/55 vs 79/50 (5 slides, unexplained)
  -> decision: use only the 270 CAMELYON16 training slides; test slides excluded
- G2: 1,180 patches (0.36%) without a hospital (normal_086, test_049), dropped
- G3: [paste the supply table]
- Next: Day 4, start the big downloads

## Day 4–5 · 23 Oct
- Compact PCam: 128,761 patches, 269 slides (≤500/slide), 2.9 GB in Drive; labels match
- Supply: RUMC 56,299 normal / 25,352 tumour; UMCU 32,279 / 14,831 (40 tumour slides)
- calcl/data.py added; tests 21 passed; loader speed: [X] patches/s
- Colour shift (mean RGB): [paste table]
- Next: Day 6, slide splits (make_splits)

## Day 6–10: 
The no-jitter model learned a colour shortcut on UMCU (predicted tumour 76% vs 42% true; accuracy on normal patches 0.32).
With colour jitter 0.5: S AUROC 0.961, G AUROC 0.910, T0 = 1.76, G4 gap 0.046 PASS. Decision: colour 0.5 is the standard recipe for all source models (pilot decision, seed 0, not reported).
