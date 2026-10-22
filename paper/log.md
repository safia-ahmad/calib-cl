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
