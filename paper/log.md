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
