# Pan-carcinoma preprocessing run: status (updated 2026-11-22 00:04)

Written by `preprocessing/beast_runner/pancan_runner.py`. Cohort: `in_cohort == True`, 17 carcinoma types, 6,940 slides. Order: `/data/processed/juan/pancan/manifest_order.csv` (stratified by type, seed 0).

## Progress

- **Done: 0 of 6,940 slides (0.0%).** pending 6940
- **GPU pauses (another user's process):** 0.0 h in total.

## Disk free

- /data/storage 6.51 TB (raw slides go here until 300 GB free), /data/scratch 3.91 TB, /data/processed 2.72 TB, /home/users 3.63 TB
- Raw slides: 

## Slides done per type

| Type | Slides | Done | Failed | Raw GB (done / all) | Output GB |
|---|---|---|---|---|---|
| BLCA | 383 | 0 | 0 | 0 / 602 | 0.0 |
| BRCA | 1055 | 0 | 0 | 0 / 1078 | 0.0 |
| CESC | 269 | 0 | 0 | 0 / 225 | 0.0 |
| COAD | 449 | 0 | 0 | 0 / 272 | 0.0 |
| ESCA | 156 | 0 | 0 | 0 / 184 | 0.0 |
| HNSC | 449 | 0 | 0 | 0 / 427 | 0.0 |
| KIRC | 511 | 0 | 0 | 0 / 467 | 0.0 |
| KIRP | 273 | 0 | 0 | 0 / 277 | 0.0 |
| LIHC | 363 | 0 | 0 | 0 / 477 | 0.0 |
| LUAD | 469 | 0 | 0 | 0 / 350 | 0.0 |
| LUSC | 473 | 0 | 0 | 0 / 374 | 0.0 |
| OV | 84 | 0 | 0 | 0 / 169 | 0.0 |
| PAAD | 183 | 0 | 0 | 0 / 214 | 0.0 |
| PRAD | 403 | 0 | 0 | 0 / 367 | 0.0 |
| STAD | 412 | 0 | 0 | 0 / 317 | 0.0 |
| THCA | 506 | 0 | 0 | 0 / 590 | 0.0 |
| UCEC | 502 | 0 | 0 | 0 / 755 | 0.0 |

## Errors

None recorded.


## Choices made unattended

- Downloads land on the fast scratch disk (`/data/scratch/juan/pancan_slides/_inbox`), where the tile loaders read them, and move to `/data/storage/juan/pancan_slides/<type>/` after processing. `/data/storage` is not read during the run, as the storage rule requires. When `/data/storage` has under 300 GB free the slide stays in `/data/scratch/juan/pancan_slides/<type>/`; the ledger column `raw_loc` says where each slide is.
- `/data/storage` had 6.0 TB free at the start (not 6.6 TB) against 7.15 TB of slides, so about 1.2 TB of slides will stay on scratch.
- 24 download streams, buffer cap 150, 16 loader workers, batch 256 (the processing test's best setting).
- Each output file holds `coords` (int32, level 0), `uni2` (1536), `conch` (512, L2-normalised), `conch_probs` (9 breast prompts, as in the test), `conch_probs8` (the 8 shared prompts of `analysis/pancan_tissue_check.py`), all float16, and `meta`. The probabilities are only for comparison: labels are recomputed from `conch`.
- A slide is retried up to 5 times (download) or 3 times (processing), then marked failed and skipped.
- The ledger is `/data/processed/juan/pancan/ledger.sqlite`; logs are in `/data/processed/juan/pancan/logs/`.
