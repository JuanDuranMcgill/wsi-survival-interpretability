# Pan-carcinoma preprocessing run: status (updated 2026-11-22 00:24)

Written by `preprocessing/beast_runner/pancan_runner.py`. Cohort: `in_cohort == True`, 17 carcinoma types, 6,940 slides. Order: `/data/processed/juan/pancan/manifest_order.csv` (stratified by type, seed 0).

## Progress

- **Done: 13 of 6,940 slides (0.2%).** downloaded 15, downloading 24, pending 6887, processing 1
- **Speed, last 13 slides:** 335 tiles/s (2.98 s per 1,000 tiles, tiling + both models, pauses excluded); the processing test gave 346 tiles/s and 2.96 s per 1,000 tiles. All 13 done: 335 tiles/s, mean 22,314 tiles per slide, 67 s compute per slide.
- **Wall rate:** 39.0 slides per hour since the first slide (includes pauses and waits).
- **ETA:** 269.0 h of GPU compute left (by raw size, 7.14 TB to go); 177.8 h at the observed wall rate. Finish about 2026-11-29 10:11.
- **Downloads:** 29 slides, 17 GB, mean 1.05 MB/s per stream, 5 dropped connections (all resumed), MD5 checked on every file. Waiting in the buffer: 15 (cap 150).
- **GPU pauses (another user's process):** 0.0 h in total.

## Disk free

- /data/storage 6.50 TB (raw slides go here until 300 GB free), /data/scratch 3.88 TB, /data/processed 2.72 TB, /home/users 3.63 TB
- Raw slides: inbox 16, none 24, storage 13

## Slides done per type

| Type | Slides | Done | Failed | Raw GB (done / all) | Output GB |
|---|---|---|---|---|---|
| BLCA | 383 | 1 | 0 | 0 / 602 | 0.0 |
| BRCA | 1055 | 2 | 0 | 1 / 1078 | 0.1 |
| CESC | 269 | 1 | 0 | 1 / 225 | 0.1 |
| COAD | 449 | 1 | 0 | 0 / 272 | 0.0 |
| ESCA | 156 | 0 | 0 | 0 / 184 | 0.0 |
| HNSC | 449 | 1 | 0 | 1 / 427 | 0.2 |
| KIRC | 511 | 0 | 0 | 0 / 467 | 0.0 |
| KIRP | 273 | 1 | 0 | 0 / 277 | 0.1 |
| LIHC | 363 | 1 | 0 | 0 / 477 | 0.1 |
| LUAD | 469 | 1 | 0 | 0 / 350 | 0.0 |
| LUSC | 473 | 1 | 0 | 1 / 374 | 0.1 |
| OV | 84 | 0 | 0 | 0 / 169 | 0.0 |
| PAAD | 183 | 0 | 0 | 0 / 214 | 0.0 |
| PRAD | 403 | 1 | 0 | 1 / 367 | 0.1 |
| STAD | 412 | 1 | 0 | 0 / 317 | 0.2 |
| THCA | 506 | 0 | 0 | 0 / 590 | 0.0 |
| UCEC | 502 | 1 | 0 | 1 / 755 | 0.1 |

## Errors

None recorded.


## Choices made unattended

- Downloads land on the fast scratch disk (`/data/scratch/juan/pancan_slides/_inbox`), where the tile loaders read them, and move to `/data/storage/juan/pancan_slides/<type>/` after processing. `/data/storage` is not read during the run, as the storage rule requires. When `/data/storage` has under 300 GB free the slide stays in `/data/scratch/juan/pancan_slides/<type>/`; the ledger column `raw_loc` says where each slide is.
- `/data/storage` had 6.5 TB free at the start (df shows 6.0 TiB) against 7.15 TB of slides, and 300 GB is kept free, so about 0.9 TB of slides will stay on scratch.
- 24 download streams, buffer cap 150, 16 loader workers, batch 256 (the processing test's best setting).
- Each output file holds `coords` (int32, level 0), `uni2` (1536), `conch` (512, L2-normalised), `conch_probs` (9 breast prompts, as in the test), `conch_probs8` (the 8 shared prompts of `analysis/pancan_tissue_check.py`), all float16, and `meta`. The probabilities are only for comparison: labels are recomputed from `conch`.
- A slide is retried up to 5 times (download) or 3 times (processing), then marked failed and skipped.
- The ledger is `/data/processed/juan/pancan/ledger.sqlite`; logs are in `/data/processed/juan/pancan/logs/`.
