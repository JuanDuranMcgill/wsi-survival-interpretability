# Timing pilot on the MEDomics-Beast (2026-11-20)

Purpose: measure what the preprocessing of our pipeline really costs on one GPU, so the
GPU budget for the pan-cancer extension (design doc, decisions 10 and 16) rests on a
measurement. No model was trained and no paper text was changed.

## What was run

- **Machine:** MEDomics-Beast, 1x RTX PRO 6000 Blackwell (96 GB), 48 CPU threads, 251 GB RAM.
  PyTorch 2.11 with CUDA 12.8. The GPU was shared in principle but **no other process used
  it during the run** (checked at the start of every slide).
- **Slides:** 98 TCGA diagnostic FFPE slides, 49 LUAD and 49 HNSC, one slide per patient,
  drawn at random (seed 0) from the GDC. 90 are scanned at 40x, 7 at 20x and 1 has no
  magnification recorded.
- **Steps per slide**, in the same order and with the same settings as `preprocessing/`:
  1. Tissue detection and tiling: Otsu mask on a 1/32 thumbnail, 256 px tiles at native
     resolution (level 0), kept if more than 20% of the tile is tissue.
  2. CONCH (`conch_ViT-B-16`) zero-shot tissue labels, batch 160, fp16.
  3. UNI-2h embeddings (1536-dim), batch 128, fp16.
  4. Outputs saved, then the raw slide deleted.
- **Download:** straight from the GDC open-access API, 8 parallel streams, up to 12 slides
  ahead of processing.

### How the pilot scripts differ from `preprocessing/`

The repo scripts hardcode cluster paths and copy each slide to `/dev/shm` or `/tmp`, so I
ran adapted copies (kept outside the repo, in `/home/users/juan/pilot/`). The differences:

- Slides are read directly from the scratch NVMe, with no copy to RAM or `/tmp`.
- Models load once for the whole run (CONCH 2.7 s, UNI-2 3.7 s) instead of once per slide.
  If each slide is its own job, as on the clusters, add about 6.5 s per slide.
- Tiles are fetched with an ordered, bounded prefetch window instead of the repo's
  `as_completed` loop. I did not compare the two, so the repo scripts may be somewhat
  slower or faster than these numbers.
- The mask and tile grid are computed once. The two repo scripts each compute them, so the
  original pipeline pays the tiling step twice (about 3 s more per slide).
- The duplicate debug grid scan in the CONCH script was dropped.

## Results per slide (n = 98)

| Quantity | Median | Range (min to max) | Mean |
|---|---|---|---|
| Slide file size | 768 MB | 12 MB to 2.83 GB | 860 MB |
| Tissue tiles | 38,244 | 1,048 to 109,604 | 42,595 |
| Download time (one stream) | 381 s | 11 s to 1,359 s | 444 s |
| Tiling (mask and grid) | 2.7 s | 0.25 to 8.8 s | 2.9 s |
| CONCH labels | 57 s | 1.8 to 162 s | 63 s |
| UNI-2 embeddings | 75 s | 2.1 to 217 s | 84 s |
| **Compute total (tiling + CONCH + UNI-2)** | **135 s** | **4.8 to 384 s** | **150 s** |
| CONCH output (labels, `.jsonl`) | 26 MB | 0.7 to 75 MB | 29 MB |
| UNI-2 output (embeddings, `.pt`) | 235 MB | 6.4 to 674 MB | 262 MB |
| **Output total per slide** | **261 MB** | **7 to 749 MB** | **291 MB** |

Share of compute time: tiling 2%, CONCH 42%, UNI-2 56%.

By cancer type (mean compute per slide): LUAD 159 s, HNSC 141 s. By scan magnification,
the 90 slides at 40x average 45,152 tiles and 159 s; the 7 slides at 20x average 15,658
tiles and 55 s. Cost is proportional to the number of tiles, about **3.5 s per 1,000 tiles**.

### GPU

| | CONCH | UNI-2 |
|---|---|---|
| Throughput, tiles per second (pooled over all slides) | 678 | 509 |
| Per-slide range, tiles per second | 591 to 692 | 490 to 530 |
| GPU utilisation, mean per slide (median, range) | 59% (31 to 63%) | 89% (50 to 97%) |
| Peak GPU memory | 12.5 GB | 12.4 GB |

The two models together run at about **291 tiles per second**, or **24 slides per hour**.
CONCH leaves the GPU about 40% idle, probably because it waits on tile loading and the
448 px resize on the CPU. I did not test that. Memory use is 13% of the card, so there is
room for larger batches or a second worker, also untested.

## Download from the GDC

- Single stream: median **1.9 MB/s** per slide (range 1.1 to 3.3 MB/s).
- Aggregate: 83 GB in about 4.0 hours of active downloading, **5.8 MB/s**, with an average of
  only 3 streams active. That is below the 8 I allowed: finished downloads waiting behind a
  slow slide filled the 12-slide buffer and blocked new ones. So 5.8 MB/s reflects my
  scheduling, not a limit of the GDC. Eight busy streams would give roughly 15 MB/s if the
  speed scales, which I did not test.
- **8 of the first 99 download attempts (8%) dropped mid-transfer.** A retry recovered 6 of
  them. I stopped before the last 2 finished, because retries do not affect the compute
  estimates. These 2 slides (one LUAD, one HNSC) are not in the 98.
- This measures this machine's link to the GDC only. The clusters may differ.

## Projections

Computed as slides x mean compute per slide (150 s), assuming a mean of about 42,600 tiles
per slide as in this sample. Total time scales linearly with the mean tile count.

| | 5,000 slides | 9,750 slides |
|---|---|---|
| GPU time, tiling + CONCH + UNI-2, one GPU | **208 h (8.7 days)** | **406 h (16.9 days)** |
| Same, using the LUAD mean (159 s) | 221 h | 431 h |
| Same, using the HNSC mean (141 s) | 195 h | 381 h |
| Output to keep (291 MB per slide) | **1.46 TB** | **2.84 TB (2.58 TiB)** |
| Raw slides downloaded and then deleted (860 MB per slide) | 4.3 TB | 8.4 TB |
| Download time at 5.8 MB/s (this run) | 8.7 days | 16.9 days |
| Download time at about 15 MB/s (8 busy streams, untested) | 3.3 days | 6.5 days |

Notes on the projections:

- **Compute is the GPU-bound part, and download can be as slow.** At the speed I achieved,
  download and GPU time are about equal (16.9 days each for 9,750 slides). They can overlap,
  but only if the download keeps up, which needs more parallel streams than I managed.
- **Storage:** `/data/processed` has about 2.5 TiB free. The 9,750-slide output (2.58 TiB)
  does not fit by a small margin. The 5,000-slide output (1.32 TiB) does. The UNI-2
  embeddings are saved as float32 (6.2 KB per tile); saving them as float16 would cut the
  output to about 160 MB per slide (1.56 TB for 9,750 slides). That is arithmetic only; I
  did not test it or check that the model's later use accepts float16.
- Several clusters running jobs in parallel shorten the calendar time but not the GPU-hours.
  Other GPUs will differ from this one.
- The model training runs, the Shapley and audit steps, and anything else in the design doc
  are **not** included.

## What was skipped, and why

- **Pathomic (pyradiomics) features: skipped, not timed, not in the projections.** The
  extraction scripts are not in the repo. `slurm/submit_radiomics.sh` calls `find_pairs.py`
  from a Trillium path, and that file is not here (design doc, decision 14, says to find
  it on Trillium or rebuild it). I do not know how much time it adds, and it may be
  substantial.
- **CONCH classes:** the repo script has nine breast-specific classes. I used them unchanged
  on LUAD and HNSC because timing does not depend on the prompts. The labels from this
  pilot are therefore **not meaningful for lung or head and neck** and should not be used.
  The design doc plans a shared pan-cancer class set.
- **Model loading per slide** and **the second tiling pass** of the repo scripts are
  excluded from the per-slide figures (see above).

## Limits of this pilot

- Only two cancer types, 98 slides. Mean tile count in other cancers may differ, and
  the projection moves with it.
- 90 of the 98 slides are 40x. A cohort with more 20x slides would cost less per slide.
- Single run, no repeats. Per-slide throughput was stable (CONCH 591 to 692 tiles/s,
  UNI-2 490 to 530), so I expect the compute figures to repeat closely; the download
  figures varied much more.
- The first slide was a smoke test run with an earlier version of the script (one download
  stream, same processing code). It is included in the 98; its download time is a
  single-stream measurement and it has no start and end timestamps for the aggregate figure.
- The slide list and outputs are patient-level and are not in git. Outputs are in
  `/data/processed/juan/pilot/` on the Beast (27 GB).
