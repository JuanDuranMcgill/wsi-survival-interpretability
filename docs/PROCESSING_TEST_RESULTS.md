# Processing test: single read, CONCH embeddings, float16 (2026-11-21)

Question: can the per-slide processing be made faster and smaller than the pilot
(`docs/PILOT_BEAST_RESULTS.md`: 150 s and 291 MB per slide), and can the CONCH tissue labels
be recomputed later without the slide? No training; the full cohort was not processed.

## What was run

- **Slides:** 20 TCGA diagnostic slides, 2 from each of 10 cancer types other than LUAD and
  HNSC (OV, ESCA, THCA, BLCA, LIHC, GBM, COAD, LUSC, BRCA, STAD), picked at random from the
  manifest's included types, slides up to 2.5 GB. 762,282 tiles, mean 38,114 per slide.
  Downloaded from GDC (SVS, as in the pilot), deleted after processing.
- **Same as the pilot:** 256 px tiles at level 0, Otsu mask on a 1/32 thumbnail, 0.8 threshold,
  CONCH `conch_ViT-B-16` and UNI-2h, their pilot transforms, autocast fp16, the same breast
  class prompts (so the CONCH labels are **not meaningful** outside BRCA; timing is unaffected).
- **Changed:**
  1. **One read per tile for both models.** Each tile is read and decoded once, and both
     transforms (CONCH 448 px, UNI-2 224 px) are applied to that image. The pilot read and
     decoded every tile twice.
  2. **DataLoader worker processes** instead of a thread pool (no GIL contention), prefetch 4.
  3. **Inputs sent as float16.** Under autocast the first layer casts to fp16 anyway; checked
     below.
  4. **CONCH image embeddings saved** (512-dim, L2-normalised, the `img_emb` of the repo
     script) next to UNI-2 (1536-dim), plus the 9 class probabilities, tile coordinates, all
     in one float16 `.pt` per slide.
- **Worker and batch sweep** on one slide (12,000 tiles each, one run per setting):

  | Workers | Batch | Tiles/s | GPU utilisation |
  |---|---|---|---|
  | 16 | 256 | **336** | 93% |
  | 24 | 384 | 312 | 89% |
  | 32 | 256 | 317 | 88% |
  | 32 | 512 | 294 | 83% |
  | 48 | 512 | 300 | 85% |

  More workers and larger batches did **not** help (the spread of about 10% is also the size
  of one-run noise). 16 workers and batch 256 were used for all 20 slides.
- GPU: the RTX PRO 6000 Blackwell, no other process on it. Peak GPU memory 15.7 GB.

## Speed against the pilot

| | Pilot (98 LUAD and HNSC slides) | This test (20 slides, 10 types) |
|---|---|---|
| Throughput, both models together | 291 tiles/s (CONCH 678, UNI-2 509, in two passes) | **346 tiles/s** (one pass) |
| Compute per 1,000 tiles (tiling + CONCH + UNI-2 + save) | 3.52 s | **2.96 s (1.19 times faster)** |
| GPU utilisation | CONCH 59%, UNI-2 89% | **97.7%** on slides with at least 20,000 tiles (92% mean, small slides lower) |
| CPU utilisation (share of 48 threads) | not recorded | about 5% |
| Per slide, mean | 150 s (42,595 tiles) | 113 s (38,114 tiles); median 103 s, range 8 to 286 s |
| Tiling per slide | 2.9 s, done once (the repo scripts do it twice) | 2.9 s mean (0.5 to 7.8 s), done once |
| Save per slide | not timed | 0.04 s |

Slides differ between the two (other cancers), so compare per tile, not per slide. I did not
separate how much of the 19% came from the single read, the worker processes, and the
removal of CONCH's idle GPU time; the CONCH idle time alone would account for roughly that
much, since CONCH was about 42% of GPU time at 59% busy.

**The GPU is now the limit** (98% busy, CPU nearly idle), so further speed has to come from
less GPU work: for example fewer tiles, `torch.compile`, or a smaller input for CONCH. I did
not test any of those, and a smaller CONCH input would change the labels.

By cancer type (2 slides each, so only indicative): OV, THCA, BLCA, LIHC, LUSC and BRCA run
at 345 to 351 tiles/s; ESCA, STAD and COAD at 334 to 336; GBM at 306 (two small slides,
startup cost).

## Size

| | Pilot | This test |
|---|---|---|
| Output per tile | 6.84 KB (UNI-2 float32 6.15 KB + CONCH labels 0.68 KB) | **4.12 KB** (0.60 times) |
| Output per slide, mean | 291 MB | **157 MB** (median 144 MB, range 6 to 409 MB) |
| What it contains | UNI-2 embeddings, CONCH class scores | UNI-2 and CONCH embeddings (float16), class probabilities, coordinates |

Both models' outputs come out of the model as float32 and were cast to float16 for saving.

## Float16 against float32

Over all 762,282 tiles, comparing each embedding with its float32 original:

| | UNI-2 | CONCH |
|---|---|---|
| Max absolute difference | 0.00098 | 0.000061 |
| Minimum cosine similarity | 0.9999998 | 0.9999998 |
| Mean cosine similarity | 0.99999998 | 1.0000000 |

**Labels recomputed from the saved float16 CONCH embedding agree with the labels computed
from float32 for 99.91% of tiles** (the other 0.09% are near-ties). So the tissue labels can
be recomputed later, including with other prompts, from the saved embeddings without the
slide. That needs the CONCH text encoder, which is cheap.

Two further checks on 1,024 tiles of the first slide:

- Sending float16 inputs gives exactly the same embeddings as the pilot's float32 inputs under
  autocast (max absolute difference 0.0, label agreement 100%).
- Autocast fp16 compute against pure float32 compute (TF32 off): UNI-2 max absolute difference
  0.0088, CONCH 0.00023, minimum cosine 0.99998, label agreement 99.90%. This is larger than the
  storage cast above, so the pilot's fp16 compute, not the fp16 file format, is the main
  source of difference from full float32.

## Projection for the 22-type cohort

Per-tile cost 2.96 s per 1,000 tiles (tiling, both models, saving), one GPU, at the pilot's
mean of 42,595 tiles per slide (this test's mean, 38,114, gives 10% less):

| Slides | GPU time | Output to keep |
|---|---|---|
| 5,000 | 175 h (7.3 days) | 0.88 TB |
| **8,562 (the 22 types)** | **300 h (12.5 days)** | **1.50 TB** |
| 9,750 | 342 h (14.3 days) | 1.71 TB |

The pilot projected 208 h, 357 to 406 h and 406 h for the same three cases, and 2.5 to 2.85
TB for the 22 types, which was tight against the free space on `/data/processed`; 1.5 TB fits.
At 113 s per slide the pipeline consumes about 32 slides per hour, 7.5 MB/s of slide data, well
within one GDC download stream set (`docs/DOWNLOAD_TEST_RESULTS.md`).

## Not done, and limits

- **Pathomic features** still not timed (scripts not in the repo).
- 20 slides, one run, one worker sweep on one slide. Per-slide throughput was stable (about
  340 to 352 tiles/s for slides with more than 20,000 tiles), so I expect the compute figures to
  repeat; the sweep ranking between settings is within noise.
- I did not run the pilot code on these 20 slides, so the pilot comparison is per tile across
  different slides, not a paired comparison.
- The saved files are patient-level and are not in git: `/data/processed/juan/proctest/`
  (20 files, 3.1 GB). Scripts are in `/home/users/juan/dltest/` on the Beast.
- **A trap for reuse:** forked DataLoader workers hang if their init function calls
  `cv2.setNumThreads` after the parent process has used OpenCV threads. I removed that call.
