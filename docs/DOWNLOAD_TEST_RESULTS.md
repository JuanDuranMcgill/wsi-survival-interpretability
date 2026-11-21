# Download speed test, MEDomics-Beast (2026-11-21)

Question: how fast can the ~8.8 TB of TCGA diagnostic slides (the 22 cancer types with at least
50 PFI events, 8,562 patients) reach the Beast, and from where? No GPU work and no tiling.

## Setup

- About 30 GB per setting, three disjoint random slide lists (87, 88 and 95 slides of 100 to
  600 MB, so that 64 streams can all stay busy). Slides over 600 MB were not in the test.
- One stream = one slide = one HTTP connection. A thread starts the next slide the moment it
  finishes one (no buffer cap, unlike the pilot). All streams were busy for most of each run.
- MD5 computed while downloading and checked against the GDC `md5sum` (GDC) or the S3 ETag
  (IDC, single-part files only). A dropped connection is resumed with an HTTP Range request.
- Sources: GDC; NCI Imaging Data Commons (IDC) on its AWS bucket and its Google (GCS) bucket,
  both public with no login. The IDC runs fetched the same slide lists as the GDC runs.
  IDC holds all 8,562 slides (matched on the slide barcode); its copy is 9.15 TB against
  8.83 TB from GDC.
- Slides were deleted after each measurement. One run per setting, so no repeat variance.

## Results

| Source | Streams | Total MB/s | Total MB/s, all streams busy | Median MB/s per stream | Slides with a drop | MD5 |
|---|---|---|---|---|---|---|
| GDC | 16 | 20.1 | 21.7 | 1.37 | 8 of 87 | 87/87 ok |
| GDC | 32 | 25.3 | 29.3 | 0.94 | 9 of 88 | 88/88 ok |
| GDC | 64 | 31.9 | 37.5 | 0.62 | 0 of 95 | 95/95 ok |
| IDC, AWS | 16 | 50.9 | 53.4 | 3.36 | 0 | 317/317 ok |
| IDC, AWS | 32 | 64.7 | 71.1 | 2.29 | 0 | 320/320 ok |
| IDC, AWS | 64 | 72.7 | 82.4 | 1.38 | 0 | 359/359 ok |
| IDC, GCS | 32 | 111.8 | 112.2 | 3.61 | 0 | 320/320 ok |
| IDC, GCS | 64 | 110.8 | 109.8 | 1.85 | 0 | 359/359 ok |

(IDC "MD5" counts files; a few files use multipart ETags and were checked by size only.
GCS was not run at 16 streams.)

What the numbers say:

- **GDC does not scale with streams.** Doubling from 16 to 32 streams gave +26% total, and
  64 streams gave +59% over 16. Each stream gets slower as streams are added, so GDC is
  limiting the total, at roughly 30 to 40 MB/s.
- **The Beast's network link caps IDC on GCS.** It reached 111 MB/s at both 32 and 64
  streams. The active port, `enp211s0f1np1`, is 1 Gb/s (about 118 MB/s in practice). A second
  port, `enp211s0f0np0`, is down. I did not touch any network setting.
- **Drops on GDC:** about 1 slide in 10 dropped mid-transfer at 16 and 32 streams, none at 64
  (one run each, so this is not a trend). Every drop resumed with a Range request and passed
  MD5, so no slide was lost. None on IDC.
- **IDC slides are not all readable by openslide 4.0.1.** From a sample of 30 (10 from each
  format group in the included set):
  - RGB, JPEG baseline (5,857 slides): 10 of 10 open.
  - RGB + YBR_ICT, JPEG 2000 (390 slides): 10 of 10 open.
  - RGB + YBR_FULL_422, JPEG 2000 (**2,290 slides, 26.7%, 1.28 TB**): **0 of 10** open
    ("Unsupported photometric interpretation YBR_FULL_422").
  - Every slide that opened was identical to the GDC SVS: same level-0 dimensions and same
    pixels in 3 regions of 512 x 512 px (16 of 16 slide and source pairs checked).
  - Another DICOM reader may handle the unreadable group; I did not test one.

## Time for the whole 22-type set (8.83 TB, arithmetic from the rates above)

| Route | Time |
|---|---|
| GDC, 16 streams | 5.1 days |
| GDC, 64 streams | 3.2 days |
| IDC AWS, 64 streams | 34 h |
| IDC GCS (IDC's 9.15 TB), at the link limit | 23 h, but 27% of slides unreadable |
| Hybrid: IDC GCS for the readable 73% plus GDC for the other 2,290 slides | 27 to 31 h |

## Recommendation for the ~8.8 TB

**Use GDC with 16 to 32 streams, resume on drop, MD5 check, and a cap on slides waiting on disk.**

- Download is no longer the bottleneck. The pilot's GPU rate is 24 slides per hour, which
  needs only 5.7 MB/s at 0.86 GB per slide. GDC at 16 streams delivers 20 MB/s (about 84
  slides per hour), more than three times that, and 64 streams about 134 per hour. The 8.8 TB
  will arrive faster than the GPU can use it, so the downloader has to wait for the GPU, not
  the other way round (cap the buffer at about 100 to 150 slides, 100 to 130 GB).
- GDC gives the same SVS files as the pilot, with no format risk, and its MD5 is checked.
- Use IDC on GCS only for a quick bulk head start, or if processing becomes more than about
  three times faster than the pilot (above about 80 slides per hour). Even then, the 2,290
  YBR_FULL_422 slides must come from GDC unless another DICOM reader is shown to work, and
  the IDC route needs the series folder and DICOM support in the pipeline.
- If the second network port on the Beast is a faster link and can be connected, the IDC
  ceiling would rise above 111 MB/s (not tested). It does not matter for the GDC route.

## Limits

- Test slides were 100 to 600 MB, so per-stream speed for the largest slides (up to 2.8 GB)
  was not measured. One run per setting, in one evening (23:12 to 00:50), so time-of-day and
  server load effects are not separated from the stream count.
- The numbers hold for this machine's link to GDC and IDC. The clusters (Narval, Trillium)
  were not tested; their links may differ.
- The slide lists and per-slide records are patient-level and are not in git. Scripts are in
  `/home/users/juan/dltest/` on the Beast (not in the repo).
