# Tissue classifier results (2026-11-22)

The IER audit needs to know what kind of tissue each tile of our slides shows, because the
model's weights, check I and the pathomic features are all organised by tissue type. The
pan-carcinoma run saves two embeddings for every tile (CONCH and UNI-2), so tiles can be
labelled later without touching the slides again. Asking CONCH directly with text prompts
did not work (balanced accuracy 0.52 on Kather's colon tiles, `docs/TISSUE_PROMPT_TEST.md`).
We therefore trained a small classifier on public tiles that pathologists had already
labelled, and tested it on hospitals, donors and slides it never saw.

Design: `docs/superpowers/specs/2026-11-22-tissue-classifier-design.md`. Code:
`analysis/tissue/`. Numbers below come from `results/pancan/tissue_classifier_tests.csv`,
`tissue_classifier_summary.json` and `tissue_plausibility_by_type.csv` unless another file is
named. All results are on held-out data at the 64 µm field of our 40x tiles.

## What works now

The classifier reads the UNI-2 embedding of a tile and decides in two steps. First it
decides whether the tile is tumour; if it is not, it decides which non-tumour tissue it is.
Both steps are logistic regressions with temperature scaling. The weights file is
`results/pancan/tissue_classifier_weights.npz`.

Four classes pass the bars fixed before any result. They are tumour (per organ), normal
epithelium, immune and fat. Stroma, muscle, necrosis and vessels do not, so the fixed merge table sends
them to "other", which belongs to no tissue compartment.

**Tumour against normal tissue passes in all 17 organs.** On TCGA-UT tiles from held-out
hospitals, the classifier finds at least 97.3% of tumour tiles in every type (lowest STAD).
At most 0.6% of normal GTEx tissue is called tumour (highest OV). It has not simply learned
that TCGA means tumour: 3.8% of non-tumour BCSS tiles (TCGA breast) are called tumour,
against 0% of GTEx breast.

Recall and precision per test set, final model:

| Class | Test set | Recall | Precision |
|---|---|---|---|
| Tumour | BCSS (TCGA breast) | 0.895 | 0.978 |
| | TIGER (breast) | 0.976 | 0.951 |
| | Brummer (kidney) | 0.964 | 0.964 |
| | Kather 7K (colon) | 0.974 | 0.740 |
| Normal epithelium | Brummer | 0.902 | 0.953 |
| | Kather 7K | 0.987 | 0.929 |
| Immune | BCSS (36 tiles) | 1.000 | 0.800 |
| | Kather 7K | 0.920 | 0.993 |
| | SegPath | 1.000 | 1.000 |
| Fat | Kather 7K | 0.967 | 0.999 |

Tumour fails its own class bar only through Kather's precision of 0.74. Calibration error
is at most 0.037 on every test set.

Pathomic thresholds (probability at which validation precision reaches 0.90, at equal class
prevalence, on BCSS, TIGER and Brummer): tumour 0.092, normal epithelium 0.024, immune 0.016,
other 0.203. The fat threshold (0.003) is not reliable, since these sets hold few fat tiles.

## Our slides

479 finished slides are labelled
(`/data/processed/juan/pancan/tissue_labels/<type>/<slide>.pt`). On average, tumour covers
16% of tiles in PAAD up to 57% in KIRP, and "other" 25% to 75%.

As a loose check, we compared each patient's tumour share with TCGA's own pathologist
estimate of percent tumour nuclei. That estimate comes from the frozen top and bottom sections,
not from our diagnostic slide, so only a weak positive relation is expected. Spearman's
correlation is positive in 12 types, highest in STAD (0.66), BLCA (0.57) and PRAD (0.54). It
is at or below zero in KIRC (−0.11), KIRP (−0.26), PAAD (−0.04) and UCEC (−0.02). OV has
only 6 patients so far. Each type has 6 to 73 patients at this point in the run.

## What did not work

1. **CONCH text prompts.** Balanced accuracy 0.52 with one prompt per class, 0.82 at best
   with prompt ensembles (`docs/TISSUE_PROMPT_TEST.md`).
2. **The same classifier on CONCH embeddings.** With all the data, at 64 µm, only immune
   passes. Tumour recall is 0.73 on Kather and 0.75 on BCSS; stroma recall is 0.54 to 0.87.
3. **Merging classes that fail.** The fixed table sends a failed class to a broader one. With
   CONCH it collapsed nearly everything into "other".
4. **Averaging each tile with its neighbours.** On held-out BCSS this made things worse.
   Stroma recall fell from 0.854 to 0.730 with a 3×3 window, and tumour precision from 0.973
   to 0.915 (`results/pancan/tissue_smoothing_bcss.csv`).
5. **SegPath muscle and vessel masks.** Smooth-muscle actin also stains myofibroblasts, so
   even tiles 80% covered include stroma and tumour-like cells. The endothelium marker covers
   at most about 3% of a tile, so no tile is a vessel. Only SegPath's leukocyte masks were used.
6. **OpenTME.** Its licence forbids using it to train a classifier like ours.

Practical lessons: Zenodo served under 1 MB/s, so SegPath came from a Hugging Face copy and
was checked against Zenodo's MD5 sums. Otsu thresholding inside annotated regions (no glass)
discarded pale stroma, so a fixed near-white cut is used there. SegPath's own split is by
image, so it was re-split by slide.

## Why UNI-2

Same tiles, same classifier, same tests, round 1 before any merge:

| | CONCH | UNI-2 |
|---|---|---|
| Classes passing | immune | normal epithelium, immune, fat |
| Tumour recall (BCSS, TIGER, Brummer, Kather) | 0.75, 0.99, 0.97, 0.73 | 0.90, 0.98, 0.97, 0.98 |
| Stroma recall (same order) | 0.87, 0.56, 0.62, 0.54 | 0.83, 0.88, 0.76, 0.89 |
| Normal epithelium recall (Brummer, Kather) | 0.69, 0.77 | 0.90, 0.99 |

Juan chose UNI-2 on 2026-11-22 after seeing both on these test sets, which the paper must
say. The SegPath immune tiles were added after that choice, so they are a fresh check:
UNI-2 recall 1.000, CONCH 0.934.

## Training data

| Source | What it labels | Tiles at 64 µm | Licence |
|---|---|---|---|
| TCGA-UT (Komura) | tumour, 17 types | 174,910 | CC BY-NC-SA 4.0 |
| GTEx (IDC), 320 slides | normal tissue, 16 organs | 89,486 | CC BY 4.0 |
| Kather NCT-CRC-HE 100K / 7K | colon, 9 classes | 100,000 / 7,180 | CC BY 4.0 |
| BCSS | TCGA breast regions | 15,673 | CC0 |
| TIGER | breast regions | 4,543 | CC BY-NC 4.0 |
| Brummer ccRCC | kidney | 28,165 | CC BY 4.0 |
| SegPath CD45RB | immune, 18 organs | 1,563 | CC BY-NC-SA 4.0 |

Licences are as listed on each source's Hugging Face, Zenodo or IDC record. TCGA hospitals keep one split across TCGA-UT, BCSS, TIGER and Brummer. GTEx is split by
donor, SegPath by slide, and the non-TCGA slides by slide.

## Open

- **Stroma** misses its bar narrowly with UNI-2 (recall 0.76 to 0.89, precision 0.64 to
  0.98), so by the fixed rule it sits in "other". One option is to keep that as the main
  analysis and also run the audit with stroma as its own compartment, reporting its accuracy.
  Juan to decide.
- Labels are re-run on all 6,940 slides when the run ends (about 2026-12-03), and the
  plausibility check is repeated then with all patients.
- Tile grids from our own slides wait until the run ends, since slide storage is not read
  during the run.
