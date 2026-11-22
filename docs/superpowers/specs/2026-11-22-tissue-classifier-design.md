# Design: tissue labels from a trained classifier (pan-carcinoma run)

Status: approved by Juan (2026-11-22), including the pass bars as proposed and sharing the
GPU with the running preprocessing job to embed the public tiles. Part of the pan-cancer extension
(`2026-11-20-pancancer-audit-extension-design.md`, decision 2: tissue types). Replaces the
zero-shot CONCH labels for the 17 carcinoma types.

## Intent and why

Our model weights tissue types, check I compares those weights with Shapley values over
tissue types, and the pathomic features are computed per tissue type. All of this is only
as meaningful as the tissue labels. The goal of the extension is a model that passes the
audit; a pass on wrong tissue labels would mean nothing to a pathologist.

The tissue prompt test (2026-11-22, `docs/TISSUE_PROMPT_TEST.md`) showed that our current
labels, one CONCH text prompt per class, are right only about half the time (balanced
accuracy 0.52 on Kather's labelled colon tiles). A classifier trained on the same CONCH
image embeddings reached 0.98. CONCH's image embedding holds the tissue information; the
text matching is the weak part. No pathologist can label our own tiles (Juan, 2026-11-22),
so training labels and tests come from public data labelled by pathologists, plus normal
tissue from donors without cancer.

## Goals

1. One tissue classifier that takes a tile's saved CONCH embedding (512 numbers) and
   returns one of the shared tissue classes, for all 17 carcinoma types.
2. A measured accuracy for every class we keep, on patients or donors the classifier never
   trained on, reported in the paper. For tumour, measured in every one of the 17 organs.
3. Calibrated probabilities, so the pathomic step's confidence threshold means something.
4. Labels for all 6,940 run slides, recomputed from the saved embeddings. No slide is
   reprocessed.

## Non-goals

- Pixel-level segmentation or cell-level labels.
- Labels for non-carcinoma cancer types.
- Beating published tissue classifiers.

## Anti-goals (failure even if it works)

- A classifier that recognises the organ, the fixative or the data source instead of the
  tissue (for example "looks like lung, so tumour").
- Choosing classes, thresholds or pass bars after seeing test results or survival results.
- Reporting accuracy on tiles from patients or donors the classifier trained on.
- Patient-level data in git. Public tiles, their embeddings and our labels stay on the
  Beast; only code, the small classifier weights and aggregate results go to git.
- Disturbing the running preprocessing job (see Constraints).

## Constraints (measured)

- **Our tiles:** 256 px at level 0. Of the first 251 finished slides, 228 are 40x
  (0.25 µm/px, a 64 µm field), 20 are 20x (0.5 µm/px, 128 µm) and 3 have no
  magnification recorded (ledger column `app_mag`, 2026-11-22).
- **Scale of public tiles:** mostly 0.5 µm/px. In the prompt test, zero-shot labels lost
  0.16 to 0.21 balanced accuracy on a 64 µm crop, but the trained classifier lost only
  0.04 (0.98 to 0.95). Scale matters less for a trained classifier, but it still has to be
  handled.
- **Download speed:** Zenodo gave 0.11 MB/s on the Beast; IDC gave 111 MB/s
  (`docs/DOWNLOAD_TEST_RESULTS.md`). Openslide could not read 27% of IDC slides (YBR_FULL_422
  DICOM), so IDC slides need a DICOM reader that handles them.
- **GPU:** the preprocessing run uses it until about 2026-12-03. Juan forbids touching the
  run. Sharing the GPU for a short embedding job needs Juan's OK.
- **Licences:** TCGA-UT and SegPath are CC-BY-NC-SA 4.0 (non-commercial), Kather CC-BY 4.0,
  BCSS CC0; GTEx asks for an acknowledgement of the GTEx Consortium. All allow academic
  research; each is cited.

## Approaches considered

- **A. Classifier trained on public labelled tiles (chosen by Juan, 2026-11-22).** Labels
  from pathologists, a measured accuracy per class. Costs downloads, a short GPU job and
  care with scale and with shortcuts. Weak spot: normal epithelium has tile-level labels
  only for colon and breast.
- **B. Better prompts only.** Averaged CONCH prompts with cancer-specific names for each of
  the 17 types. No training data, but 0.82 on colon was its best case, and outside colon
  only the tumour class could be checked.
- **C. A pathologist labels our own tiles.** The best test, at our exact scale. Dropped: no
  pathologist available.

## Data

| Source | Labels | Organs | Scale | Size | Use |
|---|---|---|---|---|---|
| TCGA-UT (Komura 2022, Zenodo 5889558) | tumour regions drawn by two pathologists on TCGA diagnostic slides; files named by TCGA barcode | 32 TCGA types, all 17 of ours | 256 px at 0.5 to 1.0 µm/px (6 levels) | 22.4 GB for our 17 types | tumour, every organ; the 0.5 µm/px level only |
| GTEx (IDC collection `gtex`, open) | whole slides of normal tissue from donors without cancer; every tile is "not tumour" | about 50 tissues, 971 donors, 25,503 slides; covers all 17 organs (head and neck through proxies, below) | to check on download | a sample only | non-tumour, every organ |
| Kather NCT-CRC-HE-100K-NONORM (Zenodo 1214456) | 9 classes: tumour, normal mucosa, stroma, lymphocytes, debris, fat, smooth muscle, mucus, background | colon (86 slides, 2 German centres) | 224 px at 0.5 µm/px | 11.7 GB | every class but vessels |
| Kather CRC-VAL-HE-7K (same record, already on the Beast) | same 9 classes, other patients | colon (50 patients) | same | 0.8 GB | colon test set |
| BCSS (Amgad 2019) | region masks: tumour, stroma, lymphocytic infiltrate, necrosis or debris, fat, blood vessel, blood, normal acinus or duct (to verify on download), others | breast, TCGA, 151 regions | 0.25 µm/px (to verify) | small | every class, including vessels, at our exact 40x scale; also non-tumour tiles from TCGA slides |

GTEx organs used, one per cancer type: bladder (BLCA), breast (BRCA), cervix (CESC), colon
(COAD), oesophagus mucosa (ESCA), kidney cortex (KIRC, KIRP), liver (LIHC), lung (LUAD,
LUSC), ovary (OV), pancreas (PAAD), prostate (PRAD), stomach (STAD), thyroid (THCA),
uterus (UCEC). GTEx has no oral or laryngeal mucosa; for HNSC, oesophagus mucosa (also a
squamous lining) and minor salivary gland stand in, and the paper says so. GTEx tissue is
fixed in PAXgene, not formalin, and comes from post-mortem donors, so it looks somewhat
different from TCGA tissue; see the shortcut tests below.

Second wave, only if a class fails its bar: SegPath (Komura 2023; masks from
immunofluorescence for epithelium, smooth muscle and myofibroblasts, endothelium and
leukocytes; 50 to 60 GB per marker) and TIGER (breast, Zenodo 6014420, 2.6 GB).

**Considered and not used in the first version: OpenTME** (Aignostics, arXiv 2604.12075;
3,634 TCGA slides of 5 types). Its tissue labels come from a company's model, not from
pathologists, access is gated (Juan would request it himself), and it is not yet clear
whether it releases tissue maps or only per-slide numbers. Possible later use: compare our
per-slide tissue shares with theirs on the same TCGA slides.

TCGA-UT and BCSS tiles come from TCGA patients, many of them in our cohort. Tissue labels
carry no outcome information, and the classifier is fixed before any survival model is
trained, so this is not outcome leakage. The paper says so.

## Classes

The 8 shared classes: tumour, normal epithelium, stroma, immune, necrosis or haemorrhage,
fat, vessels, muscle. Plus one reject class, **other** (Kather background and mucus; BCSS
classes that fit none of the 8), so that such tiles are not forced into a tissue class (in
the prompt test, mucus went to fat, necrosis and muscle). Tiles labelled other belong to
no tissue compartment.

Mapping: Kather TUM, NORM, STR, LYM, DEB, ADI, MUS map to tumour, normal epithelium,
stroma, immune, necrosis, fat, muscle; BACK and MUC map to other. BCSS tumour, stroma,
lymphocytic infiltrate, necrosis or debris, blood, fat, blood vessel and normal acinus or
duct map to tumour, stroma, immune, necrosis or haemorrhage, necrosis or haemorrhage, fat,
vessels and normal epithelium. The builder maps BCSS's remaining classes to other or drops
them, and lists the mapping. TCGA-UT maps to tumour; GTEx to not tumour.

## How it works

1. **Download** to `/data/scratch/juan/datasets/`: the 17 TCGA-UT zips, Kather NONORM,
   BCSS and a GTEx sample (about 20 donors per organ, one slide each; the builder sets the
   tile count per slide). Look for faster mirrors first (TCGA-UT and Kather are on Hugging
   Face). Check MD5 sums where given.
2. **Match our scale.** For each 0.5 µm/px tile keep two versions: the whole tile (close to
   our 20x tiles) and its central 128 px (our 40x field of 64 µm, at half the detail).
   From BCSS, cut 256 px tiles at 0.25 µm/px, labelled when one class covers at least 80%
   of the tile. GTEx tiles are cut from tissue (same Otsu mask as the run) at the 64 µm and
   128 µm fields.
3. **Embed** with exactly the run's CONCH code (`preprocessing/beast_runner/pancan_runner.py`:
   same transform, float16 input, autocast, L2-normalised embedding). Also UNI-2, for a
   comparison that is reported only; CONCH is the classifier. About 1.5 million tile
   passes; at the run's measured 346 tiles per second for both models, about 1 to 2 hours
   on the GPU.
4. **Classifier: two steps,** each a multinomial logistic regression on the 512 CONCH
   numbers, with class-balanced weights.
   - Step 1, tumour or not: tumour tiles from TCGA-UT, Kather and BCSS; non-tumour tiles
     from GTEx (all 17 organs), Kather and BCSS. Because every organ has both tumour and
     non-tumour examples, "this organ means tumour" is not a usable shortcut.
   - Step 2, which non-tumour tissue: normal epithelium, stroma, immune, necrosis, fat,
     vessels, muscle or other, trained on Kather and BCSS (GTEx tiles have no tissue
     label).
   - A small two-layer network replaces either step only if logistic regression misses
     the bars.
   - The penalty, temperature scaling (calibration) and the per-class thresholds are fit
     on validation sites and donors (TCGA-UT, BCSS, GTEx), never on a random split of
     Kather 100K, which has no patient IDs.
5. **Splits** (no test patient or donor ever in training):
   - TCGA-UT and BCSS: whole tissue-source sites (TSS code in the barcode) go to training,
     validation or test, stratified by type, as the audit's folds do.
   - GTEx: whole donors go to training, validation or test.
   - Kather: train on 100K, test on CRC-VAL-HE-7K (other patients).
6. **Tests**, all on held-out data and on the 64 µm crop (our 40x tiles), with whole-tile
   results reported alongside (our 20x tiles):
   - Per class, recall and precision on every test set that labels that class.
   - Tumour recall in each of the 17 types, on TCGA-UT held-out sites.
   - Tumour false-positive rate in each of the 17 organs: the share of held-out GTEx tiles
     called tumour.
   - Source shortcut: the share of BCSS held-out non-tumour tiles (TCGA, formalin) called
     tumour, against the same share on GTEx breast. If the classifier had learned "TCGA
     means tumour", BCSS non-tumour tiles would be called tumour far more often.
   - On TCGA-UT tumour tiles, the share given each non-tumour class, per type. Some
     stroma is expected, since pathologists' tumour regions include stroma between tumour
     cells.
   - Calibration: expected calibration error before and after temperature scaling.
7. **Plausibility on our own slides** (no ground truth, reported, not pass or fail):
   - Per type, the share of tiles per class, against TCGA's own pathologist estimates for
     the frozen top and bottom sections of the same tumours (GDC biospecimen fields percent
     tumour nuclei, percent necrosis, percent stromal cells, percent lymphocyte
     infiltration; diagnostic slides have none, checked 2026-11-22). Spearman correlation
     across patients within each type. These are different sections, so the expected
     relation is loose but positive; a type with zero or negative correlation for tumour is
     named in the paper.
   - A grid of 20 random tiles per class and type, saved as images to look at.
8. **Label the run's slides** from the saved CONCH embeddings: class and calibrated
   probabilities per tile, float16, in
   `/data/processed/juan/pancan/tissue_labels/<type>/<slide>.pt`. Fast on CPU.

## Pass bars (fixed before any result; Claude's proposal, Juan to check)

All bars are on the 64 µm crop and held-out data.

- A class is kept if, on every test set that labels it, recall is at least 0.85 and
  precision at least 0.80.
- Tumour, per organ: recall at least 0.85 on TCGA-UT held-out sites, and at most 10% of
  held-out GTEx tiles called tumour. An organ below either bar is named in the paper, not
  dropped.
- Normal epithelium: tested on colon and breast only; kept if it passes there and if
  under 5% of TCGA-UT tumour tiles are called normal epithelium in every type.
- A class that fails is handled by this fixed table, then the classifier is retrained once:
  vessels into stroma; muscle into stroma; immune into stroma; necrosis, fat, normal
  epithelium or stroma into other (their tiles leave every compartment, so the tumour
  compartment stays clean). Immune and stroma were added when the plan was written
  (Claude's call). The result is reported either way.
- Pathomic confidence threshold, one per class: the calibrated probability at which
  validation precision reaches 0.90, replacing the old 0.7.

## Outputs

- Code: `analysis/tissue/` (download and tiling of public sets, embedding, training,
  tests, labelling of our slides).
- Classifier weights (a few kilobytes, trained on public data only) in git.
- Results: `results/pancan/tissue_classifier_*.csv` and `.json`; write-up
  `docs/TISSUE_CLASSIFIER_RESULTS.md`; handoff updated.
- Downstream: the model, check I and the pathomic step read labels from
  `tissue_labels/`, not from `conch_probs` or `conch_probs8`.

## Calls made by Claude (Juan to check while reading)

- GTEx normal tissue added so that every organ has non-tumour examples (from the builder
  check: without it, the classifier could learn "lung means tumour").
- Two-step classifier (tumour or not, then tissue type), since GTEx tiles are known to be
  non-tumour but have no tissue label.
- Bars apply to the 64 µm crop (91% of our slides are 40x); whole-tile results reported.
- UNI-2 reported only; CONCH is the classifier.
- OpenTME not used in the first version.
- Calibration and thresholds fit on held-out sites and donors, not on Kather.

## Left to the builder

File layout under `analysis/tissue/`; batch sizes; which mirror to use; the DICOM reader
for IDC; tiles per GTEx slide and per TCGA-UT type if one type dominates; the penalty
grid; BCSS classes outside the 8 (to other or dropped).

## Answered by Juan (2026-11-22)

1. The GPU is shared with the running job for about 1 to 2 hours to embed the public
   tiles; the run finishes that much later.
2. The pass bars above are accepted as proposed.
