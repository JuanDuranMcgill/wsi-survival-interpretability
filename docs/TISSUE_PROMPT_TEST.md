# Tissue prompt test on labelled colorectal tiles (2026-11-22)

Question: how accurate are CONCH zero-shot labels for the 8 shared tissue classes, and do prompt
ensembles or cancer-specific prompts help? The tissue label check of 2026-11-21
(`analysis/pancan_tissue_check.py`) found confident but implausible labels on our slides. That
check had no ground truth; this test has one.

Script: `analysis/pancan_tissue_prompt_test.py`. Results: `results/pancan/tissue_prompt_test.json`.

## Data and method

- **Tiles:** Kather et al., CRC-VAL-HE-7K (Zenodo 1214456, MD5 checked): 7,180 colorectal tiles,
  224 px at 0.5 µm/px (112 µm field), Macenko-normalised, from 50 patients, in 9 labelled classes.
- **Embedding as our pipeline does:** CONCH transform (resize to 448), float16 input, autocast
  fp16, the L2-normalised image embedding.
- **Class mapping:** TUM tumour (1,233 tiles), NORM normal epithelium (741), STR stroma (421),
  LYM immune (634), DEB necrosis (339), ADI fat (1,338), MUS muscle (592). Every tile is classified
  over all 8 of our classes. Vessels has no Kather class, so a tile called vessels counts as wrong.
  BACK (847) and MUC (1,035) have no class of ours and are reported separately.
- **Prompt sets:**
  - (a) the 8 prompts of `analysis/pancan_tissue_check.py`, one per class ("an H&E image of ...").
  - (b) 4 to 7 generic names per class × CONCH's 22 prompt templates (88 to 154 prompts per class),
    text embeddings averaged per class.
  - (c) (b) plus 4 colon-specific names each for tumour (for example "colorectal adenocarcinoma")
    and normal epithelium (for example "normal colon mucosa").
- **Balanced accuracy** = mean of the 7 per-class accuracies (recall).

## Results (full 224 px tile)

| | (a) one prompt | (b) ensemble | (c) ensemble + colon |
|---|---|---|---|
| **Balanced accuracy** | **0.52** | **0.73** | **0.82** |
| Tumour (TUM) | 0.36 | 0.31 | 0.65 |
| Normal epithelium (NORM) | 0.04 | 0.49 | 0.79 |
| Stroma (STR) | 0.59 | 0.77 | 0.77 |
| Immune (LYM) | 0.92 | 0.99 | 0.99 |
| Necrosis (DEB) | 0.50 | 0.95 | 0.95 |
| Fat (ADI) | 0.97 | 0.90 | 0.90 |
| Muscle (MUS) | 0.24 | 0.67 | 0.67 |
| Balanced accuracy without the vessels class | 0.59 | 0.73 | 0.82 |
| Median top probability (mapped tiles) | 0.98 | 0.998 | 0.999 |
| Accuracy of tiles with top probability ≥ 0.7 (share of tiles) | 0.60 (84%) | 0.72 (89%) | 0.83 (93%) |

Where the errors go:

- **(a), our current prompts:** 39% of tumour tiles are called vessels (485 of 1,233). 69% of
  normal mucosa is called fat (511 of 741), and 21% immune. Muscle is split across vessels,
  stroma and fat. This matches the slide check: little tumour, almost no normal epithelium, a lot
  of vessels and muscle.
- **(b):** the vessels confusion disappears, but tumour now goes mostly to stroma (594 of 1,233),
  and normal mucosa to immune (227).
- **(c):** naming the organ fixes most of tumour and normal epithelium. The remaining errors are
  tumour → stroma (15%), normal → immune (15%), muscle → stroma (27%) and fat → muscle (8%).
- **BACK and MUC** (no class of ours): with (c), background goes to necrosis (53%) and muscle
  (26%), and mucus to fat (41%), necrosis (29%) and muscle (20%). Our pipeline keeps only tissue
  tiles, but mucin-rich tumours (for example mucinous colorectal, ovarian or stomach cancers) will
  feed mucus into necrosis, fat and muscle.
- **The 0.7 threshold does not filter errors.** Median top probability is 0.98 or higher with
  every prompt set, and 84 to 93% of tiles pass 0.7, yet only 60 to 83% of those are right.

## Field of view

Our tiles are 256 px at level 0. Most TCGA slides are 40x (0.25 µm/px), so a tile covers 64 µm,
against Kather's 112 µm. To see whether the smaller field matters I also classified the central
128 px of each Kather tile (64 µm), resized by the same transform:

| | (a) | (b) | (c) |
|---|---|---|---|
| Balanced accuracy, central 64 µm | 0.31 | 0.57 | 0.62 |
| Change from the full tile | −0.21 | −0.16 | −0.20 |

The crop also has half the pixel detail of our 40x tiles (0.5 against 0.25 µm/px), so this
overstates the loss. Field of view and resolution are confounded here, and I did not separate
them. It is a warning, not a measurement for our tiles: zero-shot labels may be worse on our
40x tiles than on the full Kather tiles.

## Reference: a classifier trained on the embeddings

A logistic regression on the same CONCH embeddings (5-fold cross-validation over the 7 mapped
classes) reaches balanced accuracy **0.98** on the full tile and **0.95** on the central 64 µm
(stroma is the weakest class, at 0.93 and 0.82). This is optimistic: Kather gives no patient
IDs, so tiles from the same patient can be in both the training and the test folds. Still, the
gap to zero-shot (0.98 against 0.82 at best) shows the embeddings hold the information; the text
prompts are the weak part.

## Recommendation

1. **Do not use the current one-prompt labels** (set a, balanced accuracy 0.52). Their errors
   (tumour → vessels, normal → fat) explain the implausible slide-level shares found on
   2026-11-21.
2. **If labels must stay zero-shot, use ensembles with type-specific tumour and normal-epithelium
   names.** That means writing (c)-style names for each of the 17 types, which reached 0.82 on
   colon. Expect lower accuracy on our 40x tiles, and on types whose look is far from colon. Drop
   or merge the vessels class, which no public set labels and which absorbed errors in (a).
3. **Better: train a small classifier (linear head) on the saved CONCH embeddings using public
   labelled tiles,** and test it on a cohort from another centre. Within Kather it reaches 0.98.
   For colon, train on NCT-CRC-HE-100K (same Zenodo record, 11.7 GB, about 30 hours to download at
   the 0.11 MB/s Zenodo gave tonight) and test on CRC-VAL-HE-7K. Covering the other carcinomas needs
   labelled sets for them (for example the pan-carcinoma sets in the handoff: OpenTME/HETA). This
   needs no slide reprocessing, since the run saves the CONCH embedding of every tile.
4. **Confidence thresholds are not a quality filter** for zero-shot CONCH labels here. If a
   threshold is kept for the pathomic step, set it on a calibrated classifier, not on softmax
   scores.

## Limits

- One cancer type (colorectal), one public dataset, Macenko-normalised tiles. TCGA slides are not
  normalised.
- Kather tiles are 20x-equivalent; ours are mostly 40x at a smaller field (see above).
- The class names in (b) and (c) are mine, written once and not tuned, and the results have no
  confidence intervals. With 339 to 1,338 tiles per class, the per-class accuracies have standard
  errors of about 1 to 3 points, but tiles from the same patient are correlated, so the real
  uncertainty is larger.
- The colon names in (c) are written for this dataset, so (c) is the best case for type-specific
  prompts. Their value for other types is untested.
- Data: `/data/scratch/juan/datasets/kather_crc_val_7k/` (tiles and the cached embeddings), on
  scratch; not in git.
