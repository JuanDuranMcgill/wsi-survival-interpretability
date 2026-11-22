# Pathomic (pyradiomics) feature pipeline

These scripts produced the pathomic feature tables the paper uses,
`radiomics_pre_corr.csv` for BLCA and BRCA. That's the file the radiomic MED3pa arms,
the surrogate importance and the permutation importance read, as `~/data/{cohort}/radiomics_pre_corr.csv`.
They were copied unchanged on 2026-11-21 from Trillium
`/scratch/sorkwos/multimodality/graph_subset_generation/`. The only change is a second
line in each file naming where it was found. The paths inside the scripts are the
original Trillium paths: `/home/sorkwos/links/scratch` is `/scratch/sorkwos`.

## What each script does

| Script | What it does | Input | Output |
|---|---|---|---|
| `submit_radiomics.sh` | Submits 4 CPU jobs (6 cores, 12 h), one per quarter of the BLCA slides | — | Logs in `radiomics_logs/` |
| `submit_radiomics_BRCA.sh` | Same for BRCA, 8 jobs | — | Logs in `radiomics_brca_logs/` |
| `find_pairs.py` | Matches each BLCA slide (`TCGA-BLCA-p2/**.svs`) with its CONCH tile file (`blca_jsons/{slide}_tiles.jsonl`), sorts the pairs, takes batch *k* of *n*, and runs `extract_radiomics.py` on each | Slides, CONCH JSONL | — |
| `find_pairs_BRCA.py` | Same for BRCA (`TCGA-BRCA-new`, `brca_jsons`), DX1 slides only | Slides, CONCH JSONL | — |
| `extract_radiomics.py` | Per slide: the tiles in each of the 8 BLCA compartments; per tile, haematoxylin and eosin channels by colour deconvolution, then pyradiomics features on each channel; per compartment, the mean and SD over tiles | One slide and its JSONL | `radiomics_output/{patient}_radiomics.npz` |
| `extract_radiomics_BRCA.py` | Same for the 9 BRCA compartments | One slide and its JSONL | `radiomics_output_BRCA/{patient}_radiomics.npz` |
| `filter_radiomics.py` | Stacks the per-patient files into a table and filters it (below). Writes the table before and after correlation filtering | `radiomics_output/*_radiomics.npz` | `radiomics_pre_corr.csv` (used by the paper), `radiomics_filtered.csv` |
| `filter_radiomics_BRCA.py` | Same for BRCA | `radiomics_output_BRCA/*.npz` | Same two CSVs in `radiomics_output_BRCA/` |
| `filter_radiomics_firstorder.py` | A later BLCA variant (2026-04-22) that keeps first-order features only | `radiomics_output/*.npz` | `radiomics_firstorder_504.csv`. Nothing in this repo reads it |

## Command order

Both cohorts ran on Trillium in the `conch_env` virtualenv, CPU only.

```bash
# BLCA (2026-04-04 to 04-08): 455 slides matched, 384 patients written
bash submit_radiomics.sh          # 4 x: python find_pairs.py --run --batch k --total 4
python filter_radiomics.py        # after all jobs finish

# BRCA (2026-04-20 to 04-21): 1,061 DX1 slides matched, 1,061 patients written
bash submit_radiomics_BRCA.sh     # 8 x: python find_pairs_BRCA.py --run --batch k --total 8
python filter_radiomics_BRCA.py
```

`find_pairs*.py --dry-run` lists the pairs without extracting anything.
`extract_radiomics*.py` skips any patient whose output file already exists.

## Settings

There is no params file. The extractor is configured in `worker_init()` in
`extract_radiomics*.py`:

```python
featureextractor.RadiomicsFeatureExtractor(force2D=True, force2Ddimension=0, binWidth=25,
                                           minimumROIDimensions=1, minimumROISize=1)
extractor.disableAllFeatures()
extractor.enableFeatureClassByName("firstorder")
extractor.enableFeatureClassByName("glcm")
```

**Image:**
- original image type only, no filters;
- the mask is the whole tile;
- each channel is rescaled to 0–255 per tile before extraction.

**Tiles:**
- 256 px tiles at level 0;
- channels from `skimage.color.rgb2hed` (haematoxylin and eosin).

**Which tiles count:**
- a tile counts toward a compartment when its CONCH top-1 label is that compartment with
  probability ≥ 0.7;
- a compartment needs at least 5 tiles, otherwise its features are NaN;
- at most 1,000 tiles per compartment, drawn with `random.sample` and no seed.

**Feature names:** `{compartment}_{H|E}_original_{firstorder|glcm}_{feature}_{mean|std}`.

**Filtering, in this order:**
1. drop patients missing more than 50% of features;
2. drop features missing in more than 20% of patients;
3. fill the remaining gaps with the column median;
4. drop features with variance at or below 0.01, then write `radiomics_pre_corr.csv`;
5. drop one feature of each pair with |r| > 0.90, then write `radiomics_filtered.csv`.

**Library versions** (`conch_env`, installed 2026-04-04, before both runs):
- pyradiomics 3.1.0+computecanada;
- SimpleITK 2.3.1;
- scikit-image 0.26.0;
- openslide-python 1.3.1.

**Outputs on Trillium scratch:**
- BLCA `radiomics_pre_corr.csv`: 384 patients × 895 features;
- BRCA: 1,061 × 927.

All features in `results/reliability_{cohort}_radiomic_q50/radiomic_importance.csv` (504
BLCA, 514 BRCA) are columns of these files. The copies the analyses read were not
compared byte for byte, since those are on Narval and were on the laptop.

## Caveats

- **Not exactly reproducible.** The 1,000-tile cap is drawn without a seed, so rerunning
  gives slightly different means and SDs for compartments with more than 1,000 tiles.
- **BLCA features don't always come from the DX1 slide.** BLCA `find_pairs.py` doesn't
  restrict to DX1 slides, unlike the BRCA script. Output is one file per patient, and
  the first slide processed wins, in sorted path order, which is effectively arbitrary.
  The graph model uses DX1 slides. From the job logs, 17 of the model's 379 BLCA patients
  have pathomic features from another diagnostic slide:

  TCGA-2F-A9KT (DX2), TCGA-2F-A9KW (DX2), TCGA-BL-A13I (DX2), TCGA-BL-A13J (DX2),
  TCGA-BL-A5ZZ (DX2), TCGA-FJ-A3Z7 (DX5), TCGA-FJ-A3ZE (DX2), TCGA-G2-A2EC (DX7),
  TCGA-G2-A2EJ (DX3), TCGA-G2-A2EK (DX3), TCGA-G2-A2EO (DX3), TCGA-G2-A2ES (DX2),
  TCGA-G2-A3IB (DX2), TCGA-G2-AA3B (DX4), TCGA-G2-AA3F (DX9), TCGA-GU-A42R (DX2),
  TCGA-YC-A9TC (DX2)

  TCGA-2F-A9KT is a case-study candidate. One more patient, TCGA-2F-A9KQ, was written
  by a test run before the batch jobs, and its slide isn't logged.
- **Not copied:** `extract_radiomics.py.bk`, a backup from earlier the same day that
  nothing calls. It differs only in comments, log messages and the thread settings.
