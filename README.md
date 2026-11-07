# Multi-Scale Interpretability for Graph-Based Survival Prediction in Whole-Slide Images

Code accompanying the paper:  
**"Towards Trustworthy Explanations of Whole-Slide Image Survival Models: The IER Audit"**

---

## Overview

This repository contains the core scripts for a three-level interpretability framework applied to a graph-based survival prediction model for whole-slide images (WSIs), validated on TCGA-BLCA (bladder cancer, N=379) and TCGA-BRCA (breast cancer, N=1,060).

```
WSI → Tiling → CONCH (region classification) → UNI-2 (embedding)
    → Region-specific graphs → Graph attention → Cross-region attention
    → ReLU fusion → Risk score (Cox loss)
         │
         ├── Analysis 1: Region fusion weights
         ├── Analysis 2: Bootstrap RF feature importance
         └── Analysis 3: MED3PA reliability profiling
```

---

## Running the IER audit on your own model

The audit was built on our TCGA runs, and the scripts still show it: some defaults
point at our cluster paths and TCGA files. A packaged tool is in progress. Until
then, most checks need only one file from your model, and the rest need a way to
re-score it with tissue compartments removed.

### 1. Export your model's held-out risk scores

Write one `.npz` file with one entry per patient. The predictions must be held
out (out-of-bag or cross-validated), never on patients the model was trained on.

| Key | Content |
|---|---|
| `patient_ids` | patient identifiers (matched on their first 12 characters, the length of a TCGA barcode) |
| `mean_risk` | the held-out risk score, higher meaning higher risk |
| `pfi_time` | follow-up time |
| `pfi_event` | 1 if the event occurred, 0 if censored |
| `oob_count` | number of held-out predictions averaged per patient (1 for a single cross-validation) |

`analysis/oob_risk.py` shows how we build this file from bootstrap rounds.

### 2. Checks that need only that file and feature tables

| Leg | Check | Script | Also needs |
|---|---|---|---|
| Context | C-index with interval; clinical-only Cox; risk added to clinical variables | `analysis/cohort_and_baseline.py` | `--clinical-csv` |
| Context | integrated Brier score against Kaplan–Meier | `analysis/integrated_brier.py` | — |
| E | surrogate importance (random forest on the risk score) | `analysis/*_radiomic_linear_med3pa.py` (writes `radiomic_importance.csv`) | `--radiomics-csv` |
| E | second estimator: permutation importance | `analysis/permutation_importance.py` | `--radiomics-csv`, `--importance-csv` |
| E | each feature against observed outcome, with false-discovery-rate control | `analysis/pathomic_vs_survival.py` | `--radiomics-csv`, `--importance-csv` |
| R | censoring-aware error per patient | `analysis/patient_error.py` | survival table (currently the TCGA-CDR spreadsheet, `--cdr-xlsx`) |
| R | MED3pa profiles; C-index inside each profile | `analysis/*_linear_med3pa.py`, `analysis/profile_metrics.py` | error file from the row above; feature CSV |
| R | per-patient conformal bounds; coverage by subgroup | `analysis/conformal_survival.py` | `--clinical-csv` |

Feature tables are CSV files indexed by patient ID. Pathomic feature names start
with their compartment, for example `adipose_E_original_firstorder_Mean_mean`.
For the subgroup coverage, the clinical table can hold `pathologic_stage`,
`pathologic_n` and `pathologic_t`; missing columns are skipped.

### 3. The interpretability check: re-score your model without compartments

Exact Shapley values over compartments re-score the model with every
combination of compartments removed (with 8 or 9 compartments, 256 or 512
combinations). This step needs your model:

- **Multiple-instance models:** removing a compartment means dropping its tiles
  from the bag. `analysis/abmil_compartment_shapley.py` does this for ABMIL and is
  the template to adapt. It reads one folder of tile features per compartment
  (see `analysis/compartment_common.py`) and survival from a CSV with columns
  `bcr_patient_barcode`, `PFI` and `PFI.time` (`--surv-csv`).
- **Our region-fusion model:** `analysis/region_shapley.py`.
- **Known-signal test:** `analysis/planted_signal.py` plants a signal in one
  compartment and retrains the model, so it needs your training procedure.
- **Tissue labels:** every tile needs a compartment label;
  `analysis/label_confidence_audit.py` checks how decisively they were assigned.

The figures in the paper are built from the outputs by
`visualization/make_audit_panels.py`; `tests/test_figure_data.py` checks each
plotted number against the paper.

---

## Licence

MIT; see [LICENSE](LICENSE).

---

## Repository Structure

```
preprocessing/
    classify_all_tiles_multi_6.py   # Tile WSIs and classify regions with CONCH
    embed_all_tiles_uni2.py         # Encode tiles with UNI-2 (1536-dim)

model/
    interpret_graphtrans_late_int_sparse_m3save.py
                                    # Main model (BLCA, 8 regions): per-region GAT,
                                    # cross-region self-attention, ReLU fusion, Cox loss,
                                    # bootstrap training + region weight recording
    interpret_graphtrans_late_int_sparse_m3save_BRCA.py
                                    # Same model, parameterized for BRCA (9 regions)

analysis/
    compute_patient_error.py        # BLCA: per-patient concordance/discordance error
                                     # from mean_risk vs TCGA-CDR PFI/PFI.time — this is
                                     # the "backbone_error" MED3PA labels are built from
    compute_patient_error_BRCA.py   # Same, for BRCA
    blca_clinical_linear_med3pa.py  # BLCA: clinical feature importance + MED3PA
    brca_clinical_linear_med3pa.py  # BRCA: clinical feature importance + MED3PA
    blca_radiomic_linear_med3pa.py  # BLCA: radiomic feature importance + MED3PA
    brca_radiomic_linear_med3pa.py  # BRCA: radiomic feature importance + MED3PA

slurm/
    submit_bootstrap.sh             # BLCA: 6 jobs x 17 rounds = 102 bootstrap rounds
    submit_bootstrap_BRCA.sh        # BRCA: 10 jobs x 10 rounds = 100 bootstrap rounds
    submit_bootstrap_BRCA_top_up.sh # BRCA: 11 extra jobs, 52 more rounds (seeds 101-152)
    blca_clinical.sh, blca_radiomic.sh,
    brca_clinical.sh, brca_radiomic.sh
                                    # Launch the analysis/*_linear_med3pa.py scripts
                                    # with the actual flags used: --fi-n-bootstrap 500000
                                    # (500k RF resamples) --med3pa-n-runs 1000
    submit_radiomics.sh, submit_radiomics_BRCA.sh
                                    # Launch pyradiomics feature extraction batches
                                    # (depend on find_pairs.py / find_pairs_BRCA.py,
                                    # not yet in this repo — see note below)

visualization/
    generate_wsi_highlights.py      # WSI thumbnails with top-2 regions highlighted
    analyze_med3pa_results.py       # Per-arm figures and tables (feature importance,
                                    # MED3PA accuracy-coverage, profile stability)
    generate_paper_figures.py       # Combined cross-arm paper figures and tables
```

---

## Key Components

### Model (`model/`)
- **AGTLayer**: graph attention layer with multi-head attention and 3-hop diffusion
- **IPGGraphFormer** (multi-region): per-region GAT + attention pooling, cross-region `nn.MultiheadAttention`, learnable ReLU-normalized fusion weights
- **Bootstrap training**: 102 rounds, region weights averaged at best epoch
- **Cox partial log-likelihood loss**

### Analysis 2 — Feature Importance (`analysis/`)
Bootstrap Random Forest regressor (500,000 resamples) trained to predict the WSI-derived survival risk score from clinical or radiomic features. Spearman correlations quantify individual feature–survival associations.

### Analysis 3 — MED3PA (`analysis/`)
1,000 bootstrap runs of a secondary Random Forest classifier trained to predict backbone model error (high/low label at 80th percentile). Stable profiles (≥5% of runs) are classified as Trust (accuracy ≥0.80) or Caution (<0.80). Significance assessed by two-sided t-test vs chance (0.50).

---

## Dependencies

```
torch, pycox, openslide-python, Pillow, numpy, pandas,
scikit-learn, scipy, matplotlib, joblib
```

Tissue classification requires [CONCH](https://github.com/mahmoodlab/CONCH) and tile embedding requires [UNI-2](https://github.com/mahmoodlab/UNI).

---

## Note

This repository is intended as a code reference. Raw WSI data (TCGA-BLCA, TCGA-BRCA) must be independently obtained through the [GDC Data Portal](https://portal.gdc.cancer.gov/).

All SLURM scripts assume `module load gcc opencv/4.12.0 && source ~/envs/conch_env/bin/activate` and `--account=def-senger`, and reference absolute cluster paths under `/home/sorkwos/links/scratch/...` — update these for any other environment.

### Known gaps (found during a repo/scratch audit, not yet ported in)
Still living only on `/scratch` and not yet in this repo: `find_pairs.py` / `find_pairs_BRCA.py` (feeds `submit_radiomics*.sh`), `extract_radiomics.py` / `_BRCA.py`, `filter_radiomics*.py`, `merge_bootstrap.py` / `_BRCA.py`, `prepare_clinical.py` / `_BRCA.py`. Ask if you want these pulled in too.
