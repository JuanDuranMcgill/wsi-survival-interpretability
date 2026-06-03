# Multi-Scale Interpretability for Graph-Based Survival Prediction in Whole-Slide Images

Code accompanying the paper:  
**"Towards Trustworthy WSI Survival Analysis: Interpretability, Explainability, and MED3PA Reliability Profiling"**  
MICCAI Interpretability Workshop 2026

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

## Repository Structure

```
preprocessing/
    classify_all_tiles_multi_6.py   # Tile WSIs and classify regions with CONCH
    embed_all_tiles_uni2.py         # Encode tiles with UNI-2 (1536-dim)

model/
    interpret_graphtrans_late_int_sparse_m3save.py
                                    # Main model: per-region GAT, cross-region
                                    # self-attention, ReLU fusion, Cox loss,
                                    # bootstrap training + region weight recording

analysis/
    blca_clinical_linear_med3pa.py  # BLCA: clinical feature importance + MED3PA
    brca_clinical_linear_med3pa.py  # BRCA: clinical feature importance + MED3PA
    blca_radiomic_linear_med3pa.py  # BLCA: radiomic feature importance + MED3PA
    brca_radiomic_linear_med3pa.py  # BRCA: radiomic feature importance + MED3PA

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
