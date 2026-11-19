# Case-study data: one patient through the audit (Narval, 2026-11-19)

This exports the data for `docs/HANDOFF_CASE_STUDY.md`, using our model's runs only. It
was built on CPU from saved outputs, with no training, GPU or MED3pa rerun.

- **Data:** `analysis/case_study.py` writes `results/case_study/{patient}.json`.
- **Images:** `visualization/case_study_images.py` writes
  `paper_figures/case_study/{patient}/`.

## What was checked

- **Splits.** All 16 discovery/evaluation splits were rebuilt the way the arm scripts
  build them. Each matches its saved error threshold, `n_eval` and evaluation prevalence.
- **Profiles.** All 159 recurring profiles (`top_profiles_eval`, run frequency ≥ 5%) were
  re-applied from their exact saved rules (`all_profiles_dir*.csv`) to the raw CSV
  values, as `apply_profile_to_X` does. Every profile reproduces its saved patient,
  event and pair counts and its mean error. Each candidate JSON records these checks
  under `checks`.

## Candidates

| Patient | Cohort | Held out in (C = clinical, P = pathomic) | Risk percentile | Error (pairs) | PFI event / days | High-error at q50 | Matching profiles (pathomic) | Differ from cohort | Images |
|---|---|---|---|---|---|---|---|---|---|
| TCGA-ZF-AA53 | BLCA | C q50, q70, q80; P q50, q80, q90 | 65 | 0.55 (161) | 0 / 1,761 | yes | 12 (3) | 0 | no |
| TCGA-FT-A3EE | BLCA | C q70, q80, q90; P q50 | 61 | 0.33 (15) | 0 / 99 | no | 9 (0) | 0 | no |
| TCGA-2F-A9KT | BLCA | C q70, q80, q90 | 79 | 0.74 (163) | 0 / 3,314 | yes | 9 (0) | 0 | no |
| TCGA-K4-A5RI | BLCA | C q80, q90; P q70, q80 | 99 | 0.20 (337) | 1 / 236 | no | 9 (2) | 0 | no |
| TCGA-ZF-A9RF | BLCA | C q50, q70, q80, q90; P q50 | 48 | 0.38 (203) | 1 / 1,565 | yes | 13 (0) | 0 | no |
| TCGA-B6-A0X1 | BRCA | C q90; P q50 | 99 | 0.06 (795) | 1 / 490 | no | 14 (7) | 0 | yes |
| TCGA-B6-A0I9 | BRCA | P q80, q90 | 98 | 0.05 (913) | 1 / 362 | no | 12 (12) | 0 | yes |
| TCGA-B6-A0RG | BRCA | C q50, q70, q80, q90 | 97 | 0.93 (121) | 0 / 2,082 | yes | 29 (0) | 0 | yes |
| TCGA-D8-A1JL | BRCA | C q50, q80, q90; P q80 | 81 | 0.63 (54) | 0 / 611 | yes | 34 (7) | 0 | yes |
| **TCGA-LL-A8F5** | BRCA | C q50, q80, q90; P q80, q90 | 98 | 0.94 (53) | 0 / 596 | yes | 41 (12) | **1** | yes |

How to read the columns:
- **Risk percentile** is the share of the cohort with an out-of-bag risk at or below the
  patient's.
- **Error** is the share of the patient's comparable pairs that the model orders wrongly.
- **High-error at q50** is the same in the clinical and pathomic arms for every
  candidate.
- **Matching profiles** counts the recurring profiles the patient meets, summed over the
  runs in which the patient is held out. A patient meets profiles only in those runs.
- **Differ from cohort** counts matching profiles whose in-profile C-index 95% interval
  excludes the cohort value (0.622 BLCA, 0.607 BRCA).

## Recommendation: TCGA-LL-A8F5 (BRCA)

- **Meets every preference in the handoff.** LL-A8F5 is a BRCA patient held out in two
  pathomic runs (q80, q90) and three clinical runs, and has images.
- **It shows each part of the audit in turn:**
  - **Risk.** The model ranks her at the 98th percentile of risk.
  - **Error.** She is censored at 596 days without progression, so the model orders most
    of her comparable pairs wrongly (error 0.94, 53 pairs). She is high-error at every
    threshold.
  - **Recurring profiles.** She falls into 12 recurring pathomic profiles, all on adipose
    eosin and stroma or TDLU haematoxylin intensity statistics. None of them has an
    in-profile C-index that differs from the cohort. The audit therefore has no subgroup
    rule that would have warned about her: a failure the profiles do not catch.
  - **The one profile that differs.** She also falls into one of the paper's three
    "differs" profiles: clinical q80, an age band (−0.76 < age z ≤ 0.25), with n = 138
    and C-index 0.72 [0.61, 0.82] against 0.607. There the model does better than in the
    cohort, yet she is one of its errors. This illustrates the caveat that a profile
    describes a group, not every patient in it.
  - **Tissue.** Her slide is striking: CONCH labels 35% of tiles necrosis or
    haemorrhage, the most of the five BRCA candidates.
- **Caveats to state in the text:**
  - Her error comes from short follow-up. She is censored at 596 days, and the
    comparisons charged to her are with patients who progressed earlier. She may yet
    progress.
  - The "muscle" example tiles do not look like muscle (CONCH label noise). The
    pathologists' questionnaire answers for this slide should be checked before showing
    them.
  - The orange band along the bottom edge of the tissue map is the scanner's edge
    artefact (visible in the thumbnail), not tissue.
- **Alternatives:**
  - **A contrasting correct prediction: TCGA-B6-A0I9.** Stage IV, progression at 362
    days, risk percentile 98, error 0.05. She is held out in pathomic q80 and q90 and
    meets 12 recurring pathomic profiles, none differing.
  - **TCGA-B6-A0X1** is similar to A0I9, but its CDR stage reads "[Discrepancy]".

## What is not available, and why

- **Per-patient Shapley attribution.** The main Shapley run's per-round local files
  (`/scratch/sorkwos/shapley_{blca,brca}/round_###_local.npz`) are gone from Narval,
  most likely through the scratch purge. Only the round JSONs in `results/shapley_*`
  remain, and they hold cohort averages. They were not recomputed: that would mean
  rerunning the Shapley rounds on GPU. The example tiles therefore come from the two
  compartments with the largest tile share. Run S's masked-attention local files
  (`shapley_masked_*`) still exist, but they are a different method, so they were not
  used.
- **The error classifier's per-patient probability.** The arm scripts never saved it;
  only the evaluation metrics are kept. Getting it would need a MED3pa rerun.
- **BLCA images.** The BLCA slides are not on Narval, only their embeddings. The laptop
  has the questionnaire thumbnails.
- **Questionnaire colours.** They are not on Narval. The tissue maps use the paper's
  compartment palette (`REGION_COLORS` in `visualization/generate_paper_figures.py`),
  recorded in each `images.json`. Recolour on the laptop if the questionnaire used
  different ones.

## Files

- **`results/case_study/{patient}.json`**, one per candidate (10). Contents:
  - risk and percentile;
  - PFI;
  - error, pairs and high-error labels per arm and threshold;
  - CDR age, stage and histology;
  - tissue shares (CONCH labels, `results/label_confidence_*_per_slide.csv`);
  - the 10 leading q50 pathomic features, raw and as cohort percentiles;
  - the clinical variables used by any recurring rule (in the model's CSV units, with
    percentiles);
  - per run: held out or not, the high-error label, and the matching profiles with rule,
    size, prevalence, C-index and interval, cohort value and whether it differs;
  - the split and profile checks.
- **`paper_figures/case_study/{patient}/`**, BRCA only:
  - `thumbnail.png`, with a 1,500 px long side;
  - `tissue_map.png`, the same size, one CONCH-coloured cell per 256 px tile the model
    used;
  - `tissue_overlay.png`, the map blended on the thumbnail;
  - `tiles/`, three 256 px level-0 tiles (40x, about 64 µm) from each of the two
    largest compartments;
  - `images.json`, with scale, colours and tile coordinates.
