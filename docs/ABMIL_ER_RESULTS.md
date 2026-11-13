# E and R on ABMIL, next to our model (Narval, 2026-11-12)

This document follows `docs/HANDOFF_ABMIL_ER.md`, steps 1 to 6. It reruns the explanation (E) and
reliability (R) checks with ABMIL's out-of-bag risk in place of our model's, keeping
everything else the same. All numbers come from `results/abmil_er_comparison.json`, which
`analysis/abmil_er_report.py` builds from summaries in `results/` only.

## What was run

- **ABMIL rounds.** These are run M's settings, with `EXP=M` and `--last-epoch`, saved to
  `/scratch/sorkwos/abmil_v3_{blca,brca}`. Each round also writes `round_###_local.npz`, which
  holds every out-of-bag patient's full-model risk at the selected epoch and at the last epoch.
  - BLCA: 10 of 10 rounds finished.
  - BRCA: 8 of 10 finished. Rounds 4 and 10 hit the 6 h limit while still training: the nodes
    were busy, so the GPUs waited on data and every round ran about 1.7 times slower than
    run M. They were not rerun.
  - Cost: about 64 GPU-hours, against the 38 estimated.
- **Round 1 matches run M** and passes every stop condition:

  | | Baseline c-index, this run | Run M | Selected epoch (both runs) | Efficiency gap | Head check | Empty coalition |
  |---|---|---|---|---|---|---|
  | BLCA | 0.6556 | 0.6557 | 20 | 6e-17 | 7e-7 | 0.5 |
  | BRCA | 0.6541 | 0.6548 | 7 | 6e-17 | 5e-7 | 0.5 |

  All other rounds also passed these checks.
- **Risk score for E and R.** E and R use the last-epoch risk (`risk_full_last`), averaged
  over each patient's out-of-bag rounds (`analysis/abmil_oob_risk.py`). Our model's E and R
  inputs also use its final epoch, and using the last epoch avoids choosing the epoch on the
  patients whose errors are measured. The selected-epoch c-index is reported alongside it as
  the link to run M.
- **Per-patient error.** Computed by `analysis/patient_error.py`, as for our model.
- **E/R runs.** All 16 ran (4 arm scripts × q50, q70, q80, q90) with the main runs' settings:
  - Clinical: 200 MED3pa runs, 200 bootstraps.
  - Pathomic: 200 runs, 1,000 bootstraps, top 30 features.

  Each run's surrogate target and error labels both come from ABMIL's file
  (`input_error_npz`, recorded in every reliability JSON). Outputs are in
  `results/reliability_abmil_{cohort}_{arm}_q{NN}/`.

## Out-of-bag concordance

| | Ours | ABMIL, last epoch (E/R input) | ABMIL, selected epoch | ABMIL mean per-round c-index, last / selected | Run M mean round baseline (selected epoch) | Rounds | Never out-of-bag |
|---|---|---|---|---|---|---|---|
| BLCA | 0.622 | 0.605 | 0.625 | 0.594 / 0.608 | 0.608 | 10 | 1 of 379 |
| BRCA | 0.607 | 0.573 | 0.612 | 0.578 / 0.627 | 0.626 | 8 | 12 of 1,060 |

At the selected epoch, ABMIL's per-round c-index matches run M's (0.608 and 0.627, against
0.608 and 0.626). At the last epoch, ABMIL is weaker than our model in both cohorts, and much
weaker in BRCA. BRCA's selected epoch is early (7 of 20 in round 1), and by the last epoch
the out-of-bag c-index has fallen by about 0.09 (0.654 to 0.564 in round 1, and the same in
run M).

## E: what the surrogate learns

The values come from the q50 folders, as in the paper. A unit's share is its share of the
summed surrogate importance. Fidelity is the surrogate's out-of-bag correlation with the risk
score, given as the mean ± SD over bootstraps.

### Pathomic: compartments

| | Ours: ranking (share %) | ABMIL: ranking (share %) | Rank agreement, Spearman ρ (8 compartments) | Fidelity, ours | Fidelity, ABMIL |
|---|---|---|---|---|---|
| BLCA | **adipose 17.0**, urothelium 13.8, immune 13.6, muscularis 12.4, necrosis 12.2, lamina 11.5 | **muscularis 15.2**, tumour 13.4, lamina 13.2, adipose 13.0, vessels 12.0, immune 11.5 | −0.31 (p = 0.46) | 0.365 ± 0.080 | 0.302 ± 0.077 |
| BRCA | **adipose 20.8**, TILs 15.1, DCIS 14.0, tumour 11.4, stroma 11.4, TDLU 10.3 | **adipose 15.2**, TILs 15.2, TDLU 14.3, stroma 12.0, muscle 11.6, DCIS 11.5 | 0.62 (p = 0.10) | 0.460 ± 0.043 | 0.355 ± 0.048 |

### Clinical: features

| | Ours: top features (share %) | ABMIL: top features (share %) | Rank agreement, Spearman ρ | Fidelity, ours | Fidelity, ABMIL |
|---|---|---|---|---|---|
| BLCA | **age 19.3**, weight 15.8, lymph nodes examined 14.0, height 13.1, T 7.7 | **age 16.7**, height 15.8, lymph nodes examined 12.7, weight 12.5, T 7.9 | 0.92 (21 features) | 0.291 ± 0.055 | 0.208 ± 0.059 |
| BRCA | **age 34.0**, lymph nodes examined 18.3, T 9.0, positive nodes 7.3, ER 4.2 | **age 21.9**, lymph nodes examined 15.5, T 14.1, stage 8.4, positive nodes 7.2 | 0.94 (25 features) | 0.319 ± 0.035 | 0.243 ± 0.039 |

- **Fidelity.** ABMIL's risk is harder to reproduce from the same features: its fidelity is
  lower in all four arms, by 0.06 to 0.10.
- **Clinical.** The two models' risks depend on clinical variables in nearly the same way.
  Age leads in both cohorts, though with a smaller share under ABMIL.
- **Pathomic.** The two models' compartment rankings differ:
  - In BLCA they are unrelated (ρ = −0.31). ABMIL's leading compartment is muscularis, where
    ours is adipose.
  - In BRCA, adipose still leads, but its share falls from 20.8% to 15.2% and ties with TILs.
  - Under ABMIL, the shares lie closer to the uniform 12.5%, so no compartment stands out.

## R: can the error be predicted?

The tables give the AUROC on the evaluation split, with a Hanley-McNeil 95% interval. The p
value is the permutation test against 0.5 (2,000 permutations). An asterisk (*) marks an AUROC
that survives Bonferroni over the 16 tests, α = 0.05/16 = 0.0031.

### BLCA

| Arm | Threshold | Ours | ABMIL |
|---|---|---|---|
| Clinical | q50 | 0.588 [0.497, 0.678] | 0.542 [0.450, 0.634] |
| Clinical | q70 | 0.523 [0.421, 0.624] | 0.486 [0.385, 0.586] |
| Clinical | q80 | 0.455 [0.342, 0.568] | 0.560 [0.442, 0.677] |
| Clinical | q90 | 0.580 [0.422, 0.738] | 0.499 [0.345, 0.653] |
| Pathomic | q50 | 0.547 [0.455, 0.639] | 0.505 [0.412, 0.597] |
| Pathomic | q70 | 0.439 [0.340, 0.537] | 0.519 [0.418, 0.621] |
| Pathomic | q80 | 0.561 [0.443, 0.678] | 0.460 [0.347, 0.574] |
| Pathomic | q90 | 0.476 [0.324, 0.628] | 0.386 [0.246, 0.526] |

### BRCA

| Arm | Threshold | Ours | ABMIL |
|---|---|---|---|
| Clinical | q50 | 0.579 [0.525, 0.634] | 0.570 [0.515, 0.625] |
| Clinical | q70 | 0.612 [0.551, 0.673]* | 0.606 [0.545, 0.666]* |
| Clinical | q80 | 0.580 [0.509, 0.651] | 0.597 [0.526, 0.668] |
| Clinical | q90 | 0.520 [0.426, 0.614] | 0.495 [0.402, 0.588] |
| Pathomic | q50 | 0.671 [0.620, 0.722]* | 0.579 [0.525, 0.634] |
| Pathomic | q70 | 0.651 [0.591, 0.710]* | 0.614 [0.553, 0.674]* |
| Pathomic | q80 | 0.643 [0.573, 0.713]* | 0.581 [0.510, 0.652] |
| Pathomic | q90 | 0.693 [0.600, 0.786]* | 0.621 [0.526, 0.717] |

### Summary

| | Ours | ABMIL |
|---|---|---|
| Survive Bonferroni (of 16) | 5 | 2 |
| Hanley-McNeil interval above 0.5 (of 16) | 7 | 7 |
| Profiles recurring in ≥ 5% of runs (BLCA clinical / BLCA pathomic / BRCA clinical / BRCA pathomic) | 159 (26 / 8 / 79 / 46) | 155 (28 / 4 / 77 / 46) |
| Profiles whose in-profile c-index interval excludes the cohort value | 3 | 3 |
| Cohort value used (BLCA / BRCA) | 0.622 / 0.607 | 0.605 / 0.573 |

The profiles whose interval excludes the cohort value are different for the two models.

| Model | Cohort | Arm | Threshold | Rule | n | In-profile c-index [95% CI] | Cohort value |
|---|---|---|---|---|---|---|---|
| Ours | BRCA | Clinical | q50 | stage ≤ 1.5 | 73 | 0.37 [0.07, 0.60] | 0.607 |
| Ours | BRCA | Clinical | q70 | stage ≤ 1.5 | 73 | 0.33 [0.19, 0.54] | 0.607 |
| Ours | BRCA | Clinical | q80 | age in a middle band | 138 | 0.72 [0.61, 0.82] | 0.607 |
| ABMIL | BLCA | Clinical | q90 | height above threshold | 61 | 0.74 [0.63, 0.83] | 0.605 |
| ABMIL | BRCA | Clinical | q90 | older, more nodes examined | 81 | 0.39 [0.19, 0.57] | 0.573 |
| ABMIL | BRCA | Pathomic | q50 | adipose eosin skewness (SD) ≤ 2.37 | 259 | 0.66 [0.58, 0.75] | 0.573 |

### What this shows about R

- **BLCA.** The error is not predictable for either model: no AUROC differs from 0.5 and
  nothing survives Bonferroni.
- **BRCA clinical.** Predictability barely changes from our model to ABMIL. The AUROCs agree
  within 0.03 at every threshold, and q70 survives Bonferroni for both models.
- **BRCA pathomic.** This is where the models differ. Our model's AUROCs are 0.64 to 0.69 and
  all four survive Bonferroni. ABMIL's are 0.58 to 0.62 and only q70 survives, although every
  Hanley-McNeil interval stays above 0.5.
- **How to read the BRCA pathomic difference.** ABMIL's errors can still be predicted from
  pathomic features, just less well. This fits the lower surrogate fidelity in E, and ABMIL's
  last-epoch risk is itself weaker in BRCA (c-index 0.573).
- **Profiles.** The models have about the same number of recurring profiles (159 against
  155). Each has three profiles whose in-profile c-index excludes its cohort value.

## Notes

- **Bonferroni and p values.** Bonferroni uses the permutation p values. With 2,000
  permutations the smallest p is about 0.0005, below α = 0.0031.
- **Evaluation split.** It is rebuilt from ABMIL's own scored patients. It has 151 patients
  in BLCA, against 151 or 152 for ours, and 419 or 420 in BRCA, against 424 for ours. BRCA
  is smaller because the 12 patients who were never out-of-bag are left out.
- **The Hanley-McNeil helper was missing.** The handoff names
  `visualization/figure_data.hanley_mcneil_ci`, but it was not in the repository, so it was
  added with the standard formula: the SE from Q1 = A/(2−A) and Q2 = 2A²/(1+A), and
  A ± 1.96 SE. `figure_data.profiles()` and `auroc_by_arm()` now take `model="main"` or
  `"abmil"`, so the paper's figures keep reading only our model's 16 folders. In
  `tests/test_figure_data.py`, `test_profiles` and `test_auroc` still pass, as do 5 other
  tests; the 2 tests that need pytest itself were not run (pytest is not installed in the
  Narval venv).
- **What is committed.** Only summaries:
  - `results/abmil_oob_risk_{cohort}[_selected_epoch].json`
  - `results/patient_error_{cohort}_abmil_oob.json`
  - `results/abmil_er_comparison.json`
  - from each reliability folder: the reliability JSON, `region_summary.csv`,
    `radiomic_importance.csv` and `clinical_importance_survival.csv`, as for our model's
    folders

  Patient-level files stay out of git:
  - `round_###_local.npz`
  - `~/data/abmil_oob_risk_*.npz`
  - `~/data/patient_error_*_abmil_oob.{npz,_per_patient.csv}`
  - `*_patient_predictions.csv`
  - `med3pa_backbone_error_labels*.csv`
