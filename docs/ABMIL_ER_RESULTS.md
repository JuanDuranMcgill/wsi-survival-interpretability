# E and R on ABMIL, next to our model (Narval, 2026-11-12 to 14)

This document follows `docs/HANDOFF_ABMIL_ER.md`, steps 1 to 6. It reruns the explanation (E) and
reliability (R) checks with ABMIL's out-of-bag risk in place of our model's, keeping
everything else the same. All numbers come from `results/abmil_er_comparison.json`, which
`analysis/abmil_er_report.py` builds from summaries in `results/` only.

## What was run

- **ABMIL rounds.** These are run M's settings, with `EXP=M` and `--last-epoch`, saved to
  `/scratch/sorkwos/abmil_v3_{blca,brca}`, 10 rounds per cohort. Each round also writes
  `round_###_local.npz`, which holds every out-of-bag patient's full-model risk at the
  selected epoch and at the last epoch.
- **BRCA rounds 4 and 10 had to be rerun.**
  - **What went wrong.** On the first attempt both hit the 6 h limit: /scratch was slow, so
    training read it at about 70 MB/s while the GPU sat idle.
  - **The fix.** They were rerun with `slurm/narval/review/abmil_local_round.sh`, which first
    copies the embeddings to the node's local SSD (249 GB in about 4 minutes) and then trains
    from there.
  - **Checkpointing.** `abmil_compartment_shapley.py` now saves a checkpoint after every epoch
    and resumes from it if a round is rerun. A round that is killed and resumed gives the same
    output as one that runs straight through.
  - **Same results.** The copy keeps the file names, so the patients and splits are the same.
    The first epochs' loss and c-index matched the interrupted runs exactly.
  - **Cost.** About 72 GPU-hours in all, against the 38 estimated, mostly because of the slow
    /scratch reads and the rounds that timed out.
- **Round 1 matches run M** and passes every stop condition:

  | | Baseline c-index, this run | Run M | Selected epoch (both runs) | Efficiency gap | Head check | Empty coalition |
  |---|---|---|---|---|---|---|
  | BLCA | 0.6556 | 0.6557 | 20 | 6e-17 | 7e-7 | 0.5 |
  | BRCA | 0.6541 | 0.6548 | 7 | 6e-17 | 5e-7 | 0.5 |

  All 20 rounds passed these checks.
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

| | Ours | ABMIL, last epoch (E/R input) | ABMIL, selected epoch | ABMIL mean per-round c-index, last / selected | Run M mean round baseline (selected epoch) | Never out-of-bag |
|---|---|---|---|---|---|---|
| BLCA | 0.622 | 0.605 | 0.625 | 0.594 / 0.608 | 0.608 | 1 of 379 |
| BRCA | 0.607 | 0.586 | 0.627 | 0.577 / 0.626 | 0.626 | 5 of 1,060 |

At the selected epoch, ABMIL reproduces run M (mean round c-index 0.608 and 0.626 in both).
At the last epoch, ABMIL is weaker than our model by 0.02 in each cohort. In BRCA the selected
epoch is early (7 of 20 in round 1), and by the last epoch the out-of-bag c-index has fallen,
by 0.09 in round 1 (the same in run M).

## E: what the surrogate learns

The values come from the q50 folders, as in the paper. A unit's share is its share of the
summed surrogate importance. Fidelity is the surrogate's out-of-bag correlation with the risk
score, given as the mean ± SD over bootstraps.

### Pathomic: compartments

| | Ours: ranking (share %) | ABMIL: ranking (share %) | Rank agreement, Spearman ρ (8 compartments) | Fidelity, ours | Fidelity, ABMIL |
|---|---|---|---|---|---|
| BLCA | **adipose 17.0**, urothelium 13.8, immune 13.6, muscularis 12.4, necrosis 12.2, lamina 11.5 | **muscularis 15.2**, tumour 13.4, lamina 13.2, adipose 13.0, vessels 12.0, immune 11.5 | −0.31 (p = 0.46) | 0.365 ± 0.080 | 0.302 ± 0.077 |
| BRCA | **adipose 20.8**, TILs 15.1, DCIS 14.0, tumour 11.4, stroma 11.4, TDLU 10.3 | **adipose 17.3**, TILs 15.2, TDLU 14.8, stroma 12.6, DCIS 11.3, muscle 10.3 | 0.71 (p = 0.05) | 0.460 ± 0.043 | 0.442 ± 0.047 |

### Clinical: features

| | Ours: top features (share %) | ABMIL: top features (share %) | Rank agreement, Spearman ρ | Fidelity, ours | Fidelity, ABMIL |
|---|---|---|---|---|---|
| BLCA | **age 19.3**, weight 15.8, lymph nodes examined 14.0, height 13.1, T 7.7 | **age 16.7**, height 15.8, lymph nodes examined 12.7, weight 12.5, T 7.9 | 0.92 (21 features) | 0.291 ± 0.055 | 0.208 ± 0.059 |
| BRCA | **age 34.0**, lymph nodes examined 18.3, T 9.0, positive nodes 7.3, ER 4.2 | **age 20.9**, T 15.4, lymph nodes examined 14.7, stage 7.9, positive nodes 6.8 | 0.95 (25 features) | 0.319 ± 0.035 | 0.272 ± 0.036 |

- **Clinical.** The two models' risks depend on clinical variables in nearly the same way
  (ρ = 0.92 and 0.95). Age leads in both cohorts, with a smaller share under ABMIL.
- **BRCA pathomic.** The picture agrees: adipose leads for both models, followed by TILs, and
  the rankings agree (ρ = 0.71). Adipose's share is lower under ABMIL, 17.3% against 20.8%.
- **BLCA pathomic.** The rankings are unrelated (ρ = −0.31). ABMIL's leading compartment is
  muscularis, where ours is adipose, and its shares are close to the uniform 12.5%.
- **Fidelity.** ABMIL's fidelity is lower in all four arms. The gap is 0.02 in BRCA pathomic
  and 0.05 to 0.08 in the other three arms.

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
| Clinical | q50 | 0.579 [0.525, 0.634] | 0.584 [0.530, 0.639]* |
| Clinical | q70 | 0.612 [0.551, 0.673]* | 0.593 [0.532, 0.654] |
| Clinical | q80 | 0.580 [0.509, 0.651] | 0.562 [0.492, 0.633] |
| Clinical | q90 | 0.520 [0.426, 0.614] | 0.527 [0.432, 0.621] |
| Pathomic | q50 | 0.671 [0.620, 0.722]* | 0.641 [0.588, 0.693]* |
| Pathomic | q70 | 0.651 [0.591, 0.710]* | 0.651 [0.591, 0.711]* |
| Pathomic | q80 | 0.643 [0.573, 0.713]* | 0.613 [0.542, 0.683]* |
| Pathomic | q90 | 0.693 [0.600, 0.786]* | 0.617 [0.522, 0.713] |

### Summary

| | Ours | ABMIL |
|---|---|---|
| Survive Bonferroni (of 16) | 5 | 4 |
| Hanley-McNeil interval above 0.5 (of 16) | 7 | 6 |
| Profiles recurring in ≥ 5% of runs (BLCA clinical / BLCA pathomic / BRCA clinical / BRCA pathomic) | 159 (26 / 8 / 79 / 46) | 170 (28 / 4 / 80 / 58) |
| Profiles whose in-profile c-index interval excludes the cohort value | 3 | 7 |
| Cohort value used (BLCA / BRCA) | 0.622 / 0.607 | 0.605 / 0.586 |

These are the profiles whose in-profile c-index interval excludes the cohort value:

| Model | Cohort | Arm | Threshold | Rule | n | In-profile c-index [95% CI] | Cohort value |
|---|---|---|---|---|---|---|---|
| Ours | BRCA | Clinical | q50 | stage ≤ 1.5 | 73 | 0.37 [0.07, 0.60] | 0.607 |
| Ours | BRCA | Clinical | q70 | stage ≤ 1.5 | 73 | 0.33 [0.19, 0.54] | 0.607 |
| Ours | BRCA | Clinical | q80 | age in a middle band | 138 | 0.72 [0.61, 0.82] | 0.607 |
| ABMIL | BLCA | Clinical | q90 | height above threshold | 61 | 0.74 [0.63, 0.83] | 0.605 |
| ABMIL | BRCA | Clinical | q80 | younger, T ≤ 1.5 | 50 | 0.78 [0.61, 0.90] | 0.586 |
| ABMIL | BRCA | Pathomic | q90 | 5 single-feature rules, each "above threshold" on a first-order feature with the `_std` suffix: 3 of adipose eosin (robust MAD, RMS, variance) and 2 of TIL haematoxylin (robust MAD, RMS) | 264–333 | 0.67–0.69, lower bounds 0.59–0.60 | 0.586 |

### What this shows about R

- **BLCA.** The error is not predictable for either model: no AUROC differs from 0.5 and
  nothing survives Bonferroni.
- **BRCA clinical.** The two models behave alike. The AUROCs agree within 0.02 at every
  threshold, and one threshold survives Bonferroni for each model (q70 for ours, q50 for
  ABMIL).
- **BRCA pathomic.** ABMIL's error is also predictable from pathomic features, a little less
  well than ours:
  - ABMIL's AUROCs are 0.61 to 0.65, against 0.64 to 0.69 for ours.
  - Three of four thresholds survive Bonferroni for ABMIL, against all four for ours.
  - The intervals overlap at every threshold.
- **Recurring profiles.** These look similar in number for the two models.
- **Profiles that differ from the cohort value.** ABMIL has more, 7 against 3, but five of
  its seven come from a single arm and threshold (BRCA pathomic q90). Those five are
  related rules on adipose and TIL intensity features, each covering 264–333 of the 422
  evaluation patients, so they must overlap heavily. In each, ABMIL ranks patients better
  than in the cohort as a whole.
- **Overall.** The reliability finding (BRCA pathomic error is predictable, BLCA error is
  not) holds for ABMIL as well as for our model.

## Notes

- **Bonferroni and p values.** Bonferroni uses the permutation p values. With 2,000
  permutations the smallest p is about 0.0005, below α = 0.0031.
- **Evaluation split.** It is rebuilt from ABMIL's own scored patients. It has 151 patients
  in BLCA, against 151 or 152 for ours, and 422 in BRCA, against 424 for ours.
- **The first BRCA run used only 8 rounds.** The version pushed in `b295542` had 8 BRCA
  rounds (12 patients never out-of-bag), and its BRCA results differ from the ones above:

  | BRCA, ABMIL | 8 rounds (`b295542`) | 10 rounds (above) |
  |---|---|---|
  | Out-of-bag c-index, last epoch | 0.573 | 0.586 |
  | Pathomic AUROC | 0.58 to 0.62 | 0.61 to 0.65 |
  | Tests surviving Bonferroni (of 16) | 2 | 4 |

  Two more rounds changed the BRCA pathomic AUROCs by up to 0.06. Differences of that size
  between the two models should therefore not be over-read. The 8-round outputs are kept in
  `/scratch/sorkwos/abmil_er_brca_8rounds/` and in git history.
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
