# Handoff: data for a one-patient case study, on Narval (2026-11-19)

The paper will get a short section between Results and Discussion, "One patient through
the audit": a half-page figure that follows one held-out TCGA patient through the three
checks. It shows the slide and its tissue labels, the model's risk, the pathomic features,
the reliability profiles, and which outputs passed or failed the audit. The figure is
built on the laptop; this handoff only exports the numbers and images it needs.

**CPU only. No training, no GPU, no rerun of MED3pa.** Read the saved outputs of the
existing runs. If something can only be had by rerunning, stop and tell Juan.

## Candidates

The 10 slides the pathologists reviewed (the patient ID is the first 12 characters):

- BLCA: TCGA-ZF-AA53, TCGA-FT-A3EE, TCGA-2F-A9KT, TCGA-K4-A5RI, TCGA-ZF-A9RF
- BRCA: TCGA-B6-A0X1, TCGA-B6-A0I9, TCGA-B6-A0RG, TCGA-D8-A1JL, TCGA-LL-A8F5

Use our model's runs only, not ABMIL's.

## 1. Held-out membership

Each of the 16 reliability runs (`results/reliability_{cohort}_{arm}_q{NN}/`) split its
patients into discovery and evaluation sets (`discovery_eval_split` in the JSON gives the
seed and fraction). For each candidate and each run, record whether the patient is in the
evaluation set. Rebuild the split exactly as the arm scripts
(`analysis/{blca,brca}_{clinical,radiomic}_linear_med3pa.py`) do, or read it from the
saved per-run files (`*_patient_predictions.csv`, `med3pa_backbone_error_labels*.csv`)
if they are still on scratch. Check the rebuilt split against the saved `n_eval`.

## 2. Per-patient numbers

For each candidate:

- **Risk.** Out-of-bag risk score (the file the reliability runs used, `input_error_npz`
  in each reliability JSON), and its percentile in the cohort.
- **Outcome.** PFI time and event.
- **Error.** Patient error (share of comparable pairs ordered wrongly), the number of
  comparable pairs, and the high-error label at q50, q70, q80 and q90.
- **Tissue.** The share of the patient's tiles in each compartment (CONCH labels).
- **Attribution.** Each compartment's share of this patient's risk score, averaged over
  the rounds in which the patient was out-of-bag, from the per-round local Shapley files
  (`round_###_local.npz` written by `analysis/region_shapley.py`), if they exist.
  If they do not, say so; do not recompute.
- **Pathomic features.** The values of the 10 leading features by surrogate importance at
  q50 (`results/reliability_{cohort}_radiomic_q50/radiomic_importance.csv`), raw and as
  a percentile in the cohort.
- **Clinical variables.** Those used by any profile rule (stage, T, N, age, receptor
  status, etc.).

## 3. Profiles the patient falls into

For each run in which the candidate is in the evaluation set, list the profiles recurring
in at least 5% of MED3pa runs (`top_profiles_eval` with `run_freq_pct >= 5`) whose rule
the patient meets. Apply each rule exactly as the script does, including any scaling of
the features (`scaler_note`). For each matching profile, record the readable rule,
`n_patients`, the high-error prevalence, the in-profile C-index with its 95% interval,
the cohort value (0.622 BLCA, 0.607 BRCA), and whether the interval excludes it. If the
error classifier's predicted probability for the patient was saved, record it too.

## 4. Images

If the candidates' WSIs are reachable from Narval:

- a thumbnail of the slide, about 1,500 px on the long side, as PNG;
- the CONCH tissue map at the same size, with the same colours as the pathologist
  questionnaire;
- three example tiles (256 px) from each of the patient's two compartments with the
  largest attribution, or the largest tile share if attribution is unavailable.

If the slides are not reachable, say so; the laptop has the questionnaire thumbnails.

## 5. Report and commit

- One JSON per candidate in `results/case_study/{patient_id}.json`, and the images in
  `paper_figures/case_study/{patient_id}/`. These ten public TCGA cases are the one
  exception to the rule against committing patient-level files; commit nothing else
  patient-level.
- `docs/CASE_STUDY_DATA.md`: one table with a row per candidate (cohort, the runs in
  which the patient is held out, risk percentile, error, high-error label at q50, the
  number of matching profiles, and whether any of them differs from the cohort). Then
  recommend one patient, preferring a BRCA patient who is held out in the pathomic runs,
  falls into at least one recurring profile, and has images. Say why.
- Commit and push.
