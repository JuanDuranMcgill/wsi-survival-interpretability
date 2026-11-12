# Handoff: co-author's analyses that need the cluster (2026-11-12)

The paper's text has been revised for a co-author's review. Four of her requests
need cluster files. This note says what to compute, where the inputs probably
are, and what to write back. Do them in the order below. Each is independent.

## Rules

- Read-only on existing results. Write new outputs to new files only; never
  overwrite anything under `results/` that already exists.
- Never commit patient-level files (`*_local.npz`, `planted_feature.npz`,
  `synthetic_labels.npz`, per-patient tables). Commit summaries (JSON/CSV of
  aggregates) and any new scripts.
- CPU work is fine to run. Ask Juan before submitting any GPU job, or more than
  about 20 jobs of any kind.
- If an input is missing, or is on the other cluster (Narval), say so and stop
  that task; do not regenerate it.
- When done, commit and push to `main`, and add a short report to
  `docs/SHIRIN_CLUSTER_RESULTS.md` with the numbers asked for below.

## 1. Patient-level bootstrap of the Shapley correlations (most important)

The paper's central interpretability contrast is a Spearman correlation across
only 8 (BLCA) or 9 (BRCA) compartments. Its current intervals come from
resampling training rounds, not patients. She asks for a patient-level
bootstrap. No retraining is needed.

- Input: the main Shapley run, 20 rounds per cohort,
  `round_###_local.npz` beside `round_###.json` (the JSONs are in
  `results/shapley_{blca,brca}/`; the npz files are on scratch, written by
  `analysis/region_shapley.py`, which saves `patient_ids`, `times`, `events`,
  `coalition_risks` (2^R x n out-of-bag patients) and `phi_local`).
- Method: B = 1000 replicates per cohort. In each replicate, resample the
  cohort's patient IDs with replacement. In every round, keep the resampled
  patients who are out-of-bag in that round, with their multiplicity (as
  weights), recompute the concordance of every coalition, then the exact
  Shapley values (`shapley_from_table`), and the per-patient local shares from
  `phi_local` as `region_shapley.py` does. Average over rounds as the aggregate
  does, then compute the three Spearman correlations over compartments:
  - fusion weight vs outcome-based Shapley (paper: BLCA +0.81, BRCA -0.60);
  - what the model uses (`local_share_mean`) vs outcome-based Shapley
    (paper: BLCA +0.88, BRCA -0.27) - this is the headline contrast;
  - fusion weight vs what the model uses (paper: BLCA +0.95, BRCA +0.73).
  Fusion weights are model parameters and do not change under the bootstrap.
  Hint: with multiplicity weights w, a coalition's concordance is
  (C_k w) . w / (w . K w), where K is the comparable-pair matrix and C_k the
  concordant-pair matrix (ties 0.5) of coalition k; precompute per round.
- Output: `results/patient_bootstrap_shapley_{blca,brca}.json` with, for each
  correlation, the point estimate (reproduce the paper's value first as a
  check), the bootstrap median and the 95% percentile interval; plus the
  BLCA minus BRCA difference for the headline contrast with its 95% interval
  (pair replicate b of BLCA with replicate b of BRCA). Commit the script as
  `analysis/patient_bootstrap_shapley.py`.

## 2. Comparable pairs per patient, by censoring status

The reliability check's per-patient error is the discordance rate over the
patient's comparable pairs (a pair is comparable when the patient with the
shorter time had an event; a patient is charged whether earlier or later).
She asks how many pairs patients have, especially censored ones.

- Input: the out-of-bag risk and survival data used by
  `analysis/patient_error.py` (summaries in `results/patient_error_{cohort}_oob.json`;
  the MED3pa runs read `patient_error_{cohort}_oob.npz`).
- Output: `results/comparable_pairs_{blca,brca}.json` with, separately for
  patients with an event and censored patients, and for all patients: n,
  median, IQR, min, max, and the number with 0 and with fewer than 10
  comparable pairs. Repeat on the MED3pa evaluation split if it can be
  rebuilt exactly (60/40 stratified on the median-threshold label, seed
  20260908; check that it gives 151 and 424 evaluation patients).

## 3. Scan magnification by cohort and site

Tiles were cut at each slide's native resolution with no resampling, so a tile
covers a different physical area at 20x and 40x. She asks for the
magnification mix per cohort and per site, and its place in the site analysis.

- Find the slide files or their metadata (`openslide.objective-power`,
  `aperio.AppMag`, MPP) for every slide used (379 BLCA, 1,060 BRCA). Site is
  the TCGA tissue-source-site code (characters 6-7 of the barcode).
- Output: `results/magnification_{blca,brca}.json`: counts by magnification
  for the cohort and per site, and the number of slides with no recorded
  value.
- Then extend the site analysis (the partial R-squared of site in the
  out-of-bag risk score adjusted for stage, T and N, plus ER and PR in BRCA;
  see `analysis/cohort_and_baseline.py` or `analysis/review_checks.py`) with
  magnification: the partial R-squared of magnification with the same
  adjustment, and the partial R-squared of site after also adjusting for
  magnification. Paper values to reproduce first: site explains 15% (BLCA)
  and 14% (BRCA).
- Do not rerun anything with resampled tiles; that is out of scope.

## 4. Explainability and reliability checks on ABMIL: first check, then report

She asks to run the surrogate (E) and MED3pa (R) checks on the ABMIL model,
which need only per-patient out-of-bag risk scores. As far as we can tell from
the code, `analysis/abmil_compartment_shapley.py` does not save per-patient
risks (no `_local.npz`), so they may not exist.

- Search scratch for any saved ABMIL per-patient out-of-bag risks (any run:
  `abmil_*`, `abmil_v2_*`, planted runs' `abmil/`).
- Report what exists. If nothing usable exists, estimate what a rerun would
  take: saving per-patient risks needs a small change to the script and 10
  GPU rounds per cohort (about 40 min BLCA and 85 min BRCA per round on
  Trillium). Do not start it; Juan decides.

## Report back

In `docs/SHIRIN_CLUSTER_RESULTS.md`: for task 1 the three correlations per
cohort with intervals and the difference interval; for task 2 the median and
IQR of pairs for event and censored patients and the counts with 0 and fewer
than 10; for task 3 the magnification mix per cohort, how many sites mix
magnifications, and the partial R-squared values; for task 4 what exists and
the estimate.
