# Handoff: explainability (E) and reliability (R) checks on ABMIL, on Narval (2026-11-12)

The co-author asked us to support the claim that the audit transfers to other models
by running the E and R checks on ABMIL. Both need per-patient out-of-bag (OOB) risk
scores, which no ABMIL run saved (`docs/SHIRIN_CLUSTER_RESULTS.md`, task 4). So:
save them in a rerun of ABMIL, then run E and R exactly as for our model.

Juan has approved the 20 GPU jobs in step 2 and the CPU jobs in step 5. Ask before
anything else that uses a GPU, and never commit patient-level files (`*_local.npz`,
per-patient tables, the risk and error npz files).

## 1. Save per-patient risks in `analysis/abmil_compartment_shapley.py`

After each round's coalition table is built, also write `round_###_local.npz` beside
`round_###.json`, as `analysis/region_shapley.py` does: `patient_ids` (same
`case_id` helper), `times`, `events` and `risk_full`, the full-coalition risk per
out-of-bag patient at the selected epoch (plus `risk_full_last` when the last epoch is
also scored). Change nothing else, so the round JSONs stay comparable with run M
(`results/abmil_v2_{blca,brca}/`). Test on synthetic data, then commit.

## 2. Rerun ABMIL, 10 rounds per cohort, one round per job (20 GPU jobs)

- Fresh folders `/scratch/sorkwos/abmil_v3_{blca,brca}`; same settings as run M
  (`EXP=M` through `slurm/narval/review/review_round.sh`, with `ROUNDS_DIR` set to the
  new folder and `ROUND_START=ROUND_END=r`).
- Submit all 20 at once; their wall-clock is then about one round. Narval's recent
  ABMIL rounds (run N) took 74 min on average for BLCA (max 134) and 155 min for BRCA
  (max 216), so ask for 4 h (BLCA) and 6 h (BRCA), not the launcher's 12 h and 16 h.
  Total about 38 GPU-hours.
- Check round 1 before trusting the rest: the usual stop conditions (baseline > 0.5,
  efficiency gap < 1e-9, head check < 1e-3, empty coalition = 0.5), and the baseline
  should be near run M's round 1 (0.656 BLCA, 0.655 BRCA; small differences between
  clusters are expected).

## 3. ABMIL out-of-bag risk per patient

Average each patient's `risk_full` over the rounds in which the patient is out-of-bag,
the way `analysis/oob_risk.py` builds the main model's risk. Report the resulting
out-of-bag concordance per cohort (run M's mean round baselines were 0.608 BLCA and
0.626 BRCA) and how many patients are never out-of-bag (expect a handful).

## 4. Patient error for ABMIL

Run `analysis/patient_error.py` on the ABMIL out-of-bag risks, giving
`patient_error_{cohort}_abmil_oob.npz` (keep it on scratch or in `/home/sorkwos/data/`,
not in git).

## 5. E and R on ABMIL (CPU, 16 runs)

Run the four arm scripts `analysis/{blca,brca}_{clinical,radiomic}_linear_med3pa.py`
at the four thresholds (q50, q70, q80, q90), with the same settings as the main runs
and `--error-npz` pointing to the ABMIL file, writing to
`results/reliability_abmil_{cohort}_{arm}_q{NN}/`. First read the scripts and make sure
every input that comes from the model (the surrogate's target as well as the error
labels) is ABMIL's; if anything still reads the main model's saved rounds
(`--save-root`, `--round-count`), point it at ABMIL or tell Juan.

## 6. Report

Commit the summaries (not patient-level files) and write `docs/ABMIL_ER_RESULTS.md`,
putting each ABMIL number next to our model's:

- ABMIL out-of-bag concordance per cohort.
- E: compartment ranking by surrogate importance and the leading compartment's share,
  for clinical and pathomic surrogates; out-of-bag fidelity of the surrogate.
- R: AUROC per arm and threshold with Hanley-McNeil 95% intervals
  (`visualization/figure_data.hanley_mcneil_ci`), how many survive Bonferroni over the
  16 tests, the number of profiles recurring in at least 5% of runs, and how many have
  an in-profile c-index interval excluding the cohort value.
