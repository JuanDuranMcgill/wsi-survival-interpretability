# Runbook: overnight re-run

Follow in order. Parts 1 and 2 are cheap and gate everything else, so do not
skip ahead. Every step writes to `results/` and gets committed.

Decisions already locked by Juan:
- **Endpoint: PFI.** Already corrected in the paper. Do not switch to OS.
- **Primary error metric: symmetric pairwise concordance.** IPCW Brier is
  computed too, as extra supporting information, not as the label.
- **Label threshold: not yet chosen.** Step 1 prints the distribution; the
  threshold gets picked from that evidence. Do not assume 0.8.
- **Title: deferred.** Do not touch it.
- **Target: Artificial Intelligence in Medicine** (Elsevier). Affects text
  structure only, not these runs.

---

## Step 0. Pull and check the environment

```bash
cd <repo>
git pull
python -c "import numpy, pandas, sklearn; print(numpy.__version__, sklearn.__version__)"
```

New code added for this revision, all tested locally:
- `analysis/patient_error.py` — replaces `compute_patient_error.py`
- `analysis/profile_metrics.py` — imbalance-aware metrics, import this
- `analysis/fusion_weight_stats.py` — Δw and consistency tests

Note `numpy>=2` removed `ndarray.ptp()`. The new code uses `np.ptp(arr)`, which
works on both. If you touch other scripts, watch for the same.

---

## Step 1. Recompute per-patient error  (CPU, minutes)

This is the blocking fix. The old `compute_patient_error.py` gave every
censored patient `error = 0.0`, because `counts[i]` only incremented when
`events[i] == 1`. So only patients with an event could ever be labelled
high-error, making the label a partial proxy for the outcome.

Set these three per cluster first. The values below are Narval; see
`envs/narval.md`. On Trillium the CDR file was under a UUID subdirectory of
`/home/sorkwos/` and the npz files were under `/scratch/sorkwos/`.

```bash
export CDR_XLSX=/home/sorkwos/TCGA-CDR-SupplementalTableS1.xlsx
export BLCA_NPZ=$HOME/data/blca/final_med3pa_input.npz
export BRCA_NPZ=$HOME/data/brca/final_med3pa_input.npz

python analysis/patient_error.py --cohort blca \
  --final-npz "$BLCA_NPZ" \
  --cdr-xlsx  "$CDR_XLSX" \
  --out results/patient_error_blca.json

python analysis/patient_error.py --cohort brca \
  --final-npz "$BRCA_NPZ" \
  --cdr-xlsx  "$CDR_XLSX" \
  --out results/patient_error_brca.json
```

Verify the paths first; adjust if the BRCA final npz lives elsewhere.

**Then STOP and report these four numbers per cohort, before running Step 2:**

1. `global_cindex` — must match the paper's 0.600 (BLCA) / 0.632 (BRCA). If it
   does not, the risk vector or the alignment is wrong and nothing downstream
   is valid.
2. `censored_fraction`
3. `frac_exactly_zero_error` — should now be small. Under the old definition it
   was roughly the censored fraction.
4. For each of `label_at_q50/70/80/90`: `prevalence`,
   `majority_class_accuracy`, and especially
   `frac_of_high_error_that_had_event`.

That last number is the diagnostic. Compare it to the cohort event rate. If it
is near 1.0 the label is still an outcome proxy and we must rethink. If it sits
near the base rate, the confound is fixed. **The threshold gets chosen from
these numbers, so do not proceed until Juan has seen them.**

---

## Step 2. Reliability re-run  (CPU, hours)

Modify the four `analysis/*_med3pa.py` drivers. Do not rewrite them; make these
targeted changes.

**2a. Point them at the new error file.** They currently read
`med3pa_error_input.npz`. Step 1 writes a drop-in replacement with the same
`patient_ids` / `error` keys at `results/patient_error_<cohort>.npz`.

**2b. Use the locked discovery/evaluation split.**

```python
from analysis.profile_metrics import discovery_eval_split
disc, ev = discovery_eval_split(patient_ids, frac_discovery=0.6, stratify=y_med3pa)
```

Discover profiles on `disc` only. Evaluate every reported number on `ev`, which
discovery never sees. This is the single most important change: all three
reviewers raised it and it is what makes the numbers defensible without CPTAC.

**2c. Replace the metrics.** Delete every `t`-test against 0.50. Use:

```python
from analysis.profile_metrics import classifier_metrics, profile_report
cm = classifier_metrics(y_true_eval, y_score_eval)          # AUROC, balanced acc, majority baseline
pr = profile_report(mask_eval, y_true_eval, patient_error_eval,
                    times_eval, events_eval, risks_eval)     # N, prevalence, real c-index + CI
```

`classifier_metrics` gives AUROC with a permutation p-value against 0.50, which
is the valid null for AUROC. `profile_report` gives the per-profile patient N,
the high-error prevalence inside the profile, and the **actual survival-model
c-index in that subgroup with a bootstrap CI**, which the paper never reported
and which is the number a clinician would actually want.

**2d. Threshold sensitivity.** The drivers already take
`--med3pa-error-quantile`. Run the chosen threshold plus three neighbours:

```bash
for q in 0.5 0.7 0.8 0.9; do
  python analysis/blca_clinical_linear_med3pa.py --med3pa-error-quantile $q \
    --outdir results/reliability_blca_clinical_q$q
done
```

Repeat for the other three arms. This converts an arbitrary cutoff into a
reported robustness check, which answers the "it was a bit random" objection
directly.

**2e. Inverse-transform the clinical thresholds.** Profile rules currently print
in z-scored units, which is why "stage ≤ 3.5" and "T stage ≤ 1.5" read as tree
artefacts. Save the `StandardScaler` and map thresholds back to clinical
categories before writing them out.

**2f. Fold in the BLCA consistency fix.** BLCA's Table 1 came from the
102-round run but its downstream analyses read the 25-round
`/scratch/sorkwos/med3pa_bootstrap/`. Point the BLCA drivers at the 102-round
`med3pa_bootstrap_intermediate/` so the paper describes one training run. Report
both if they differ materially.

Output one JSON per arm per threshold. Include the input directory, round count,
split seed, and N in every file.

---

## Step 3. Fusion weight statistics  (CPU, minutes)

```bash
# Narval, after rsyncing the round trees from Trillium:
python analysis/fusion_weight_stats.py --cohort blca \
  --npz-glob "$HOME/data/blca/rounds/job_*/round_*/epoch_*.npz" \
  --out results/fusion_weight_test_blca.json

python analysis/fusion_weight_stats.py --cohort brca \
  --npz-glob "$HOME/data/brca/rounds/job_*/round_*/epoch_*.npz" \
  --out results/fusion_weight_test_brca.json
```

Run with `--pick last` and `--pick best` and report both. Which epoch per round
fed Table 1 is an unstated methodological choice, and we need to state it.

This also settles the BRCA Table 1 discrepancy: the log summaries covered only
the 52 top-up rounds and disagreed with the paper by up to 0.69 pp on
necrosis/haemorrhage. The npz files cover all rounds, so this is the
authoritative recomputation. **Report the per-region means and N.**

The permutation test asks whether the *same* regions are consistently weighted
higher across independent rounds. Validated on synthetic data: p ≈ 0.0005 with
real signal, p ≈ 0.67 on pure noise.

---

## Step 4. Cheap analyses  (CPU, minutes each)

Write these; they are small and self-contained.

**4a. Fusion weight versus region-level RF importance.** Reviewer 1's exact ask.
Correlate mean fusion weight `w̄_r` against the summed RF importance of features
belonging to region `r`. The pathomic feature names carry the region prefix
(`Adip·H`, `TIL·E`, `Imm·H`), so group on that. Report Spearman and Pearson per
cohort. This addresses why adipose dominates pathomic importance (19.6% /
25.8%) while ranking 4th in BLCA and 7th in BRCA by fusion weight.
→ `results/weight_vs_importance.json`

**4b. Cohort descriptives.** Exclusions and why, final N, events versus
censored, median follow-up, patients with multiple slides and how they were
handled, which patient sets fed training / validation / explainability /
reliability, and a TCGA source-site breakdown for the site-effect question.
→ `results/cohort_descriptives.json`

**4c. Clinical-only Cox baseline.** Same splits, same PFI endpoint, clinical
variables only. c-index with CI. This is what tells a reader whether the WSI
model adds anything over clinical data alone.
→ `results/clinical_cox_baseline.json`

**4d. c-index confidence intervals.** Bootstrap CIs for both cohorts, and state
explicitly whether the reported value is training, validation, out-of-bag or
held-out.
→ `results/cindex.json`

---

## Step 5. GPU jobs  (overnight, SLURM)

**5a. Region ablation.** For each region r, zero its contribution to the patient
embedding at inference and measure the change in c-index. Use the saved
checkpoints; no retraining needed. This is now the decisive interpretability
experiment, because the logit audit ruled out the ε-floor explanation.
→ `results/region_ablation.json`

**5b. Fusion-layer parameterisation check** (see the architecture note in
`RERUN_PLAN.md`). Two extra training arms, 20 rounds each is enough:
- `region_weights` initialised at **0.0** instead of 1.0
- **softmax** fusion instead of `ReLU + ε` then normalise

Both test whether the near-uniformity is a property of the parameterisation
rather than of the data. Report the resulting weight spread against the
observed 1.5 pp (BLCA) and 0.9 pp (BRCA).
→ `results/fusion_parameterisation_check.json`

**5c. Explainability re-run.** Add permutation importance alongside impurity
importance, state which one the paper reports, correct for multiple testing
across the 504 / 514 pathomic features, handle correlated features, and test the
adipose features against actual PFI survival rather than only against the risk
score. Also drop 500,000 resamples to 10,000 and confirm the OOB fidelity is
unchanged; CIs converge long before 10,000 and the large number invites
disbelief for no gain.
→ `results/explainability_<cohort>_<featureset>.json`

---

## Step 6. Report back

Commit everything under `results/` and post the JSON contents. Do not edit the
paper; the numbers get pulled into LaTeX from these files.

Priority if time runs short: Steps 1, 2 and 3. Those close the reviewers' core
objection and the biggest interpretability question. Steps 4 and 5 are
important but survivable as follow-ups.

---

## Cluster portability

**Every path in this runbook was written for Trillium.** Trillium and Narval have
separate `/home`, `/scratch` and `/project`, so nothing transfers implicitly.
Resolve paths per cluster before running; see `envs/narval.md` for the Narval
locations. Set `CDR_XLSX` to wherever the TCGA-CDR spreadsheet actually is.

Move data between clusters with rsync over ssh rather than through git. The npz
files pair TCGA barcodes with model risk scores, which is patient-level data and
is excluded by `.gitignore` on purpose:

```bash
rsync -avP sorkwos@trillium.scinet.utoronto.ca:<trillium-path> ~/data/<cohort>/
```

**Version skew to watch.** The three new analysis modules were tested against
numpy 2.0 / pandas 2.3 / sklearn 1.6. Narval's venv has numpy 2.4 / pandas 3.0 /
sklearn 1.8. pandas 3.0 is a major release with behaviour changes, so if
`patient_error.py` throws on the `read_excel` or `.at[]` lookups, that is the
likely cause and not a logic error. Report the traceback rather than rewriting
the metric.
