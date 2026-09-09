# Re-run plan for the revision

Consolidated from all three reviewer reports, Shirin Enger's 21 Overleaf comments,
and two defects found by reading this repo. Ordered so the cheap things that
unblock the most run first.

Rules for every job below:
- Write results to `results/<job>.json` or `.csv`, committed. No number reaches
  the paper by hand.
- Record the input run directory and round count in the output file.
- Compute goes through SLURM, never the login node.

---

## Part 0. Two defects found in the code that must be fixed before anything else

### 0a. The paper reports the wrong survival endpoint

`interpret_graphtrans_late_int_sparse_m3save.py:315-316` and the BRCA twin both use:

```python
t = float(row["PFI.time"].values[0])
e = int(row["PFI"].values[0])
```

That is **progression-free interval**, not overall survival. The paper's Datasets
section says "with overall survival annotations from the TCGA Clinical Data
Resource". Every c-index in the paper is a PFI c-index.

**Action:** decide whether to (i) correct the paper to say PFI, or (ii) re-run on
`OS`/`OS.time`. Option (i) is free. Option (ii) changes every number in the paper.
Note that PFI is the endpoint the TCGA-CDR paper actually recommends for BRCA,
so (i) is defensible and probably correct.

**Closes:** nothing on its own, but it is a factual error a reviewer will catch.

### 0b. Every censored patient is assigned error exactly 0

`compute_patient_error.py`:

```python
errors = np.zeros(N)
counts = np.zeros(N)
for i in range(N):
    for j in range(N):
        if times[i] < times[j] and events[i] == 1:   # requires events[i] == 1
            counts[i] += 1
            if risks[i] <= risks[j]:
                errors[i] += 1
valid = counts > 0
errors[valid] = errors[valid] / counts[valid]        # counts[i]==0 keeps errors[i]==0
```

`counts[i]` can only increment when `events[i] == 1`. So **every censored patient
keeps `errors[i] = 0.0`** from initialisation, meaning "perfectly predicted".

Consequences:
- The high-error label (top 20% of the error distribution) can only ever be
  assigned to patients who had an event.
- The reliability label is therefore substantially a proxy for *did this patient
  progress*, which is why stage, T stage, ER and PR status predict it so well.
- More than half of each cohort is likely censored, so the 80th percentile is
  computed over a distribution where the majority of values are exactly 0.

This is a stronger version of Reviewer 2's objection. They guessed the label was
arbitrary; it is actually correlated with outcome by construction.

**Action:** redefine the per-patient error so censored patients are included and
scored on the information they carry. Two defensible options:

1. **Symmetric pairwise concordance.** Count patient `i` in every comparable pair
   they participate in, as the earlier-event member *or* the later member. A
   censored patient at time `t` is comparable to any patient with an event before
   `t`. This keeps the concordance interpretation and gives censored patients real
   counts.
2. **IPCW Brier score** at a fixed horizon (for example median follow-up),
   which handles censoring explicitly and yields a per-patient squared error.

Recommend option 1: it stays closest to the c-index the paper already reports and
needs no new horizon choice.

**Also report:** the fraction censored per cohort, and the fraction of patients
with `counts == 0` under both old and new definitions.

**Closes:** Shirin 1:41pm (define exactly how error is calculated, account for
censoring), Reviewer 3 (unclear patient-level error definition).

---

## Part 1. Cheap, no GPU, run these first

### Job 1. Recompute per-patient error, fixed definition
Rewrite `compute_patient_error.py` per 0b, for both cohorts. Vectorise it; the
current double loop is O(N^2) in Python and BRCA is N=1060.
Output: `results/patient_error_{blca,brca}.json` with per-patient error, censoring
flag, comparable-pair count, plus cohort-level censoring fraction.
**Blocks Job 2. Do this first.**

### Job 2. Reliability re-run, corrected evaluation
Re-run all four MED3pa arms with the new error labels. Required outputs:
- **AUROC** and **balanced accuracy** for the secondary classifier, not plain accuracy
- AUROC tested against 0.50, which is the valid null for that metric
- The **majority-class baseline accuracy** stated explicitly per arm
- Per profile: patient N, high-error prevalence inside the profile, coverage
- Per profile: the **actual survival-model error** within that subgroup, with CI
- **Profile discovery and profile evaluation on disjoint patient splits.** Discover
  on a calibration set, evaluate on a test set never touched during discovery.
- Clinical thresholds **inverse-transformed out of z-scored units** into clinical
  categories, using the saved scaler
Output: `results/reliability_{arm}.json`
**Closes:** Shirin 1:41pm, 1:45pm, 1:48pm; Reviewer 1 (reliability conclusions),
Reviewer 2 (the 0.50 baseline), Reviewer 3 (Trust/Caution overinterpreted, error
definition, held-out validation).

### Job 3. Statistical test on the fusion weight spread
From the saved per-round `region_weights` in the npz files, test whether
`Δw = w_max − w_min` exceeds what the uniform baseline `1/R` would produce.
Paired test across rounds, per cohort. Also report ranking stability, meaning how
often each region occupies each rank across rounds.
Output: `results/fusion_weight_test.json`
**Closes:** Reviewer 1 (test Δw statistically), Shirin 1:36pm (report uncertainty
and ranking stability).

### Job 4. Fusion weight versus region-level RF importance
Correlate mean fusion weight `w̄_r` against the total RF importance mass of
features belonging to region `r`. Per cohort. This is the experiment Reviewer 1
handed over, and it addresses why adipose dominates the pathomic importance
(19.6% / 25.8%) while ranking 4th in BLCA and 7th in BRCA by fusion weight.
Output: `results/weight_vs_importance.json`
**Closes:** Reviewer 1 (linkage between interpretability and explainability).

### Job 5. Cohort descriptives
Exclusions and why, final N, events versus censored, median follow-up, patients
with multiple slides and how they were handled, and which patient sets fed
training, validation, explainability and reliability. Also a TCGA source-site
breakdown to check for site effects.
Output: `results/cohort_descriptives.json`
**Closes:** Shirin 1:32pm, and partly Shirin 1:18pm.

### Job 6. Clinical-only Cox baseline
Fit a standard Cox model on the clinical variables alone, same splits, same
endpoint. Report c-index with CI. This is the comparison that tells a reader
whether the WSI model adds anything.
Output: `results/clinical_cox_baseline.json`
**Closes:** Shirin 1:32pm (compare against a simple clinical Cox model).

### Job 7. c-index confidence intervals and provenance
Bootstrap CIs for both cohorts' c-index, and state explicitly whether the
reported value is training, validation, out-of-bag or held-out.
Output: `results/cindex.json`
**Closes:** Shirin 1:18pm.

---

## Part 2. Needs GPU, run overnight

### Job 8. Region ablation
For each region `r`, zero its contribution to the patient embedding and measure
the change in c-index. This is the test that decides whether the small but real
fusion-weight differences matter functionally. It is now the decisive
interpretability experiment, because the logit audit showed the weights are not
an ε artefact.
Output: `results/region_ablation.json`
**Closes:** Reviewer 3 (ablation or perturbation), Shirin 1:36pm.

### Job 9. BLCA downstream re-run on the 102-round models
BLCA's Table 1 came from the 102-round run but its explainability and reliability
results came from a 25-round run (`/scratch/sorkwos/med3pa_bootstrap/`). Re-run
the BLCA downstream analyses against the 102-round models so the paper describes
one training run. Fold this into Job 2 rather than running it separately.
**Closes:** an inconsistency not yet raised by any reviewer, which is the best
time to fix it.

### Job 10. Explainability re-run with proper importance statistics
Re-run the surrogate RF with:
- **permutation importance** alongside the existing impurity importance
- an explicit statement of which importance method the paper reports
- correction for **correlated features** (report clustered importance or note the
  correlation structure)
- **multiple-testing correction** across the 504 / 514 pathomic features
- the adipose features additionally tested **against actual survival**, not only
  against the risk score
Also drop the 500,000 resamples to something defensible. CIs converge long before
10,000, and 500,000 × 100 trees invites disbelief for no gain. Re-run at 10,000
and confirm the OOB fidelity is unchanged.
Output: `results/explainability_{cohort}_{featureset}.json`
**Closes:** Shirin 1:37pm, 1:44pm; Reviewer 2 (circularity), Reviewer 1
(explainability depth).

---

## Part 3. Needs data or people, not tonight

### Job 11. CONCH classification validation
Confusion matrix, precision, recall and F1 against pathologist-labelled patches.
Requires a labelled set. The current review covered a handful of slides
qualitatively. Also test whether the adipose signal comes partly from
tumour-containing or mixed patches, and state what happens when a tissue class is
absent from a slide.
**Closes:** Shirin 1:33pm, Reviewer 3 (insufficient CONCH validation).
**Blocker:** needs pathologist time. Start the ask now, it is the long pole.

### Job 12. External validation on CPTAC-BRCA
Derive MED3pa profiles on TCGA, evaluate on CPTAC without refitting. Needs the
cohort downloaded and the full tiling, CONCH and UNI-2 pipeline run on it.
**Closes:** all three reviewers' strongest objection.
**Blocker:** days of compute plus data access, not an overnight job. If it cannot
be done, the fallback is the locked discovery/evaluation split already specified
in Job 2.

---

## Part 4. Figures, after the numbers land

### Job 13. Regenerate Figures 3 and 4
- Figure 3: split it or simplify. Four panels with bar length, fill colour, edge
  colour and significance stars is too many encodings. Show effect sizes with
  confidence intervals instead of stars.
- Figure 4: eight panels with inset text boxes is unreadable at print size. Move
  profile details to a table and keep the figure to the accuracy-versus-coverage
  landscape. Remove every marker testing against 0.50.
- Both: increase caption font size.
**Closes:** Shirin 1:40pm, 1:48pm; Reviewer 2 (crowded figures).

---

## Decisions, locked

1. **Endpoint: PFI.** Already corrected in the paper. No re-run needed.
2. **Error metric: symmetric pairwise concordance** as primary, **IPCW Brier**
   computed alongside as extra supporting information. Both implemented in
   `analysis/patient_error.py`.
3. **Label threshold: chosen from evidence, not now.** Fixing the censoring
   defect changes the error distribution completely, so the old 80th percentile
   means something different. Step 1 of the runbook prints quantiles, a
   histogram, and an outcome-confound diagnostic at the 50th, 70th, 80th and
   90th percentiles. Pick after seeing those. Expected landing point: median
   split as primary plus a sensitivity sweep across thresholds, which turns an
   arbitrary cutoff into a reported robustness check.
4. **Title: deferred** to the end of the revision.
5. **Target: Artificial Intelligence in Medicine** (Elsevier, ScienceDirect).
   Implies four text changes, none of which block the re-runs: convert LNCS to
   `elsarticle`, add a Discussion section and shorten the Conclusion (which
   also closes Shirin's 1:49pm comment), add 3 to 5 Highlights of at most 85
   characters, and add the data availability, competing interests and CRediT
   statements.

---

## Architecture: recommendation is do not change it

Juan asked whether architecture changes are on the table. They are, but the
recommendation is **no changes to the reported model**, for one reason: "a
learnable global fusion weight barely differentiates between tissue regions" is
a legitimate finding, and tuning the architecture until the weights look more
interesting, then reporting that, is engineering the result. A reviewer will ask
why the parameterisation changed and the answer would be unflattering.

What the logit audit established, so this is not speculation:
- `region_weights` is initialised to `torch.ones(R)`, so the model starts at
  exactly `1/R`.
- Across 49,140 logged readings no logit ever went negative or approached
  ε = 0.01. The smallest was 0.76, roughly 76× the floor.
- So the near-uniform weights are **not** an ε artefact. The parameter simply
  stays in a tight band around its initialisation, and sum-to-one normalisation
  compresses that band to within about 1.5 pp of uniform.

Instead of changing the model, run two cheap diagnostic arms as a robustness
check (Step 5b in the runbook), 20 rounds each:

1. **Initialise `region_weights` at 0.0** instead of 1.0. If the weights still
   land near uniform, "stuck at initialisation" is ruled out and the flatness is
   a property of the data. If they differentiate more, then initialisation was
   the constraint, which is a real and reportable methodological finding.
2. **Softmax fusion** instead of `ReLU + ε` then normalise. Tests whether the
   flatness depends on the parameterisation at all.

Both are reported as supplementary robustness checks, with the original
architecture remaining the paper's model. That is defensible, cheap, and
converts a weakness into a characterisation of the fusion layer.

The one architecture change worth *proposing as future work* rather than
running: make the fusion weights **patient-specific** (a gating network over
region embeddings) instead of a single global parameter. That would give
per-patient region attributions, which is a genuinely stronger interpretability
claim. But it is a different model and a different paper, so it belongs in the
Discussion, not in this revision.
