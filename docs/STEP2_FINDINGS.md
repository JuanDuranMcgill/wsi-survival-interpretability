# Step 2 findings: the corrected reliability analysis

All 16 arms (2 cohorts x 2 feature sets x 4 label thresholds) completed. Every
number below is from `results/reliability_*/`, computed on the evaluation half
of a locked patient split that profile discovery never saw.

Three corrections separate this from the original analysis:
1. Per-patient error is computed from **out-of-bag** predictions, not the
   in-sample-contaminated `mean_risk` whose c-index was 0.90.
2. Error is scored over **every comparable pair a patient participates in**, so
   censored patients receive real counts instead of an automatic zero.
3. Profiles are **discovered on 60% and evaluated on the held-out 40%**.

---

## 1. The secondary classifier is weak, and works in only one cohort

AUROC on the evaluation set, with a 2000-fold permutation test against 0.50:

| Arm | q50 | q70 | q80 | q90 |
|---|---|---|---|---|
| BLCA clinical | 0.588 | 0.523 | 0.455 | 0.580 |
| BLCA radiomic | 0.547 | 0.439 | 0.561 | 0.476 |
| BRCA clinical | **0.579** | **0.612** | **0.580** | 0.520 |
| BRCA radiomic | **0.671** | **0.651** | **0.643** | **0.693** |

Bold marks p < 0.05. **7 of 16 arms are significant, and all seven are BRCA.
No BLCA arm reaches significance at any threshold.** Four arms have AUROC below
0.50, meaning the classifier is worse than chance on held-out patients.

Where the signal is real it is modest: AUROC 0.58 to 0.69. The BRCA radiomic arm
is the strongest and is significant at every threshold, with average precision
roughly double its baseline (0.189 against 0.097 at q90).

## 2. Accuracy was misleading in both directions

The original analysis reported accuracy against a 0.50 null. The corrected
output shows why that fails, and it fails twice over:

| Arm | plain accuracy | majority-class baseline | AUROC |
|---|---|---|---|
| BLCA clinical q90 | 0.815 | 0.901 | 0.580 (ns) |
| BRCA radiomic q90 | 0.854 | 0.903 | 0.693 (p<0.001) |

Both accuracies look high and both are **below** the trivial baseline. Yet one
arm has no signal and the other has the strongest signal in the study. Accuracy
against 0.50 inflated the weak arms and simultaneously hid the real one.

## 3. The central claim does not survive: profiles do not stratify survival performance

This is the number that matters. For each profile we computed the survival
model's actual c-index within that subgroup, with a bootstrap 95% CI, and asked
whether it differs from the cohort-wide value (0.6215 BLCA, 0.6067 BRCA).

- **159 profiles examined across all 16 arms.**
- **3 have a 95% CI that excludes the cohort-wide c-index.**
- **About 8 would be expected by chance** at alpha = 0.05 with no multiple-testing
  correction.

Fewer than chance. And two of the three are the same rule
(`pathologic_stage <= 1.5`, N=73) appearing at two thresholds, so there are only
two distinct candidates out of 159.

In-profile c-index spans per arm are narrow and centred on the cohort value. The
widest, BRCA clinical q50, runs 0.368 to 0.721, but every constituent CI is wide
because subgroup event counts are small.

**Conclusion: even in the arms where the error label is predictable, that does
not translate into subgroups where the survival model measurably performs better
or worse.** Predicting a derived error label and identifying subgroups of
differential model reliability are not the same thing, and here only the first
happens, weakly, in one cohort.

## 4. Surrogate fidelity drops when explaining honest predictions

Out-of-bag Spearman fidelity of the Random Forest surrogate:

| Arm | original (in-sample target) | corrected (OOB target) |
|---|---|---|
| BLCA clinical | 0.444 | 0.291 |
| BRCA clinical | 0.378 | 0.319 |
| BLCA radiomic | — | 0.358 to 0.436 |
| BRCA radiomic | — | 0.450 to 0.472 |

Expected: an in-sample risk score is a smoother, more learnable target than an
out-of-bag one. It also means the explainability section's reach was overstated,
since it was explaining a partly memorised signal.

---

## What this means for the paper

The reliability section's original claims cannot be restated. Trust profiles at
0.92 to 0.96 accuracy were majority-class baselines plus in-sample
contamination, and the corrected analysis finds no survival-performance
stratification at all.

There is a stronger paper available in this. The three defects we found are not
specific to this pipeline; each is an easy mistake with a specific signature:

1. **In-sample contamination.** A bootstrap pipeline that saves predictions for
   the full cohort at every round yields a per-patient risk vector that looks
   excellent (c-index 0.90 to 0.99) while the honest out-of-bag value is 0.61.
   Detectable by comparing the two, which the bootstrap design makes free.
2. **Censoring-blind error definitions.** Scoring a patient only over pairs
   where they are the earlier-event member assigns zero error to every censored
   patient, so a top-quantile label becomes reachable only by patients who had
   an event. Detectable by checking what fraction of high-error patients had an
   event against the cohort base rate: it was 0.667 against 0.430 before the
   fix and 0.307 after.
3. **Accuracy under class imbalance.** A top-quintile label makes a degenerate
   classifier 0.80 accurate. Detectable by always reporting the majority-class
   baseline alongside accuracy.

Reframing around those three pitfalls, with the corrected analysis as the worked
example and the diagnostics as the contribution, makes the negative result the
point rather than a retreat. That is a methods paper for a field currently
publishing a lot of patient-level reliability claims, and it is more original
than the original framing.

## Open items

- **No multiple-testing correction across the 159 profiles.** Should be added
  before any profile is discussed individually.
- **Top-K feature selection in the radiomic arms** may run before the
  discovery/evaluation split. If so the evaluation set leaked into feature
  selection and those four arms need re-running. Unconfirmed.
- **BRCA q90 has only ~41 high-error patients in the evaluation half**
  (0.097 x 424). The strongest AUROC in the study rests on that, so it needs a
  stability check.
- **Region ablation and the fusion-parameterisation arms (Step 5) have not
  run.** Those are what the interpretability section still needs.
