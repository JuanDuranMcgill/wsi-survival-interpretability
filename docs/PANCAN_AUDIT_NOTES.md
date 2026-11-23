# Pan-carcinoma audit: what the dry runs taught us (2026-11-23)

Code: `analysis/pancan/` (plan `docs/superpowers/plans/2026-11-23-pancan-training-audit.md`). Everything below
was run on the 920 slides finished on 2026-11-23, on the Beast's CPU, while the preprocessing run used the GPU.
These are checks of the pipeline, not results: the cohort was incomplete and the runs were short.

## Check I depends on the number of compartments

The faithful test asks for an exact one-sided permutation p below 0.05 over the compartments. With the main labels
there are 4 compartments, so the smallest possible p is 1/24 = 0.042: only a perfect ranking of fusion weights
against risk-score Shapley shares passes. With the stroma-kept labels (5 compartments), a Spearman rho of 0.9 or
more passes (p = 5/120). This follows from the rule, not from data.

## Check R passes on a model with no skill

On the dry-run model (one round, 2 epochs, 282 out-of-bag patients, within-type C 0.504), "errors are predictable"
passed: AUROC 0.738 at q50 and 0.730 at q90, permutation p 0.0005. With the risk scores shuffled within each
cancer type (a model with no skill), the error classifier reached a mean AUROC of 0.674 (q50) and 0.695 (q90), and
60% (q50) and 40% (q90) of 10 shuffles passed. "Differs" passed in 0 of 2 (q50) and 1 of 2 (q90) shuffles. A
patient's error depends on how many comparable pairs the patient has, which follows cancer type and follow-up
time. `check_r.py` therefore reports this no-skill reference next to every verdict; the pass rules are unchanged
(Juan, 2026-11-23).

## The planted test cannot single out a compartment whose signal the others share

Planted outcome as in the paper (projection of each patient's mean stroma embedding on a random direction, oracle
within-type C 0.755), our model, stroma-kept labels, 5 hospital-held-out rounds of 30 epochs: the model learned it
(out-of-bag within-type C 0.639), but stroma ranked 4th by fusion weight and 3rd by Shapley value and local share
(Shapley +4.3 points, against tumour +5.0 and immune +4.4). Fusion weights stayed near 20% each. The same
projection of the other compartments correlates 0.58 to 0.68 with stroma's, so the other compartments really
carry the planted signal.

How much of a patient's stroma embedding is its own (cross-validated ridge R2 of the planted signal):

| Planted signal | From the other compartments | From stroma itself |
|---|---|---|
| Paper's (random direction) | 0.36 | 0.56 |
| Stroma-only, strongest direction after removing 10 PCs of each other compartment | 0.67 | 0.63 |
| Stroma-only, after removing 50 PCs (`planted.py make ... --specific`) | -0.02 | 0.16 |

A signal that lives only in stroma is barely learnable from 920 patients. The stroma-only planted test is ready
(`/data/processed/juan/pancan/planted/stroma_stroma_c75_specific.csv` on 920 slides; within-type oracle C 0.745)
and runs on the full cohort after the preprocessing ends, with its learnability checked first. That a slide's
compartments share most of their embedding variation is itself a finding about compartment-level attribution.
