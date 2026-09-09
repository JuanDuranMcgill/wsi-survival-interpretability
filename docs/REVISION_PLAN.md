# Revision plan: minimal-change path to the AIM submission

Decided after reading all 16 Step 2 arms. The paper stays an **interpretability
paper** with the IER audit framing. Reviewer 1 already endorsed that framing
("This paper's strongest contribution is the proposed trustworthiness audit
framework"), so we keep it rather than rebuilding the paper's identity.

Only the **R** leg changed materially. **I** is stronger than before and **E**
needs wording, not restructuring.

---

## Section-by-section scope

| Section | Change | Status |
|---|---|---|
| Title, framing | none | done |
| Introduction | none beyond the operational definitions already pushed | done |
| Methods 2.1 Datasets | PFI endpoint corrected | done |
| Methods 2.2 CONCH | prompts and no-threshold statement added | done |
| Methods 2.3 Graph model | relative-not-absolute weights, ε floor | done |
| Methods 2.4 Interpretability | uniform baseline framing | done |
| Methods 2.5 Explainability | surrogate target is the risk score | done |
| **Methods 2.6 Reliability** | **rewrite**: OOB predictions, symmetric pairwise error, locked split, imbalance-aware metrics | pending |
| **Results 3.1 (I)** | **grows**: add the logit audit and the Δw consistency test; add the ablation when Step 5 lands | pending |
| Results 3.2 (E) | update fidelity numbers to the OOB-target values | pending |
| **Results 3.3 (R)** | **shrinks**: one honest subsection, simplified table | pending |
| Table 2 | drop the significance column, add N, prevalence, in-profile c-index with CI | pending |
| Figure 4 | simplify to the accuracy-vs-coverage landscape; remove all 0.50 markers | pending |
| Discussion | drafted; needs the R subsection filled | partial |
| Conclusion | drafted, short | done |

Net effect: 3.1 grows, 3.3 shrinks, Table 2 and Figure 4 get simpler. Less total
work than the current draft, not more.

---

## What interpretability now carries

Interpretability is the paper's lead claim, so it needs to be the most rigorous
section. Three components, two already established:

1. **The weights are near-uniform.** $1.5$ pp spread around $1/8$ in BLCA,
   $0.95$ pp around $1/9$ in BRCA. Verified against the saved per-round npz for
   both cohorts; both Table 1 columns reproduce.
2. **It is not an ε artefact.** Every logged logit across all rounds and epochs
   in both cohorts: none negative, minimum $0.76$ against $\varepsilon = 0.01$.
   This rules out the obvious explanation and is a concrete, quantitative
   interpretability result.
3. **Whether the ordering is consistent, and whether it matters functionally.**
   Not yet run. `analysis/fusion_weight_stats.py` answers the first;
   the region ablation answers the second.

**Step 5's region ablation is now the highest-value remaining run.** It converts
"the weights are near-uniform" from an observation into a tested claim and is
what Reviewer 3 asked for directly. Priority above everything else outstanding.

---

## What the reliability section can honestly claim

The original claims are gone: Trust profiles at 0.92 to 0.96 were majority-class
baselines plus in-sample contamination. But the corrected analysis is not empty,
and the section should not read as a retraction.

### Holds

**Patient-level error is partially predictable in BRCA, and better from slide
morphology than from clinical variables.**

| BRCA arm | q50 | q70 | q80 | q90 |
|---|---|---|---|---|
| clinical | 0.579 | 0.612 | 0.580 | 0.520 |
| radiomic | **0.671** | **0.651** | **0.643** | **0.693** |

The radiomic arm is significant at every threshold (p = 0.0005 at three of four)
and beats the clinical arm at every threshold. Average precision at q90 is
$0.189$ against a $0.097$ baseline, close to double. Consistency across the
entire threshold sweep is the strongest evidence here: a spurious result would
not survive four different label definitions.

That is a real finding and a novel one. The failure modes of a WSI survival
model are, in BRCA, morphologically identifiable from the same slides the model
reads.

**The cohort asymmetry is itself informative.** No BLCA arm reaches significance
at any threshold. BLCA has $N = 379$ against BRCA's $1{,}060$, and the
evaluation half is $151$ against $424$, so power is the obvious candidate and
should be discussed as such rather than presented as a property of bladder
cancer.

### Does not hold

**Predicting the error label is not the same as identifying subgroups with
different survival performance, and only the first happens.**

Across 159 profiles, 3 have a 95% CI on the in-profile survival c-index that
excludes the cohort-wide value, against roughly 8 expected by chance with no
correction. Two of the three are the same rule at two thresholds.

This distinction is the section's methodological contribution, and it matters
beyond this paper. A MED3pa profile identifies where a confidence model
separates high- from low-error cases. It does not follow that the underlying
model measurably performs differently there, and in a censored survival setting
with small subgroup event counts the second claim is much harder to support than
the first. Reporting the in-profile c-index with a CI, which we do here, is what
makes the difference visible.

### Framing for the first-application claim

This is the first application of MED3pa outside its originating group and the
first in computational pathology. The result is that the confidence-prediction
step transfers to WSI survival data in the larger cohort, while the
profile-to-performance inference requires more care than the original setting
implies. Both halves are useful to report, and the second is a contribution to
how MED3pa outputs should be read rather than a criticism of the method.

---

## Statistical items: both RESOLVED, Results 3.3 is unblocked

**No feature-selection leakage.** Verified from the committed drivers: in both
radiomic scripts the discovery/evaluation split happens before feature
importance (line 593), and importance is fitted on `X[disc_mask]` only (line
600), with top-K indices then applied as fixed columns to both halves. The
clinical arms have no selection step. The 0.64 to 0.69 AUROCs stand.

**Multiple-testing correction applied.** Across the 16 AUROC tests, 5 survive
Bonferroni (alpha = 0.00313) and 6 survive Benjamini-Hochberg at FDR 0.05. **All
four BRCA radiomic arms survive Bonferroni at every label threshold**, which is
the most conservative correction available. No BLCA arm is significant even
uncorrected. Reported p of 0.0005 is the permutation floor, so those are upper
bounds.

**The 159-profile negative needs no correction**, since 3 observed against
roughly 8 expected by chance uncorrected means any correction can only reduce
the count. State it that way rather than applying one.

Detail in `STEP2_FINDINGS.md`, addendum section.

One item still open:
- **BRCA radiomic q90 rests on roughly 41 high-error evaluation patients**
  ($0.097 \times 424$). It survives Bonferroni, but the study's strongest AUROC
  resting on that few positives deserves a stability check or a stated caveat.

---

## Run order from here

1. ~~Resolve the top-K leakage question~~ **DONE, no leak.**
2. **Step 5a, region ablation.** Now the only real blocker. Blocks Results 3.1,
   which is the lead section. GPU. Highest priority.
3. ~~Step 3, fusion weight consistency test~~ **DONE**, run locally from the
   round npz already on the Mac. p = 5e-5 in both cohorts, both epoch choices;
   CIs exclude uniform for 7/8 BLCA and 8/9 BRCA regions.
4. **Step 5b, fusion parameterisation arms** (init at 0.0, softmax). Supporting
   robustness checks for 3.1.
5. Multiple-testing correction and the q90 stability check.
6. Steps 4a to 4d, the cheap analyses that close the remaining supervisor
   comments.

Text work that does not depend on any of the above, and can proceed in parallel:
Methods 2.6, the Discussion R subsection skeleton, Table 2 restructuring, and
Figure 4 simplification.
