# Draft: reframing the AIIM manuscript (2026-10-26)

**Status:** applied to Overleaf on 2026-10-27, approved by Juan; co-authors still to see it. Changes beyond this draft:
- The same reframing in the manuscript's own highlights block.
- Results 3.2 opening, the end of the planted-driver section, Future work (DPsurv cited) and two appendix captions.
- A trim of the Introduction so the answer is not stated three times.
- The `number` field dropped from the Zhou entry for elsarticle-num.

It compiles with 0 errors and no undefined citations; the PDF is 67 pages.

## What changes, and why

- **The central (I) result is reframed, not changed.** The paper's Shapley value uses out-of-bag concordance as its payoff, so it measures what carries *outcome signal*, not what the model *uses*. The result files already hold the label-free measure: exact per-patient Shapley values with the patient's own risk score as the payoff (`local_attribution_share`).
  - The weights agree with what the model uses in **both** cohorts.
  - What the model uses agrees with what carries outcome signal **only in BLCA**.
  - The new story answers the authors' question directly: compartment weights do make the model's use of tissue readable, but a readable weight is not a prognostic statement.
- **The performance half of the question gets an explicit answer:** separating compartments did not improve prediction.
- **The novelty claim is narrowed** to what is actually new, and HIPPO, Albuquerque et al. and Zhou et al. are cited.

### New numbers and their sources (all in `results/`)

| Run (file) | Weights vs model use, BLCA | BRCA | Model use vs concordance Shapley, BLCA | BRCA |
|---|---|---|---|---|
| Main (`shapley_{c}/region_shapley_{c}.json`) | +0.95 (p<0.001) | +0.73 (p=0.025) | +0.88 (p=0.004) | −0.27 (p=0.49) |
| Masked (`region_shapley_masked_{c}.json`) | +0.90 (p=0.002) | +0.92 (p=0.001) | +0.71 (p=0.047) | −0.15 (p=0.70) |
| Diffusion (`region_shapley_graph_{c}.json`) | +0.86 (p=0.007) | +0.78 (p=0.013) | +0.74 (p=0.037) | −0.18 (p=0.64) |
| Held-out sites (`region_shapley_sites_{c}.json`) | +0.60 (p=0.12) | +0.85 (p=0.004) | +0.50 (p=0.21) | −0.12 (p=0.77) |

Keys: `agreement_across_regions.fusion_weight_vs_local_share` and `.local_share_vs_shapley`. The definition is in `analysis/region_shapley.py`, lines 284–309.

**Last-epoch numbers checked:** the manuscript's +0.64 and −0.35 (lines 1093–1095) reproduce exactly from the committed round files in `results/shapley_last_{blca,brca}/` (keys `fusion_weights_last` and `phi_perf_last`).

---

## A. Abstract (replaces the current abstract; 246 words, counting each number as one word)

```latex
\noindent\textbf{Background.}
Splitting whole-slide images (WSIs) into tissue compartments, each fused with
one learned weight, promises better survival prediction and a readable
statement of which tissues a model uses. We tested both with an
Interpretability, Explainability and Reliability (IER) audit.

\noindent\textbf{Methods.}
On The Cancer Genome Atlas bladder (BLCA, $N=379$) and breast (BRCA,
$N=1{,}060$) cohorts we compared our compartment model with an attention model
(ABMIL) and clinical variables. (I) Exact Shapley values over compartments,
computed against each patient's risk score and against outcome, tested the
weights across bootstrap splits, held-out hospitals and planted drivers. (E) A
surrogate related the risk score to clinical and pathomic features. (R) MED3pa error profiling, adapted to censored outcomes,
assessed reliability.

\noindent\textbf{Results.}
Compartments did not improve prediction (concordance $0.622$ and $0.607$;
ABMIL $0.608$ and $0.626$) and added $0.007$ and $0.002$ to clinical variables. (I) The weights matched what the
model used in both cohorts ($\rho=+0.95$ and $+0.73$), but what it used carried
outcome signal only in BLCA ($\rho=+0.88$ against $-0.27$); in BRCA it relied
most on DCIS and necrosis, which added little concordance. Outcome-based Shapley values found a planted driver more
often than the weights; summed attention never did. (E) No surrogate feature
survived false-discovery-rate control against outcome. (R) Error was
predictable in BRCA (AUROC $0.671$), yet only $3$ of $159$ subgroups showed
different concordance.

\noindent\textbf{Conclusions.}
Compartment weights can show faithfully what a model uses without showing what
is prognostic. Reporting them as biology requires an outcome-based check, which
the IER audit provides.
```

**Title:** keep the current title. If you want the question visible, one option is:
*Do tissue-compartment weights explain whole-slide image survival models? An interpretability, explainability and reliability audit.*

## B. Highlights (replace `highlights.txt`; each at most 85 characters)

```
- Splitting slides into tissue compartments did not improve survival prediction
- Compartment weights matched what the model used, in bladder and breast cancer
- What the model used predicted outcome in bladder cancer, not in breast cancer
- With a known driver, Shapley values found it more often than weights or attention
- Predictable patient error rarely meant different model performance (3 of 159)
```

## C. Introduction

**C1. Replace lines 167–170** ("The architecture audited here is our own…") with:

```latex
This work began with a design question. Whole-slide images contain distinct
tissue compartments, and we built a survival model that encodes each
compartment separately and fuses them through one learned weight per
compartment. We expected two benefits: better prediction than a model that
pools all tiles, and a weight that states, without a post-hoc surrogate, which
tissue the model uses.
```

**C2. Replace lines 186–197**, from "What has not been examined, to our knowledge…" to "…showed different model performance.", with:

```latex
better on faithfulness~\cite{hense2024xmil}. At the level of named tissue
regions, adding and removing tissue in counterfactual slides has been used to
explain weakly supervised pathology models, including prognostic models in
breast cancer and melanoma, and was more informative than
attention~\cite{kaczmarzyk2024hippo}; planted confounders have been used to test
attention maps in pathology~\cite{albuquerque2024characterizing}, following
benchmarks that plant ground truth to test attribution
methods~\cite{zhou2022feature}. Two distinctions matter for a compartment-level
weight. It asserts something about tumour biology that a pathologist can agree
or disagree with, so it has to be tested against outcome as well as against
the model. And it can be faithful to the model without the tissue it points to
being prognostic. We therefore tested each learned per-compartment weight
twice: against what the model's own risk score depends on, and against what its
concordance depends on. In both cohorts the weights matched the first; only in
BLCA did the first match the second. We then tested this on held-out hospitals,
with removal made exactly equal to absence, on a standard attention model and on
outcomes whose driving compartment is known. The other two legs gave similar
cautions: the surrogate's leading features described the model's output rather
than survival, and subgroups where error was predictable almost never showed
different model performance.
```

(The first line repeats the end of the existing sentence that this passage continues.)

**C3. Replace lines 199–205** ("The contribution of this paper is the audit framework…") with:

```latex
The answer to the design question is therefore mixed, and the audit is what
established it (Table~\ref{tab:audit}). Separating compartments did not improve
prediction. It did make the model's use of tissue readable, but a readable
weight is not yet a prognostic statement. Each leg of the audit takes a quantity
the model reports and adds a check of whether it means what it appears to mean;
the last column of Table~\ref{tab:audit} records what each check changed for our
model. Two preliminary checks come first: we compare the model with a
clinical-variable model, with a standard attention model and with a model that
gives every patient the same prediction, and we check whether the tissue labels
that all three legs depend on are assigned reliably.
```

**C4. Contributions (lines 300–321).** Change "makes three contributions" to "makes four contributions". Add a new first item, and replace item 2:

```latex
\item \textbf{A tested answer to the compartment question.} Separating
compartments did not improve discrimination over a standard attention model or
add materially to clinical variables, its signal was spread across
compartments, and its weights were faithful to the model in both cohorts but
pointed to prognostic tissue in only one.
```

```latex
\item \textbf{A compartment-level test of faithfulness and of prognostic
relevance.} The faithfulness of attention has been examined for
patches~\cite{cai2025attrimil,hense2024xmil}, and Shapley-based explanations of
multimodal survival models attribute risk to learned
features~\cite{eijpe2026dimafx}. We treat the eight or nine named tissue
compartments as players and compute exact Shapley values with two payoffs: each
patient's risk score, which measures what the model uses, and out-of-bag
concordance, which measures what carries outcome signal. Comparing the model's
weights with both separates a faithful weight from a prognostic one. We check
the outcome-based test on outcomes driven by a chosen compartment, where the
answer is known, and repeat it on held-out hospitals and on a standard attention
model.
```

**C5. Replace lines 324–332**, from "what is new is the compartment-level test…" to "…known to drive the outcome.", with:

```latex
conformal bounds and MED3pa); what is new is their combination over a complete
partition of named compartments: exact Shapley values with a model-based and an
outcome-based payoff, compared with the model's learned weights and with
attention, and checked against planted compartment-level drivers, held-out
hospitals and retraining. Closest to our work, counterfactual tissue removal has
been used to explain pathology models~\cite{kaczmarzyk2024hippo}, attention has
been characterised by tissue or cell composition~\cite{chen2022pancancer},
survival scores have been built from tissue
composition~\cite{kather2019predicting}, and model-agnostic interpretability has
been proposed for multiple-instance learning at the level of
instances~\cite{early2022model}. To our knowledge none separates whether a
compartment-level weight is faithful to the model from whether the tissue it
points to is prognostic.
```

## D. Table 1, row (I), last column (lines 251–256)

```latex
Weights are reproducible ($p<10^{-4}$) and match what the model uses in both
cohorts ($\rho=+0.95$, $+0.73$); what it uses carries outcome signal in BLCA
($\rho=+0.88$) but not BRCA ($-0.27$). With a known driver, Shapley values found
it most often; summed attention never ranked it first. Signal is spread across
compartments; no single one clearly suffices. In BRCA the weights keep nearly the
same order on shuffled outcomes, so the model's use of tissue there does not
follow the outcome. (\S\ref{sec:res_interp}) \\
```

## E. Methods: Exact Shapley values (lines 660–663)

Replace "…we use it as the measure of how much the model relies on that compartment." with:

```latex
the other compartments; it measures how much the model's discrimination depends
on that compartment, which we call its outcome-based contribution. Because this
payoff uses the outcome, we also computed, for every patient, exact Shapley values
with that patient's risk score as the payoff, which uses no outcome; a
compartment's mean share of the absolute per-patient attribution measures how
much the model's output depends on it, which we call what the model uses.
Comparing the weights with this share tests whether they are faithful to the
model; comparing the share with the outcome-based value tests whether what the
model uses carries outcome signal.
```

## F. Results 3.2, "Exact Shapley values" (lines 1038–1064)

**F1. New subsection title:**

```latex
\subsubsection*{Exact Shapley values: the weights show what the model uses}
```

**F2.** In lines 1046–1049, change "the weights hold" to "the weights match the outcome-based ranking" and "the weights do not hold" to "they do not".

**F3.** Replace the sentence at lines 1058–1060, "Shapley values and the concordance lost on deletion agree… not between two measures of behaviour.", with:

```latex
Shapley values and the concordance lost on deletion agree closely in both
cohorts ($\rho = +1.00$ and $+0.87$), so the two outcome-based measures agree
with each other.
```

**F4. Insert after line 1064** ("…weak ($-0.35$, $p = 0.39$)."):

```latex
The weights are nonetheless faithful to the model in both cohorts. Against each
compartment's share of the per-patient attribution of the risk score, which uses
no outcome, they agree at $\rho = +0.95$ ($p < 0.001$) in BLCA and $+0.73$
($p = 0.025$) in BRCA, and between $+0.60$ and $+0.92$ with the diffusion, with
masked removal and on held-out hospitals. What differs between the cohorts is
whether the tissue the model uses carries outcome signal. The attribution share
agrees with the outcome-based Shapley value at $\rho = +0.88$ ($p = 0.004$) in
BLCA and $-0.27$ ($p = 0.49$) in BRCA, and between $+0.50$ and $+0.74$ in BLCA
against $-0.18$ to $-0.12$ in BRCA in the other three runs. In BRCA the model's
output depends most on DCIS ($18.2\%$ of the attribution) and necrosis
($12.6\%$), the two highest-weighted compartments, while adipose and muscle,
which carry the most concordance, receive $9.7\%$ and $10.5\%$
(Table~\ref{tab:shapley}). The disagreement in BRCA is therefore not between the
weights and the model, but between what the model uses and what predicts
outcome.
```

**F5. Table `tab:shapley`:** add a column "Model use (\%)" from `local_attribution_share` (mean, with the 95% interval over rounds):

| BLCA | Model use | BRCA | Model use |
|---|---|---|---|
| Invasive UC (tumour) | 14.7 [13.3, 16.0] | Necrosis or haemorrhage | 12.6 [11.5, 13.7] |
| Necrosis | 13.5 [11.7, 15.4] | DCIS | 18.2 [14.9, 21.5] |
| Inflammatory infiltrates | 13.9 [12.2, 15.5] | TILs | 10.4 [9.0, 11.9] |
| Perivesical adipose | 12.8 [12.0, 13.7] | Normal glands (TDLU) | 9.9 [8.6, 11.2] |
| Normal urothelium | 12.5 [11.0, 14.0] | Invasive carcinoma | 9.5 [8.5, 10.6] |
| Muscularis propria | 10.6 [9.4, 11.8] | Muscle | 10.5 [8.5, 12.5] |
| Lamina propria | 11.8 [10.6, 12.9] | Adipose | 9.7 [8.6, 10.7] |
| Blood vessels | 10.2 [8.7, 11.8] | Fibrous stroma | 9.4 [8.0, 10.9] |
| | | Blood vessels | 9.8 [7.9, 11.7] |

Rows are in the table's existing order (by weight). Add this to the caption:

```latex
Model use is each compartment's mean share of the absolute per-patient Shapley
attribution of the risk score, a payoff that uses no outcome.
```

**F6. Figure `fig:weight_vs_shapley` caption (lines 1110–1112):** replace "If the weights measured what the model relies on, points would rise from left to right." with:

```latex
If the weights tracked outcome-based contribution, points would rise from left
to right.
```

**Recommended, needs a figure-script change:** add panels (c, d) plotting weight against model use, which rises in both cohorts. The script is `visualization/make_weight_vs_shapley_figure.py`.

## G. Discussion

**G1. Replace lines 1503–1509** (subsection 4.1, first paragraph) with:

```latex
We asked whether separating tissue compartments improves survival prediction
and interpretability. It did not improve prediction. It did make the model's use
of tissue readable: the learned weights matched what the model's output depended
on in both cohorts. But in BRCA what the model used did not carry outcome
signal, and nothing in the weights showed which case applied. The audit is how
we found this, and it is the part of the work that transfers to other models.
The three legs answer different questions, and none substitutes for another.
```

**G2. Replace lines 1514–1521**, from "The weights were reproducible…" to "…not a property the architecture guarantees.", with:

```latex
The weights were reproducible, ordered and different from uniform in both
cohorts, and they were faithful: they matched each compartment's share of what
the model's risk score depended on in BLCA and BRCA alike. What they could not
show is whether that tissue is prognostic. In BLCA the tissue the model used
also carried its concordance; in BRCA it did not, on bootstrap splits, on
held-out hospitals and with removed compartments masked out of the attention.
Whether a faithful weight points to prognostic tissue is therefore an empirical
question that an outcome-based intervention has to answer for each model, not a
property the architecture guarantees.
```

**G3. Lines 1523–1530** ("A mismatch is in fact the expected case…"): keep the argument, and change "shows what the weights follow instead" to "shows why the model's use of tissue in BRCA is not prognostic". The model uses the rarest compartments, and it orders them the same way when the outcome is shuffled.

**G4. Lines 1534–1537:** replace "…risks a confident attribution to the wrong tissue, which is why we present the check rather than the architecture as this paper's result." with:

```latex
without the outcome-based check risks reading a faithful description of the
model as a statement about tumour biology.
```

## H. Conclusion (replaces lines 1769–1790)

```latex
We asked whether separating whole-slide images into tissue compartments improves
survival prediction and makes the model interpretable, and answered it with the
\textbf{IER} audit on TCGA-BLCA and TCGA-BRCA, for our own architecture and a
standard attention model. Separating compartments did not improve prediction:
the model performed close to the attention model, added $0.007$ and $0.002$
concordance to clinical variables, and spread its signal across compartments
rather than concentrating it in one.

\textbf{(I)} The per-compartment weights were faithful: in both cohorts they
matched what the model's risk score depended on. They were not, on their own,
prognostic statements. What the model used carried outcome signal in BLCA but
not in BRCA, on bootstrap splits, on held-out hospitals and with masked removal,
and nothing in the weights showed which case applied. On outcomes whose driving
compartment was known, outcome-based Shapley values recovered it more often than
the weights, deletion or gradient attributions, while the summed attention of a
standard model never ranked it first. \textbf{(E)} The surrogate explained the
model's risk score rather than the underlying biology: adipose morphology led it
in both cohorts, but no single feature survived false-discovery-rate control
against outcome. \textbf{(R)} A predictable error label did not identify a
subgroup of differential performance. Error was predictable from slide
morphology in BRCA, but only three of $159$ profiles showed a detectably
different in-profile concordance, and conformal bounds that held on average were
over-optimistic in advanced BRCA.
```

Keep the existing final paragraph (lines 1792–1795).

## I. Cover letter, second paragraph of findings

Replace "The central finding is that a learned compartment weight is not a measure of what the model relies on. Tested against exact Shapley values over all compartment coalitions, our model's weights tracked reliance in bladder cancer but not in breast cancer." with:

```latex
The central finding is that a learned compartment weight can describe faithfully
what a model uses without saying what is prognostic. Tested with exact Shapley
values over all compartment coalitions, our model's weights matched what its
risk score depended on in both bladder and breast cancer, but the tissue it used
carried outcome signal only in bladder cancer.
```

## J. New references (add to `references.bib`; all checked)

```bibtex
@misc{kaczmarzyk2024hippo,
  title  = {Explainable {AI} for computational pathology identifies model limitations and tissue biomarkers},
  author = {Kaczmarzyk, Jakub R. and Kim, Chanwoo and Gadgil, Soham and Savant, Deepika and Zhao, Zhen and Saltz, Joel H. and Lee, Su-In and Koo, Peter K.},
  year   = {2024},
  eprint = {2409.03080},
  archivePrefix = {arXiv},
  note   = {arXiv:2409.03080, version 3, December 2025}
}

@misc{albuquerque2024characterizing,
  title  = {Characterizing the Interpretability of Attention Maps in Digital Pathology},
  author = {Albuquerque, Tom{\'e} and Y{\"u}ce, Anil and Herrmann, Markus D. and Gomariz, Alvaro},
  year   = {2024},
  eprint = {2407.02484},
  archivePrefix = {arXiv}
}

@inproceedings{zhou2022feature,
  title     = {Do Feature Attribution Methods Correctly Attribute Features?},
  author    = {Zhou, Yilun and Booth, Serena and Ribeiro, Marco Tulio and Shah, Julie},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {36},
  number    = {9},
  pages     = {9623--9633},
  year      = {2022},
  doi       = {10.1609/aaai.v36i9.21196}
}
```

**Also fix line 1752:** "appears to be unexplored" is contradicted by DPsurv (Xing et al., arXiv:2510.00053, 2025), which aggregates component-wise relative risks. Cite it and soften the wording.

## Not in this draft (separate decisions)

- **ABMIL comparison:** the ABMIL and main-model concordances come from different runs. A paired difference on the shared splits would make "did not improve" airtight.
- **Planted-driver statistics:** the Wilcoxon tests run over 80 rounds but rest on 8 outcome draws. Test at the outcome level.
- **(R) leg:** report the AUROC of the risk score alone as a baseline for the 0.671.
- **Length:** about 3,000 words could move to supplementary material (conformal bounds, ablation, robustness repeats).
